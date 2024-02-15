#Define types for different strategies of reading sparse data
abstract type ChunkStrategy{S} end
#Read bounding box and extract appropriate values
struct CanStepRange end
struct NoStepRange end

@kwdef struct NoBatch{S} <: ChunkStrategy{S} 
    allow_steprange::S = NoStepRange()
    density_threshold::Float64 = 0.5
end
#Split contiguous streaks into ranges and read the separately
@kwdef struct SubRanges{S} <: ChunkStrategy{S} 
    allow_steprange::S = NoStepRange()
    density_threshold::Float64 = 0.5
end
#Split dataset according to chunk and read chunk by chunk
@kwdef struct ChunkRead{S} <: ChunkStrategy{S} 
    allow_steprange::S = NoStepRange()
    density_threshold::Float64 = 0.5
end
(to::Type{<:ChunkStrategy})(from::ChunkStrategy) = to(from.allow_steprange,from.density_threshold)
batchstrategy(x) = batchstrategy(haschunks(x))

allow_steprange(::ChunkStrategy{S}) where S = allow_steprange(S)
allow_steprange(::Type{CanStepRange}) = true
allow_steprange(::Type{NoStepRange}) = false
allow_steprange(::CanStepRange) = true
allow_steprange(::NoStepRange) = false
allow_steprange(a) = allow_steprange(batchstrategy(a))

allow_multi_chunk_access(::ChunkRead) = false
allow_multi_chunk_access(::SubRanges) = true

density_threshold(a) = density_threshold(batchstrategy(a))
density_threshold(a::ChunkStrategy) = a.density_threshold

struct MultiRead{I}
    indexlist::I
end

struct MRArray{T,N,A} <: AbstractArray{T,N}
    a::A
end
function MRArray(a::NTuple{N,MultiRead}) where N 
    allvecs = getproperty.(a,:indexlist)
    MRArray{Any,N,typeof(allvecs)}(allvecs)
end
mapflatten(f,x) = foldl((x,y)->(x...,f(y)),x, init=())
Base.size(a::MRArray) = mapflatten(length,a.a)
Base.IndexStyle(::Type{<:MRArray}) = IndexCartesian()
function merge_indices(ret,a,i) 
    merge_indices((ret...,first(a)[first(i)]...),Base.tail(a),Base.tail(i))
end
merge_indices(ret,::Tuple{},::Tuple{}) = ret
function Base.getindex(a::MRArray{<:Any,N},I::Vararg{Int, N}) where N
    merge_indices((),a.a,I)
end

function has_chunk_gap(cs,ids::AbstractVector{<:Integer})
    #Find largest jump in indices
    minind,maxind = extrema(ids)
    maxind - minind > first(cs)
end
#Return true for all multidimensional indices for now, could be optimised in the future
has_chunk_gap(cs,ids) = true

#Compute the number of possible indices in the hyperrectangle
span(v::AbstractVector{<:Integer}) = 1 -(-(extrema(v)...))
function span(v::AbstractVector{CartesianIndex{N}}) where N
    minind,maxind = extrema(v)
    prod((maxind-minind+oneunit(minind)).I)
end
function span(v::AbstractArray{Bool})
    minind,maxind = extrema(view(CartesianIndices(size(v)),v))
    prod((maxind-minind+oneunit(minind)).I)
end
#The number of indices to actually be read
numind(v::AbstractArray{Bool}) = sum(v)
numind(v::Union{AbstractVector{<:Integer},AbstractVector{<:CartesianIndex}})=length(v)

function is_sparse_index(ids; density_threshold = 0.5)
    indexdensity = numind(ids) / span(ids)
    return indexdensity < density_threshold
end

function process_index(i, cs, strategy::Union{ChunkRead,SubRanges})
    outsize, tempsize, outinds,tempinds,datainds,cs = process_index(i,cs, NoBatch(strategy))
    outsize, tempsize, (MultiRead([outinds]),), (MultiRead([tempinds]),), (MultiRead([datainds]),), cs
end


function process_index(i::AbstractVector{<:Integer}, cs, ::ChunkRead)
    csnow = first(cs)
    chunksdict = Dict{Int,Vector{Pair{Int,Int}}}()
    # Look for affected chunks
    for (outindex,dataindex) in enumerate(i)
        cI = findchunk(csnow,dataindex)
        a = get!(()->Pair{Int,Int}[],chunksdict,cI)
        push!(a,(dataindex=>outindex))
    end
    tempinds,datainds,outinds = Tuple{Vector{Int}}[], Tuple{UnitRange{Int}}[], Tuple{Vector{Int}}[]
    maxtempind = -1
    for (cI,a) in chunksdict
        dataind = extrema(first,a)
        tempind = first.(a) .- first(dataind) .+ 1
        push!(outinds, (map(last,a),))
        push!(datainds, (first(dataind):last(dataind),))
        push!(tempinds, (tempind,))
        maxtempind = max(maxtempind,maximum(tempind))
    end
    (length(i),), ((maxtempind),), (MultiRead(outinds),), (MultiRead(tempinds),), (MultiRead(datainds),), Base.tail(cs)
end

function find_subranges_sorted(inds,allow_steprange=false)
    t = allow_steprange ? Union{UnitRange{Int},StepRange{}} : UnitRange{Int}
    rangelist = t[]
    outputinds = UnitRange{Int}[]
    current_step = 0
    current_base = 1
    for iind in 1:length(inds)-1
        next_step = inds[iind+1] - inds[iind]
        if (next_step == current_step) || (next_step == 0)
            nothing
        else
            if !allow_steprange && next_step != 1
                #Need to close the range
                push!(rangelist,inds[current_base]:inds[iind])
                push!(outputinds,current_base:iind)
                current_base = iind + 1
                current_step = 0
                continue
            end
            if current_step === 0
                # Just set the step (hasnt been set before)
                current_step = inds[iind+1] - inds[iind]
            else
                #Need to close the range
                if current_step == 1
                    push!(rangelist,inds[current_base]:inds[iind])
                else
                    push!(rangelist,inds[current_base]:current_step:inds[iind])
                end
                push!(outputinds,current_base:iind)
                current_base = iind + 1
                current_step = 0
                continue
            end
        end
    end
    push!(rangelist,inds[current_base]:last(inds))
    push!(outputinds,current_base:length(inds))
    rangelist, outputinds
end

##Implement NCDatasets behavior of splitting list of indices into ranges
function process_index(i::AbstractVector{<:Integer}, cs, s::SubRanges)
    if issorted(i)
        rangelist, outputinds = find_subranges_sorted(i,allow_steprange(s))
        datainds = tuple.(rangelist)
        tempinds = map(rangelist,outputinds) do rl,oi
            v = view(i,oi)
            r = map(x->(x-first(v))÷step(rl)+1,v)
            (r,)
        end
        outinds = tuple.(outputinds)
        tempsize = maximum(length(rangelist))
        (length(i),), (tempsize,), (MultiRead(outinds),), (MultiRead(tempinds),), (MultiRead(datainds),), Base.tail(cs)
    else
        p = sortperm(i)
        i_sorted = view(i,p)
        rangelist, outputinds = find_subranges_sorted(i_sorted,allow_steprange(s))
        datainds = tuple.(rangelist)
        tempinds = map(rangelist,outputinds) do rl,oi
            v = view(i_sorted,oi)
            r = map(x->(x-first(v))÷step(rl)+1,v)
            (r,)
        end
        outinds = map(outputinds) do oi
            (view(p,oi),)
        end
        tempsize = maximum(length(rangelist))
        (length(i),), (tempsize,), (MultiRead(outinds),), (MultiRead(tempinds),), (MultiRead(datainds),), Base.tail(cs)
    end
end
function process_index(i::AbstractArray{Bool,N}, cs, cr::ChunkRead) where N
    process_index(findall(i),cs,cr)
end
function process_index(i::StepRange{<:Integer}, cs, ::ChunkStrategy{CanStepRange})
    (length(i),), (length(i),), (Colon(),), (Colon(),), (i,), Base.tail(cs)
end
function process_index(i::AbstractVector{<:CartesianIndex{N}}, cs, ::ChunkRead) where N
    csnow, csrem = splitcs(i,cs)
    chunksdict = Dict{CartesianIndex{N},Vector{Pair{CartesianIndex{N},Int}}}()
    # Look for affected chunks
    for (outindex,dataindex) in enumerate(i)
        cI = CartesianIndex(findchunk.(csnow,dataindex.I))
        a = get!(()->Pair{CartesianIndex{N},Int}[],chunksdict,cI)
        push!(a,(dataindex=>outindex))
    end
    tempinds,datainds,outinds = Tuple{Vector{CartesianIndex{N}}}[], NTuple{N,UnitRange{Int}}[], Tuple{Vector{Int}}[]
    tempsize = map(_->0,csnow)
    for (cI,a) in chunksdict
        datamin,datamax = extrema(first,a)
        aa = first.(a)
        tempind = aa .- datamin .+ oneunit(CartesianIndex{N})
        push!(outinds, tuple(map(last,a)))
        push!(datainds, range.(datamin.I,datamax.I))
        push!(tempinds, tuple(tempind))
        s = datamax.I .- datamin.I .+ 1
        tempsize = max.(s,tempsize)
    end
    (length(i),), tempsize, (MultiRead(outinds),), (MultiRead(tempinds),), (MultiRead(datainds),), csrem
end



