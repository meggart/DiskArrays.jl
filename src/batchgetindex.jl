#Define types for different strategies of reading sparse data

#Read bounding box and extract appropriate values
struct NoBatch{S<:Bool} 
    allow_steprange::Val{S}
end

#Split contiguous streaks into ranges and read the separately
struct SubRanges{S}
    allow_steprange::Val{S}
end

#Split dataset according to chunk and read chunk by chunk
struct ChunkRead{S} 
    allow_steprange::Val{S}
end

get_batchstrategy(_) = ChunkRead(Val(false))
allow_steprange(::NoBatch{S}) where S = S
allow_steprange(a::SubRanges{S}) where S = S
allow_steprange(a::ChunkRead{S}) where S = S


struct MultiRead{I}
    indexlist::I
end

struct MRArray{T,N,A} <: AbstractArray{T,N}
    a::A
end
tuple_type_length(::Type{<:NTuple{N}}) where N = N
function MRArray(a::NTuple{N,MultiRead}) where N 
    allvecs = getproperty.(a,:indexlist)
    # M = sum(tuple_type_length,eltype.(allvecs))
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
    largest_jump = foldl(ids,init=(0,first(ids))) do (largest,last),next
        largest = max(largest,next-last)
        (largest,next)
    end |> first
    largest_jump > first(cs)
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

function process_index(i, cs, ::ChunkRead)
    outsize, tempsize, outinds,tempinds,datainds,cs = process_index(i,cs, NoBatch())
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
    for (cI,a) in chunksdict
        dataind = extrema(first,a)
        tempind = first.(a) .- first(dataind) .+ 1
        push!(outinds, (map(last,a),))
        push!(datainds, (first(dataind):last(dataind),))
        push!(tempinds, (tempind,))
    end
    tempsize = maximum(length,tempinds)
    (length(i),), ((tempsize),), (MultiRead(outinds),), (MultiRead(tempinds),), (MultiRead(datainds),), Base.tail(cs)
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
                # Just set the step (hanst been set before)
                current_step = inds[iind+1] - inds[iind]
            else
                #Need to close the range
                push!(rangelist,inds[current_base]:inds[iind])
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
        rangelist, offsetlist = find_subranges_sorted(i,allow_steprange(s))
        datainds = MultiRead(rangelist)
        outinds = view.(Ref(i),)
    end
    (length(i),), ((tempsize),), (MultiRead(outinds),), (MultiRead(tempinds),), (MultiRead(datainds),), Base.tail(cs)
end

function process_index(i::AbstractArray{Bool,N}, cs, ::ChunkRead) where N
    process_index(findall(i),cs,ChunkRead())
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
# Define fallbacks for reading and writing sparse data
#= function _readblock!(A::AbstractArray, A_ret, r::AbstractVector...)
    if need_batch(A,r)
        # Fall back to batchgetindex to do the readblock
        A_ret .= batchgetindex(A, r...)
    else
        # Read the whole ranges in one go and subset what is required
        mi, ma = map(minimum, r), map(maximum, r)
        output_array_size = map((a, b) -> b - a + 1, mi, ma)
        A_temp = similar(A_ret, output_array_size)
        readranges = map(:, mi, ma)
        readblock!(A, A_temp, readranges...)
        subset = map(ir -> ir .- (minimum(ir) .- 1), r)
        A_ret .= view(A_temp, subset...)
    end
    return nothing
end

function _writeblock!(A::AbstractArray, A_ret, r::AbstractVector...)
    #Check how sparse the vectors are, we look at the largest stride in the inputs
    need_batch = map(approx_chunksize(eachchunk(A)), r) do cs, ids
        length(ids) == 1 && return false
        largest_jump = maximum(diff(ids))
        mi, ma = extrema(ids)
        return largest_jump > cs && length(ids) / (ma - mi) < 0.5
    end
    if any(need_batch)
        batchsetindex!(A, A_ret, r...)
    else
        mi, ma = map(minimum, r), map(maximum, r)
        A_temp = similar(A_ret, map((a, b) -> b - a + 1, mi, ma))
        A_temp[map(ir -> ir .- (minimum(ir) .- 1), r)...] = A_ret
        writeblock!(A, A_temp, map(:, mi, ma)...)
    end
    return nothing
end =#

macro implement_batchgetindex(t)
    t = esc(t)
    quote
        # Define fallbacks for reading and writing sparse data
        function DiskArrays.readblock!(A::$t, A_ret, r::AbstractVector...)
            return _readblock!(A, A_ret, r...)
        end

        function DiskArrays.writeblock!(A::$t, A_ret, r::AbstractVector...)
            return _writeblock!(A, A_ret, r...)
        end
    end
end
