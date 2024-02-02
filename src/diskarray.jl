"""
    AbstractDiskArray <: AbstractArray

Abstract DiskArray type that can be inherited by Array-like data structures that
have a significant random access overhead and whose access pattern follows
n-dimensional (hyper)-rectangles.
"""
abstract type AbstractDiskArray{T,N} <: AbstractArray{T,N} end

"""
    readblock!(A::AbstractDiskArray, A_ret, r::AbstractUnitRange...)

The only function that should be implemented by a `AbstractDiskArray`. This function
"""
function readblock!() end

"""
    writeblock!(A::AbstractDiskArray, A_in, r::AbstractUnitRange...)

Function that should be implemented by a `AbstractDiskArray` if write operations
should be supported as well.
"""
function writeblock!() end


function allow_multi_chunk_access(a)
    false
end

"""
    resolve_indices(a, i)

Determines a list of tuples used to perform the read or write operations. The returned values are:

- `outsize` size of the output array
- `temp_size` size of the temp array passed to `readblock`
- `output_indices` indices for copying into the output array
- `temp_indices` indices for reading from temp array
- `data_indices` indices for reading from data array
"""
Base.@assume_effects :foldable resolve_indices(a, i, nb=Val{false}()) = _resolve_indices(eachchunk(a).chunks,i,(),(),(),(),(),nb)
Base.@assume_effects :foldable resolve_indices(a::AbstractVector,i::Tuple{AbstractVector{<:Integer}},nb=Val{false}()) = _resolve_indices(eachchunk(a).chunks,i,(),(),(),(),(),nb)
Base.@assume_effects :foldable need_batch(a,i) = _need_batch(eachchunk(a).chunks,i,allow_multi_chunk_access(a))
function _need_batch(cs, i, am)
    nb, csrem = need_batch_index(first(i),cs,am)
    nb ? true : _need_batch(csrem,Base.tail(i),am)
end
_need_batch(::Tuple{},::Tuple{},_) = false
_need_batch(::Tuple{},_,_) = false
_need_batch(_,::Tuple{},_) = false
need_batch_index(::Union{Integer,UnitRange,Colon},cs,_) = false, Base.tail(cs)
function need_batch_index(i, cs,allow_multi)
    csnow,csrem = splitcs(i,cs)
    nb = (allow_multi || has_chunk_gap(approx_chunksize.(csnow),i)) && is_sparse_index(i)
    nb, csrem
end
function _resolve_indices(cs,i,output_size,temp_sizes,output_indices,temp_indices,data_indices,nb)
    inow = first(i)
    outsize, tempsize, outinds,tempinds,datainds,cs = process_index(inow, cs, nb)
    output_size = (output_size...,outsize...)
    output_indices = (output_indices...,outinds...)
    temp_sizes = (temp_sizes...,tempsize...)
    temp_indices = (temp_indices...,tempinds...)
    data_indices = (data_indices...,datainds...)
    _resolve_indices(cs,Base.tail(i),output_size,temp_sizes,output_indices,temp_indices, data_indices, nb)
end
_resolve_indices(::Tuple{},::Tuple{},output_size,temp_sizes,output_indices,temp_indices,data_indices) = output_size,temp_sizes,output_indices,temp_indices,data_indices
#No dimension left in array, only singular indices allowed
function _resolve_indices(::Tuple{},i,output_size,temp_sizes,output_indices,temp_indices,data_indices)
    inow = first(i)
    if inow isa Integer
        inow == 1 || throw(ArgumentError("Trailing indices must be 1"))
        _resolve_indices((),Base.tail(i),output_size,temp_sizes,output_indices,temp_indices,data_indices)
    elseif inow isa AbstractVector
         (length(inow)==1 && first(inow)==1) || throw(ArgumentError("Trailing indices must be 1"))
         output_size = (output_size...,1)
         output_indices = (output_indices...,1)
         _resolve_indices((),Base.tail(i),output_size,temp_sizes,output_indices,temp_indices,data_indices)
    else
        throw(ArgumentError("Trailing indices must be 1"))
    end
end
#Still dimensions left, but no indices available
function _resolve_indices(cs,::Tuple{},output_size,temp_sizes,output_indices,temp_indices,data_indices)
    csnow = first(cs)
    arraysize_from_chunksize(csnow) == 1 || throw(ArgumentError("Wrong indexing"))
    data_indices = (data_indices...,1:1)
    temp_sizes = (temp_sizes...,1)
    temp_indices = (temp_indices...,1)
    _resolve_indices(Base.tail(cs),(),output_size,temp_sizes,output_indices,temp_indices,data_indices)
end

resolve_indices(a, ::Tuple{Colon}) = (length(a),), size(a), (Colon(),), (Colon(),), Base.OneTo.(size(a))
resolve_indices(a, i::Tuple{<:CartesianIndex}) = resolve_indices(a, only(i).I)
resolve_indices(a, i::Tuple{<:CartesianIndices}) = resolve_indices(a, only(i).indices)

function resolve_indices(a, i::Tuple{<:AbstractVector{<:Integer}})
    inds = first(i)
    toread = view(CartesianIndices(size(a)),inds)
    cindmin,cindmax = extrema(toread)
    indmin,indmax = cindmin.I,cindmax.I
    tempsize = indmax .- indmin .+ 1
    tempoffset = cindmin - oneunit(cindmin)
    datainds = range.(indmin,indmax)
    tempinds = toread .- tempoffset
    length.(i),tempsize,(Colon(),),(tempinds,),datainds
end
#outsize, tempsize, outinds,tempinds,datainds,cs
process_index(i, cs, ::Val{false}) = process_index(i,cs)
function process_index(i, cs, ::Val{true})
    outsize, tempsize, outinds,tempinds,datainds,cs = process_index(i,cs)
    outsize, tempsize, MultiRead(outinds), MultiRead(tempinds), MultiRead(datainds)
end
process_index(inow::Integer, cs) = ((), 1, (), (1,),(inow:inow,), Base.tail(cs))
function process_index(::Colon, cs)
    s = arraysize_from_chunksize(first(cs))
    (s,), (s,), (Colon(),), (Colon(),), (Base.OneTo(s),), Base.tail(cs)
end
function process_index(i::AbstractUnitRange, cs)
    (length(i),), (length(i),), (Colon(),), (Colon(),), (i,), Base.tail(cs)
end
function process_index(i::AbstractVector{<:Integer}, cs, ::Val{false})
    indmin,indmax = extrema(i)
    (length(i),), ((indmax-indmin+1),), (Colon(),), ((i.-(indmin-1)),), (indmin:indmax,), Base.tail(cs) 
end
function process_index(i::AbstractArray{Bool,N}, cs, ::Val{false}) where N
    csnow, csrem = splitcs(i,cs)
    s = arraysize_from_chunksize.(csnow)
    cindmin,cindmax = extrema(view(CartesianIndices(s),i))
    indmin,indmax = cindmin.I,cindmax.I
    tempsize = indmax .- indmin .+ 1
    tempinds = view(i,range.(indmin,indmax)...)
    (sum(i),), tempsize, (Colon(),),(tempinds,), range.(indmin,indmax), csrem
end
function process_index(i::AbstractVector{<:CartesianIndex{N}}, cs, ::Val{false}) where N
    csnow, csrem = splitcs(i,cs)
    s = arraysize_from_chunksize.(csnow)
    cindmin,cindmax = extrema(view(CartesianIndices(s),i))
    indmin,indmax = cindmin.I,cindmax.I
    tempsize = indmax .- indmin .+ 1
    tempoffset = cindmin - oneunit(cindmin)
    tempinds = i .- tempoffset
    (length(i),), tempsize, (Colon(),), (tempinds,), range.(indmin,indmax), csrem
end
splitcs(i::AbstractVector{<:CartesianIndex},cs) = splitcs(first(i).I,(),cs)
splitcs(i::AbstractArray{Bool},cs) = splitcs(size(i),(),cs)
splitcs(_,cs) = (first(cs),), Base.tail(cs)
splitcs(si,csnow,csrem) = splitcs(Base.tail(si),(csnow...,first(csrem)),Base.tail(csrem))
splitcs(::Tuple{},csnow,csrem) = (csnow,csrem)


viewifnecessary(a,::Tuple{Vararg{Colon}}) = a
viewifnecessary(a,i) = view(a,i...)
maybe_unwrap(a,_) = a
maybe_unwrap(a,::Tuple{Vararg{<:Integer}}) = a[1]


function getindex_disk(a, i::Union{Integer,CartesianIndex}...)
    checkscalar(i)
    outputarray = Array{eltype(a)}(undef,map(_->1,size(a))...)
    i = Base.to_indices(a,i)
    j = map(1:ndims(a)) do d
        d<=length(i) ? (i[d]:i[d]) : 1:1
    end
    readblock!(a, outputarray, j...)
    only(outputarray)
end

getindex_disk(a, i...) = getindex_disk!(nothing, a, i...)
function create_outputarray(out,a,output_size)
    size(out) == output_size || throw(ArgumentError("Expected output array size of $output_size"))
    out
end
create_outputarray(::Nothing,a,output_size) = Array{eltype(a)}(undef, output_size...)
function getindex_disk!(out, a, i...)
    if need_batch(a,i)
        output_size, temparray_size, output_indices, temparray_indices, data_indices = resolve_indices(a,i,batch=true)
        println("Doing batch stuff")

    else
        output_size, temparray_size, output_indices, temparray_indices, data_indices = resolve_indices(a,i)
        #@debug output_size, temparray_size, output_indices, temparray_indices, data_indices
        outputarray = create_outputarray(out,a,output_size)
        temparray = Array{eltype(a)}(undef, temparray_size...)  
        readblock!(a, temparray, data_indices...)
        transfer_results!(outputarray, temparray, output_indices, temparray_indices)
    end
end

function transfer_results!(outputarray, temparray, output_indices, temparray_indices)
    outputarray[output_indices...] = view(temparray,temparray_indices...)
    outputarray
end
function transfer_results!(o,t,oi::Tuple{Vararg{Int}},ti::Tuple{Vararg{Int}}) 
    o[oi...] = t[ti...]
    o
end
function transfer_results_write!(v, temparray, output_indices, temparray_indices)
    temparray[temparray_indices...] = view(v,output_indices...) 
    temparray
end
function transfer_results_write!(v,t,oi::Tuple{Vararg{Int}},ti::Tuple{Vararg{Int}}) 
    t[ti...] = v[oi...]
    t
end

function setindex_disk!(a::AbstractArray{T}, v::T, i...) where {T<:AbstractArray}
    checkscalar(i)
    return setindex_disk!(a, [v], i...)
end

function setindex_disk!(a::AbstractArray, v::AbstractArray, i...)
    output_size, temparray_size, output_indices, temparray_indices, data_indices = resolve_indices(a,i)
    @debug output_size, temparray_size, output_indices, temparray_indices, data_indices
    temparray = Array{eltype(a)}(undef, temparray_size...)
    transfer_results_write!(v, temparray, output_indices, temparray_indices)
    writeblock!(a,temparray,data_indices...)
end


maybe2range(i::AbstractRange) = i
function maybe2range(inds::T)::Union{T,StepRange{Int,Int},UnitRange{Int}} where T<:AbstractVector{Int}
    Base.has_offset_axes(inds) && throw(ArgumentError("Indexing with Offset Arrays is not allowed"))
    if length(inds) == 1 
        return only(inds):only(inds)
    end
    rstep = inds[2]-inds[1]
    for i in 3:length(inds)
        if inds[i] - inds[i-1] != rstep
            return inds
        end
    end
    if rstep == 1
        return first(inds):last(inds)
    else
        return first(inds):rstep:last(inds)
    end
end

"Find the indices of elements containing integers in a Tuple"
findints(x) = _findints((), 1, x...)
_findints(c, i, x::Integer, rest...) = _findints((c..., i), i + 1, rest...)
_findints(c, i, x, rest...) = _findints(c, i + 1, rest...)
_findints(c, i) = c
#Normal indexing for a full subset of an array


include("chunks.jl")

macro implement_getindex(t)
    t = esc(t)
    quote
        Base.getindex(a::$t, i...) = getindex_disk(a, i...)

        function Base.getindex(a::$t, i::ChunkIndex)
            cs = eachchunk(a)
            inds = cs[i.I]
            return wrapchunk(i.chunktype, a[inds...], inds)
        end
        function DiskArrays.ChunkIndices(a::$t; offset=false)
            return ChunkIndices(
                Base.OneTo.(size(eachchunk(a))), offset ? OffsetChunks() : OneBasedChunks()
            )
        end
    end
end

macro implement_setindex(t)
    t = esc(t)
    quote
        Base.setindex!(a::$t, v::AbstractArray, i...) = setindex_disk!(a, v, i...)

        # Add an extra method if a single number is given
        function Base.setindex!(a::$t{<:Any,N}, v, i...) where {N}
            return Base.setindex!(a, fill(v, ntuple(i -> 1, N)...), i...)
        end

        function Base.setindex!(a::$t, v::AbstractArray, i::ChunkIndex)
            cs = eachchunk(a)
            inds = cs[i.I]
            return setindex_disk!(a, v, inds...)
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", X::AbstractDiskArray)
    return println(io, "Disk Array with size ", join(size(X), " x "))
end
function Base.show(io::IO, X::AbstractDiskArray)
    return println(io, "Disk Array with size ", join(size(X), " x "))
end
