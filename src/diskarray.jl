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
    allow_multi_chunk_access(batchstrategy(a))
end

include("batchgetindex.jl")

"""
    resolve_indices(a, i)

Determines a list of tuples used to perform the read or write operations. The returned values are:

- `outsize` size of the output array
- `temp_size` size of the temp array passed to `readblock`
- `output_indices` indices for copying into the output array
- `temp_indices` indices for reading from temp array
- `data_indices` indices for reading from data array
"""
Base.@assume_effects :removable resolve_indices(a,i) = resolve_indices(a,i,batchstrategy(a))
Base.@assume_effects :removable resolve_indices(a, i, batch_strategy) = _resolve_indices(eachchunk(a).chunks, i, DiskIndex((),(),(),(),()), batch_strategy)
Base.@assume_effects :removable resolve_indices(a::AbstractVector, i::Tuple{AbstractVector{<:Integer}}, batch_strategy) = _resolve_indices(eachchunk(a).chunks, i, DiskIndex((), (), (), (), ()), batch_strategy)
resolve_indices(a, ::Tuple{Colon}, _) = DiskIndex((length(a),), size(a), (Colon(),), (Colon(),), map(s->1:s,size(a)),ReshapeOutArray())
resolve_indices(a, i::Tuple{<:CartesianIndex}, batch_strategy=NoBatch()) =
    resolve_indices(a, only(i).I, batch_strategy)
function resolve_indices(a, i::Tuple{<:AbstractVector{<:Integer}}, batchstrategy)
    cI = CartesianIndices(a)
    resolve_indices(a,(view(cI,only(i)),),batchstrategy)
end

Base.@assume_effects :foldable need_batch(a, i) = _need_batch(eachchunk(a).chunks, i, batchstrategy(a))

function _need_batch(cs, i, batchstrat)
    nb, csrem = need_batch_index(first(i), cs, batchstrat)
    nb ? true : _need_batch(csrem, Base.tail(i), batchstrat)
end
_need_batch(::Tuple{}, ::Tuple{}, _) = false
_need_batch(::Tuple{}, _, _) = false
_need_batch(_, ::Tuple{}, _) = false

need_batch_index(::Union{Integer,UnitRange,Colon}, cs, _) = false, Base.tail(cs)
need_batch_index(i::CartesianIndices{N}, cs, _) where N = false, last(splitcs(i, cs))
need_batch_index(::StepRange, cs, ::ChunkStrategy{CanStepRange}) = false, Base.tail(cs)
function need_batch_index(i, cs, batchstrat)
    csnow, csrem = splitcs(i, cs)
    allow_multi = allow_multi_chunk_access(batchstrat)
    density_thresh = density_threshold(batchstrat)
    nb = (allow_multi || has_chunk_gap(approx_chunksize.(csnow), i)) && is_sparse_index(i; density_threshold=density_thresh)
    nb, csrem
end


struct DiskIndex{N,M,A<:Tuple,B<:Tuple,C<:Tuple}
    output_size::NTuple{N,Int}
    temparray_size::NTuple{M,Int}
    output_indices::A
    temparray_indices::B
    data_indices::C
end
DiskIndex(output_size,temparray_size,output_indices,temparray_indices,data_indices) = 
    DiskIndex(output_size, temparray_size,output_indices,temparray_indices,data_indices)
@inline function merge_index(a::DiskIndex,b::DiskIndex)
    DiskIndex(
        (a.output_size...,b.output_size...),
        (a.temparray_size...,b.temparray_size...),
        (a.output_indices...,b.output_indices...),
        (a.temparray_indices...,b.temparray_indices...),
        (a.data_indices...,b.data_indices...),
    )
end

function _resolve_indices(cs, i, indices_pre::DiskIndex, nb)
    inow = first(i)
    indices_new, cs_rem = process_index(inow, cs, nb)
    _resolve_indices(cs_rem, Base.tail(i), merge_index(indices_pre,indices_new), nb)
end
_resolve_indices(::Tuple{}, ::Tuple{}, indices::DiskIndex, nb) = indices
#No dimension left in array, only singular indices allowed
function _resolve_indices(::Tuple{}, i, indices_pre::DiskIndex, nb)
    inow = first(i)
    (length(inow) == 1 && only(inow) == 1) || throw(ArgumentError("Trailing indices must be 1"))
    indices_new = DiskIndex(size(inow),(),size(inow),(),())
    indices = merge_index(indices_pre,indices_new)
    _resolve_indices((), Base.tail(i), indices, nb)
end
#Still dimensions left, but no indices available
function _resolve_indices(cs, ::Tuple{}, indices_pre::DiskIndex, nb) 
    csnow = first(cs)
    arraysize_from_chunksize(csnow) == 1 || throw(ArgumentError("Indices can only be omitted for trailing singleton dimensions"))
    indices_new = DiskIndex((),(1,),(),(1,),(1:1,))
    indices = merge_index(indices_pre,indices_new)
    _resolve_indices(Base.tail(cs), (), indices, nb)
end




#outsize, tempsize, outinds,tempinds,datainds,cs
process_index(i, cs, ::NoBatch) = process_index(i, cs)
process_index(inow::Integer, cs) = DiskIndex((), (1,), (), (1,), (inow:inow,)), Base.tail(cs)
function process_index(::Colon, cs)
    s = arraysize_from_chunksize(first(cs))
    DiskIndex((s,), (s,), (Colon(),), (Colon(),), (1:s,),), Base.tail(cs)
end
function process_index(i::AbstractUnitRange{<:Integer}, cs, ::NoBatch)
    DiskIndex((length(i),), (length(i),), (Colon(),), (Colon(),), (i,)), Base.tail(cs)
end
function process_index(i::AbstractUnitRange{<:Integer}, cs, ::ChunkRead)
    DiskIndex((length(i),), (length(i),), (Colon(),), (Colon(),), (i,)), Base.tail(cs)
end
function process_index(i::AbstractUnitRange{<:Integer}, cs, ::SubRanges)
    DiskIndex((length(i),), (length(i),), (Colon(),), (Colon(),), (i,)), Base.tail(cs)
end
function process_index(i::AbstractArray{<:Integer}, cs, ::NoBatch)
    indmin, indmax = extrema(i)
    DiskIndex(size(i), ((indmax - indmin + 1),), map(_->Colon(),size(i)), ((i .- (indmin - 1)),), (indmin:indmax,)), Base.tail(cs)
end
function process_index(i::AbstractArray{Bool,N}, cs, ::NoBatch) where {N}
    csnow, csrem = splitcs(i, cs)
    s = arraysize_from_chunksize.(csnow)
    cindmin, cindmax = extrema(view(CartesianIndices(s), i))
    indmin, indmax = cindmin.I, cindmax.I
    tempsize = indmax .- indmin .+ 1
    tempinds = view(i, range.(indmin, indmax)...)
    DiskIndex((sum(i),), tempsize, (Colon(),), (tempinds,), range.(indmin, indmax)), csrem
end
function process_index(i::AbstractArray{<:CartesianIndex{N}}, cs, ::NoBatch) where {N}
    csnow, csrem = splitcs(i, cs)
    s = arraysize_from_chunksize.(csnow)
    cindmin, cindmax = extrema(view(CartesianIndices(s), i))
    indmin, indmax = cindmin.I, cindmax.I
    tempsize = indmax .- indmin .+ 1
    tempoffset = cindmin - oneunit(cindmin)
    tempinds = i .- tempoffset
    outinds = map(_->Colon(),size(i))
    DiskIndex(size(i), tempsize, outinds, (tempinds,), range.(indmin, indmax)), csrem
end
function process_index(i::CartesianIndices{N}, cs, ::NoBatch) where {N}
    _, csrem = splitcs(i, cs)
    cols = map(_ -> Colon(), i.indices)
    DiskIndex(length.(i.indices), length.(i.indices), cols, cols, i.indices), csrem
end
splitcs(i::AbstractArray{<:CartesianIndex}, cs) = splitcs(first(i).I, (), cs)
splitcs(i::AbstractArray{Bool}, cs) = splitcs(size(i), (), cs)
splitcs(i::CartesianIndices, cs) = splitcs(i.indices, (), cs)
splitcs(_, cs) = (first(cs),), Base.tail(cs)
splitcs(si, csnow, csrem) = splitcs(Base.tail(si), (csnow..., first(csrem)), Base.tail(csrem))
splitcs(::Tuple{}, csnow, csrem) = (csnow, csrem)



function getindex_disk(a, i::Union{Integer,CartesianIndex}...)
    checkscalar(i)
    outputarray = Array{eltype(a)}(undef, map(_ -> 1, size(a))...)
    i = Base.to_indices(a, i)
    j = map(1:ndims(a)) do d
        d <= length(i) ? (i[d]:i[d]) : 1:1
    end
    readblock!(a, outputarray, j...)
    only(outputarray)
end

function create_outputarray(out, a, output_size)
    size(out) == output_size || throw(ArgumentError("Expected output array size of $output_size"))
    out
end
create_outputarray(::Nothing, a, output_size) = Array{eltype(a)}(undef, output_size...)

getindex_disk(a, i...) = getindex_disk!(nothing, a, i...)

function getindex_disk_batch!(out,a,i)
    indices = resolve_indices(a, i)
    moutput_indices = MRArray(indices.output_indices)
    mtemparray_indices = MRArray(indices.temparray_indices)
    mdata_indicess = MRArray(indices.data_indices)
    outputarray = create_outputarray(out, a, indices.output_size)
    temparray = Array{eltype(a)}(undef, indices.temparray_size...)
    for ii in eachindex(moutput_indices)
        data_indices = mdata_indicess[ii]
        output_indices = moutput_indices[ii]
        temparray_indices = mtemparray_indices[ii]
        vtemparray = maybeshrink(temparray, a, data_indices)
        readblock!(a, vtemparray, data_indices...)
        transfer_results!(outputarray, temparray, output_indices, temparray_indices)
    end
    outputarray
end

function getindex_disk_nobatch!(out,a,i)
    indices = resolve_indices(a, i, NoBatch(allow_steprange(a), 1.0))
    #@debug output_size, temparray_size, output_indices, temparray_indices, data_indices
    outputarray = create_outputarray(out, a, indices.output_size)
    temparray = Array{eltype(a)}(undef, indices.temparray_size...)
    readblock!(a, temparray, indices.data_indices...)
    transfer_results!(outputarray, temparray, indices.output_indices, indices.temparray_indices)
    outputarray
end

function getindex_disk!(out, a, i...)
    if need_batch(a, i)
        getindex_disk_batch!(out,a,i)
    else
        getindex_disk_nobatch!(out,a,i)
    end
end

function transfer_results!(outputarray, temparray, output_indices, temparray_indices)
    outputarray[output_indices...] = view(temparray, temparray_indices...)
    outputarray
end
function transfer_results!(o, t, oi::Tuple{Vararg{Int}}, ti::Tuple{Vararg{Int}})
    o[oi...] = t[ti...]
    o
end
function transfer_results_write!(v, temparray, output_indices, temparray_indices)
    temparray[temparray_indices...] = view(v, output_indices...)
    temparray
end
function transfer_results_write!(v, t, oi::Tuple{Vararg{Int}}, ti::Tuple{Vararg{Int}})
    t[ti...] = v[oi...]
    t
end

function setindex_disk!(a::AbstractArray{T}, v::T, i...) where {T<:AbstractArray}
    checkscalar(i)
    return setindex_disk!(a, [v], i...)
end

function maybeshrink(temparray, a, indices)
    if all(size(temparray) .== length.(indices))
        temparray
    else
        view(temparray, map(i->1:length(i),indices)...)
    end
end

function setindex_disk_batch!(a,v,i)
    batch_strategy = batchstrategy(a)
    indices = resolve_indices(a, i, batch_strategy)
    moutput_indices = MRArray(indices.output_indices)
    mtemparray_indices = MRArray(indices.temparray_indices)
    mdata_indicess = MRArray(indices.data_indices)
    temparray = Array{eltype(a)}(undef, indices.temparray_size...)
    for (output_indices, temparray_indices, data_indices) in zip(moutput_indices, mtemparray_indices, mdata_indicess)
        transfer_results_write!(v, temparray, output_indices, temparray_indices)
        vtemparray = maybeshrink(temparray, a, data_indices)
        writeblock!(a, vtemparray, data_indices...)
    end
end

function setindex_disk_nobatch!(a,v,i)
    indices = resolve_indices(a, i, NoBatch())
    temparray = Array{eltype(a)}(undef, indices.temparray_size...)
    transfer_results_write!(v, temparray, indices.output_indices, indices.temparray_indices)
    writeblock!(a, temparray, indices.data_indices...)
end

function setindex_disk!(a::AbstractArray, v::AbstractArray, i...)
    if need_batch(a, i)
        setindex_disk_batch!(a,v,i)
    else
        setindex_disk_nobatch!(a,v,i)
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
                map(s->1:s,size(eachchunk(a))), offset ? OffsetChunks() : OneBasedChunks()
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
