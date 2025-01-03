
"""
    ConcatDiskArray <: AbstractDiskArray

Joins multiple AbstractArrays or AbstractDiskArrays in lazy concatination.
"""
struct ConcatDiskArray{T,N,P,C,HC} <: AbstractDiskArray{T,N}
    parents::P
    startinds::NTuple{N,Vector{Int}}
    size::NTuple{N,Int}
    chunks::C
    haschunks::HC
end
function ConcatDiskArray(arrays::AbstractArray{<:AbstractArray{<:Any,N},M}) where {N,M}
    T = mapreduce(eltype,promote_type, init = eltype(first(arrays)),arrays)
        
    function othersize(x, id)
        return (x[1:(id - 1)]..., x[(id + 1):end]...)
    end
    if N > M
        newshape = (size(arrays)..., ntuple(_ -> 1, N - M)...)
        arrays1 = reshape(arrays, newshape)
        D = N
    elseif N < M
        arrays1 = map(arrays) do a
            newshape = (size(a)..., ntuple(_ -> 1, M - N)...)
            reshape(a, newshape)
        end
        D = M
    else
        arrays1 = arrays
        D = M
    end
    arraysizes = map(size, arrays1)
    si = ntuple(D) do id
        a = reduce(arraysizes; dims=id, init=ntuple(zero, D)) do i, j
            if all(iszero, i)
                j
            elseif othersize(i, id) == othersize(j, id)
                j
            else
                error("Dimension sizes don't match")
            end
        end
        I = ntuple(D) do i
            i == id ? Colon() : 1
        end
        ari = map(i -> i[id], arraysizes[I...])
        sl = sum(ari)
        r = cumsum(ari)
        pop!(pushfirst!(r, 0))
        r .+ 1, sl
    end

    startinds = map(first, si)
    sizes = map(last, si)

    chunks = concat_chunksize(D, arrays1)
    hc = haschunks(first(arrays1))

    return ConcatDiskArray{T,D,typeof(arrays1),typeof(chunks),typeof(hc)}(arrays1, startinds, sizes, chunks, hc)
end
function ConcatDiskArray(arrays::AbstractArray)
    # Validate array eltype and dimensionality
    all(a -> ndims(a) == ndims(first(arrays)), arrays) ||
        error("Arrays don't have the same dimensions")
    return error("Should not be reached")
end

Base.size(a::ConcatDiskArray) = a.size

function readblock!(a::ConcatDiskArray, aout, inds::AbstractUnitRange...)
    # Find affected blocks and indices in blocks
    _concat_diskarray_block_io(a, inds...) do outer_range, array_range, I
        aout[outer_range...] = a.parents[I][array_range...]
    end
end

function writeblock!(a::ConcatDiskArray, aout, inds::AbstractUnitRange...)
    _concat_diskarray_block_io(a, inds...) do outer_range, array_range, I
        a.parents[I][array_range...] = aout[outer_range...]
    end
end

function _concat_diskarray_block_io(f, a::ConcatDiskArray, inds...)
    # Find affected blocks and indices in blocks
    blockinds = map(inds, a.startinds, size(a.parents)) do i, si, s
        bi1 = max(searchsortedlast(si, first(i)), 1)
        bi2 = min(searchsortedfirst(si, last(i) + 1) - 1, s)
        bi1:bi2
    end
    map(CartesianIndices(blockinds)) do cI
        myar = a.parents[cI]
        mysize = size(myar)
        array_range = map(cI.I, a.startinds, mysize, inds) do ii, si, ms, indstoread
            max(first(indstoread) - si[ii] + 1, 1):min(last(indstoread) - si[ii] + 1, ms)
        end
        outer_range = map(cI.I, a.startinds, array_range, inds) do ii, si, ar, indstoread
            (first(ar) + si[ii] - first(indstoread)):(last(ar) + si[ii] - first(indstoread))
        end
        # aout[outer_range...] = a.parents[cI][array_range...]
        f(outer_range, array_range, cI)
    end
end

haschunks(c::ConcatDiskArray) = c.haschunks

function concat_chunksize(N, parents)
    oldchunks = map(eachchunk, parents)
    newchunks = ntuple(N) do i
        sliceinds = Base.setindex(ntuple(_ -> 1, N), :, i)
        v = map(c -> c.chunks[i], oldchunks[sliceinds...])
        init = RegularChunks(approx_chunksize(first(v)), 0, 0)
        reduce(mergechunks, v; init=init)
    end

    return GridChunks(newchunks...)
end

function eachchunk(aconc::ConcatDiskArray{T,N}) where {T,N}
    aconc.chunks
end

function mergechunks(a::RegularChunks, b::RegularChunks)
    if a.s == 0 || (a.cs == b.cs && length(last(a)) == a.cs)
        RegularChunks(a.cs, a.offset, a.s + b.s)
    else
        mergechunks_irregular(a, b)
    end
end

mergechunks(a::ChunkType, b::ChunkType) = mergechunks_irregular(a, b)
function mergechunks_irregular(a, b)
    return IrregularChunks(; chunksizes=filter(!iszero, [length.(a); length.(b)]))
end

function cat_disk(dims, As::AbstractArray...)
    if length(dims) == 1
    dims = only(dims)
    cat_disk(dims, As...)
    else
        throw(ArgumentError("Block concatenation is not yet implemented for DiskArrays."))
    end
end

function cat_disk(dims::Int, As::AbstractArray...)
    sz = map(ntuple(identity, dims)) do i
        i == dims ? length(As) : 1
    end
    cdas = reshape(collect(As), sz)
    return ConcatDiskArray(cdas)
end

# Implementation macro

macro implement_cat(t)
    t = esc(t)
    quote
        # Allow mixed lazy cat of other arrays and disk arrays to still be lazy
        # TODO this could be better. allowing non-AbstractDiskArray in
        # the macro makes this kind of impossible to avoid dispatch problems
        Base.cat(A1::$t, As::AbstractArray...; dims) = cat_disk(dims, A1, As...)
        function Base.cat(A1::AbstractArray, A2::$t, As::AbstractArray...; dims)
            return cat_disk(dims, A1, A2, As...)
        end
        function Base.cat(A1::$t, A2::$t, As::AbstractArray...; dims)
            return cat_disk(dims, A1, A2, As...)
        end
        function Base.vcat(
            A1::Union{$t{<:Any,1},$t{<:Any,2}}, As::Union{$t{<:Any,1},$t{<:Any,2}}...
        )
            return cat_disk(1, A1, As...)
        end
        function Base.hcat(
            A1::Union{$t{<:Any,1},$t{<:Any,2}}, As::Union{$t{<:Any,1},$t{<:Any,2}}...
        )
            return cat_disk(2, A1, As...)
        end
    end
end
