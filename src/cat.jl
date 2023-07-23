
"""
    ConcatDiskArray <: AbstractDiskArray

Joins multiple AbstractArrays or AbstractDiskArrays in lazy concatination.
"""
struct ConcatDiskArray{T,N,P} <: AbstractDiskArray{T,N}
    parents::P
    startinds::NTuple{N,Vector{Int}}
    size::NTuple{N,Int}
end
function ConcatDiskArray(arrays::AbstractArray{<:AbstractArray{T,N},M}) where {T,N,M}
    function othersize(x, id)
        (x[1:id-1]..., x[id+1:end]...)
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

    return ConcatDiskArray{T,D,typeof(arrays1)}(arrays1, startinds, sizes)
end
function ConcatDiskArray(arrays::AbstractArray)
    # Validate array eltype and dimensionality
    all(t -> t == eltype(first(arrays)), eltypes) || error("Arrays don't have the same element type")
    all(s -> length(s) == ndims(first(arrays)), sizes) || error("Arrays don't have the same dimensions")
    error("Should not be reached")
end

Base.size(a::ConcatDiskArray) = a.size

function readblock!(a::ConcatDiskArray, aout, inds::AbstractUnitRange...)
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
            (first(ar)+si[ii]-first(indstoread)):(last(ar)+si[ii]-first(indstoread))
        end
        aout[outer_range...] = a.parents[cI][array_range...]
    end
end

function writeblock!(a::ConcatDiskArray, aout, inds::AbstractUnitRange...)
    error("No method yet for writing into a ConcatDiskArray")
end

haschunks(::ConcatDiskArray) = Chunked()

function eachchunk(aconc::ConcatDiskArray{T,N}) where {T,N}
    s = size(aconc)
    oldchunks = map(eachchunk, aconc.parents)
    newchunks = ntuple(N) do i
        sliceinds = Base.setindex(ntuple(_ -> 1, N), :, i)
        v = map(c -> c.chunks[i], oldchunks[sliceinds...])
        init = RegularChunks(approx_chunksize(first(v)), 0, 0)
        reduce(mergechunks, v; init = init)
    end

    return GridChunks(newchunks...)
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
    IrregularChunks(chunksizes = filter(!iszero, [length.(a); length.(b)]))
end

function cat_disk(As::AbstractDiskArray...; dims::Int)
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
        function Base.cat(A1::$t, As::$t...; dims::Int)
            return cat_disk(A1, As...; dims)
        end
    end
end
