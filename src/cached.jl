
# Force disk any abstractarray into a different chunking pattern.
# This is useful in `zip` and other operations that can iterate
# over multiple arrays with different patterns.

"""
    CachedDiskArray <: AbstractDiskArray

    CachedDiskArray(A::AbstractArray; maxsize=1000)

Wrap some disk array `A` with a caching mechanism that will 
keep chunks up to a total of `maxsize` megabytes, dropping
the least used chunks when `maxsize` is exceeded.
"""
struct CachedDiskArray{T,N,A<:AbstractArray{T,N},C} <: AbstractDiskArray{T,N}
    parent::A
    cache::C
end
function CachedDiskArray(A::AbstractArray{T,N}; maxsize=1000) where {T,N}
    by(x) = sizeof(x) ÷ 1_000_000 # In Megabytes
    CachedDiskArray(A, LRU{Tuple,Any}(; by, maxsize))
end

Base.parent(A::CachedDiskArray) = A.parent
Base.size(A::CachedDiskArray) = size(parent(A))
# These could be more efficient with memory in some cases, but this is simple
readblock!(A::CachedDiskArray, data, I...) = _readblock_cached!(A, data, I...)
readblock!(A::CachedDiskArray, data, I::AbstractVector...) = _readblock_cached!(A, data, I...)
# TODO we need to invalidate caches when we write
# writeblock!(A::CachedDiskArray, data, I...) = writeblock!(parent(A), data, I...)

haschunks(A::CachedDiskArray) = haschunks(parent(A))
eachchunk(A::CachedDiskArray) = eachchunk(parent(A))

function _readblock_cached!(A::CachedDiskArray{T,N}, data, I...) where {T,N}
    chunks = eachchunk(A)
    chunk_inds = findchunk.(chunks.chunks, I)
    needed_chunks = chunks[chunk_inds...]

    chunk_arrays = map(needed_chunks) do c
        if haskey(A.cache, c)
            A.cache[c]
        else
            chunk_data = Array{T,N}(undef, length.(c)...)
            A.cache[c] = readblock!(parent(A), chunk_data, c...)
        end
    end
    out = ConcatDiskArray(chunk_arrays)

    out_inds = map(I, first(needed_chunks)) do i, nc
        i .- first(nc) .+ 1 
    end

    data .= view(out, out_inds...)

    return data
end

"""
    cache(A::AbstractArray; maxsize=1000)

Wrap internal disk arrays with `CacheDiskArray`.

This function is intended to be extended by package that want to
re-wrap the disk array afterwards, such as YAXArrays.jl or Rasters.jl.
"""
cache(A::AbstractArray; maxsize=1000) = CachedDiskArray(A; maxsize)
