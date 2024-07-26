
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
struct CachedDiskArray{T,N,A<:AbstractArray{T,N},C} <: ChunkTiledDiskArray{T,N}
    parent::A
    cache::C
end
function CachedDiskArray(A::AbstractArray{T,N}; maxsize=1000) where {T,N}
    by(x) = sizeof(x) ÷ 1_000_000 # In Megabytes
    CachedDiskArray(A, LRU{ChunkIndex{N,OffsetChunks},OffsetArray{T,N,Array{T,N}}}(; by, maxsize))
end

Base.parent(A::CachedDiskArray) = A.parent
Base.size(A::CachedDiskArray) = size(parent(A))
# TODO we need to invalidate caches when we write
# writeblock!(A::CachedDiskArray, data, I...) = writeblock!(parent(A), data, I...)

haschunks(A::CachedDiskArray) = haschunks(parent(A))
eachchunk(A::CachedDiskArray) = eachchunk(parent(A))
function getchunk(A::CachedDiskArray, i::ChunkIndex)
    get!(A.cache, i) do
        inds = eachchunk(A)[i.I]
        chunk = parent(A)[inds...]
        wrapchunk(chunk, inds)
    end
end
Base.getindex(A::CachedDiskArray, i::ChunkIndex{N,OffsetChunks}) where {N} = getchunk(A, i)
Base.getindex(A::CachedDiskArray, i::ChunkIndex{N,OneBasedChunks}) where {N} = parent(getchunk(A, i))


"""
    cache(A::AbstractArray; maxsize=1000)

Wrap internal disk arrays with `CacheDiskArray`.

This function is intended to be extended by package that want to
re-wrap the disk array afterwards, such as YAXArrays.jl or Rasters.jl.
"""
cache(A::AbstractArray; maxsize=1000) = CachedDiskArray(A; maxsize)
