
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
    CachedDiskArray(A, LRU{ChunkIndex{N,OffsetChunks},OffsetArray{T,N,Array{T,N}}}(; by, maxsize))
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
    data_offset = OffsetArray(data,map(i->first(i)-1,I)...)
    foreach(CartesianIndices(chunk_inds)) do ci
        chunkindex = ChunkIndex(ci,offset=true)
        chunk = get!(A.cache, chunkindex) do
            res = parent(A)[chunkindex]
            res
        end
        inner_indices = map(axes(chunk),axes(data_offset)) do ax1, ax2
            max(first(ax1),first(ax2)):min(last(ax1),last(ax2))
        end
        for ii in CartesianIndices(inner_indices)
            data_offset[ii] = chunk[ii]
        end
    end
end

"""
    cache(A::AbstractArray; maxsize=1000)

Wrap internal disk arrays with `CacheDiskArray`.

This function is intended to be extended by package that want to
re-wrap the disk array afterwards, such as YAXArrays.jl or Rasters.jl.
"""
cache(A::AbstractArray; maxsize=1000) = CachedDiskArray(A; maxsize)
