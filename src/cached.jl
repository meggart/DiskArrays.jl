
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
readblock!(A::CachedDiskArray, data, I...) = _readblock_cached(A, data, I...)
readblock!(A::CachedDiskArray, data, I::AbstractVector...) = _readblock_cached(A, data, I...)
# TODO we need to invalidate caches when we write
# writeblock!(A::CachedDiskArray, data, I...) = writeblock!(parent(A), data, I...)

haschunks(A::CachedDiskArray) = haschunks(parent(A))
eachchunk(A::CachedDiskArray) = eachchunk(parent(A))

function _readblock_cached(A, data, I...)
    if haskey(A.cache, I)
        data .= A.cache[I]
    else
        readblock!(parent(A), data, I...) 
        A.cache[I] = copy(data)
    end
    return data
end

function cached(A::AbstractArray)
    isdiskarray(A) || throw(ArgumentError("Array `$(typeof(A))` is not a disk array"))
    CachedDiskArray(A)
end
