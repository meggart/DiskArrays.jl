# Force disk any abstractarray into a different chunking pattern.
# This is useful in `zip` and other operations that can iterate
# over multiple arrays with different patterns.

struct RechunkedDiskArray{T,N,A<:AbstractArray{T,N},C} <: AbstractDiskArray{T,N}
    parent::A
    chunks::C
end

Base.parent(A::RechunkedDiskArray) = A.parent
Base.size(A::RechunkedDiskArray) = size(parent(A))
Base.getindex(A::RechunkedDiskArray, I...) = getindex(parent(A), I...)
Base.setindex!(A::RechunkedDiskArray, I...) = setindex!(parent(A), I...)

haschunks(::RechunkedDiskArray) = Chunked()
eachchunk(A::RechunkedDiskArray) = A.chunks
