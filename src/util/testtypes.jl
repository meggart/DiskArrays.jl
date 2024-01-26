module TestTypes
export _DiskArray, UnchunkedDiskArray, getindex_count, setindex_count, trueparent
import ..DiskArrays

# Define a data structure that can be used for testing
struct _DiskArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    getindex_count::Ref{Int}
    setindex_count::Ref{Int}
    parent::A
    chunksize::NTuple{N,Int}
end
_DiskArray(a; chunksize=size(a)) = _DiskArray(Ref(0), Ref(0), a, chunksize)

# Apply the all in one macro rather than inheriting
DiskArrays.@implement_diskarray _DiskArray

Base.size(a::_DiskArray) = size(a.parent)
DiskArrays.haschunks(::_DiskArray) = DiskArrays.Chunked()
DiskArrays.eachchunk(a::_DiskArray) = DiskArrays.GridChunks(a, a.chunksize)
getindex_count(a::_DiskArray) = a.getindex_count[]
setindex_count(a::_DiskArray) = a.setindex_count[]
trueparent(a::_DiskArray) = a.parent
getindex_count(a::DiskArrays.ReshapedDiskArray) = getindex_count(a.parent)
setindex_count(a::DiskArrays.ReshapedDiskArray) = setindex_count(a.parent)
trueparent(a::DiskArrays.ReshapedDiskArray) = trueparent(a.parent)
getindex_count(a::DiskArrays.PermutedDiskArray) = getindex_count(a.a.parent)
setindex_count(a::DiskArrays.PermutedDiskArray) = setindex_count(a.a.parent)
function trueparent(
    a::DiskArrays.PermutedDiskArray{T,N,<:PermutedDimsArray{T,N,perm,iperm}}
) where {T,N,perm,iperm}
    return permutedims(trueparent(a.a.parent), perm)
end
function DiskArrays.readblock!(a::_DiskArray, aout, i::AbstractUnitRange...)
    ndims(a) == length(i) || error("Number of indices is not correct")
    all(r -> isa(r, AbstractUnitRange), i) || error("Not all indices are unit ranges")
    # println("reading from indices ", join(string.(i)," "))
    a.getindex_count[] += 1
    return aout .= a.parent[i...]
end
function DiskArrays.writeblock!(a::_DiskArray, v, i::AbstractUnitRange...)
    ndims(a) == length(i) || error("Number of indices is not correct")
    all(r -> isa(r, AbstractUnitRange), i) || error("Not all indices are unit ranges")
    # println("Writing to indices ", join(string.(i)," "))
    a.setindex_count[] += 1
    return view(a.parent, i...) .= v
end

struct UnchunkedDiskArray{T,N,P<:AbstractArray{T,N}} <: DiskArrays.AbstractDiskArray{T,N}
    p::P
end
DiskArrays.haschunks(::UnchunkedDiskArray) = DiskArrays.Unchunked()
Base.size(a::UnchunkedDiskArray) = size(a.p)
function DiskArrays.readblock!(a::UnchunkedDiskArray, aout, i::AbstractUnitRange...)
    ndims(a) == length(i) || error("Number of indices is not correct")
    all(r -> isa(r, AbstractUnitRange), i) || error("Not all indices are unit ranges")
    # println("reading from indices ", join(string.(i)," "))
    return aout .= a.p[i...]
end
end