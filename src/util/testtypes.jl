module TestTypes

import ..DiskArrays

export AccessCountDiskArray, ChunkedDiskArray, UnchunkedDiskArray, getindex_count, setindex_count, trueparent

"""
    AccessCountDiskArray(A; chunksize)
    
An array that counts `getindex` and `setindex` calls, to debug
and optimise chunk access.

`getindex_count(A)` and `setindex_count(A)` can be used to check the
the counters.
"""
struct AccessCountDiskArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    getindex_count::Ref{Int}
    setindex_count::Ref{Int}
    parent::A
    chunksize::NTuple{N,Int}
end
AccessCountDiskArray(a; chunksize=size(a)) = AccessCountDiskArray(Ref(0), Ref(0), a, chunksize)

Base.size(a::AccessCountDiskArray) = size(a.parent)

# Apply the all in one macro rather than inheriting
DiskArrays.@implement_diskarray AccessCountDiskArray

DiskArrays.haschunks(::AccessCountDiskArray) = DiskArrays.Chunked()
DiskArrays.eachchunk(a::AccessCountDiskArray) = DiskArrays.GridChunks(a, a.chunksize)
function DiskArrays.readblock!(a::AccessCountDiskArray, aout, i::AbstractUnitRange...)
    ndims(a) == length(i) || error("Number of indices is not correct")
    all(r -> isa(r, AbstractUnitRange), i) || error("Not all indices are unit ranges")
    # println("reading from indices ", join(string.(i)," "))
    a.getindex_count[] += 1
    return aout .= a.parent[i...]
end
function DiskArrays.writeblock!(a::AccessCountDiskArray, v, i::AbstractUnitRange...)
    ndims(a) == length(i) || error("Number of indices is not correct")
    all(r -> isa(r, AbstractUnitRange), i) || error("Not all indices are unit ranges")
    # println("Writing to indices ", join(string.(i)," "))
    a.setindex_count[] += 1
    return view(a.parent, i...) .= v
end

getindex_count(a::AccessCountDiskArray) = a.getindex_count[]
setindex_count(a::AccessCountDiskArray) = a.setindex_count[]
trueparent(a::AccessCountDiskArray) = a.parent

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

"""
    ChunkedDiskArray(A; chunksize)
    
A generic `AbstractDiskArray` that can wrap any other `AbstractArray`, with custom `chunksize`.
"""
struct ChunkedDiskArray{T,N,A<:AbstractArray{T,N}} <: DiskArrays.AbstractDiskArray{T,N}
    parent::A
    chunksize::NTuple{N,Int}
end
ChunkedDiskArray(a; chunksize=size(a)) = ChunkedDiskArray(a, chunksize)

Base.size(a::ChunkedDiskArray) = size(a.parent)

DiskArrays.haschunks(::ChunkedDiskArray) = DiskArrays.Chunked()
DiskArrays.eachchunk(a::ChunkedDiskArray) = DiskArrays.GridChunks(a, a.chunksize)
DiskArrays.readblock!(a::ChunkedDiskArray, aout, i::AbstractUnitRange...) = aout .= a.parent[i...]
DiskArrays.writeblock!(a::ChunkedDiskArray, v, i::AbstractUnitRange...) = view(a.parent, i...) .= v

"""
    UnchunkedDiskArray(A)

A disk array without chunking, that can wrap any other `AbstractArray`.
"""
struct UnchunkedDiskArray{T,N,P<:AbstractArray{T,N}} <: DiskArrays.AbstractDiskArray{T,N}
    p::P
end

Base.size(a::UnchunkedDiskArray) = size(a.p)

DiskArrays.haschunks(::UnchunkedDiskArray) = DiskArrays.Unchunked()
function DiskArrays.readblock!(a::UnchunkedDiskArray, aout, i::AbstractUnitRange...)
    ndims(a) == length(i) || error("Number of indices is not correct")
    all(r -> isa(r, AbstractUnitRange), i) || error("Not all indices are unit ranges")
    return aout .= a.p[i...]
end

end
