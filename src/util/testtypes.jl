module TestTypes

import ..DiskArrays

export AccessCountDiskArray, ChunkedDiskArray, UnchunkedDiskArray, getindex_count, setindex_count, trueparent,
  getindex_log, setindex_log

"""
    AccessCountDiskArray(A; chunksize)
    
An array that counts `getindex` and `setindex` calls, to debug
and optimise chunk access.

`getindex_count(A)` and `setindex_count(A)` can be used to check the
the counters.
"""
struct AccessCountDiskArray{T,N,A<:AbstractArray{T,N},RS} <: DiskArrays.AbstractDiskArray{T,N}
    getindex_log::Vector{Any}
    setindex_log::Vector{Any}
    parent::A
    chunksize::NTuple{N,Int}
    batchstrategy::RS
end
DiskArrays.batchstrategy(a::AccessCountDiskArray) = a.batchstrategy
AccessCountDiskArray(a; chunksize=size(a),batchstrategy=DiskArrays.ChunkRead(DiskArrays.NoStepRange(),0.5)) = 
  AccessCountDiskArray([], [], a, chunksize,batchstrategy)

Base.size(a::AccessCountDiskArray) = size(a.parent)

# Apply the all in one macro rather than inheriting
DiskArrays.@implement_diskarray_skip_zip AccessCountDiskArray

DiskArrays.haschunks(::AccessCountDiskArray) = DiskArrays.Chunked()
DiskArrays.eachchunk(a::AccessCountDiskArray) = DiskArrays.GridChunks(a, a.chunksize)
function DiskArrays.readblock!(a::AccessCountDiskArray, aout, i::OrdinalRange...)
    ndims(a) == length(i) || error("Number of indices is not correct")
    foreach(i) do r
        isa(r,AbstractUnitRange) || DiskArrays.allow_steprange(a) || error("StepRange passed although trait is false")
    end
    # println("reading from indices ", join(string.(i)," "))
    push!(a.getindex_log, i)
    return aout .= a.parent[i...]
end
function DiskArrays.writeblock!(a::AccessCountDiskArray, v, i::OrdinalRange...)
    ndims(a) == length(i) || error("Number of indices is not correct")
    foreach(i) do r  
        isa(r,AbstractUnitRange) || DiskArrays.allow_steprange(a) || error("StepRange passed although trait is false")
    end
    # println("Writing to indices ", join(string.(i)," "))
    push!(a.setindex_log, i)
    return view(a.parent, i...) .= v
end

getindex_count(a::AccessCountDiskArray) = length(a.getindex_log)
setindex_count(a::AccessCountDiskArray) = length(a.setindex_log)
getindex_log(a::AccessCountDiskArray) = a.getindex_log
setindex_log(a::AccessCountDiskArray) = a.setindex_log
trueparent(a::AccessCountDiskArray) = a.parent

getindex_count(a::DiskArrays.ReshapedDiskArray) = getindex_count(a.parent)
setindex_count(a::DiskArrays.ReshapedDiskArray) = setindex_count(a.parent)
getindex_log(a::DiskArrays.ReshapedDiskArray) = getindex_log(a.parent)
setindex_log(a::DiskArrays.ReshapedDiskArray) = setindex_log(a.parent)
trueparent(a::DiskArrays.ReshapedDiskArray) = trueparent(a.parent)

getindex_count(a::DiskArrays.PermutedDiskArray) = getindex_count(a.a.parent)
setindex_count(a::DiskArrays.PermutedDiskArray) = setindex_count(a.a.parent)
getindex_log(a::DiskArrays.PermutedDiskArray) = getindex_log(a.a.parent)
setindex_log(a::DiskArrays.PermutedDiskArray) = setindex_log(a.a.parent)
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
