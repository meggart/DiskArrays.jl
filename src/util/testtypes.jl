module TestTypes

import ..DiskArrays

export AccessCountDiskArray, ChunkedDiskArray, TestBackend, UnchunkedDiskArray, getindex_count,
    reset_test_backend, setindex_count, trueparent, getindex_log, setindex_log

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
AccessCountDiskArray(a; chunksize=size(a), batchstrategy=DiskArrays.ChunkRead(DiskArrays.NoStepRange(), 0.5)) =
    AccessCountDiskArray([], [], a, chunksize, batchstrategy)

Base.parent(a::AccessCountDiskArray) = a.parent
Base.size(a::AccessCountDiskArray) = size(parent(a))

# Apply the all in one macro rather than inheriting

DiskArrays.haschunks(a::AccessCountDiskArray) = DiskArrays.Chunked(a.batchstrategy)
DiskArrays.eachchunk(a::AccessCountDiskArray) = DiskArrays.GridChunks(a, a.chunksize)
function DiskArrays.readblock!(a::AccessCountDiskArray, aout, i::OrdinalRange...)
    ndims(a) == length(i) || error("Number of indices is not correct")
    foreach(i) do r
        isa(r, AbstractUnitRange) || DiskArrays.allow_steprange(a) || error("StepRange passed although trait is false")
    end
    # println("reading from indices ", join(string.(i)," "))
    push!(a.getindex_log, i)
    return aout .= parent(a)[i...]
end
function DiskArrays.writeblock!(a::AccessCountDiskArray, v, i::OrdinalRange...)
    ndims(a) == length(i) || error("Number of indices is not correct")
    foreach(i) do r
        isa(r, AbstractUnitRange) || DiskArrays.allow_steprange(a) || error("StepRange passed although trait is false")
    end
    # println("Writing to indices ", join(string.(i)," "))
    push!(a.setindex_log, i)
    return view(parent(a), i...) .= v
end

getindex_count(a::AccessCountDiskArray) = length(a.getindex_log)
setindex_count(a::AccessCountDiskArray) = length(a.setindex_log)
getindex_log(a::AccessCountDiskArray) = a.getindex_log
setindex_log(a::AccessCountDiskArray) = a.setindex_log
trueparent(a::AccessCountDiskArray) = parent(a)

getindex_count(a::DiskArrays.AbstractDiskArray) = getindex_count(parent(a))
setindex_count(a::DiskArrays.AbstractDiskArray) = setindex_count(parent(a))
getindex_log(a::DiskArrays.AbstractDiskArray) = getindex_log(parent(a))
setindex_log(a::DiskArrays.AbstractDiskArray) = setindex_log(parent(a))
function trueparent(a::DiskArrays.AbstractDiskArray) 
    if parent(a) === a
        a
    else
        trueparent(parent(a))
    end
end

trueparent(a::DiskArrays.PermutedDiskArray{T,N,perm,iperm}) where {T,N,perm,iperm} =
    permutedims(trueparent(parent(a)), perm)

"""
    ChunkedDiskArray(A; chunksize)
    
A generic `AbstractDiskArray` that can wrap any other `AbstractArray`, with custom `chunksize`.
"""
struct ChunkedDiskArray{T,N,A<:AbstractArray{T,N}} <: DiskArrays.AbstractDiskArray{T,N}
    parent::A
    chunksize::NTuple{N,Int}
end
ChunkedDiskArray(a; chunksize=size(a)) = ChunkedDiskArray(a, chunksize)

Base.parent(a::ChunkedDiskArray) = a.parent
Base.size(a::ChunkedDiskArray) = size(parent(a))

DiskArrays.haschunks(::ChunkedDiskArray) = DiskArrays.Chunked()
DiskArrays.eachchunk(a::ChunkedDiskArray) = DiskArrays.GridChunks(a, a.chunksize)
DiskArrays.readblock!(a::ChunkedDiskArray, aout, i::AbstractUnitRange...) = aout .= parent(a)[i...]
DiskArrays.writeblock!(a::ChunkedDiskArray, v, i::AbstractUnitRange...) = view(parent(a), i...) .= v

"""
    UnchunkedDiskArray(A)

A disk array without chunking, that can wrap any other `AbstractArray`.
"""
struct UnchunkedDiskArray{T,N,P<:AbstractArray{T,N}} <: DiskArrays.AbstractDiskArray{T,N}
    parent::P
end

Base.parent(a::UnchunkedDiskArray) = a.parent
Base.size(a::UnchunkedDiskArray) = size(parent(a))

DiskArrays.haschunks(::UnchunkedDiskArray) = DiskArrays.Unchunked()
function DiskArrays.readblock!(a::UnchunkedDiskArray, aout, i::AbstractUnitRange...)
    ndims(a) == length(i) || error("Number of indices is not correct")
    all(r -> isa(r, AbstractUnitRange), i) || error("Not all indices are unit ranges")
    return aout .= parent(a)[i...]
end

"""
    TestBackend

A `ComputeBackend` subtype for testing that backend dispatch works correctly.
Wraps the default implementation while counting how many times it is invoked.
"""
struct TestBackend <: DiskArrays.ComputeBackend
    call_count::Threads.Atomic{Int}
end

TestBackend() = TestBackend(Threads.Atomic{Int}(0))

reset_test_backend(tb::TestBackend) = (tb.call_count[]=0; tb)

function DiskArrays.diskarrays_sum_impl(f, a::DiskArrays.AbstractDiskArray, tb::TestBackend)
    Threads.atomic_add!(tb.call_count, 1)
    DiskArrays._diskarrays_sum_default(f, a)
end

end
