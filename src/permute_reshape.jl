import Base: _throw_dmrs
import DiskArrays: splittuple
#Reshaping is really not trivial, because the access pattern would completely change for reshaped arrays,
#rectangles would not remain rectangles in the parent array. However, we can support the case where only
#singleton dimensions are added, later we could allow more special cases like joining two dimensions to one
struct ReshapedDiskArray{T,N,P<:AbstractArray,M} <: AbstractDiskArray{T,N}
    parent::P
    keepdim::NTuple{M,Int}
    newsize::NTuple{N,Int}
end
Base.size(r::ReshapedDiskArray) = r.newsize
haschunks(a::ReshapedDiskArray) = haschunks(a.parent)
eachchunk(a::ReshapedDiskArray{<:Any,N}) where N = map(eachchunk(a.parent)) do j
    r = toRanges(j)
    inow::Int = 0
    ntuple(N) do i
        if i in a.keepdim
            inow+=1
            r[inow]
        else
            1:1
        end
    end::NTuple{N,UnitRange{Int}}
end
function DiskArrays.readblock!(a::ReshapedDiskArray,aout,i...)
  inew = tuple_tuple_getindex(i,a.keepdim)
  DiskArrays.readblock!(a.parent,reshape(aout,map(length,inew)),inew...)
  nothing
end
tuple_tuple_getindex(t,i) = _ttgi((),t,i...)
_ttgi(o,t,i1,irest...) = _ttgi((o...,t[i1]),t,irest...)
_ttgi(o,t,i1) = (o...,t[i1])
function DiskArrays.writeblock!(a::ReshapedDiskArray,v,i...)
  inew = tuple_tuple_getindex(i,a.keepdim)
  DiskArrays.writeblock!(a.parent,reshape(v,map(length,inew)),inew...)
  nothing
end
function reshape_disk(parent, dims)
    n = length(parent)
    ndims(parent) > length(dims) && error("For DiskArrays, reshape is restricted to adding singleton dimensions")
    prod(dims) == n || _throw_dmrs(n, "size", dims)
    ipassed::Int=0
    s = size(parent)
    keepdim = map(s) do snow
        while true
            ipassed += 1
            d = dims[ipassed]
            if d>1
                d != snow && error("For DiskArrays, reshape is restricted to adding singleton dimensions")
                return ipassed
            end
        end
    end
    ReshapedDiskArray{eltype(parent),length(dims),typeof(parent),ndims(parent)}(parent,keepdim,dims)
end

import Base: _throw_dmrs
import DiskArrays: splittuple, toRanges
import Base.PermutedDimsArrays: genperm
struct PermutedDiskArray{T,N,P<:PermutedDimsArray} <: AbstractDiskArray{T,N}
    a::P
end
function permutedims_disk(a,perm)
  pd = PermutedDimsArray(a,perm)
  PermutedDiskArray{eltype(a),ndims(a),typeof(pd)}(pd)
end
Base.size(r::PermutedDiskArray) = size(r.a)
haschunks(a::PermutedDiskArray) = haschunks(a.a.parent)
_getperm(a::PermutedDiskArray) = _getperm(a.a)
_getperm(::PermutedDimsArray{<:Any,<:Any,perm}) where perm = perm
_getiperm(a::PermutedDiskArray) = _getiperm(a.a)
_getiperm(::PermutedDimsArray{<:Any,<:Any,<:Any,iperm}) where iperm = iperm
function eachchunk(a::PermutedDiskArray)
  cc = eachchunk(a.a.parent)
  perm = _getperm(a.a)
  GridChunks(a,genperm(cc.chunksize,perm),offset=genperm(cc.offset,perm))
end
function DiskArrays.readblock!(a::PermutedDiskArray,aout,i...)
  iperm = _getiperm(a)
  inew = genperm(i, iperm)
  DiskArrays.readblock!(a.a.parent,PermutedDimsArray(aout,iperm),inew...)
  nothing
end
function DiskArrays.writeblock!(a::PermutedDiskArray,v,i...)
  iperm = _getiperm(a)
  inew = genperm(i, iperm)
  DiskArrays.writeblock!(a.a.parent,PermutedDimsArray(v,iperm),inew...)
  nothing
end

macro implement_reshape(t)
  quote
    Base._reshape(parent::$t, dims::NTuple{N,Int}) where N =
      reshape_disk(parent, dims)
  end
end

macro implement_permutedims(t)
  quote
    Base.permutedims(parent::$t, perm) =
      permutedims_disk(parent, perm)
  end
end
