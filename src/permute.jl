struct PermutedDiskArray{T,N,P<:PermutedDimsArray{T,N}} <: AbstractDiskArray{T,N}
    a::P
end

# Base methods

Base.size(r::PermutedDiskArray) = size(r.a)

# DiskArrays interface

function permutedims_disk(a, perm)
    pd = PermutedDimsArray(a, perm)
    return PermutedDiskArray{eltype(a),ndims(a),typeof(pd)}(pd)
end
haschunks(a::PermutedDiskArray) = haschunks(a.a.parent)
function eachchunk(a::PermutedDiskArray)
    cc = eachchunk(a.a.parent)
    perm = _getperm(a.a)
    return GridChunks(genperm(cc.chunks, perm)...)
end
function DiskArrays.readblock!(a::PermutedDiskArray, aout, i::OrdinalRange...)
    iperm = _getiperm(a)
    inew = genperm(i, iperm)
    DiskArrays.readblock!(a.a.parent, PermutedDimsArray(aout, iperm), inew...)
    return nothing
end
function DiskArrays.writeblock!(a::PermutedDiskArray, v, i::OrdinalRange...)
    iperm = _getiperm(a)
    inew = genperm(i, iperm)
    DiskArrays.writeblock!(a.a.parent, PermutedDimsArray(v, iperm), inew...)
    return nothing
end

_getperm(a::PermutedDiskArray) = _getperm(a.a)
_getperm(::PermutedDimsArray{<:Any,<:Any,perm}) where {perm} = perm
_getiperm(a::PermutedDiskArray) = _getiperm(a.a)
_getiperm(::PermutedDimsArray{<:Any,<:Any,<:Any,iperm}) where {iperm} = iperm

# Implementaion macros

macro implement_permutedims(t)
    quote
        Base.permutedims(parent::$t, perm) = permutedims_disk(parent, perm)
    end
end
