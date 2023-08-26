import Base: _throw_dmrs
import Base.PermutedDimsArrays: genperm

# Reshaping is really not trivial, because the access pattern would completely change for reshaped arrays,
# rectangles would not remain rectangles in the parent array. However, we can support the case where only
# singleton dimensions are added, later we could allow more special cases like joining two dimensions to one

struct ReshapedDiskArray{T,N,P<:AbstractArray{T},M} <: AbstractDiskArray{T,N}
    parent::P
    keepdim::NTuple{M,Int}
    newsize::NTuple{N,Int}
end

# Base methods

Base.size(r::ReshapedDiskArray) = r.newsize

# DiskArrays interface

haschunks(a::ReshapedDiskArray) = haschunks(a.parent)
function eachchunk(a::ReshapedDiskArray{<:Any,N}) where {N}
    pchunks = eachchunk(a.parent)
    inow::Int = 0
    outchunks = ntuple(N) do idim
        if in(idim, a.keepdim)
            inow += 1
            pchunks.chunks[inow]
        else
            RegularChunks(1, 0, size(a, idim))
        end
    end
    return GridChunks(outchunks...)
end
function DiskArrays.readblock!(a::ReshapedDiskArray, aout, i::OrdinalRange...)
    inew = tuple_tuple_getindex(i, a.keepdim)
    DiskArrays.readblock!(a.parent, reshape(aout, map(length, inew)), inew...)
    return nothing
end
function DiskArrays.writeblock!(a::ReshapedDiskArray, v, i::OrdinalRange...)
    inew = tuple_tuple_getindex(i, a.keepdim)
    DiskArrays.writeblock!(a.parent, reshape(v, map(length, inew)), inew...)
    return nothing
end
function reshape_disk(parent, dims)
    n = length(parent)
    ndims(parent) > length(dims) &&
        error("For DiskArrays, reshape is restricted to adding singleton dimensions")
    prod(dims) == n || _throw_dmrs(n, "size", dims)
    ipassed::Int = 0
    keepdim = map(size(parent)) do s
        while true
            ipassed += 1
            d = dims[ipassed]
            if d > 1
                d != s && error(
                    "For DiskArrays, reshape is restricted to adding singleton dimensions",
                )
                return ipassed
            else
                # For existing trailing 1s
                d == s == 1 && return ipassed
            end
        end
    end
    return ReshapedDiskArray{eltype(parent),length(dims),typeof(parent),ndims(parent)}(
        parent, keepdim, dims
    )
end

tuple_tuple_getindex(t, i) = _ttgi((), t, i...)
_ttgi(o, t, i1, irest...) = _ttgi((o..., t[i1]), t, irest...)
_ttgi(o, t, i1) = (o..., t[i1])

# Implementaion macro

macro implement_reshape(t)
    t = esc(t)
    quote
        function Base._reshape(A::$t, dims::NTuple{N,Int}) where {N}
            return reshape_disk(A, dims)
        end
        # For ambiguity
        function Base._reshape(A::DiskArrays.AbstractDiskArray{<:Any,1}, dims::Tuple{Int64})
            reshape_disk(A, dims)
        end
    end
end
