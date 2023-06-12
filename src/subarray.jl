struct SubDiskArray{T,N,P,I,L} <: AbstractDiskArray{T,N}
    v::SubArray{T,N,P,I,L}
end

# Base methods
function Base.view(a::SubDiskArray, i...)
    return SubDiskArray(view(a.v, i...))
end
Base.view(a::SubDiskArray, i::CartesianIndices) = view(a, i.indices...)
Base.size(a::SubDiskArray) = size(a.v)
Base.parent(a::SubDiskArray) = a.v.parent

is_wrapper(::Type{<:SubDiskArray}) = true
parent_type(::Type{<:SubDiskArray{<:Any,<:Any,P}}) where P = P

_replace_colon(s, ::Colon) = Base.OneTo(s)
_replace_colon(s, r) = r

# Diskarrays.jl interface
function readblock!(a::SubDiskArray, aout, i::OrdinalRange...)
    pinds = parentindices(view(a.v, i...))
    inds, resh = interpret_indices_disk(parent(a.v), pinds)
    return readblock!(parent(a.v), reshape(aout, map(length, inds)...), inds...)
end
function writeblock!(a::SubDiskArray, v, i::OrdinalRange...)
    pinds = parentindices(view(a.v, i...))
    inds, resh = interpret_indices_disk(parent(a.v), pinds)
    return writeblock!(parent(a.v), reshape(v, map(length, inds)...), inds...)
end
eachchunk(a::SubDiskArray) = eachchunk_view(haschunks(a.v.parent), a.v)
function eachchunk_view(::Chunked, vv)
    pinds = parentindices(vv)
    iomit = findints(pinds)
    chunksparent = eachchunk(parent(vv))
    newchunks = [
        subsetchunks(chunksparent.chunks[i], pinds[i]) for
        i in 1:length(pinds) if !in(i, iomit)
    ]
    return GridChunks(newchunks...)
end
eachchunk_view(::Unchunked, a) = estimate_chunksize(a)
haschunks(a::SubDiskArray) = haschunks(parent(a.v))

# Implementaion macro

macro implement_subarray(t)
    t = esc(t)
    quote
        function Base.view(a::$t, i...)
            i2 = _replace_colon.(size(a), i)
            return SubDiskArray(SubArray(a, i2))
        end
        Base.view(a::$t, i::CartesianIndices) = view(a, i.indices...)
    end
end
