"""
    AbstractDiskArray <: AbstractArray

Abstract DiskArray type that can be inherited by Array-like data structures that
have a significant random access overhead and whose access pattern follows
n-dimensional (hyper)-rectangles.
"""
abstract type AbstractDiskArray{T,N} <: AbstractArray{T,N} end

"""
    readblock!(A::AbstractDiskArray, A_ret, r::AbstractUnitRange...)

The only function that should be implemented by a `AbstractDiskArray`. This function
"""
function readblock!() end

"""
    writeblock!(A::AbstractDiskArray, A_in, r::AbstractUnitRange...)

Function that should be implemented by a `AbstractDiskArray` if write operations
should be supported as well.
"""
function writeblock!() end


function allow_multi_chunk_access(a)
    false
end

# This is for filtering "true" batch getindex functions with vector indexing from
# excess singleton dimensions with values like [1] and 1:1
is_vector_arg(j::AbstractArray) = length(j) != 1
is_vector_arg(_) = false
is_vector_arg(::AbstractRange) = false

function getindex_disk(a, i...)
    checkscalar(i)
    i_multi = resolve_multiindex(a,i)
    inds, trans = interpret_indices_disk(a, i_multi)
    inds = map(maybe2range,inds)
    chunk_gaps = any(map(has_chunk_gap,approx_chunksize(eachchunk(a)),inds))
    sparse_index = any(map(is_sparse_index,inds))
    if sparse_index && (chunk_gaps || allow_multi_chunk_access(a))
        batchgetindex(a, i...)
    else
        data = Array{eltype(a)}(undef, map(length, inds)...)
        readblock!(a, data, inds...)
        #Transform the output to match shape of indices
        trans(data)
    end
end

function setindex_disk!(a::AbstractArray{T}, v::T, i...) where {T<:AbstractArray}
    checkscalar(i)
    return setindex_disk!(a, [v], i...)
end

function setindex_disk!(a::AbstractArray, v::AbstractArray, i...)
    checkscalar(i)
    if any(j -> isa(j, AbstractArray) && !isa(j, AbstractRange), i)
        batchsetindex!(a, v, i...)
    else
        inds, trans = interpret_indices_disk(a, i)
        data = reshape(v, map(length, inds))
        writeblock!(a, data, inds...)
        v
    end
end

"""
Function that translates a list of user-supplied indices into plain ranges and
integers for reading blocks of data. This function respects additional indexing
rules like omitting additional trailing indices.

The passed array handle A must implement methods for `Base.size` and `Base.ndims`
The function returns two values:

  1. a tuple whose length equals `ndims(A)` containing only unit
  ranges and integers. This contains the minimal "bounding box" of data that
  has to be read from disk.
  2. A callable object which transforms the hyperrectangle read from disk to
  the actual shape that represents the Base getindex behavior.
"""
function interpret_indices_disk(A, r::Tuple)
    throw(ArgumentError("Indices of type $(typeof(r)) are not yet supported"))
end

#Read the entire array and reshape to 1D in the end
function interpret_indices_disk(A, ::Tuple{Colon})
    return map(Base.OneTo, size(A)), Reshaper(prod(size(A)))
end

interpret_indices_disk(A, r::Tuple{<:CartesianIndex}) = interpret_indices_disk(A, r[1].I)

function interpret_indices_disk(A, r::Tuple{<:CartesianIndices})
    return interpret_indices_disk(A, r[1].indices)
end

extract_indices_and_dropdims(sa,r) = _convert_index((),(),1,sa,r)
_convert_index(i::Integer,s::Integer) = i:i
_convert_index(i::AbstractVector, s::Integer) = i
_convert_index(i::MultiIndex{<:Any,<:Any,D},s::Integer) where D = first(i.bb[D]):last(i.bb[D])
_convert_index(::Colon, s::Integer) = Base.OneTo(Int(s))
function extract_indices_and_dropdims(inds,intdims,inow,s,r)
    inew,


function interpret_indices_disk(
    A, r::NTuple{N,Union{Integer,AbstractVector,Colon}}
) where {N}
    if ndims(A) == N
        inds = map(_convert_index, r, size(A))
        resh = DimsDropper(findints(r))
        return inds, resh
    elseif ndims(A) < N
        n_add_dim = sum((ndims(A) + 1):N) do i
            first(r[i]) == 1 || throw(BoundsError(A, r))
            isa(r[i], AbstractArray)
        end
        _, rshort = commonlength(size(A), r)
        inds, resh1 = interpret_indices_disk(A, rshort)
        if n_add_dim > 0
            ladddim = ntuple(_ -> 1, n_add_dim)
            oldsize = result_size(inds, resh1)
            resh2 = transformstack(resh1, Reshaper((oldsize..., ladddim...)))
            inds, resh2
        else
            inds, resh1
        end
    else
        size(A, N + 1) == 1 || throw(BoundsError(A, r))
        return interpret_indices_disk(A, (r..., 1))
    end
end


function interpret_indices_disk(A::AbstractVector, r::NTuple{1,AbstractVector})
    inds = map(_convert_index, r, size(A))
    resh = DimsDropper(findints(r))
    return inds, resh
end

# function interpret_indices_disk(A, r::Tuple{<:AbstractArray{<:Bool}})
#   ba = r[1]
#   if ndims(A)==ndims(ba)
#     inds = getbb(ba)
#     resh = a -> a[view(ba,inds...)]
#     return inds, resh
#   elseif ndims(ba)==1
#     interpret_indices_disk(A,(reshape(ba,size(A)),))
#   else
#     throw(BoundsError(A, r))
#   end
# end

function interpret_indices_disk(A, r::NTuple{1,AbstractVector})
    lininds = first(r)
    cartinds = CartesianIndices(A)
    mi, ma = extrema(view(cartinds, lininds))
    inds = map((i1, i2) -> i1:i2, mi.I, ma.I)
    resh = a -> map(lininds) do ii
        a[cartinds[ii] - mi + oneunit(mi)]
    end
    return inds, resh
end

struct Reshaper{I}
    reshape_indices::I
end
(r::Reshaper)(a) = reshape(a, r.reshape_indices)
result_size(_, r::Reshaper) = r.reshape_indices
struct DimsDropper{D}
    d::D
end
(d::DimsDropper)(a) = length(d.d) == ndims(a) ? a[1] : dropdims(a; dims=d.d)
function result_size(inds, d::DimsDropper)
    return getindex.(Ref(inds), filter(!in(d.d), ntuple(identity, length(inds))))
end

struct TransformStack{S}
    s::S
end
transformstack(_::Union{Reshaper,DimsDropper,typeof(identity)}, s2::Reshaper) = s2
transformstack(s...) = TransformStack(filter(!=(identity), s))
(s::TransformStack)(a) = ∘(s.s...)(a)

maybe2range(i::AbstractRange) = i
function maybe2range(inds::T)::Union{T,StepRange{Int,Int},UnitRange{Int}} where T<:AbstractVector{Int}
    Base.has_offset_axes(inds) && throw(ArgumentError("Indexing with Offset Arrays is not allowed"))
    if length(inds) == 1 
        return only(inds):only(inds)
    end
    rstep = inds[2]-inds[1]
    for i in 3:length(inds)
        if inds[i] - inds[i-1] != rstep
            return inds
        end
    end
    if rstep == 1
        return first(inds):last(inds)
    else
        return first(inds):rstep:last(inds)
    end
end


#Some helper functions
"For two given tuples return a truncated version of both so they have common length"
commonlength(a, b) = _commonlength((first(a),), (first(b),), Base.tail(a), Base.tail(b))
commonlength(::Tuple{}, b) = (), ()
commonlength(a, ::Tuple{}) = (), ()
commonlength(a::Tuple{}, ::Tuple{}) = (), ()
function _commonlength(a1, b1, a, b)
    return _commonlength((a1..., first(a)), (b1..., first(b)), Base.tail(a), Base.tail(b))
end
_commonlength(a1, b1, ::Tuple{}, b) = (a1, b1)
_commonlength(a1, b1, a, ::Tuple{}) = (a1, b1)
_commonlength(a1, b1, a::Tuple{}, ::Tuple{}) = (a1, b1)

"Find the indices of elements containing integers in a Tuple"
findints(x) = _findints((), 1, x...)
_findints(c, i, x::Integer, rest...) = _findints((c..., i), i + 1, rest...)
_findints(c, i, x, rest...) = _findints(c, i + 1, rest...)
_findints(c, i) = c
#Normal indexing for a full subset of an array


include("chunks.jl")

macro implement_getindex(t)
    t = esc(t)
    quote
        Base.getindex(a::$t, i...) = getindex_disk(a, i...)

        function Base.getindex(a::$t, i::ChunkIndex)
            cs = eachchunk(a)
            inds = cs[i.I]
            return wrapchunk(i.chunktype, a[inds...], inds)
        end
        function DiskArrays.ChunkIndices(a::$t; offset=false)
            return ChunkIndices(
                Base.OneTo.(size(eachchunk(a))), offset ? OffsetChunks() : OneBasedChunks()
            )
        end
    end
end

macro implement_setindex(t)
    t = esc(t)
    quote
        Base.setindex!(a::$t, v::AbstractArray, i...) = setindex_disk!(a, v, i...)

        # Add an extra method if a single number is given
        function Base.setindex!(a::$t{<:Any,N}, v, i...) where {N}
            return Base.setindex!(a, fill(v, ntuple(i -> 1, N)...), i...)
        end

        function Base.setindex!(a::$t, v::AbstractArray, i::ChunkIndex)
            cs = eachchunk(a)
            inds = cs[i.I]
            return setindex_disk!(a, v, inds...)
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", X::AbstractDiskArray)
    return println(io, "Disk Array with size ", join(size(X), " x "))
end
function Base.show(io::IO, X::AbstractDiskArray)
    return println(io, "Disk Array with size ", join(size(X), " x "))
end
