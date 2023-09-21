using OffsetArrays: OffsetArray

struct BlockedIndices{C<:GridChunks}
    c::C
end

# Base methods

Base.length(b::BlockedIndices) = prod(last.(last.(b.c.chunks)))
Base.IteratorEltype(::Type{<:BlockedIndices}) = Base.HasEltype()
Base.IteratorSize(::Type{<:BlockedIndices{<:GridChunks{N}}}) where {N} = Base.HasShape{N}()
Base.size(b::BlockedIndices)::NTuple{<:Any,Int} = map(last ∘ last, b.c.chunks)
Base.eltype(b::BlockedIndices) = CartesianIndex{ndims(b.c)}
function Base.iterate(a::BlockedIndices)
    chunkiter = Iterators.Stateful(a.c)
    ii = iterate(chunkiter)
    ii === nothing && return nothing
    innerinds = Iterators.Stateful(CartesianIndices(first(ii)))
    ind = iterate(innerinds)
    ind === nothing && return nothing
    return first(ind), (chunkiter, innerinds)
end
function Base.iterate(::BlockedIndices, i)
    chunkiter, innerinds = i
    r = iterate(innerinds)
    if r === nothing
        ii = iterate(chunkiter)
        ii === nothing && return nothing
        innerinds = Iterators.Stateful(CartesianIndices(first(ii)))
        r = iterate(innerinds)
        r === nothing && return nothing
        return first(r), (chunkiter, innerinds)
    else
        return first(r), (chunkiter, innerinds)
    end
    return first(r), (chunkiter, innerinds)
end

# Implementaion macros
@noinline function _iterate_disk(a::AbstractArray{T}, i::I) where {T,I<:Tuple{A,B,C}} where {A,B,C}
    datacur::A, bi::B, bstate::C = i
    (chunkiter, innerinds) = bstate
    cistateold = length(chunkiter)
    biter = iterate(bi, bstate)
    if biter === nothing
        return nothing
    else
        innernow, bstatenew = biter
        (chunkiter, innerinds) = bstatenew
        if length(chunkiter) !== cistateold
            curchunk = innerinds.itr.indices
            datacur = OffsetArray(a[curchunk...], innerinds.itr)
            return datacur[innernow]::T, (datacur, bi, bstatenew)::I
        else
            return datacur[innernow]::T, (datacur, bi, bstatenew)::I
        end
    end
end
@noinline function _iterate_disk(a)
    bi = BlockedIndices(eachchunk(a))
    it = iterate(bi)
    isnothing(it) && return nothing
    innernow, (chunkiter, innerinds) = it
    curchunk = innerinds.itr.indices
    datacur = OffsetArray(a[curchunk...], innerinds.itr)
    return datacur[innernow], (datacur, bi, (chunkiter, innerinds))
end

macro implement_iteration(t)
    t = esc(t)
    quote
        Base.eachindex(a::$t) = BlockedIndices(eachchunk(a))
        Base.iterate(a::$t) = _iterate_disk(a)
        Base.iterate(a::$t, i) = _iterate_disk(a, i)
    end
end
