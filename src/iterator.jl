using OffsetArrays: OffsetArray

struct BlockedIndices{C<:GridChunks}
    c::C
end

# Base methods

Base.length(b::BlockedIndices) = prod(last.(last.(b.c.chunks)))
Base.IteratorEltype(::Type{<:BlockedIndices}) = Base.HasEltype()
Base.IteratorSize(::Type{<:BlockedIndices{<:GridChunks{N}}}) where {N} = Base.HasShape{N}()
Base.size(b::BlockedIndices) = last.(last.(b.c.chunks))
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
    end
    return first(r), (chunkiter, innerinds)
end

# Implementaion macros

macro implement_iteration(t)
    quote
        Base.eachindex(a::$t) = BlockedIndices(eachchunk(a))
        function Base.iterate(a::$t)
            bi = BlockedIndices(eachchunk(a))
            it = iterate(bi)
            isnothing(it) && return nothing
            innernow, (chunkiter, innerinds) = it
            curchunk = innerinds.itr.indices
            datacur = OffsetArray(a[curchunk...], innerinds.itr)
            return datacur[innernow], (datacur, bi, (chunkiter, innerinds))
        end
        function Base.iterate(a::$t, i)
            datacur, bi, bstate = i
            (chunkiter, innerinds) = bstate
            cistateold = chunkiter.taken
            biter = iterate(bi, bstate)
            if biter === nothing
                return nothing
            end
            innernow, bstatenew = biter
            (chunkiter, innerinds) = bstatenew
            if chunkiter.taken !== cistateold
                curchunk = innerinds.itr.indices
                datacur = OffsetArray(a[curchunk...], innerinds.itr)
            end
            return datacur[innernow], (datacur, bi, bstatenew)
        end
    end
end
