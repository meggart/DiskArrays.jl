import Base.Broadcast: Broadcasted, AbstractArrayStyle, DefaultArrayStyle, flatten

# DiskArrays broadcast style

struct ChunkStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end

Base.BroadcastStyle(::ChunkStyle{N}, ::ChunkStyle{M}) where {N,M} = ChunkStyle{max(N, M)}()
function Base.BroadcastStyle(::ChunkStyle{N}, ::DefaultArrayStyle{M}) where {N,M}
    return ChunkStyle{max(N, M)}()
end
function Base.BroadcastStyle(::DefaultArrayStyle{M}, ::ChunkStyle{N}) where {N,M}
    return ChunkStyle{max(N, M)}()
end

struct BroadcastDiskArray{T,N,BC<:Broadcasted{<:ChunkStyle{N}}} <: AbstractDiskArray{T,N}
    bc::BC
end
function BroadcastDiskArray(bcf::B) where {B<:Broadcasted{<:ChunkStyle{N}}} where {N}
    ElType = Base.Broadcast.combine_eltypes(bcf.f, bcf.args)
    return BroadcastDiskArray{ElType,N,B}(bcf)
end

# Base methods

Base.size(bc::BroadcastDiskArray) = size(bc.bc)
function DiskArrays.readblock!(a::BroadcastDiskArray, aout, i::OrdinalRange...)
    argssub = map(arg -> subsetarg(arg, i), a.bc.args)
    return aout .= a.bc.f.(argssub...)
end
Base.broadcastable(bc::BroadcastDiskArray) = bc.bc
function Base.copy(bc::Broadcasted{ChunkStyle{N}}) where {N}
    return BroadcastDiskArray(flatten(bc))
end
Base.copy(a::BroadcastDiskArray) = copyto!(zeros(eltype(a), size(a)), a.bc)
function Base.copyto!(dest::AbstractArray, bc::Broadcasted{ChunkStyle{N}}) where {N}
    bcf = flatten(bc)
    gcd = common_chunks(size(bcf), dest, bcf.args...)
    foreach(gcd) do cnow
        # Possible optimization would be to use a LRU cache here, so that data has not
        # to be read twice in case of repeating indices
        argssub = map(i -> subsetarg(i, cnow), bcf.args)
        dest[cnow...] .= bcf.f.(argssub...)
    end
    return dest
end

# DiskArrays interface

haschunks(a::BroadcastDiskArray) = Chunked()
function eachchunk(a::BroadcastDiskArray)
    return common_chunks(size(a.bc), a.bc.args...)
end
function common_chunks(s, args...)
    N = length(s)
    chunkedars = filter(i -> haschunks(i) === Chunked(), collect(args))
    all(ar -> isa(eachchunk(ar), GridChunks), chunkedars) ||
        error("Currently only chunks of type GridChunks can be merged by broadcast")
    if isempty(chunkedars)
        totalsize = sum(sizeof ∘ eltype, args)
        return estimate_chunksize(s, totalsize)
    else
        allcs = eachchunk.(chunkedars)
        tt = ntuple(N) do n
            csnow = filter(allcs) do cs
                ndims(cs) >= n && first(first(cs.chunks[n])) < last(last(cs.chunks[n]))
            end
            isempty(csnow) && return RegularChunks(1, 0, s[n])
            cs = first(csnow).chunks[n]
            if all(s -> s.chunks[n] == cs, csnow)
                return cs
            else
                return merge_chunks(csnow, n)
            end
        end
        return GridChunks(tt...)
    end
end

# Utility methods

to_ranges(r::Tuple) = r
to_ranges(r::CartesianIndices) = r.indices

function merge_chunks(csnow, n)
    chpos = [1 for ch in csnow]
    chunk_offsets = Int[0]
    while true
        # Get the largest chunk end point
        cur_chunks = map(chpos, csnow) do i, ch
            ch.chunks[n][i]
        end
        chend = maximum(last.(cur_chunks))
        # @show chpos chend# cur_chunks 
        # Find the position where the end of a chunk matches the new chunk endpoint
        newchpos = map(chpos, csnow) do i, ch
            found = findnext(x -> (@show last(x); last(x) == chend), ch.chunks[n], i)
            found === nothing && error("Chunks do not align in dimension $n")
            found
        end
        # If we can't find this end point for all chunk lists, error
        firstcs = csnow[1].chunks[n]
        # Define the new combined chunk offset
        newchunkoffset = last(firstcs[newchpos[1]])
        # Update positions in each list of chunks
        chpos = newchpos .+ 1
        # If this is the last chunk, break
        chpos[1] >= length(firstcs) && break
        # Add our new offset
        push!(chunk_offsets, newchunkoffset)
    end
    push!(chunk_offsets, last(last(csnow[1].chunks[n])))
    return IrregularChunks(chunk_offsets)
end

subsetarg(x, a) = x
function subsetarg(x::AbstractArray, a)
    ashort = maybeonerange(size(x), a)
    return view(x, ashort...) # Maybe making a copy here would be faster, need to check...
end
repsingle(s, r) = s == 1 ? (1:1) : r
function maybeonerange(out, sizes, ranges)
    s1, sr = splittuple(sizes...)
    r1, rr = splittuple(ranges...)
    return maybeonerange((out..., repsingle(s1, r1)), sr, rr)
end
maybeonerange(out, ::Tuple{}, ranges) = out
maybeonerange(sizes, ranges) = maybeonerange((), sizes, ranges)
splittuple(x1, xr...) = x1, xr

# Implementation macro

macro implement_broadcast(t)
    quote
        # Broadcasting with a DiskArray on LHS
        function Base.copyto!(dest::$t, bc::Broadcasted{Nothing})
            foreach(eachchunk(dest)) do c
                ar = [bc[i] for i in CartesianIndices(to_ranges(c))]
                dest[to_ranges(c)...] = ar
            end
            return dest
        end
        Base.BroadcastStyle(T::Type{<:$t}) = ChunkStyle{ndims(T)}()
        function DiskArrays.subsetarg(x::$t, a)
            ashort = maybeonerange(size(x), a)
            return x[ashort...]
        end

        # This is a heavily allocating implementation, but working for all cases.
        # As performance optimization one might:
        # Allocate the array only once if all chunks have the same size
        # Use FillArrays, if the DiskArray accepts these
        function Base.fill!(dest::$t, value)
            foreach(eachchunk(dest)) do c
                ar = fill(value, length.(to_ranges(c)))
                dest[to_ranges(c)...] = ar
            end
            return dest
        end
    end
end
