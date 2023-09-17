"""
    eachchunk(a)

Returns an iterator with `CartesianIndices` elements that mark the index range of each chunk within an array.
"""
function eachchunk end

abstract type ChunkType <: AbstractVector{UnitRange} end

findchunk(a::ChunkType, i::AbstractUnitRange) = findchunk(a, first(i))::Int:findchunk(a, last(i))::Int
findchunk(a::ChunkType, ::Colon) = 1:length(a)

"""
    RegularChunks <: ChunkType

Defines chunking along a dimension where the chunks have constant size and a potential
offset for the first chunk. The last chunk is truncated to fit the array size. 
"""
struct RegularChunks <: ChunkType
    cs::Int
    offset::Int
    s::Int
end

# Base methods

function Base.getindex(r::RegularChunks, i::Int)
    @boundscheck checkbounds(r, i)
    return max((i - 1) * r.cs + 1 - r.offset, 1):min(i * r.cs - r.offset, r.s)
end
Base.size(r::RegularChunks, _) = div(r.s + r.offset - 1, r.cs) + 1
Base.size(r::RegularChunks) = (size(r, 1),)

# DiskArrays interface

function subsetchunks(r::RegularChunks, subs::AbstractUnitRange)
    snew = length(subs)
    newoffset = mod(first(subs) - 1 + r.offset, r.cs)
    r = RegularChunks(r.cs, newoffset, snew)
    # In case the new chunk is trivial and has length 1, we shorten the chunk size
    if length(r) == 1
        r = RegularChunks(snew, 0, snew)
    end
    return r
end

function subsetchunks(r::RegularChunks, subs::AbstractRange)
    if rem(r.cs, step(subs)) == 0
        newcs = r.cs ÷ abs(step(subs))
        if step(subs) > 0
            newoffset = mod(first(subs) - 1 + r.offset, r.cs) ÷ step(subs)
            return RegularChunks(newcs, newoffset, length(subs))
        elseif step(subs) < 0
            r2 = subsetchunks(r, last(subs):first(subs))::ChunkType
            newoffset = (r.cs - length(last(r2))) ÷ (-step(subs))
            return RegularChunks(newcs, newoffset, length(subs))
        end
    else
        return subsetchunks_fallback(r, subs)
    end
end
findchunk(r::RegularChunks, i::Int) = div(i + r.offset - 1, r.cs) + 1

subsetchunks(r, subs) = subsetchunks_fallback(r, subs)

approx_chunksize(r::RegularChunks) = r.cs
grid_offset(r::RegularChunks) = r.offset
max_chunksize(r::RegularChunks) = r.cs

"""
    IrregularChunks <: ChunkType

Defines chunks along a dimension where chunk sizes are not constant but arbitrary
"""
struct IrregularChunks <: ChunkType
    offsets::Vector{Int}
end

"""
    IrregularChunks(; chunksizes)

Returns an IrregularChunks object for the given list of chunk sizes
"""
function IrregularChunks(; chunksizes)
    offs = pushfirst!(cumsum(chunksizes), 0)
    # push!(offs, last(offs)+1)
    return IrregularChunks(offs)
end

function Base.getindex(r::IrregularChunks, i::Int)
    @boundscheck checkbounds(r, i)
    return (r.offsets[i] + 1):r.offsets[i + 1]
end
Base.size(r::IrregularChunks) = (length(r.offsets) - 1,)
function subsetchunks(r::IrregularChunks, subs::UnitRange)
    c1 = findchunk(r, first(subs))
    c2 = findchunk(r, last(subs))
    offsnew = r.offsets[c1:(c2 + 1)]
    firstoffset = first(subs) - r.offsets[c1] - 1
    offsnew[end] = last(subs)
    offsnew[2:end] .= offsnew[2:end] .- firstoffset
    offsnew .= offsnew .- first(offsnew)
    return IrregularChunks(offsnew)
end
findchunk(r::IrregularChunks, i::Int) = searchsortedfirst(r.offsets, i) - 1
function approx_chunksize(r::IrregularChunks)
    return round(Int, sum(diff(r.offsets)) / (length(r.offsets) - 1))
end
grid_offset(r::IrregularChunks) = 0
max_chunksize(r::IrregularChunks) = maximum(diff(r.offsets))

struct GridChunks{N,C<:Tuple{Vararg{<:ChunkType,N}}} <: AbstractArray{NTuple{N,UnitRange{Int64}},N}
    chunks::C
end
GridChunks(ct::ChunkType...) = GridChunks(ct)
GridChunks(a, chunksize; offset=(_ -> 0).(size(a))) = GridChunks(size(a), chunksize; offset)
function GridChunks(a::Tuple, chunksize; offset=(_ -> 0).(a))
    gcs = map(a, chunksize, offset) do s, cs, of
        RegularChunks(cs, of, s)
    end
    return GridChunks(gcs)
end

# Base methods

function Base.getindex(g::GridChunks{N}, i::Vararg{Int,N}) where {N}
    @boundscheck checkbounds(g, i...)
    return getindex.(g.chunks, i)
end
Base.size(g::GridChunks) = length.(g.chunks)

Base.:(==)(g1::GridChunks, g2::GridChunks) = g1.chunks == g2.chunks

function subsetchunks_fallback(r, subs)
    #This is a fallback method that should work for Regular and Irregular chunks r 
    #Assuming the desired subset is sorted
    # We simply compute the chunk for every element in subs and collect everything together 
    #again in either a Regular or IrregularChunk
    rev = if issorted(subs)
        false
    elseif issorted(subs; rev=true)
        true
    else
        throw(ArgumentError("Can only subset chunks for sorted indices"))
    end
    cs = zeros(Int, length(r))
    for i in subs
        cs[findchunk(r, i)] += 1
    end
    # Find first and last chunk where elements are extracted
    i1 = findfirst(!iszero, cs)
    i2 = findlast(!iszero, cs)
    chunks = cs[i1:i2]
    if rev
        reverse!(chunks)
    end
    chunktype_from_chunksizes(chunks)
end

"""
    chunktype_from_chunksizes(chunks)

Utility function that constructs either a `RegularChunks` or an
`IrregularChunks` object based on a vector of chunk sizes given as worted Integers. Wherever
possible it will try to create a regular chunks object.  
"""
function chunktype_from_chunksizes(chunks)
    if length(chunks) == 1
        #only a single chunk is affected
        return RegularChunks(chunks[1], 0, chunks[1])
    elseif length(chunks) == 2
        #Two affected chunks
        chunksize = max(chunks[1], chunks[2])
        return RegularChunks(chunksize, chunksize - chunks[1], sum(chunks))
    elseif all(==(chunks[2]), view(chunks, (2):(length(chunks)-1))) &&
        chunks[end] <= chunks[2] &&
        chunks[1] <= chunks[2]
        #All chunks have the same size, only first and last chunk can be shorter
        chunksize = chunks[2]
        return RegularChunks(chunksize, chunksize - chunks[1], sum(chunks))
    else
        #Chunks are Irregular
        return IrregularChunks(; chunksizes=filter(!iszero, chunks))
    end
end

# DiskArrays interface

"""
    approx_chunksize(g::GridChunks)

Returns the aproximate chunk size of the grid. For the dimension with regular chunks, this will be the exact chunk size
while for dimensions with irregular chunks this is the average chunks size. Useful for downstream applications that want to
distribute computations and want to know about chunk sizes. 
"""
approx_chunksize(g::GridChunks) = approx_chunksize.(g.chunks)

"""
    grid_offset(g::GridChunks)

Returns the offset of the grid for the first chunks. Expect this value to be non-zero for views into regular-gridded
arrays. Useful for downstream applications that want to distribute computations and want to know about chunk sizes. 
"""
grid_offset(g::GridChunks) = grid_offset.(g.chunks)

"""
    max_chunksize(g::GridChunks)

Returns the maximum chunk size of an array for each dimension. Useful for pre-allocating arrays to make sure they can hold
a chunk of data. 
"""
max_chunksize(g::GridChunks) = max_chunksize.(g.chunks)

# Define the approx default maximum chunk size (in MB)
"The target chunk size for processing for unchunked arrays in MB, defaults to 100MB"
const default_chunk_size = Ref(100)

"""
A fallback element size for arrays to determine a where elements have unknown
size like strings. Defaults to 100MB
"""
const fallback_element_size = Ref(100)

# Here we implement a fallback chunking for a DiskArray although this should normally
# be over-ridden by the package that implements the interface

function eachchunk(a::AbstractArray)
    return estimate_chunksize(a)
end

# Chunked trait

struct Chunked end
struct Unchunked end

function haschunks end
haschunks(x) = Unchunked()

struct OffsetChunks end
struct OneBasedChunks end
wrapchunk(::OneBasedChunks, x, _) = x
wrapchunk(::OffsetChunks, x, inds) = OffsetArray(x, inds...)

"""
    ChunkIndex{N}

This can be used in indexing operations when one wants to extract a full data chunk from a DiskArray. Useful for 
iterating over chunks of data. d[ChunkIndex(1,1)] will extract the first chunk of a 2D-DiskArray
"""
struct ChunkIndex{N,O}
    I::CartesianIndex{N}
    chunktype::O
end
function ChunkIndex(i::CartesianIndex; offset=false)
    return ChunkIndex(i, offset ? OffsetChunks() : OneBasedChunks())
end
ChunkIndex(i::Integer...; offset=false) = ChunkIndex(CartesianIndex(i); offset)

"""
    ChunkIndices{N}

Represents an iterator 
"""
struct ChunkIndices{N,RT<:Tuple{Vararg{Any,N}},O} <: AbstractArray{ChunkIndex{N},N}
    I::RT
    chunktype::O
end
Base.size(i::ChunkIndices) = length.(i.I)
function Base.getindex(A::ChunkIndices{N}, I::Vararg{Int,N}) where {N}
    return ChunkIndex(CartesianIndex(getindex.(A.I, I)), A.chunktype)
end
Base.eltype(::Type{<:ChunkIndices{N}}) where {N} = ChunkIndex{N}

"""
    element_size(a::AbstractArray)

Returns the approximate size of an element of a in bytes. This falls back to calling `sizeof` on 
the element type or to the value stored in `DiskArrays.fallback_element_size`. Methods can be added for 
custom containers. 
"""
function element_size(a::AbstractArray)
    if isbitstype(eltype(a))
        return sizeof(eltype(a))
    elseif isbitstype(Base.nonmissingtype(eltype(a)))
        return sizeof(Base.nonmissingtype(eltype(a)))
    else
        @warn "Can not determine size of element type. Using DiskArrays.fallback_element_size[] = $(fallback_element_size[]) bytes"
        return fallback_element_size[]
    end
end

estimate_chunksize(a::AbstractArray) = estimate_chunksize(size(a), element_size(a))
function estimate_chunksize(s, si)
    ii = searchsortedfirst(cumprod(collect(s)), default_chunk_size[] * 1e6 / si)
    cs = ntuple(length(s)) do idim
        if idim < ii
            return s[idim]
        elseif idim > ii
            return 1
        else
            sbefore = idim == 1 ? 1 : prod(s[1:(idim - 1)])
            return floor(Int, default_chunk_size[] * 1e6 / si / sbefore)
        end
    end
    return GridChunks(s, cs)
end

include("batchgetindex.jl")
