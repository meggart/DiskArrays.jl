"""
    eachchunk(a)

Returns an iterator with `CartesianIndices` elements that mark the index range of each chunk within an array.
"""
function eachchunk end

abstract type ChunkType <: AbstractVector{UnitRange} end


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
  if rem(r.cs,step(subs)) == 0
      newcs = r.cs ÷ abs(step(subs))
      if step(subs) > 0
          newoffset = mod(first(subs)-1+r.offset,r.cs) ÷ step(subs)
          RegularChunks(newcs,newoffset,length(subs))
      elseif step(subs) < 0
          r2 = subsetchunks(r,last(subs):first(subs))
          newoffset = (r.cs - length(last(r2))) ÷ (-step(subs))
          RegularChunks(newcs,newoffset,length(subs))
      end
  else
      subsetchunks_fallback(r,subs)
  end
end
findchunk(r::RegularChunks, i::Int) = div(i+r.offset-1,r.cs)+1


subsetchunks(r,subs) = subsetchunks_fallback(r,subs)

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

# Base methods

function Base.getindex(r::IrregularChunks, i::Int)
    @boundscheck checkbounds(r, i)
    return (r.offsets[i] + 1):r.offsets[i + 1]
end
Base.size(r::IrregularChunks) = (length(r.offsets) - 1,)

# DiskArrays interface

function subsetchunks(r::IrregularChunks, subs::UnitRange)
    c1 = searchsortedfirst(r.offsets, first(subs)) - 1
    c2 = searchsortedfirst(r.offsets, last(subs))
    offsnew = r.offsets[c1:c2]
    firstoffset = first(subs) - r.offsets[c1] - 1
    offsnew[end] = last(subs)
    offsnew[2:end] .= offsnew[2:end] .- firstoffset
    offsnew .= offsnew .- first(offsnew)
    return IrregularChunks(offsnew)
end
findchunk(r::IrregularChunks,i::Int) = searchsortedfirst(r.offsets, i)-1
function approx_chunksize(r::IrregularChunks)
    return round(Int, sum(diff(r.offsets)) / (length(r.offsets) - 1))
end
grid_offset(r::IrregularChunks) = 0
max_chunksize(r::IrregularChunks) = maximum(diff(r.offsets))

struct GridChunks{N} <: AbstractArray{NTuple{N,UnitRange{Int64}},N}
    chunks::Tuple{Vararg{ChunkType,N}}
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


function subsetchunks_fallback(r, subs)
  #This is a fallback method that should work for Regular and Irregular chunks r 
  #Assuming the desired subset is sorted
  # We simply compute the chunk for every element in subs and collect everything together 
  #again in either a Regular or IrregularChunk
  rev = if issorted(subs)
      false
  elseif issorted(subs,rev=true)
      true
  else
      throw(ArgumentError("Can only subset chunks for sorted indices"))
  end
  cs = zeros(Int,length(r))
  for i in subs
      cs[findchunk(r,i)] += 1
  end
  # Find first and last chunk where elements are extracted
  i1 = findfirst(!iszero,cs)
  i2 = findlast(!iszero,cs)
  if i2==i1
      #only a single chunk is affected
      return RegularChunks(length(subs),0,length(subs))
  elseif i2-i1 == 1
      #Two affected chunks
      l1,l2 = rev ? (i2,i1) : (i1,i2)
      chunksize = max(cs[l1],cs[l2])
      RegularChunks(chunksize,chunksize-cs[l1],length(subs))
  elseif all(==(cs[i1+1]),view(cs,i1+1:i2-1)) && cs[i2] <= cs[i1+1] && cs[i1] <= cs[i1+1]
      #All chunks have the same size, only first and last chunk can be shorter
      l1 = rev ? cs[i2] : cs[i1]
      chunksize = cs[i1+1]
      RegularChunks(chunksize,chunksize-l1,length(subs))
  else
      #Chunks are Irregular
      chunks = rev ? reverse(cs) : cs

      IrregularChunks(chunksizes = filter(!iszero,chunks))
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
