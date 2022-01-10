"""
    eachchunk(a)

Returns an iterator with `CartesianIndices` elements that mark the index range of each chunk within an array.
"""
function eachchunk end

abstract type ChunkType <: AbstractVector{UnitRange} end

"""
    RegularChunks

Defines chunking along a dimension where the chunks have constant size and a potential
offset for the first chunk. The last chunk is truncated to fit the array size. 
"""
struct RegularChunks <: ChunkType
    cs::Int
    offset::Int
    s::Int
end
function Base.getindex(r::RegularChunks,i::Int) 
  @boundscheck checkbounds(r, i)
  max((i-1)*r.cs+1-r.offset,1):min(i*r.cs-r.offset, r.s)
end
Base.size(r::RegularChunks,_) = div(r.s + r.offset - 1,r.cs) +1
Base.size(r::RegularChunks) = (size(r,1),)
function subsetchunks(r::RegularChunks, subs::AbstractUnitRange)
  snew = length(subs)
  newoffset = mod(first(subs)-1+r.offset,r.cs)
  RegularChunks(r.cs, newoffset, snew)
end
function subsetchunks(r::RegularChunks, subs::AbstractRange)
  #This is a method only to make "reverse" work and should error for all other cases
  if step(subs) == -1 && first(subs)==r.s && last(subs)==1
    lastlen = length(last(r))
    newoffset = r.cs-lastlen
    return RegularChunks(r.cs, newoffset, r.s)
  end
end
approx_chunksize(r::RegularChunks) = r.cs
grid_offset(r::RegularChunks) = r.offset
max_chunksize(r::RegularChunks) = r.cs

"""
    IrregularChunks

Defines chunks along a dimension where chunk sizes are not constant but arbitrary
"""
struct IrregularChunks <: ChunkType
  offsets::Vector{Int}
end
function Base.getindex(r::IrregularChunks,i::Int) 
  @boundscheck checkbounds(r, i)
  (r.offsets[i]+1):r.offsets[i+1]
end
Base.size(r::IrregularChunks) = (length(r.offsets)-1,)
function subsetchunks(r::IrregularChunks, subs::UnitRange)
  c1 = searchsortedfirst(r.offsets, first(subs))-1
  c2 = searchsortedfirst(r.offsets, last(subs))
  offsnew = r.offsets[c1:c2]
  firstoffset = first(subs)-r.offsets[c1]-1
  offsnew[end] = last(subs)
  offsnew[2:end] .= offsnew[2:end] .- firstoffset
  offsnew .= offsnew .- first(offsnew)
  IrregularChunks(offsnew)
end
approx_chunksize(r::IrregularChunks) = round(Int,sum(diff(r.offsets))/(length(r.offsets)-1))
grid_offset(r::IrregularChunks) = 0
max_chunksize(r::IrregularChunks) = maximum(diff(r.offsets))


"""
    IrregularChunks(;chunksizes)

Returns an IrregularChunks object for the given list of chunk sizes
"""
function IrregularChunks(;chunksizes)
  offs = pushfirst!(cumsum(chunksizes),0)
  #push!(offs,last(offs)+1)
  IrregularChunks(offs)
end


struct GridChunks{N} <: AbstractArray{UnitRange,N}
  chunks::Tuple{Vararg{ChunkType,N}}
end
function Base.getindex(g::GridChunks{N},i::Vararg{Int, N}) where N 
  @boundscheck checkbounds(g, i...)
  getindex.(g.chunks,i)
end
Base.size(g::GridChunks) = length.(g.chunks)
GridChunks(ct::ChunkType...) = GridChunks(ct)
GridChunks(a, chunksize; offset = (_->0).(size(a))) = GridChunks(size(a), chunksize; offset)
function GridChunks(a::Tuple, chunksize; offset = (_->0).(a))
  gcs = map(a,chunksize, offset) do s, cs, of
      RegularChunks(cs,of,s)
  end
  GridChunks(gcs)
end

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



#Define the approx default maximum chunk size (in MB)
"The target chunk size for processing for unchunked arrays in MB, defaults to 100MB"
const default_chunk_size = Ref(100)

"""
A fallback element size for arrays to determine a where elements have unknown
size like strings. Defaults to 100MB
"""
const fallback_element_size = Ref(100)

#Here we implement a fallback chunking for a DiskArray although this should normally
#be over-ridden by the package that implements the interface

function eachchunk(a::AbstractArray)
  estimate_chunksize(a)
end

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
  ii = searchsortedfirst(cumprod(collect(s)),default_chunk_size[]*1e6/si)
  cs = ntuple(length(s)) do idim
    if idim<ii
      return s[idim]
    elseif idim>ii
      return 1
    else
      sbefore = idim == 1 ? 1 : prod(s[1:idim-1])
      return floor(Int,default_chunk_size[]*1e6/si/sbefore)
    end
  end
  GridChunks(s,cs)
end
