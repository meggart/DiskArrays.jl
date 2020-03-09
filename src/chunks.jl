"""
    eachchunk(a)

Returns an iterator with `CartesianIndices` elements that mark the index range of each chunk within an array.
"""
function eachchunk end

struct GridChunks{N}
    parentsize::NTuple{N,Int}
    chunksize::NTuple{N,Int}
    chunkgridsize::NTuple{N,Int}
    offset::NTuple{N,Int}
end
GridChunks(a, chunksize; offset = ntuple(_->0,ndims(a))) = GridChunks(Int64.(size(a)), Int64.(chunksize), getgridsize(a,chunksize,offset),offset)
function getgridsize(a,chunksize,offset)
  map(size(a),chunksize,offset) do s,cs,of
    fld1(s+of,cs)
  end
end
function Base.show(io::IO, g::GridChunks)
  print(io,"Regular ",join(g.chunksize,"x")," chunks over a ", join(g.parentsize,"x"), " array.")
end
Base.size(g::GridChunks) = g.chunkgridsize
Base.size(g::GridChunks, dim) = g.chunkgridsize[dim]
Base.IteratorSize(::Type{GridChunks{N}}) where N = Base.HasShape{N}()
Base.eltype(::Type{GridChunks{N}}) where N = CartesianIndices{N,NTuple{N,UnitRange{Int64}}}
Base.length(c::GridChunks) = prod(size(c))
@inline function _iterate(g,r)
    if r === nothing
        return nothing
    else
        ichunk, state = r
        outinds = map(ichunk.I, g.chunksize, g.parentsize,g.offset) do ic, cs, ps, of
            max((ic-1)*cs+1-of,1):min(ic*cs-of, ps)
        end |> CartesianIndices
        outinds, state
    end
end
function Base.iterate(g::GridChunks)
    r = iterate(CartesianIndices(g.chunkgridsize))
    _iterate(g,r)
end
function Base.iterate(g::GridChunks, state)
    r = iterate(CartesianIndices(g.chunkgridsize), state)
    _iterate(g,r)
end

#Define the approx default maximum chunk size (in MB)
const default_chunk_size = Ref(100)

#Here we implement a fallback chunking for a DiskArray although this should normally
#be over-ridden by the package that implements the interface

function eachchunk(a::AbstractArray)
  cs = estimate_chunksize(a)
  GridChunks(a,cs)
end

struct Chunked end
struct Unchunked end
function haschunks end
haschunks(x) = Unchunked()

estimate_chunksize(a::AbstractArray) = estimate_chunksize(size(a), sizeof(eltype(a)))
function estimate_chunksize(s, si)
  ii = searchsortedfirst(cumprod(collect(s)),default_chunk_size[]*1e6/si)
  ntuple(length(s)) do idim
    if idim<ii
      return s[idim]
    elseif idim>ii
      return 1
    else
      sbefore = idim == 1 ? 1 : prod(s[1:idim-1])
      return floor(Int,default_chunk_size[]*1e6/si/sbefore)
    end
  end
end
