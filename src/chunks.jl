import ChunkedArrayBase: eachchunk, GridChunks

#Define the approx default maximum chunk size (in MB)
const default_chunk_size = Ref(100)

#Here we implement a fallback chunking for a DiskArray although this should normally
#be over-ridden by the package that implements the interface

function eachchunk(a::AbstractDiskArray)
  cs = estimate_chunksize(a)
  GridChunks(a,cs)
end

function estimate_chunksize(a::AbstractDiskArray)
  s = size(a)
  si = sizeof(eltype(a))
  ii = searchsortedfirst(cumprod(collect(s)),default_chunk_size[]*1e6/si)
  ntuple(ndims(a)) do idim
    if idim<ii
      return size(a,idim)
    elseif idim>ii
      return 1
    else
      sbefore = idim == 1 ? 1 : prod(s[1:idim-1])
      return floor(Int,default_chunk_size[]*1e6/si/sbefore)
    end
  end
end
