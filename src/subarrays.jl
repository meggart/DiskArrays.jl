struct SubDiskArray{T,N} <: AbstractDiskArray{T,N}
  v::SubArray{T,N}
end
function Base.view(a::AbstractDiskArray,i...)
  SubDiskArray(SubArray(a,i))
end
function Base.view(a::SubDiskArray,i...)
  SubDiskArray(view(a.v,i...))
end
function readblock!(a::SubDiskArray,aout,i...)
  pinds = parentindices(view(a.v,i...))
  inds,resh = interpret_indices_disk(parent(a.v),pinds)
  readblock!(parent(a.v),reshape(aout,map(length,inds)...),inds...)
end
function writeblock!(a::SubDiskArray,v,i...)
  pinds = parentindices(view(a.v,i...))
  inds,resh = interpret_indices_disk(parent(a.v),pinds)
  writeblock!(parent(a.v),reshape(v,map(length,inds)...),inds...)
end
Base.size(a::SubDiskArray) = size(a.v)
eachchunk(a::SubDiskArray) = eachchunk_view(haschunks(a.v.parent),a.v)
function eachchunk_view(::Chunked, vv)
  pinds = parentindices(vv)
  iomit = findints(pinds)
  chunksparent = eachchunk(parent(vv))
  parentoffset, parentsize = chunksparent.offset, chunksparent.chunksize
  offsets = ([mod1(first(pinds[i]),parentsize[i])+parentoffset[i]-1 for i in 1:length(pinds) if !in(i,iomit)]...,)
  newcs = ([parentsize[i] for i in 1:length(pinds) if !in(i,iomit)]...,)
  GridChunks(vv,newcs,offset = offsets)
end
eachchunk_view(::Unchunked, a) = GridChunks(a,estimate_chunksize(a))
haschunks(a::SubDiskArray) = haschunks(parent(a.v))
