struct SubDiskArray{T,N} <: AbstractDiskArray{T,N}
  v::SubArray{T,N}
end
replace_colon(s,::Colon) = Base.OneTo(s)
replace_colon(s,r) = r
function Base.view(a::AbstractDiskArray,i...)
  i2 = replace_colon.(size(a),i)
  SubDiskArray(SubArray(a,i2))
end
Base.view(a::AbstractDiskArray, i::CartesianIndices) = view(a,i.indices...)
function Base.view(a::SubDiskArray,i...)
  SubDiskArray(view(a.v,i...))
end
Base.view(a::SubDiskArray, i::CartesianIndices) = view(a,i.indices...)
function readblock!(a::SubDiskArray,aout,i::OrdinalRange...)
  pinds = parentindices(view(a.v,i...))
  inds,resh = interpret_indices_disk(parent(a.v),pinds)
  readblock!(parent(a.v),reshape(aout,map(length,inds)...),inds...)
end
function writeblock!(a::SubDiskArray,v,i::OrdinalRange...)
  pinds = parentindices(view(a.v,i...))
  inds,resh = interpret_indices_disk(parent(a.v),pinds)
  writeblock!(parent(a.v),reshape(v,map(length,inds)...),inds...)
end
Base.size(a::SubDiskArray) = size(a.v)
eachchunk(a::SubDiskArray) = eachchunk_view(haschunks(a.v.parent),a.v)
Base.parent(a::SubDiskArray) = a.v.parent
function eachchunk_view(::Chunked, vv)
  pinds = parentindices(vv)
  iomit = findints(pinds)
  chunksparent = eachchunk(parent(vv))
  newchunks = [subsetchunks(chunksparent.chunks[i], pinds[i]) for i in 1:length(pinds) if !in(i,iomit)]
  GridChunks(newchunks...)
end
eachchunk_view(::Unchunked, a) = GridChunks(a,estimate_chunksize(a))
haschunks(a::SubDiskArray) = haschunks(parent(a.v))
