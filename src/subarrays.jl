const SubDiskArray = SubArray{<:Any,<:Any,<:AbstractDiskArray}
#Allow getindex for SubArrays
function Base.getindex(a::SubDiskArray,i...)
  pinds = parentindices(view(a,i...))
  getindex_disk(parent(a), pinds...)
end
#Remove amiguity
function Base.getindex(a::SubDiskArray, i::Vararg{Int64,N}) where N
  pinds = parentindices(view(a,i...))
  getindex_disk(parent(a), pinds...)
end
function Base.setindex!(a::SubDiskArray, v, i...)
  pinds = parentindices(view(a,i...))
  setindex_disk!(parent(a),v,pinds...)
end
function Base.show(io::IO, ::MIME"text/plain", X::SubDiskArray)
  println(io, "Disk Array view with size ", join(size(X)," x "))
end
eachchunk(a::SubDiskArray) = eachchunk_view(haschunks(a),a)
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
haschunks(a::SubDiskArray) = haschunks(parent(a))

@implement_mapreduce SubDiskArray
@implement_broadcast SubDiskArray
@implement_iteration SubDiskArray
