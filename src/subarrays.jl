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
