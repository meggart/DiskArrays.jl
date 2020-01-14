function Base.mapreduce(f,op,v::AbstractDiskArray)
  mapreduce(op,eachchunk(v)) do cI
    a = v[cI.indices...]
    mapreduce(f,op,a)
  end
end
function Base.mapreducedim!(f, op, R::AbstractArray, A::AbstractDiskArray)
  sR = size(R)
  foreach(eachchunk(A)) do cI
    a = A[cI.indices...]
    ainds = map((cinds, arsize)->arsize==1 ? Base.OneTo(1) : cinds, cI.indices,size(R))
    #Maybe the view into R here is problematic and a copy would be faster
    Base.mapreducedim!(f,op,view(R,ainds...),a)
  end
  R
end
