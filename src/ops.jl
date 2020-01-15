toRanges(r::Tuple) = r
toRanges(r::CartesianIndices) = r.indices
function Base.mapreduce(f,op,v::AbstractDiskArray)
  mapreduce(op,eachchunk(v)) do cI
    a = v[toRanges(cI)...]
    mapreduce(f,op,a)
  end
end
function Base.mapreducedim!(f, op, R::AbstractArray, A::AbstractDiskArray)
  sR = size(R)
  foreach(eachchunk(A)) do cI
    a = A[toRanges(cI)...]
    ainds = map((cinds, arsize)->arsize==1 ? Base.OneTo(1) : cinds, toRanges(cI),size(R))
    #Maybe the view into R here is problematic and a copy would be faster
    Base.mapreducedim!(f,op,view(R,ainds...),a)
  end
  R
end
