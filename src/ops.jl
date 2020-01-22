toRanges(r::Tuple) = r
toRanges(r::CartesianIndices) = r.indices
macro implement_mapreduce(t)
quote
function Base._mapreduce(f,op,v::$t)
  mapreduce(op,eachchunk(v)) do cI
    a = v[toRanges(cI)...]
    mapreduce(f,op,a)
  end
end
function Base.mapreducedim!(f, op, R::AbstractArray, A::$t)
  sR = size(R)
  foreach(eachchunk(A)) do cI
    a = A[toRanges(cI)...]
    ainds = map((cinds, arsize)->arsize==1 ? Base.OneTo(1) : cinds, toRanges(cI),size(R))
    #Maybe the view into R here is problematic and a copy would be faster
    Base.mapreducedim!(f,op,view(R,ainds...),a)
  end
  R
end

function Base.mapfoldl_impl(f, op, nt::NamedTuple{()}, itr::$t)
  cc = eachchunk(itr)
  isempty(cc) && return Base.mapreduce_empty_iter(f, op, itr, IteratorEltype(itr))
  Base.mapfoldl_impl(f,op,nt,itr,cc)
end
function Base.mapfoldl_impl(f, op, nt::NamedTuple{()}, itr::$t, cc)
    y = first(cc)
    a = itr[toRanges(y)...]
    init = mapfoldl(f,op,a)
    Base.mapfoldl_impl(f,op,(init=init,),itr,Iterators.drop(cc,1))
end
function Base.mapfoldl_impl(f, op, nt::NamedTuple{(:init,)}, itr::$t, cc)
  init = nt.init
  for y in cc
    a = itr[toRanges(y)...]
    init = mapfoldl(f,op,a,init=init)
  end
  init
end
end
end

import Base.Broadcast: Broadcasted

macro implement_broadcast(t)
quote
#Broadcasting with a DiskArray on LHS
function Base.copyto!(dest::$t, bc::Broadcasted{Nothing})
  foreach(eachchunk(dest)) do c
    ar = [bc[i] for i in CartesianIndices(c)]
    dest[toRanges(c)...] = ar
  end
end

#This is a heavily allocating implementation, but working for all cases.
#As performance optimization one might:
#Allocate the array only once if all chunks have the same size
#Use FillArrays, if the DiskArray accepts these
function Base.fill!(dest::$t, value)
  foreach(eachchunk(dest)) do c
    ar = fill(value,length.(toRanges(c)))
    dest[toRanges(c)...] = ar
  end
end
end
end
