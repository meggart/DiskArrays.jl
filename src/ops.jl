import Base.Broadcast: Broadcasted, AbstractArrayStyle, DefaultArrayStyle, flatten
toRanges(r::Tuple) = r
toRanges(r::CartesianIndices) = r.indices

struct ChunkStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end

Base.BroadcastStyle(::ChunkStyle{N}, ::ChunkStyle{M}) where {N,M}= ChunkStyle{max(N,M)}()
Base.BroadcastStyle(::ChunkStyle{N}, ::DefaultArrayStyle{M}) where {N,M}= ChunkStyle{max(N,M)}()
Base.BroadcastStyle(::DefaultArrayStyle{M},::ChunkStyle{N}) where {N,M}= ChunkStyle{max(N,M)}()

struct BroadcastDiskArray{T,N,BC<:Broadcasted{<:ChunkStyle{N}}} <: AbstractDiskArray{T,N}
  bc::BC
end
function BroadcastDiskArray(bcf::B) where {B<:Broadcasted{<:ChunkStyle{N}}} where N
  ElType = Base.Broadcast.combine_eltypes(bcf.f, bcf.args)
  BroadcastDiskArray{ElType,N,B}(bcf)
end
Base.size(bc::BroadcastDiskArray) = size(bc.bc)
function DiskArrays.readblock!(a::BroadcastDiskArray,aout,i::OrdinalRange...)
  argssub = map(arg->subsetarg(arg,i),a.bc.args)
  aout .= a.bc.f.(argssub...)
end
Base.broadcastable(bc::BroadcastDiskArray) = bc.bc
haschunks(a::BroadcastDiskArray) = Chunked()
function eachchunk(a::BroadcastDiskArray)
  cs,off = common_chunks(size(a.bc),a.bc.args...)
  GridChunks(a.bc,cs,offset=off)
end
function Base.copy(bc::Broadcasted{ChunkStyle{N}}) where N
  BroadcastDiskArray(flatten(bc))
end
Base.copy(a::BroadcastDiskArray) = copyto!(zeros(eltype(a),size(a)),a.bc)
function Base.copyto!(dest::AbstractArray, bc::Broadcasted{ChunkStyle{N}}) where N
  bcf = flatten(bc)
  cs,off = common_chunks(size(bcf),dest,bcf.args...)
  gcd = GridChunks(bcf,cs,offset=off)
  foreach(gcd) do cnow
    #Possible optimization would be to use a LRU cache here, so that data has not
    #to be read twice in case of repeating indices
    argssub = map(i->subsetarg(i,cnow.indices),bcf.args)
    dest[cnow.indices...] .= bcf.f.(argssub...)
  end
  dest
end
function common_chunks(s,args...)
  N = length(s)
  chunkedars = filter(i->haschunks(i)===Chunked(),collect(args))
  all(ar->isa(eachchunk(ar),GridChunks), chunkedars) || error("Currently only chunks of type GridChunks can be merged by broadcast")
  if isempty(chunkedars)
    totalsize = sum(sizeof ∘ eltype, args)
    return (estimate_chunksize(s,totalsize),ntuple(zero,N))
  else

    allcs = map(ar->(eachchunk(ar).chunksize,eachchunk(ar).offset),chunkedars)
    tt = ntuple(N) do n
      csnow = filter(cs->length(cs[1])>=n && cs[1][n]>1,allcs)
      isempty(csnow) && return (1, 0)
      cs = (csnow[1][1][n],csnow[1][2][n])
      all(s->(s[1][n],s[2][n]) == cs,csnow) || error("Chunks do not align in dimension $n")
      return cs
    end
    return map(i->i[1],tt),map(i->i[2],tt)
  end
end
subsetarg(x, ranges) = x
function subsetarg(x::Tuple, ranges)
  length(x) !== 1 && length(ranges) !== 1 && throw(DimensionMismatch("DiskArrays can only broadcast a Tuple of length > 1 with a 1 dimensional Array"))
  x
end
function subsetarg(x::AbstractArray,ranges)
  (length(ranges) === ndims(x) || length(x) === 1) || throw(DimensionMismatch("Can only broadcast with arrays of matching size"))
  ashort = maybeonerange(size(x),ranges)
  view(x,ashort...) #Maybe making a copy here would be faster, need to check...
end
repsingle(s,r) = s==1 ? (1:1) : r
function maybeonerange(out,sizes,ranges)
  s1,sr = splittuple(sizes...)
  r1,rr = splittuple(ranges...)
  maybeonerange((out...,repsingle(s1,r1)),sr,rr)
end
maybeonerange(out,::Tuple{},ranges) = out
maybeonerange(sizes,ranges) = maybeonerange((),sizes,ranges)
splittuple(x1,xr...) = x1,xr

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

macro implement_broadcast(t)
  quote
    #Broadcasting with a DiskArray on LHS
    function Base.copyto!(dest::$t, bc::Broadcasted{Nothing})
      foreach(eachchunk(dest)) do c
        ar = [bc[i] for i in CartesianIndices(toRanges(c))]
        dest[toRanges(c)...] = ar
      end
      dest
    end
    Base.BroadcastStyle(T::Type{<:$t}) = ChunkStyle{ndims(T)}()
    function DiskArrays.subsetarg(x::$t,a)
      ashort = maybeonerange(size(x),a)
      x[ashort...]
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
      dest
    end
  end
end
