import Base.Broadcast: Broadcasted, AbstractArrayStyle, DefaultArrayStyle, Broadcasted, flatten
toRanges(r::Tuple) = r
toRanges(r::CartesianIndices) = r.indices

struct ChunkStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end

Base.BroadcastStyle(::ChunkStyle{N}, ::ChunkStyle{M}) where {N,M}= ChunkStyle{max(N,M)}()
Base.BroadcastStyle(::ChunkStyle{N}, ::DefaultArrayStyle{M}) where {N,M}= ChunkStyle{max(N,M)}()
Base.BroadcastStyle(::DefaultArrayStyle{M},::ChunkStyle{N}) where {N,M}= ChunkStyle{max(N,M)}()

struct BroadcastDiskArray{T,N,BC<:Broadcasted{<:ChunkStyle{N}}} <: AbstractDiskArray{T,N}
    bc::BC
end
Base.size(bc::BroadcastDiskArray) = size(bc.bc)
function DiskArrays.readblock!(a::BroadcastDiskArray,aout,i...)
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
    bcf = flatten(bc)
    ElType = Base.Broadcast.combine_eltypes(bcf.f, bcf.args)
    BroadcastDiskArray{ElType,N,typeof(bcf)}(bcf)
end
function Base.copyto!(dest::AbstractArray, bc::Broadcasted{ChunkStyle{N}}) where N
    bcf = flatten(bc)
    cs,off = common_chunks(size(bcf),dest,bcf.args...)
    gcd = GridChunks(bcf,cs,offset=off)
    foreach(gcd) do cnow
        argssub = map(i->subsetarg(i,cnow.indices),bc.args)
        dest[cnow.indices...] .= bc.f.(argssub...)
    end
    dest
end

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
    ar = [bc[i] for i in CartesianIndices(c)]
    dest[toRanges(c)...] = ar
  end
  dest
end

Base.BroadcastStyle(::Type{<:$t{<:Any,N}}) where N = ChunkStyle{N}()
Base.BroadcastStyle(::Type{<:$t{<:Any,N}}) where N = ChunkStyle{N}()

subsetarg(x::Number,a) = x
function subsetarg(x::AbstractArray,a)
    ashort = map((s,r)->s==1 ? (1:1) : r,size(x),a)
    view(x,ashort...) #Maybe making a copy here would be faster, need to check...
end
function subsetarg(x::$t,a)
    ashort = map((s,r)->s==1 ? (1:1) : r,size(x),a)
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
