"""
    MultiIndex

Type representing an Index spanning multiple dimensions, for example a `Matrix{Bool}` or a `Vector{CartesianIndex{3}}`
where a single index array consumes multiple dimensions of the array to be indexed. Here the `indices` are referenced
into the struct and the respective dimension is stored. 
"""
struct MultiIndex{N,P,D}
    indices::P
    dim::Val{D}
    has_chunk_gap::Bool
    is_sparse_index::Bool
    bb::NTuple{N,Tuple{Int,Int}}
end
Base.ndims(::MultiIndex{N}) where N = N

"""
    resolve_multiindex(a,i)

For index types that map to multiple sub-dimensions of the array this will convert them to MultiIndex indices, so
that the resulting index has a 1:1 mapping between index and array dimension
"""
function resolve_multiindex(a,i)
    cs = eachchunk(a)
    _resolve_multiindex(cs,i)
end
_resolve_multiindex(_,::Tuple{})  = ()
_resolve_multiindex(cs,i) = _resolve_multiindex(cs,(),first(i),Base.tail(i))
splittail(::Tuple{}) = ()
splittail(itail) = first(itail),Base.tail(itail)
function _resolve_multiindex(cs,result,inow,itail)
    new_indices, new_chunks = maybemultiindex(cs,inow)
    nextinds = splittail(itail)
    new_chunks,(result...,new_indices...),nextinds...    
end
function _resolve_multiindex(cs,result,inow)
    new_indices, _ = maybemultiindex(cs,inow)
    result...,new_indices...  
end

maybemultiindex(cs,inds) = inds,Base.tail(cs)

function maybemultiindex(x::AbstractArray{<:Bool},cs)
    bb = getbb(x)
    n = sum(x)
    is_sparse_index = n/length(x) < 0.5
    #TODO check for chunk jumps/omitted chunks, for now set chunk gap to true
    chunk_gaps = true
    newinds = ntuple(ndims(x)) do d
        MultiIndex(x,Val(d),chunk_gaps,is_sparse_index,bb)
    end
    rem_cs = consumechunks(newinds,cs)
    newinds,rem_cs
end

function maybemultiindex(x::AbstractVector{<:CartesianIndex},cs)
    bb = getbb(x)
    n = length(x)
    size_bb = prod(length.(range.(bb)))
    is_sparse_index = n/size_bb(x) < 0.5
    #TODO check for chunk jumps/omitted chunks, for now set chunk gap to true
    chunk_gaps = true
    newinds = ntuple(ndims(x)) do d
        MultiIndex(x,Val(d),chunk_gaps,is_sparse_index,bb)
    end
    rem_cs = consumechunks(newinds,cs)
    newinds,rem_cs
end

consumechunks(newinds,cs) = consumechunks(Base.tail(newinds),Base.tail(cs))
consumechunks(::Tuple{},cs) = cs

getbb(ar::AbstractArray{<:CartesianIndex}) = extrema(ar)

function getbb(ar::AbstractArray{<:Bool})
  maxval = CartesianIndex(size(ar))
  minval = CartesianIndex{ndims(ar)}()
  reduceop = (i1,i2)->begin i2===nothing ? i1 : (min(i1[1],i2),max(i1[2],i2)) end
  mi,ma = mapfoldl(reduceop,
    zip(CartesianIndices(ar),ar),
    init = (maxval,minval)) do ii
    ind,val = ii
    val ? ind : nothing
  end
  map(identity, mi.I,ma.I)
end