"""
    AbstractDiskArray <: AbstractArray

Abstract DiskArray type that can be inherited by Array-like data structures that
have a significant random access overhead and whose access pattern follows
n-dimensional (hyper)-rectangles.
"""
abstract type AbstractDiskArray{T,N} <: AbstractArray{T,N} end

"""
    readblock!(A::AbstractDiskArray, A_ret, r::AbstractUnitRange...)

The only function that should be implemented by a `AbstractDiskArray`. This function
"""
function readblock!() end

"""
    writeblock!(A::AbstractDiskArray, A_in, r::AbstractUnitRange...)

Function that should be implemented by a `AbstractDiskArray` if write operations
should be supported as well.
"""
function writeblock!() end


function allow_multi_chunk_access(a)
    false
end

"""
    resolve_indices(a, i)

Determines a list of tuples used to perform the read or write operations. The returned values are:

- `outsize` size of the output array
- `temp_size` size of the temp array passed to `readblock`
- `output_indices` indices for copying into the output array
- `temp_indices` indices for reading from temp array
- `data_indices` indices for reading from data array

"""
Base.@assume_effects :foldable resolve_indices(a, i) = _resolve_indices(eachchunk(a).chunks,i,(),(),(),(),())
function _resolve_indices(cs,i,output_size,temp_sizes,output_indices,temp_indices,data_indices)
    inow = first(i)
    outsize, tempsize, outinds,tempinds,datainds,cs = process_index(inow, cs)
    output_size = (output_size...,outsize...)
    output_indices = (output_indices...,outinds...)
    temp_sizes = (temp_sizes...,tempsize...)
    temp_indices = (temp_indices...,tempinds...)
    data_indices = (data_indices...,datainds...)
    _resolve_indices(cs,Base.tail(i),output_size,temp_sizes,output_indices,temp_indices, data_indices)
end
_resolve_indices(::Tuple{},::Tuple{},output_size,temp_sizes,output_indices,temp_indices,data_indices) = output_size,temp_sizes,output_indices,temp_indices,data_indices
#No dimension left in array, only singular indices allowed
function _resolve_indices(::Tuple{},i,output_size,temp_sizes,output_indices,temp_indices,data_indices)
    inow = first(i)
    if inow isa Integer
        inow == 1 || throw(ArgumentError("Trailing indices must be 1"))
        _resolve_indices((),Base.tail(i),output_size,temp_sizes,output_indices,temp_indices,data_indices)
    elseif inow isa AbstractVector
         (length(inow)==1 && first(inow)==1) || throw(ArgumentError("Trailing indices must be 1"))
         output_size = (output_size...,1)
         output_indices = (output_indices...,1)
         _resolve_indices((),Base.tail(i),output_size,temp_sizes,output_indices,temp_indices,data_indices)
    else
        throw(ArgumentError("Trailing indices must be 1"))
    end
end
#Still dimensions left, but no indices available
function _resolve_indices(cs,::Tuple{},output_size,temp_sizes,output_indices,temp_indices,data_indices)
    csnow = first(cs)
    arraysize_from_chunksize(csnow) == 1 || throw(ArgumentError("Wrong indexing"))
    data_indices = (data_indices...,1:1)
    temp_sizes = (temp_sizes...,1)
    temp_indices = (temp_indices...,1)
    _resolve_indices(Base.tail(cs),(),output_size,temp_sizes,output_indices,temp_indices,data_indices)
end

resolve_indices(a, ::Tuple{Colon}) = (length(a),), size(a), (Colon(),), (Colon(),), Base.OneTo.(size(a))
resolve_indices(a, i::Tuple{<:CartesianIndex}) = resolve_indices(a, only(i).I)
resolve_indices(a, i::Tuple{<:CartesianIndices}) = resolve_indices(a, only(i).indices)
function resolve_indices(a, i::Tuple{<:AbstractVector})
    inds = first(i)
    toread = view(CartesianIndices(size(a)),inds)
    cindmin,cindmax = extrema(toread)
    indmin,indmax = cindmin.I,cindmax.I
    tempsize = indmax .- indmin .+ 1
    tempoffset = cindmin - oneunit(cindmin)
    datainds = range.(indmin,indmax)
    tempinds = toread .- tempoffset
    length.(i),tempsize,(Colon(),),(tempinds,),datainds
end
#outsize, tempsize, outinds,tempinds,datainds,cs
process_index(inow::Integer, cs) = ((), 1, (), (1,),(inow:inow,), Base.tail(cs))
function process_index(::Colon, cs)
    s = arraysize_from_chunksize(first(cs))
    (s,), (s,), (Colon(),), (Colon(),), (Base.OneTo(s),), Base.tail(cs)
end
function process_index(i::AbstractUnitRange, cs)
    (length(i),), (length(i),), (Colon(),), (Colon(),), (i,), Base.tail(cs)
end
function process_index(i::AbstractVector{<:Integer}, cs)
    indmin,indmax = extrema(i)
    (length(i),), ((indmax-indmin+1),), (Colon(),), ((i.-(indmin-1)),), (indmin:indmax,), Base.tail(cs) 
end
function process_index(i::AbstractArray{Bool,N}, cs) where N
    csnow, csrem = splitcs(size(i),(),cs)
    s = arraysize_from_chunksize.(csnow)
    cindmin,cindmax = extrema(view(CartesianIndices(s),i))
    indmin,indmax = cindmin.I,cindmax.I
    tempsize = indmax .- indmin .+ 1
    tempinds = view(i,range.(indmin,indmax)...)
    (sum(i),), tempsize, (Colon(),),(tempinds,), range.(indmin,indmax), csrem
end
function process_index(i::AbstractVector{<:CartesianIndex{N}}, cs) where N
    csnow, csrem = splitcs(first(i).I,(),cs)
    s = arraysize_from_chunksize.(csnow)
    cindmin,cindmax = extrema(view(CartesianIndices(s),i))
    indmin,indmax = cindmin.I,cindmax.I
    tempsize = indmax .- indmin .+ 1
    tempoffset = cindmin - oneunit(cindmin)
    tempinds = i .- tempoffset
    (length(i),), tempsize, (Colon(),), (tempinds,), range.(indmin,indmax), csrem
end
splitcs(si,csnow,csrem) = splitcs(Base.tail(si),(csnow...,first(csrem)),Base.tail(csrem))
splitcs(::Tuple{},csnow,csrem) = (csnow,csrem)


viewifnecessary(a,::Tuple{Vararg{Colon}}) = a
viewifnecessary(a,i) = view(a,i...)
maybe_unwrap(a,_) = a
maybe_unwrap(a,::Tuple{Vararg{<:Integer}}) = a[1]


function getindex_disk(a, i::Union{Integer,CartesianIndex}...)
    checkscalar(i)
    outputarray = Array{eltype(a)}(undef,map(_->1,size(a))...)
    i = Base.to_indices(a,i)
    j = map(1:ndims(a)) do d
        d<=length(i) ? (i[d]:i[d]) : 1:1
    end
    readblock!(a, outputarray, j...)
    only(outputarray)
end

function getindex_disk(a, i...)
    # i_multi = resolve_multiindex(a,i)
    output_size, temparray_size, output_indices, temparray_indices, data_indices = resolve_indices(a,i)
    @debug output_size, temparray_size, output_indices, temparray_indices, data_indices
    # inds, trans = interpret_indices_disk(a, i_multi)
    # inds = map(maybe2range,inds)
    # chunk_gaps = any(map(has_chunk_gap,approx_chunksize(eachchunk(a)),inds))
    # sparse_index = any(map(is_sparse_index,inds))
    # if sparse_index && (chunk_gaps || allow_multi_chunk_access(a))
    #     batchgetindex(a, i...)
    # else
    outputarray = Array{eltype(a)}(undef, output_size...)  
    temparray = Array{eltype(a)}(undef, temparray_size...)  
    readblock!(a, temparray, data_indices...)
    transfer_results!(outputarray, temparray, output_indices, temparray_indices)
end

function transfer_results!(outputarray, temparray, output_indices, temparray_indices)
    outputarray[output_indices...] = view(temparray,temparray_indices...)
    outputarray
end
function transfer_results!(o,t,oi::Tuple{Vararg{Int}},ti::Tuple{Vararg{Int}}) 
    o[oi...] = t[ti...]
    o
end

function setindex_disk!(a::AbstractArray{T}, v::T, i...) where {T<:AbstractArray}
    checkscalar(i)
    return setindex_disk!(a, [v], i...)
end

function setindex_disk!(a::AbstractArray, v::AbstractArray, i...)
    checkscalar(i)
    if any(j -> isa(j, AbstractArray) && !isa(j, AbstractRange), i)
        batchsetindex!(a, v, i...)
    else
        inds, trans = interpret_indices_disk(a, i)
        data = reshape(v, map(length, inds))
        writeblock!(a, data, inds...)
        v
    end
end

"""
Function that translates a list of user-supplied indices into plain ranges and
integers for reading blocks of data. This function respects additional indexing
rules like omitting additional trailing indices.

The passed array handle A must implement methods for `Base.size` and `Base.ndims`
The function returns two values:

  1. a tuple whose length equals `ndims(A)` containing only unit
  ranges and integers. This contains the minimal "bounding box" of data that
  has to be read from disk.
  2. A callable object which transforms the hyperrectangle read from disk to
  the actual shape that represents the Base getindex behavior.
"""
function interpret_indices_disk(A, r::Tuple)
    throw(ArgumentError("Indices of type $(typeof(r)) are not yet supported"))
end

#Read the entire array and reshape to 1D in the end
function interpret_indices_disk(A, ::Tuple{Colon})
    return map(Base.OneTo, size(A)), Reshaper(prod(size(A)))
end

interpret_indices_disk(A, r::Tuple{<:CartesianIndex}) = interpret_indices_disk(A, r[1].I)

function interpret_indices_disk(A, r::Tuple{<:CartesianIndices})
    return interpret_indices_disk(A, r[1].indices)
end

extract_indices_and_dropdims(sa,r) = _convert_index((),(),1,sa,r)
_convert_index(i::Integer,s::Integer) = i:i
_convert_index(i::AbstractVector, s::Integer) = i
_convert_index(i::MultiIndex{<:Any,<:Any,D},s::Integer) where D = first(i.bb[D]):last(i.bb[D])
_convert_index(::Colon, s::Integer) = Base.OneTo(Int(s))



function interpret_indices_disk(
    A, r::NTuple{N,Union{Integer,AbstractVector,Colon}}
) where {N}
    if ndims(A) == N
        inds = map(_convert_index, r, size(A))
        resh = DimsDropper(findints(r))
        return inds, resh
    elseif ndims(A) < N
        n_add_dim = sum((ndims(A) + 1):N) do i
            first(r[i]) == 1 || throw(BoundsError(A, r))
            isa(r[i], AbstractArray)
        end
        _, rshort = commonlength(size(A), r)
        inds, resh1 = interpret_indices_disk(A, rshort)
        if n_add_dim > 0
            ladddim = ntuple(_ -> 1, n_add_dim)
            oldsize = result_size(inds, resh1)
            resh2 = transformstack(resh1, Reshaper((oldsize..., ladddim...)))
            inds, resh2
        else
            inds, resh1
        end
    else
        size(A, N + 1) == 1 || throw(BoundsError(A, r))
        return interpret_indices_disk(A, (r..., 1))
    end
end


function interpret_indices_disk(A::AbstractVector, r::NTuple{1,AbstractVector})
    inds = map(_convert_index, r, size(A))
    resh = DimsDropper(findints(r))
    return inds, resh
end

# function interpret_indices_disk(A, r::Tuple{<:AbstractArray{<:Bool}})
#   ba = r[1]
#   if ndims(A)==ndims(ba)
#     inds = getbb(ba)
#     resh = a -> a[view(ba,inds...)]
#     return inds, resh
#   elseif ndims(ba)==1
#     interpret_indices_disk(A,(reshape(ba,size(A)),))
#   else
#     throw(BoundsError(A, r))
#   end
# end

function interpret_indices_disk(A, r::NTuple{1,AbstractVector})
    lininds = first(r)
    cartinds = CartesianIndices(A)
    mi, ma = extrema(view(cartinds, lininds))
    inds = map((i1, i2) -> i1:i2, mi.I, ma.I)
    resh = a -> map(lininds) do ii
        a[cartinds[ii] - mi + oneunit(mi)]
    end
    return inds, resh
end

struct Reshaper{I}
    reshape_indices::I
end
(r::Reshaper)(a) = reshape(a, r.reshape_indices)
result_size(_, r::Reshaper) = r.reshape_indices
struct DimsDropper{D}
    d::D
end
(d::DimsDropper)(a) = length(d.d) == ndims(a) ? a[1] : dropdims(a; dims=d.d)
function result_size(inds, d::DimsDropper)
    return getindex.(Ref(inds), filter(!in(d.d), ntuple(identity, length(inds))))
end

struct TransformStack{S}
    s::S
end
transformstack(_::Union{Reshaper,DimsDropper,typeof(identity)}, s2::Reshaper) = s2
transformstack(s...) = TransformStack(filter(!=(identity), s))
(s::TransformStack)(a) = ∘(s.s...)(a)

maybe2range(i::AbstractRange) = i
function maybe2range(inds::T)::Union{T,StepRange{Int,Int},UnitRange{Int}} where T<:AbstractVector{Int}
    Base.has_offset_axes(inds) && throw(ArgumentError("Indexing with Offset Arrays is not allowed"))
    if length(inds) == 1 
        return only(inds):only(inds)
    end
    rstep = inds[2]-inds[1]
    for i in 3:length(inds)
        if inds[i] - inds[i-1] != rstep
            return inds
        end
    end
    if rstep == 1
        return first(inds):last(inds)
    else
        return first(inds):rstep:last(inds)
    end
end


#Some helper functions
"For two given tuples return a truncated version of both so they have common length"
commonlength(a, b) = _commonlength((first(a),), (first(b),), Base.tail(a), Base.tail(b))
commonlength(::Tuple{}, b) = (), ()
commonlength(a, ::Tuple{}) = (), ()
commonlength(a::Tuple{}, ::Tuple{}) = (), ()
function _commonlength(a1, b1, a, b)
    return _commonlength((a1..., first(a)), (b1..., first(b)), Base.tail(a), Base.tail(b))
end
_commonlength(a1, b1, ::Tuple{}, b) = (a1, b1)
_commonlength(a1, b1, a, ::Tuple{}) = (a1, b1)
_commonlength(a1, b1, a::Tuple{}, ::Tuple{}) = (a1, b1)

"Find the indices of elements containing integers in a Tuple"
findints(x) = _findints((), 1, x...)
_findints(c, i, x::Integer, rest...) = _findints((c..., i), i + 1, rest...)
_findints(c, i, x, rest...) = _findints(c, i + 1, rest...)
_findints(c, i) = c
#Normal indexing for a full subset of an array


include("chunks.jl")

macro implement_getindex(t)
    t = esc(t)
    quote
        Base.getindex(a::$t, i...) = getindex_disk(a, i...)

        function Base.getindex(a::$t, i::ChunkIndex)
            cs = eachchunk(a)
            inds = cs[i.I]
            return wrapchunk(i.chunktype, a[inds...], inds)
        end
        function DiskArrays.ChunkIndices(a::$t; offset=false)
            return ChunkIndices(
                Base.OneTo.(size(eachchunk(a))), offset ? OffsetChunks() : OneBasedChunks()
            )
        end
    end
end

macro implement_setindex(t)
    t = esc(t)
    quote
        Base.setindex!(a::$t, v::AbstractArray, i...) = setindex_disk!(a, v, i...)

        # Add an extra method if a single number is given
        function Base.setindex!(a::$t{<:Any,N}, v, i...) where {N}
            return Base.setindex!(a, fill(v, ntuple(i -> 1, N)...), i...)
        end

        function Base.setindex!(a::$t, v::AbstractArray, i::ChunkIndex)
            cs = eachchunk(a)
            inds = cs[i.I]
            return setindex_disk!(a, v, inds...)
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", X::AbstractDiskArray)
    return println(io, "Disk Array with size ", join(size(X), " x "))
end
function Base.show(io::IO, X::AbstractDiskArray)
    return println(io, "Disk Array with size ", join(size(X), " x "))
end
