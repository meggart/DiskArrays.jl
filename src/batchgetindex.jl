struct ReIndexer{I}
    inds::Val{I}
end
struct Colon_From{N} end
getind(_, i, j) = last(i)[j]
getind(data, i, ::Colon_From{N}) where {N} = shrinkaxis(first(i)[N], axes(data, N))

function getrangeinsert(i1)
    lastcol = 0
    lasticol = 0
    allcols = ()

    while (nextcol = findnext(i -> !isa(i, Integer), i1, lastcol + 1)) !== nothing
        if nextcol == lastcol + 1
            lasticol = lasticol + 1
        else
            lasticol = lasticol + 2
        end
        lastcol = nextcol
        allcols = (allcols..., nextcol => lasticol)
    end
    return allcols
end

carttotuple(i::CartesianIndex) = i.I
carttotuple(i::Tuple) = i
carttotuple(i::Integer) = (Int(i),)

getnd(::Type{<:Tuple{Vararg{Any,N}}}) where {N} = N

function create_indexvector(a, i)
    inds = ()
    idim = 1
    ibcdim = 1
    for ind in i
        if isa(ind, AbstractArray) && !isa(ind, AbstractUnitRange)
            if eltype(ind) <: Bool
                o = carttotuple.(findall(ind))
                outshape = (ones(Int, ibcdim - 1)..., length(o))
                inds = (inds..., reshape(o, outshape))
                idim = idim + ndims(ind)
                ibcdim = ibcdim + 1
            elseif eltype(ind) <: Union{CartesianIndex,Tuple,Integer}
                o = carttotuple.(ind)
                N = getnd(eltype(o))
                outshape = (ones(Int, ibcdim - 1)..., size(ind)...)
                inds = (inds..., reshape(o, outshape))
                idim = idim + N
                ibcdim = ibcdim + ndims(ind)
            else
                error("")
            end
        elseif isa(ind, Colon)
            inds = (inds..., Ref((1:size(a, idim),)))
            idim = idim + 1
        else
            inds = (inds..., Ref((ind,)))
            idim = idim + 1
        end
    end
    broadcast(inds...) do i...
        tuple(Iterators.flatten(i)...)
    end
end

function batchgetindex(a, i::AbstractVector{Int})
    ci = CartesianIndices(size(a))
    return batchgetindex(a, ci[i])
end

# indexing with vector of integers from NCDatasets 0.12.17 (MIT)

# computes the shape of the array of size `sz` after applying the indexes
# size(a[indexes...]) == _shape_after_slice(size(a),indexes...)

# the difficulty here is to make the size inferrable by the compiler
@inline _shape_after_slice(sz,indexes...) = __sh(sz,(),1,indexes...)
@inline __sh(sz,sh,n,i::Integer,indexes...) = __sh(sz,sh,               n+1,indexes...)
@inline __sh(sz,sh,n,i::Colon,  indexes...) = __sh(sz,(sh...,sz[n]),    n+1,indexes...)
@inline __sh(sz,sh,n,i,         indexes...) = __sh(sz,(sh...,length(i)),n+1,indexes...)
@inline __sh(sz,sh,n) = sh


# convert e.g. vector indices to a list of ranges
# [1,2,3,6,7] into [1:3, 6:7]
# 1:10 into [1:10]
to_range_list(index::Integer,len) = index
to_range_list(index::Colon,len) = [1:len]
to_range_list(index::AbstractRange,len) = [index]

function to_range_list(index::Vector{T},len) where T <: Integer
    grow(istart) = istart[begin]:(istart[end]+step(istart))

    baseindex = 1
    indices_ranges = UnitRange{T}[]

    while baseindex <= length(index)
        range = index[baseindex]:index[baseindex]
        range_test = grow(range)
        index_view = @view index[baseindex:end]

        while checkbounds(Bool,index_view,length(range_test)) &&
            (range_test[end] == index_view[length(range_test)])

            range = range_test
            range_test = grow(range_test)
        end

        push!(indices_ranges,range)
        baseindex += length(range)
    end

    # make sure we did not lose any indices
    @assert reduce(vcat,indices_ranges,init=T[]) == index
    return indices_ranges
end

_range_indices_dest(ri_dest) = ri_dest
_range_indices_dest(ri_dest,i::Integer,rest...) = _range_indices_dest(ri_dest,rest...)
function _range_indices_dest(ri_dest,v,rest...)
    baseindex = 0
    ind = similar(v,0)
    for r in v
        rr = 1:length(r)
        push!(ind,baseindex .+ rr)
        baseindex += length(r)
    end

    _range_indices_dest((ri_dest...,ind),rest...)
end

# rebase in source indices to the destination array
# for example if we load range 1:3 and 7:10 we will write to ranges 1:3 and 4:7
range_indices_dest(ri...) = _range_indices_dest((),ri...)

function batchgetindex(a::TA,indices::Vararg{Union{Int,Colon,AbstractRange{<:Integer},Vector{Int}},N}) where TA <: AbstractArray{T,N} where {T,N}
    sz_source = size(a)
    ri = to_range_list.(indices,sz_source)
    sz_dest = _shape_after_slice(sz_source,indices...)
    ri_dest = range_indices_dest(ri...)

    @debug "transform vector of indices to ranges" ri_dest ri

    dest = Array{eltype(a),length(sz_dest)}(undef,sz_dest)
    for R in CartesianIndices(length.(ri))
        ind_source = ntuple(i -> ri[i][R[i]],N)
        ind_dest = ntuple(i -> ri_dest[i][R[i]],length(ri_dest))
        dest[ind_dest...] = a[ind_source...]
    end
    return dest
end



function batchgetindex(a, i...)
    indvec = create_indexvector(a, i)
    return disk_getindex_batch(a, indvec)
end

function prepare_disk_getindex_batch(ar, indstoread)
    i1 = first(indstoread)
    inserts = getrangeinsert(i1)
    inds = collect(Any, 1:ndims(indstoread))
    offsets = zeros(Int, ndims(indstoread))
    for i in inserts
        insert!(inds, last(i), Colon_From{first(i)}())
        insert!(offsets, last(i), first(i1[first(i)]) - 1)
    end
    outindexer = ReIndexer(Val((inds...,)))
    it = eltype(indstoread)
    affected_chunk_dict = Dict{
        ChunkIndex{ndims(ar),OffsetChunks},Vector{Tuple{it,NTuple{ndims(indstoread),Int}}}
    }()
    for ii in CartesianIndices(indstoread)
        for ci in
            ChunkIndices(findchunk.(eachchunk(ar).chunks, indstoread[ii]), OffsetChunks())
            v = get!(affected_chunk_dict, ci) do
                it[]
            end
            push!(v, (indstoread[ii], ii.I))
        end
    end
    outsize = collect(size(indstoread))
    for (iax, iins) in inserts
        insert!(outsize, iins, length(i1[iax]))
    end
    return (; outsize, offsets, affected_chunk_dict, indexer=outindexer)
end

function disk_getindex_batch!(outar, ar, indstoread; prep=nothing)
    if prep === nothing
        prep = prepare_disk_getindex_batch(ar, indstoread)
    end
    size(outar) == (prep.outsize...,) || throw(
        DimensionMismatch("Output size $(prep.outsize) expected but got $(size(outar))")
    )
    for (chunk, inds) in prep.affected_chunk_dict
        data = ar[chunk]
        filldata!(outar, data, inds, prep.indexer)
    end
    return parent(outar)
end

function disk_getindex_batch(ar, indstoread)
    prep = prepare_disk_getindex_batch(ar, indstoread)
    outar = OffsetArray(Array{eltype(ar)}(undef, prep.outsize...), prep.offsets...)
    return disk_getindex_batch!(outar, ar, indstoread; prep=prep)
end

function filldata!(outar, data, inds, ::ReIndexer{M}) where {M}
    for i in inds
        inew = map(j -> getind(data, i, j), M)
        tofill = data[shrinkaxis.(i[1], axes(data))...]
        outar[inew...] = tofill
    end
end

function batchsetindex!(a, v, i::AbstractVector{Int})
    ci = CartesianIndices(size(a))
    return batchsetindex!(a, v, ci[i])
end
function batchsetindex!(a, v, i...)
    indvec = create_indexvector(a, i)
    return disk_setindex_batch!(a, v, indvec)
end

function disk_setindex_batch!(ar, v, indstoread)
    prep = prepare_disk_getindex_batch(ar, indstoread)
    size(v) == (prep.outsize...,) ||
        throw(DimensionMismatch("Output size $(prep.outsize) expected but got $(size(v))"))
    for (chunk, inds) in prep.affected_chunk_dict
        data = ar[chunk]
        writedata!(v, data, inds, prep.indexer)
        ar[chunk] = data
    end
    return v
end
function writedata!(v, data, inds, ::ReIndexer{M}) where {M}
    for i in inds
        inew = map(j -> getind(data, i, j), M)
        data[shrinkaxis.(i[1], axes(data))...] = v[inew...]
    end
end

function shrinkaxis(a, b)
    return max(first(a), first(b)):min(last(a), last(b))
end
shrinkaxis(a::Int, _) = a

# Define fallbacks for reading and writing sparse data
function _readblock!(A::AbstractArray, A_ret, r::AbstractVector...)
    #Check how sparse the vectors are, we look at the largest stride in the inputs
    need_batch = map(approx_chunksize(eachchunk(A)), r) do cs, ids
        length(ids) == 1 && return false
        largest_jump = maximum(diff(ids))
        mi, ma = extrema(ids)
        return largest_jump > cs && length(ids) / (ma - mi) < 0.5
    end
    # What TODO?: necessary to avoid infinite recursion
    need_batch = false
    if any(need_batch)
        A_ret .= batchgetindex(A, r...)
    else
        mi, ma = map(minimum, r), map(maximum, r)
        A_temp = similar(A_ret, map((a, b) -> b - a + 1, mi, ma))
        readblock!(A, A_temp, map(:, mi, ma)...)
        A_ret .= view(A_temp, map(ir -> ir .- (minimum(ir) .- 1), r)...)
    end
    return nothing
end

function _writeblock!(A::AbstractArray, A_ret, r::AbstractVector...)
    #Check how sparse the vectors are, we look at the largest stride in the inputs
    need_batch = map(approx_chunksize(eachchunk(A)), r) do cs, ids
        length(ids) == 1 && return false
        largest_jump = maximum(diff(ids))
        mi, ma = extrema(ids)
        return largest_jump > cs && length(ids) / (ma - mi) < 0.5
    end
    if any(need_batch)
        batchsetindex!(A, A_ret, r...)
    else
        mi, ma = map(minimum, r), map(maximum, r)
        A_temp = similar(A_ret, map((a, b) -> b - a + 1, mi, ma))
        A_temp[map(ir -> ir .- (minimum(ir) .- 1), r)...] = A_ret
        writeblock!(A, A_temp, map(:, mi, ma)...)
    end
    return nothing
end

macro implement_batchgetindex(t)
    t = esc(t)
    quote
        # Define fallbacks for reading and writing sparse data
        function DiskArrays.readblock!(A::$t, A_ret, r::AbstractVector...)
            return _readblock!(A, A_ret, r...)
        end

        function DiskArrays.writeblock!(A::$t, A_ret, r::AbstractVector...)
            return _writeblock!(A, A_ret, r...)
        end
    end
end
