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
function batchgetindex(a, i...)
    indvec = create_indexvector(a, i)
    return disk_getindex_batch(a, indvec)
end

function prepare_disk_getindex_batch(ar, indstoread)
    i1 = first(indstoread)
    inserts = getrangeinsert(i1)
    inds = collect(Any, 1:ndims(indstoread))
    for i in inserts
        insert!(inds, last(i), Colon_From{first(i)}())
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
    return (; outsize, affected_chunk_dict, indexer=outindexer)
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
    return outar
end

function disk_getindex_batch(ar, indstoread)
    prep = prepare_disk_getindex_batch(ar, indstoread)
    outar = zeros(eltype(ar), prep.outsize...)
    return disk_getindex_batch!(outar, ar, indstoread; prep=prep)
end

function filldata!(outar, data, inds, ::ReIndexer{M}) where {M}
    for i in inds
        inew = map(j -> getind(data, i, j), M)
        outar[inew...] = data[shrinkaxis.(i[1], axes(data))...]
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
function readblock!(A::AbstractDiskArray, A_ret, r::AbstractVector...)
    #Check how sparse the vectors are, we look at the largest stride in the inputs
    need_batch = map(approx_chunksize(eachchunk(A)), r) do cs, ids
        length(ids) == 1 && return false
        largest_jump = maximum(diff(ids))
        mi, ma = extrema(ids)
        return largest_jump > cs && length(ids) / (ma - mi) < 0.5
    end
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

function writeblock!(A::AbstractDiskArray, A_ret, r::AbstractVector...)
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
