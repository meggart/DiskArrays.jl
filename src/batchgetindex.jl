struct ReIndexer{I}
    inds::Val{I}
end
indmask(::ReIndexer{I}) where I = I
struct Colon_From{N} end
getind(_, i, j) = last(i)[j]
getind(data, i, ::Colon_From{N}) where N = shrinkaxis(first(i)[N],axes(data,N))

function getrangeinsert(i1)
    lastcol = 0
    lasticol = 0
    allcols = ()

    while (nextcol = findnext(i->!isa(i,Integer),i1,lastcol+1)) !== nothing
        if nextcol == lastcol + 1
            lasticol = lasticol+1
        else
            lasticol = lasticol+2
        end
        lastcol = nextcol
        allcols = (allcols...,nextcol=>lasticol)
    end
    allcols
end

carttotuple(i::CartesianIndex) = i.I
carttotuple(i::Tuple) = i
carttotuple(i::Integer) = (Int(i),)


getnd(::Type{<:Tuple{Vararg{Any,N}}}) where N = N


function to_batchgetindex(a,i::AbstractVector{Int})
    ci = CartesianIndices(size(a))
    to_batchgetindex(a,ci[i])
end
function to_batchgetindex(a,i...)
    inds = ()
    idim = 1
    ibcdim = 1
    for ind in i
        if isa(ind,AbstractArray) && !isa(ind,AbstractUnitRange)
            if eltype(ind) <: Bool
                o = carttotuple.(findall(ind))
                outshape = (ones(Int,ibcdim-1)...,length(o))
                inds = (inds...,reshape(o,outshape))
                idim = idim + ndims(ind)
                ibcdim = ibcdim+1
            elseif eltype(ind) <: Union{CartesianIndex,Tuple,Integer}
                o = carttotuple.(ind)
                N = getnd(eltype(o))
                outshape = (ones(Int,ibcdim-1)...,size(ind)...)
                inds = (inds...,reshape(o,outshape))
                idim = idim+N
                ibcdim = ibcdim + ndims(ind)
            else
                error("")
            end
        elseif isa(ind,Colon)
            inds = (inds...,Ref((1:size(a,idim),)))
            idim = idim+1
        else
            inds = (inds...,Ref((ind,)))
            idim = idim+1
        end
    end
    indvec = broadcast(inds...) do i...
        tuple(Iterators.flatten(i)...)
    end
    disk_getindex_batch(a,indvec)
end
                





function disk_getindex_batch(ar,indstoread)

    i1 = first(indstoread)
    inserts = getrangeinsert(i1)
    inds = collect(Any,1:ndims(indstoread))
    for i in inserts
        insert!(inds,last(i),Colon_From{first(i)}())
    end

    outindexer = ReIndexer(Val((inds...,)))
    it = eltype(indstoread)
    affected_chunk_dict = Dict{ChunkIndex{ndims(ar),OffsetChunks},Vector{Tuple{it,NTuple{ndims(indstoread),Int}}}}()
    for ii in CartesianIndices(indstoread)
        for ci in ChunkIndices(findchunk.(eachchunk(ar).chunks,indstoread[ii]),OffsetChunks())
            v = get!(affected_chunk_dict,ci) do 
                it[]
            end
            push!(v,(indstoread[ii],ii.I))
        end
    end
    outsize = collect(size(indstoread))
    for (iax,iins) in inserts
        insert!(outsize,iins,length(i1[iax]))
    end
    outar = zeros(eltype(ar),outsize...)
    for (chunk,inds) in affected_chunk_dict
        data = ar[chunk]
        filldata!(outar,data,inds,outindexer)
    end
    outar
end

function filldata!(outar,data,inds,::ReIndexer{M}) where M
    for i in inds
        inew = map(j -> getind(data, i, j), M)
        outar[inew...] = data[shrinkaxis.(i[1],axes(data))...]
    end
end

function shrinkaxis(a,b) 
    max(first(a),first(b)):min(last(a),last(b))
end
shrinkaxis(a::Int,_) = a
