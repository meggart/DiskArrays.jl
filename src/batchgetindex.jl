function has_chunk_gap(cs,ids::AbstractVector{<:Integer})
    #Find largest jump in indices
    largest_jump = foldl(ids,init=(0,first(ids))) do (largest,last),next
        largest = max(largest,next-last)
        (largest,next)
    end |> first
    largest_jump > first(cs)
end
#Return true for all multidimensional indices for now, could be optimised in the future
has_chunk_gap(cs,ids) = true

#Compute the number of possible indices in the hyperrectangle
span(v::AbstractVector{<:Integer}) = 1 -(-(extrema(v)...))
function span(v::AbstractVector{CartesianIndex{N}}) where N
    minind,maxind = extrema(v)
    prod((maxind-mindind+oneunit(minind)).I)
end
function span(v::AbstractArray{Bool})
    mindind,maxind = extrema(view(CartesianIndices(size(v)),v))
    prod((maxind-mindind+oneunit(minind)).I)
end
#The number of indices to actually be read
numind(v::AbstractArray{Bool}) = sum(v)
numind(v::Union{AbstractVector{<:Integer},AbstractVector{<:CartesianIndex}})=length(v)

function is_sparse_index(ids; density_threshold = 0.5)
    indexdensity = numind(ids) / span(ids)
    return indexdensity < density_threshold
end

# Define fallbacks for reading and writing sparse data
#= function _readblock!(A::AbstractArray, A_ret, r::AbstractVector...)
    if need_batch(A,r)
        # Fall back to batchgetindex to do the readblock
        A_ret .= batchgetindex(A, r...)
    else
        # Read the whole ranges in one go and subset what is required
        mi, ma = map(minimum, r), map(maximum, r)
        output_array_size = map((a, b) -> b - a + 1, mi, ma)
        A_temp = similar(A_ret, output_array_size)
        readranges = map(:, mi, ma)
        readblock!(A, A_temp, readranges...)
        subset = map(ir -> ir .- (minimum(ir) .- 1), r)
        A_ret .= view(A_temp, subset...)
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
end =#

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
