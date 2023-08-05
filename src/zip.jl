
# Rechunk using the chunks of the first 
function zip_disk(As...)
    chunks = reduce(As; init=nothing) do acc, A
        if acc == nothing && haschunks(A) == Chunked() 
            eachchunk(A) 
        else
            nothing
        end
    end
    if isnothing(chunks)
        chunks = eachchunk(first(As))
    end
    rechunked = map(As) do A
        RechunkedDiskArray(A, chunks)
    end
    return Base.Iterators.Zip(rechunked)
end

macro implement_zip(t)
    t = esc(t)
    quote
        Base.zip(A1::$t, As::AbstractArray...) = zip_disk(A1, As...)
        Base.zip(A1::AbstractArray, A2::$t, As::AbstractArray...) = zip_disk(A1, A2, As...)
        Base.zip(A1::$t, A2::$t, As::AbstractArray...) = zip_disk(A1, A2, As...)
    end
end
