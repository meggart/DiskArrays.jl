
# Rechunk using the chunks of the first Chunked array
# This forces the iteration order to be the same for 
# all arrays.
function zip_disk(As...)
    # Get the chunkes of the first Chunked array
    chunks = reduce(As; init=nothing) do acc, A
        if acc == nothing && haschunks(A) == Chunked() 
            eachchunk(A) 
        else
            nothing
        end
    end
    if isnothing(chunks)
        return Base.Iterators.Zip(As)
    else
        rechunked = map(As) do A
            RechunkedDiskArray(A, chunks)
        end
        return Base.Iterators.Zip(rechunked)
    end
end

_zip_error() = throw(ArgumentError("Cannot `zip` a disk array with an iterator"))

Base.zip(A1::AbstractDiskArray, A2::AbstractDiskArray, As::AbstractArray...) = zip_disk(A1, A2, As...)
Base.zip(A1::AbstractDiskArray, A2::AbstractArray, As::AbstractArray...) = zip_disk(A1, A1, As...)
Base.zip(A1::AbstractArray, A2::AbstractDiskArray, As::AbstractArray...) = zip_disk(A1, A2, As...)

Base.zip(::AbstractDiskArray, x, xs...) = _zip_error()
Base.zip(x, ::AbstractDiskArray, xs...) = _zip_error()
Base.zip(x::AbstractDiskArray, ::AbstractDiskArray, xs...) = _zip_error()

macro implement_zip(t)
    t = esc(t)
    quote
        Base.zip(A1::$t, A2::$t, As::AbstractArray...) = $zip_disk(A1, A1, As...)
        Base.zip(A1::$t, A2::AbstractArray, As::AbstractArray...) = $zip_disk(A1, A2, As...)
        Base.zip(A1::AbstractArray, A2::$t, As::AbstractArray...) = $zip_disk(A1, A2, As...)

        Base.zip(A1::AbstractDiskArray, A2::$t, As::AbstractArray...) = $zip_disk(A1, A2, As...)
        Base.zip(A1::$t, A2::AbstractDiskArray, As::AbstractArray...) = $zip_disk(A1, A2, As...)

        Base.zip(::$t, x, xs...) = $_zip_error()
        Base.zip(x, ::$t, xs...) = $_zip_error()
        Base.zip(x, ::$t, xs...) = $_zip_error()
    end
end

