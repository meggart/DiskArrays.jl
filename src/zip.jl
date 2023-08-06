
struct DiskZip{Is<:Tuple}
    is::Is
end
Base.iterate(dz::DiskZip) = Base.iterate(Iterators.Zip(dz.is))
Base.iterate(dz::DiskZip, i) = Base.iterate(Iterators.Zip(dz.is), i)
Base.first(dz::DiskZip) = Base.first(Iterators.Zip(dz.is))
Base.last(dz::DiskZip) = Base.last(Iterators.Zip(dz.is))
Base.length(dz::DiskZip) = Base.length(Iterators.Zip(dz.is))
Base.size(dz::DiskZip) = Base.size(Iterators.Zip(dz.is))
Base.IteratorSize(::Type{DiskZip{Is}}) where {Is<:Tuple} =
    Base.IteratorSize(Iterators.Zip{Is})
Base.IteratorEltype(::Type{DiskZip{Is}}) where {Is<:Tuple} =
    Base.IteratorEltype(Iterators.Zip{Is})

# Rechunk using the chunks of the first Chunked array
# This forces the iteration order to be the same for
# all arrays.

function DiskZip(As::AbstractArray{<:Any,N}...) where N
    # Get the chunkes of the first Chunked array
    chunks = reduce(As; init=nothing) do acc, A
        if isnothing(acc) && (haschunks(A) isa Chunked)
            eachchunk(A)
        else
            acc
        end
    end
    if isnothing(chunks)
        return DiskZip(As)
    else
        rechunked = map(As) do A
            RechunkedDiskArray(A, chunks)
        end
        return DiskZip(rechunked)
    end
end
# For now we only allow zip on exact same-sized arrays
DiskZip(As...) = throw(ArgumentError("zip on disk arrays only works with other same-sized AbstractArray"))

# Collect zipped disk arrays in the right order
function Base.collect(z::DiskZip)
    shp = Base._similar_shape(z, Iterators.IteratorSize(z))
    out = Base._similar_for(1:1, eltype(z), z, Iterators.IteratorSize(z), shp)
    itr = iterate(z)
    for I in eachindex(first(z.is))
        out[I] = first(itr)
        itr = iterate(z, last(itr))
    end
    return out
end

_zip_error() = throw(ArgumentError("Cannot `zip` a disk array with an iterator"))

Base.zip(A1::AbstractDiskArray, A2::AbstractDiskArray, As::AbstractArray...) = DiskZip(A1, A2, As...)
Base.zip(A1::AbstractDiskArray, A2::AbstractArray, As::AbstractArray...) = DiskZip(A1, A1, As...)
Base.zip(A1::AbstractArray, A2::AbstractDiskArray, As::AbstractArray...) = DiskZip(A1, A2, As...)

Base.zip(::AbstractDiskArray, x, xs...) = _zip_error()
Base.zip(x, ::AbstractDiskArray, xs...) = _zip_error()
Base.zip(x::AbstractDiskArray, ::AbstractDiskArray, xs...) = _zip_error()

macro implement_zip(t)
    t = esc(t)
    quote
        Base.zip(A1::$t, A2::$t, As::AbstractArray...) = $DiskZip(A1, A1, As...)
        Base.zip(A1::$t, A2::AbstractArray, As::AbstractArray...) = $DiskZip(A1, A2, As...)
        Base.zip(A1::AbstractArray, A2::$t, As::AbstractArray...) = $DiskZip(A1, A2, As...)

        Base.zip(A1::AbstractDiskArray, A2::$t, As::AbstractArray...) = $DiskZip(A1, A2, As...)
        Base.zip(A1::$t, A2::AbstractDiskArray, As::AbstractArray...) = $DiskZip(A1, A2, As...)

        Base.zip(::$t, x, xs...) = $_zip_error()
        Base.zip(x, ::$t, xs...) = $_zip_error()
        Base.zip(x, ::$t, xs...) = $_zip_error()
    end
end
