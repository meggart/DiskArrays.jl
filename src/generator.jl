
struct DiskGenerator{I,F}
    f::F
    iter::I
end
# Copied from `iterate(::Generator, s...) in julia 1.9
function Base.iterate(dg::DiskGenerator, s...)
    y = iterate(dg.iter, s...)
    y === nothing && return nothing
    y = y::Tuple{Any, Any} # try to give inference some idea of what to expect about the behavior of the next line
    return (dg.f(y[1]), y[2])
end
Base.isempty(dg::DiskGenerator) = Base.isempty(dg.iter)
Base.length(dg::DiskGenerator) = Base.length(dg.iter)
Base.ndims(dg::DiskGenerator) = Base.ndims(dg.iter)
Base.size(dg::DiskGenerator) = Base.size(dg.iter)
Base.keys(dg::DiskGenerator) = Base.keys(dg.iter)
Base.IteratorSize(::Type{DiskGenerator{I,F}}) where {I,F} =
    Base.IteratorSize(Iterators.Generator{I,F})
Base.IteratorEltype(::Type{DiskGenerator{I,F}}) where {I,F} =
    Base.IteratorEltype(Iterators.Generator{I,F})

# Collect zipped disk arrays in the right order
# Copied from `collect(::Generator) in julia 1.9
function Base.collect(itr::DiskGenerator{<:AbstractArray{<:Any,N}}) where N
    
    y = iterate(itr)
    shp = axes(itr.iter)
    if y === nothing
        et = Base.@default_eltype(itr)
        return similar(Array{et,N}, shp)
    end
    v1, st = y
    dest = similar(Array{typeof(v1),N}, shp)
    i = y
    for I in eachindex(itr.iter)
        dest[I] = first(i)
        i = iterate(itr, last(i))
    end
    return dest
end

macro implement_generator(t)
    t = esc(t)
    quote
        Base.Generator(f, A::$t) = $DiskGenerator(f, A)
    end
end
