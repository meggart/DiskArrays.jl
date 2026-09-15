
# Implementation macro

function Base._mapreduce(f, op, ::IndexCartesian, v::AbstractDiskArray)
    mapreduce(op, eachchunk(v)) do cI
        a = v[to_ranges(cI)...]
        mapreduce(f, op, a)
    end
end
function Base.mapreducedim!(f, op, R::AbstractArray, a::AbstractDiskArray)
    diskarrays_mapreducedim_impl(f, op, R, a, get_backend(compute_backend))
end

function diskarrays_mapreducedim_impl(f, op, R, a::AbstractDiskArray, ::ComputeBackend)
    _diskarrays_mapreducedim_default!(f, op, R, a)
end

function _diskarrays_mapreducedim_default!(f, op, R, a::AbstractDiskArray)
    foreach(eachchunk(a)) do cI
        aview = a[to_ranges(cI)...]
        ainds = map(
            (cinds, arsize) -> arsize == 1 ? Base.OneTo(1) : cinds,
            to_ranges(cI),
            size(R),
        )
        Base.mapreducedim!(f, op, view(R, ainds...), aview)
    end
    return R
end

function Base.mapfoldl_impl(f, op, nt::NamedTuple{()}, itr::AbstractDiskArray)
    cc = eachchunk(itr)
    isempty(cc) &&
        return Base.mapreduce_empty_iter(f, op, itr, Base.IteratorEltype(itr))
    return Base.mapfoldl_impl(f, op, nt, itr, cc)
end
function Base.mapfoldl_impl(f, op, nt::NamedTuple{()}, itr::AbstractDiskArray, cc)
    y = first(cc)
    a = itr[to_ranges(y)...]
    init = mapfoldl(f, op, a)
    return Base.mapfoldl_impl(f, op, (init=init,), itr, Iterators.drop(cc, 1))
end
function Base.mapfoldl_impl(f, op, nt::NamedTuple{(:init,)}, itr::AbstractDiskArray, cc)
    init = nt.init
    for y in cc
        a = itr[to_ranges(y)...]
        init = mapfoldl(f, op, a; init=init)
    end
    return init
end

Base.mapreduce(f, op, a::AbstractDiskArray; dims=:, init=Base._InitialValue(), kwargs...) =
    diskarrays_mapreduce_impl(f, op, a, dims, init, get_backend(compute_backend); kwargs...)

diskarrays_mapreduce_impl(f, op, a, dims, init, backend::ComputeBackend) =
    _diskarrays_mapreduce_impl(f, op, a, dims, init, backend)

_diskarrays_mapreduce_impl(f, op, a::AbstractDiskArray, ::Colon, ::Base._InitialValue, ::ComputeBackend) =
    Base.mapfoldl(f, op, a)

_diskarrays_mapreduce_impl(f, op, a::AbstractDiskArray, ::Colon, init, ::ComputeBackend) =
    Base.mapfoldl_impl(f, op, init, a)

_diskarrays_mapreduce_impl(f, op, a::AbstractDiskArray, dims, init, ::ComputeBackend) =
    Base.mapreducedim!(f, op, fill(init, Base.reduced_indices(a, dims)), a)

_diskarrays_mapreduce_impl(f, op, a::AbstractDiskArray, dims, ::Base._InitialValue, ::ComputeBackend) =
    Base.mapreducedim!(f, op, Base.reducedim_init(f, op, a, dims), a)



# ── prod, all, any, minimum, maximum: current chunk iteration ─────────
# ── Public convenience wrappers (sum, prod, all, any, min, max) ─────
# These are actually redundant and would usually fall back to mapreduce except the ability
# to short-circuit for any and all. We still define these here for DefaultBackend
# to provide other backends the possibility to overload
for fname in (:sum, :prod, :all, :any, :minimum, :maximum)
    fnameimpl = Symbol("diskarrays_$(fname)_impl")
    fnamedef = Symbol("_diskarrays_$(fname)_default")
    @eval begin
        function Base.$fname(f::Function, a::AbstractDiskArray; kwargs...)
            $(fnameimpl)(f, a, get_backend(compute_backend); kwargs...)
        end
        Base.$fname(a::AbstractDiskArray; kwargs...) = Base.$fname(identity, a; kwargs...)

        $(fnameimpl)(f, a::AbstractDiskArray, ::ComputeBackend; dims=:) =
            $(fnamedef)(f, a; dims)

        function $(fnamedef)(f, a::AbstractDiskArray; dims=:)
            if dims === Colon()
                $fname(eachchunk(a)) do chunk
                    $fname(f, a[chunk...])
                end
            else
                # Here we call the base fallback, which will run into the mapreducedim! implementation above
                invoke($fname, Tuple{typeof(f),AbstractArray{eltype(a),ndims(a)}}, f, a; dims)
            end
        end
    end
end

Base.count(v::AbstractDiskArray) = count(identity, v::AbstractDiskArray)
Base.count(f, v::AbstractDiskArray) = diskarrays_count_impl(f, v, get_backend(compute_backend))
function diskarrays_count_impl(f, v::AbstractDiskArray, ::DefaultBackend)
    sum(eachchunk(v)) do chunk
        count(f, v[chunk...])
    end
end

Base.unique(v::AbstractDiskArray) = unique(identity, v)
Base.unique(f, v::AbstractDiskArray) = diskarrays_unique_impl(f, v, get_backend(compute_backend))
function diskarrays_unique_impl(f, v::AbstractDiskArray, ::DefaultBackend)
    reduce((unique(f, v[c...]) for c in eachchunk(v))) do acc, u
        unique!(f, append!(acc, u))
    end
end


function Base.extrema(f::Function, a::AbstractDiskArray; kwargs...)
    diskarrays_extrema_impl(f, a, get_backend(compute_backend); kwargs...)
end
Base.extrema(a::AbstractDiskArray; kwargs...) = extrema(identity, a; kwargs...)

diskarrays_extrema_impl(f, a::AbstractDiskArray, ::DefaultBackend; kwargs...) =
    invoke(extrema, Tuple{typeof(f),AbstractArray{eltype(a),ndims(a)}}, f, a; kwargs...)


# Stubs for functions that will be created once Statistics.jl is loaded
function diskarrays_mean_impl end
function diskarrays_median_impl end