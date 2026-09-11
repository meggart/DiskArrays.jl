
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

Base.mapreduce(f, op, a::AbstractDiskArray; dims=:, init=Base._InitialValue()) =
    diskarrays_mapreduce_impl(f, op, a, dims, init, get_backend(compute_backend))

diskarrays_mapreduce_impl(f, op, a::AbstractDiskArray, ::Colon, ::Base._InitialValue, ::ComputeBackend) =
    Base.mapfoldl(f, op, a)

diskarrays_mapreduce_impl(f, op, a::AbstractDiskArray, ::Colon, init, ::ComputeBackend) =
    Base.mapfoldl_impl(f, op, init, a)

diskarrays_mapreduce_impl(f, op, a::AbstractDiskArray, dims, init, ::ComputeBackend) =
    Base.mapreducedim!(f, op, fill(init, Base.reduced_indices(a, dims)), a)

diskarrays_mapreduce_impl(f, op, a::AbstractDiskArray, dims, ::Base._InitialValue, ::ComputeBackend) =
    Base.mapreducedim!(f, op, Base.reducedim_init(f, op, a, dims), a)



# ── prod, all, any, minimum, maximum: current chunk iteration ─────────
# ── Public convenience wrappers (sum, prod, all, any, min, max) ─────
for fname in (:sum, :prod, :all, :any, :minimum, :maximum)
    fnameimpl = Symbol("diskarrays_$(fname)_impl")
    fnamedef = Symbol("_diskarrays_$(fname)_default")
    @eval begin
        function Base.$fname(f::Function, a::AbstractDiskArray)
            $(fnameimpl)(f, a, get_backend(compute_backend))
        end
        Base.$fname(a::AbstractDiskArray) = Base.$fname(identity, a)

        $(fnameimpl)(f, a::AbstractDiskArray, ::ComputeBackend) =
            $(fnamedef)(f, a)

        function $(fnamedef)(f, a::AbstractDiskArray)
            $fname(eachchunk(a)) do chunk
                $fname(f, a[chunk...])
            end
        end
    end
end

# diskarrays_mapreduce_impl(f, op, a::AbstractDiskArray, ::ComputeBackend) =
#     _diskarrays_mapreduce_default(f, op, a)

# function _diskarrays_mapreduce_default(f, op, a::AbstractDiskArray)
#     Base.mapfoldl(f, op, a)
# end



Base.count(v::AbstractDiskArray) = count(identity, v::AbstractDiskArray)
function Base.count(f, v::AbstractDiskArray)
    sum(eachchunk(v)) do chunk
        count(f, v[chunk...])
    end
end

Base.unique(v::AbstractDiskArray) = unique(identity, v)
function Base.unique(f, v::AbstractDiskArray)
    reduce((unique(f, v[c...]) for c in eachchunk(v))) do acc, u
        unique!(f, append!(acc, u))
    end
end
