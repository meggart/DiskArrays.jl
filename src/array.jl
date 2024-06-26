
macro implement_array_methods(t)
    t = esc(t)
    quote
        Base.Array(a::$t) = $_Array(a)
        Base.collect(a::$t) = $_Array(a)
        Base.copyto!(dest::$t, source::AbstractArray) = $_copyto!(dest, source)
        Base.copyto!(dest::AbstractArray, source::$t) = $_copyto!(dest, source)
        Base.copyto!(dest::$t, source::$t) = $_copyto!(dest, source)
        function Base.copyto!(
            dest::$t, Rdest::CartesianIndices, src::AbstractArray, Rsrc::CartesianIndices
        )
            return $_copyto!(dest, Rdest, src, Rsrc)
        end
        function Base.copyto!(
            dest::AbstractArray, Rdest::CartesianIndices, src::$t, Rsrc::CartesianIndices
        )
            return $_copyto!(dest, Rdest, src, Rsrc)
        end
        function Base.copyto!(
            dest::$t, Rdest::CartesianIndices, src::$t, Rsrc::CartesianIndices
        )
            return $_copyto!(dest, Rdest, src, Rsrc)
        end
        # For ambiguity
        copyto!(dest::PermutedDimsArray, src::$t) = DiskArrays._copyto!(dest, src)
        function copyto!(dest::PermutedDimsArray{T,N}, src::$t{T,N}) where {T,N}
            return $_copyto!(dest, src)
        end

        Base.reverse(a::$t; dims=:) = $_reverse(a, dims)
        Base.reverse(a::$t{<:Any,1}) = $_reverse1(a)
        Base.reverse(a::$t{<:Any,1}, start::Integer, stop::Integer=lastindex(a)) = $_reverse1(a, start, stop)

        # Here we extend the unexported `_replace` method, but we replicate 
        # much less Base functionality by extending it rather than `replace`.
        function Base._replace!(new::Base.Callable, res::AbstractArray, A::$t, count::Int)
            return $_replace!(new, res, A, count)
        end
        function Base._replace!(new::Base.Callable, res::$t, A::AbstractArray, count::Int)
            return $_replace!(new, res, A, count)
        end
        function Base._replace!(new::Base.Callable, res::$t, A::$t, count::Int)
            return $_replace!(new, res, A, count)
        end
    end
end

# Use broadcast to copy to a new Array
_Array(a::AbstractArray{T,N}) where {T,N} = a[ntuple(_ -> :, Val{N}())...]
_Array(a::AbstractArray{T,0}) where {T} = fill(a[])

# Use broadcast to copy
function _copyto!(dest::AbstractArray{<:Any,N}, source::AbstractArray{<:Any,N}) where {N}
    return dest .= source
end
function _copyto!(dest::AbstractArray, source::AbstractArray)
    # TODO make this more specific so we are reshaping the Non-DiskArray more often.
    reshape(dest, size(source)) .= source
    return dest
end

function _copyto!(dest, Rdest, src, Rsrc)
    if size(Rdest) != size(Rsrc)
        throw(ArgumentError("source and destination must have same size (got $(size(Rsrc)) and $(size(Rdest)))"))
    end

    if isempty(Rdest)
        # This check is here to catch #168
        return dest
    end
    view(dest, Rdest) .= view(src, Rsrc)
end
# Use a view for lazy reverse
_reverse(a, ::Colon) = _reverse(a, ntuple(identity, ndims(a)))
_reverse(a, dims::Int) = _reverse(a, (dims,))
function _reverse(A, dims::Tuple)
    rev_axes = map(ntuple(identity, ndims(A)), axes(A)) do d, a
        ax = StepRange(a)
        d in dims ? reverse(ax) : ax
    end
    return view(A, rev_axes...)
end
_reverse1(a) = _reverse(a, 1)
function _reverse1(a, start::Int, stop::Int) 
    inds = [firstindex(a):start-1; stop:-1:start; stop+1:lastindex(a)]
    return view(a, inds)
end

# Use broadcast instead of a loop. 
# The `count` argument is disallowed as broadcast is not sequential.
function _replace!(new, res::AbstractArray, A::AbstractArray, count::Int)
    count < length(res) &&
        throw(ArgumentError("`replace` on DiskArrays objects cannot use a count value"))
    return broadcast!(new, res, A)
end
