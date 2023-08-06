
macro implement_array_methods(t)
    t = esc(t)
    quote
        Base.Array(a::$t) = $_Array(a)
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
        function copyto!(
            dest::$t{T,2}, 
            Rdest::CartesianIndices{2,R} where R<:Tuple{OrdinalRange{Int64, Int64}, OrdinalRange{Int64, Int64}}, 
            src::SparseArrays.AbstractSparseMatrixCSC{T}, 
            Rsrc::CartesianIndices{2,R} where R<:Tuple{OrdinalRange{Int64, Int64}, OrdinalRange{Int64, Int64}}
        ) where T
            return $_copyto!(dest, Rdest, src, Rsrc)
        end
        copyto!(dest::PermutedDimsArray, src::$t) = DiskArrays._copyto!(dest, src)
        copyto!(dest::PermutedDimsArray{T,N}, src::$t{T,N}) where {T,N} = 
            $_copyto!(dest, src)
        copyto!(dest::$t{T,2}, src::SparseArrays.CHOLMOD.Dense{T}) where T<:Union{Float64,ComplexF64} =
            $_copyto!(dest, src)
        copyto!(dest::$t{T}, src::SparseArrays.CHOLMOD.Dense{T}) where T<:Union{Float64,ComplexF64} =
            $_copyto!(dest, src)
        copyto!(dest::$t, src::SparseArrays.CHOLMOD.Dense) = 
            $_copyto!(dest, src)
        copyto!(dest::$t{<:Any,2}, src::LinearAlgebra.AbstractQ) =
            $_copyto!(dest, src)
        copyto!(dest::SparseArrays.AbstractCompressedVector, src::$t{<:Any,1}) =
            $_copyto!(dest, src)
        copyto!(dest::$t{<:Any,2}, src::SparseArrays.AbstractSparseMatrixCSC) =
            $_copyto!(dest, src)

        Base.reverse(a::$t, dims=:) = $_reverse(a, dims)
        # For ambiguity
        Base.reverse(a::$t{<:Any,1}, dims::Integer) = $_reverse(a, dims)

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
function _Array(a::AbstractArray{T,N}) where {T,N}
    dest = Array{T,N}(undef, size(a))
    dest .= a
    return dest
end

# Use broadcast to copy
function _copyto!(dest::AbstractArray{<:Any,N}, source::AbstractArray{<:Any,N}) where {N}
    return dest .= source
end
function _copyto!(dest::AbstractArray, source::AbstractArray)
    # TODO make this more specific so we are reshaping the Non-DiskArray more often.
    reshape(dest, size(source)) .= source
    return dest
end
_copyto!(dest, Rdest, src, Rsrc) = view(dest, Rdest) .= view(src, Rsrc)

# Use a view for lazy reverse
_reverse(a, dims::Colon) = _reverse(a, ntuple(identity, ndims(a)))
_reverse(a, dims::Int) = _reverse(a, (dims,))
function _reverse(A, dims::Tuple)
    rev_axes = map(ntuple(identity, ndims(A)), axes(A)) do d, a
        ax = StepRange(a)
        d in dims ? reverse(ax) : ax
    end
    return view(A, rev_axes...)
end

# Use broadcast instead of a loop. 
# The `count` argument is disallowed as broadcast is not sequential.
function _replace!(new, res::AbstractArray, A::AbstractArray, count::Int)
    count < length(res) &&
        throw(ArgumentError("`replace` on DiskArrays objects cannot use a count value"))
    return broadcast!(new, res, A)
end
