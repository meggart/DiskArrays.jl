# Manual control over scalar indexing
const ALLOW_SCALAR = Ref{Bool}(true)

"""
    allow_scalar(x::Bool)

Specify if a disk array can do scalar indexing, (with all `Int` arguments).

Setting `allow_scalar(false)` can help identify the cause of poor performance.
"""
allow_scalar(x::Bool) = ALLOW_SCALAR[] = x

"""
    can_scalar()

Check if DiskArrays is set to allow scalar indexing, with [`allow_scalar`](@ref).

Returns a `Bool`.
"""
can_scalar() = ALLOW_SCALAR[]

function _scalar_error()
    return error(
        "Scalar indexing with `Int` is very slow, and currently is disallowed. Run DiskArrays.allow_scalar(true) to allow",
    )
end

# Checks if an index is scalar at all, and then if scalar indexing is allowed. 
# Syntax as for `checkbounds`.
checkscalar(::Type{Bool}, I::Tuple) = checkscalar(Bool, I...)
checkscalar(::Type{Bool}, I...) = !all(map(i -> i isa Int, I)) || can_scalar()
checkscalar(I::Tuple) = checkscalar(I...)
checkscalar(I...) = checkscalar(Bool, I...) || _scalar_error()
