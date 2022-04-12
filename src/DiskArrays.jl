module DiskArrays

export AbstractDiskArray, interpret_indices_disk

include("diskarray.jl")
include("array.jl")
include("chunks.jl")
include("ops.jl")
include("iterator.jl")
include("subarrays.jl")
include("permute_reshape.jl")

# The all-in-one macro

macro implement_diskarray(t)
    quote
        @implement_getindex $t
        @implement_setindex $t
        @implement_broadcast $t
        @implement_iteration $t
        @implement_mapreduce $t
        @implement_reshape $t
        @implement_array_methods $t
        @implement_permutedims $t
    end
end

@implement_diskarray AbstractDiskArray

end # module
