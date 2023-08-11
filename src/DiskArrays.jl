module DiskArrays

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end DiskArrays

export AbstractDiskArray, interpret_indices_disk, eachchunk, ChunkIndex, ChunkIndices

include("scalar.jl")
include("diskarray.jl")
include("array.jl")
include("broadcast.jl")
include("iterator.jl")
include("mapreduce.jl")
include("permute.jl")
include("reshape.jl")
include("subarray.jl")
include("cat.jl")
include("generator.jl")

# The all-in-one macro

macro implement_diskarray(t)
    t = esc(t)
    quote
        @implement_getindex $t
        @implement_setindex $t
        @implement_broadcast $t
        @implement_iteration $t
        @implement_mapreduce $t
        @implement_reshape $t
        @implement_array_methods $t
        @implement_permutedims $t
        @implement_subarray $t
        @implement_batchgetindex $t
        @implement_cat $t
        @implement_generator $t
    end
end

@implement_diskarray AbstractDiskArray

end # module
