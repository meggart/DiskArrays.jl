module DiskArrays

using LRUCache: LRUCache, LRU

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end DiskArrays

export AbstractDiskArray, eachchunk, ChunkIndex, ChunkIndices

include("scalar.jl")
include("diskarray.jl")
include("array.jl")
include("broadcast.jl")
include("iterator.jl")
include("mapreduce.jl")
include("permute.jl")
include("reshape.jl")
include("subarray.jl")
include("rechunk.jl")
include("cat.jl")
include("generator.jl")
include("zip.jl")
include("show.jl")
include("cached.jl")

# The all-in-one macro

macro implement_diskarray(t)
    # Need to do this for dispatch ambiguity
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
        @implement_cat $t
        @implement_zip $t
        @implement_show $t
        @implement_generator $t
    end
end

# https://github.com/JuliaIO/DiskArrays.jl/issues/175
macro implement_diskarray_skip_zip(t)
    # Need to do this for dispatch ambiguity
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
        @implement_cat $t
        @implement_show $t
        @implement_generator $t
    end
end

# We need to skip the `implement_zip` macro for dispatch
@implement_diskarray_skip_zip AbstractDiskArray

#And we define the test types
include("util/testtypes.jl")


end # module
