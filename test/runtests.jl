using DiskArrays
using DiskArrays: ReshapedDiskArray, PermutedDiskArray
using Test
using Statistics
using Aqua

# Run with any code changes
# using JET
# JET.report_package(DiskArrays)

if VERSION >= v"1.9.0"
    # These dont resolve even though the suggested methods exist
    # Aqua.test_ambiguities([DiskArrays, Base, Core])
    Aqua.test_unbound_args(DiskArrays)
    Aqua.test_stale_deps(DiskArrays)
    Aqua.test_undefined_exports(DiskArrays)
    Aqua.test_project_extras(DiskArrays)
    Aqua.test_deps_compat(DiskArrays)
end

@testset "allow_scalar" begin
    DiskArrays.allow_scalar(false)
    @test DiskArrays.can_scalar() == false
    @test DiskArrays.checkscalar(Bool, 1, 2, 3) == false
    @test DiskArrays.checkscalar(Bool, 1, 2:5, :) == true
    DiskArrays.allow_scalar(true)
    @test DiskArrays.can_scalar() == true
    @test DiskArrays.checkscalar(Bool, 1, 2, 3) == true
    @test DiskArrays.checkscalar(Bool, :, 2:5, 3) == true
end

# Define a data structure that can be used for testing
struct _DiskArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    getindex_count::Ref{Int}
    setindex_count::Ref{Int}
    parent::A
    chunksize::NTuple{N,Int}
end
_DiskArray(a; chunksize=size(a)) = _DiskArray(Ref(0), Ref(0), a, chunksize)

# Apply the all in one macro rather than inheriting
DiskArrays.@implement_diskarray Main._DiskArray

Base.size(a::_DiskArray) = size(a.parent)
DiskArrays.haschunks(::_DiskArray) = DiskArrays.Chunked()
DiskArrays.eachchunk(a::_DiskArray) = DiskArrays.GridChunks(a, a.chunksize)
getindex_count(a::_DiskArray) = a.getindex_count[]
setindex_count(a::_DiskArray) = a.setindex_count[]
trueparent(a::_DiskArray) = a.parent
getindex_count(a::ReshapedDiskArray) = getindex_count(a.parent)
setindex_count(a::ReshapedDiskArray) = setindex_count(a.parent)
trueparent(a::ReshapedDiskArray) = trueparent(a.parent)
getindex_count(a::PermutedDiskArray) = getindex_count(a.a.parent)
setindex_count(a::PermutedDiskArray) = setindex_count(a.a.parent)
function trueparent(
    a::PermutedDiskArray{T,N,<:PermutedDimsArray{T,N,perm,iperm}}
) where {T,N,perm,iperm}
    return permutedims(trueparent(a.a.parent), perm)
end
function DiskArrays.readblock!(a::_DiskArray, aout, i::AbstractUnitRange...)
    ndims(a) == length(i) || error("Number of indices is not correct")
    all(r -> isa(r, AbstractUnitRange), i) || error("Not all indices are unit ranges")
    # println("reading from indices ", join(string.(i)," "))
    a.getindex_count[] += 1
    return aout .= a.parent[i...]
end
function DiskArrays.writeblock!(a::_DiskArray, v, i::AbstractUnitRange...)
    ndims(a) == length(i) || error("Number of indices is not correct")
    all(r -> isa(r, AbstractUnitRange), i) || error("Not all indices are unit ranges")
    # println("Writing to indices ", join(string.(i)," "))
    a.setindex_count[] += 1
    return view(a.parent, i...) .= v
end

struct UnchunkedDiskArray{T,N,P<:AbstractArray{T,N}} <: AbstractDiskArray{T,N}
    p::P
end
DiskArrays.haschunks(::UnchunkedDiskArray) = DiskArrays.Unchunked()
Base.size(a::UnchunkedDiskArray) = size(a.p)
function DiskArrays.readblock!(a::UnchunkedDiskArray, aout, i::AbstractUnitRange...)
    ndims(a) == length(i) || error("Number of indices is not correct")
    all(r -> isa(r, AbstractUnitRange), i) || error("Not all indices are unit ranges")
    # println("reading from indices ", join(string.(i)," "))
    return aout .= a.p[i...]
end

function test_getindex(a)
    @test a[2, 3, 1] == 10
    @test a[2, 3] == 10
    @test a[2, 3, 1, 1] == 10
    @test a[:, 1] == [1, 2, 3, 4]
    @test a[1:2, 1:2, 1, 1] == [1 5; 2 6]
    @test a[2:2:4, 1:2:5] == [2 10 18; 4 12 20]
    @test a[end:-1:1, 1, 1] == [4, 3, 2, 1]
    @test a[[1, 3, 4], [1, 3], 1] == [1 9; 3 11; 4 12]
    @test a[2, 3, 1, 1:1] == [10]
    @test a[2, 3, 1, [1], [1]] == fill(10, 1, 1)
    @test a[:, 3, 1, [1]] == reshape(9:12, 4, 1)
    # Test bitmask indexing
    m = falses(4, 5, 1)
    m[2, :, 1] .= true
    @test a[m] == [2, 6, 10, 14, 18]
    # Test linear indexing
    @test a[11:15] == 11:15
    @test a[20:-1:9] == 20:-1:9
    @test a[[3, 5, 8]] == [3, 5, 8]
    @test a[2:4:14] == [2, 6, 10, 14]
    # Test that readblock was called exactly onces for every getindex
    @test getindex_count(a) == 16
    @testset "allow_scalar" begin
        DiskArrays.allow_scalar(false)
        @test_throws ErrorException a[2, 3, 1]
        DiskArrays.allow_scalar(true)
        @test a[2, 3, 1] == 10
    end
end

function test_setindex(a)
    a[1, 1, 1] = 1
    a[1, 2] = 2
    a[1, 3, 1, 1] = 3
    a[2, :] = [1, 2, 3, 4, 5]
    a[3, 3:4, 1, 1] = [3, 4]
    # Test bitmask indexing
    m = falses(4, 5, 1)
    m[4, :, 1] .= true
    a[m] = [10, 11, 12, 13, 14]
    # Test that readblock was called exactly onces for every getindex
    @test setindex_count(a) == 6
    @test trueparent(a)[1, 1:3, 1] == [1, 2, 3]
    @test trueparent(a)[2, :, 1] == [1, 2, 3, 4, 5]
    @test trueparent(a)[3, 3:4, 1] == [3, 4]
    @test trueparent(a)[4, :, 1] == [10, 11, 12, 13, 14]
    a[1:2:4, 1:2:5, 1] = [1 2 3; 5 6 7]
    @test trueparent(a)[1:2:4, 1:2:5, 1] == [1 2 3; 5 6 7]
    @test setindex_count(a) == 7
    a[[2, 4], 1:2, 1] = [1 2; 5 6]
    @test trueparent(a)[[2, 4], 1:2, 1] == [1 2; 5 6]
    @test setindex_count(a) == 8
end

function test_view(a)
    v = view(a, 2:3, 2:4, 1)

    v[1:2, 1] = [1, 2]
    v[1:2, 2:3] = [4 4; 4 4]
    @test v[1:2, 1] == [1, 2]
    @test v[1:2, 2:3] == [4 4; 4 4]
    @test trueparent(a)[2:3, 2] == [1, 2]
    @test trueparent(a)[2:3, 3:4] == [4 4; 4 4]
    @test getindex_count(a) == 2
    @test setindex_count(a) == 2
end

function test_reductions(af)
    data = rand(10, 20, 2)
    for f in (
        minimum,
        maximum,
        sum,
        (i, args...; kwargs...) -> all(j -> j > 0.1, i, args...; kwargs...),
        (i, args...; kwargs...) -> any(j -> j < 0.1, i, args...; kwargs...),
        (i, args...; kwargs...) -> mapreduce(x -> 2 * x, +, i, args...; kwargs...),
    )
        a = af(data)
        @test isapprox(f(a), f(data))
        @test getindex_count(a) <= 10
        # And test reduction along dimensions
        a = _DiskArray(data; chunksize=(5, 4, 2))
        @test all(isapprox.(f(a; dims=2), f(data; dims=2)))
        # The minimum and maximum functions do some initialization, which will increase
        # the number of reads
        @test f in (minimum, maximum) || getindex_count(a) <= 12
        a = _DiskArray(data; chunksize=(5, 4, 2))
        @test all(isapprox.(f(a; dims=(1, 3)), f(data; dims=(1, 3))))
        @test f in (minimum, maximum) || getindex_count(a) <= 12
    end
end

function test_broadcast(a_disk1)
    a_disk2 = _DiskArray(rand(1:10, 1, 9); chunksize=(1, 3))
    a_mem = reshape(1:2, 1, 1, 2)

    s = a_disk1 .+ a_disk2 .* Ref(2) ./ (2,)
    # Test lazy broadcasting
    @test s isa DiskArrays.BroadcastDiskArray
    @test s === DiskArrays.BroadcastDiskArray(s.bc)
    @test getindex_count(a_disk1) == 0
    @test setindex_count(a_disk1) == 0
    @test getindex_count(a_disk2) == 0
    @test setindex_count(a_disk2) == 0
    @test size(s) == (10, 9, 2)
    @test eltype(s) == Float64
    # Lets merge another broadcast
    s2 = s ./ a_mem
    @test s isa DiskArrays.BroadcastDiskArray
    @test getindex_count(a_disk1) == 0
    @test getindex_count(a_disk2) == 0
    @test size(s) == (10, 9, 2)
    @test eltype(s) == Float64
    # And now do the computation with Array as a sink
    aout = zeros(10, 9, 2)
    aout .= s2 .* 2 ./ Ref(2)
    # Test if the result is correct
    @test aout == (trueparent(a_disk1) .+ trueparent(a_disk2)) ./ a_mem
    @test getindex_count(a_disk1) == 6
    @test getindex_count(a_disk2) == 6
    # Now use another DiskArray as the output
    aout = _DiskArray(zeros(10, 9, 2); chunksize=(5, 3, 2))
    aout .= s ./ a_mem
    @test trueparent(aout) == (trueparent(a_disk1) .+ trueparent(a_disk2)) ./ a_mem
    @test setindex_count(aout) == 6
    @test getindex_count(a_disk1) == 12
    @test getindex_count(a_disk2) == 12
    # Test reduction of broadcasted expression
    r = sum(s2; dims=(1, 2))
    @test all(
        isapprox.(
            sum((trueparent(a_disk1) .+ trueparent(a_disk2)) ./ a_mem; dims=(1, 2)), r
        ),
    )
    @test getindex_count(a_disk1) == 18
    @test getindex_count(a_disk2) == 18
end

@testset "GridChunks object" begin
    using DiskArrays: GridChunks, RegularChunks, IrregularChunks, subsetchunks
    a1 = RegularChunks(5, 2, 10)
    @test_throws BoundsError a1[0]
    @test_throws BoundsError a1[4]
    @test a1[1] == 1:3
    @test a1[2] == 4:8
    @test a1[3] == 9:10
    @test length(a1) == 3
    @test size(a1) == (3,)
    v1 = subsetchunks(a1, 1:10)
    v2 = subsetchunks(a1, 4:9)
    @test v1 === a1
    @test v2 === RegularChunks(5, 0, 6)
    a2 = RegularChunks(2, 0, 20)
    @test a2[1] == 1:2
    @test a2[2] == 3:4
    @test a2[10] == 19:20
    @test length(a2) == 10
    @test size(a2) == (10,)
    @test_throws BoundsError a2[0]
    @test_throws BoundsError a2[11]
    @test_throws ArgumentError RegularChunks(0,2,10)
    @test_throws ArgumentError RegularChunks(2,-1,10)
    @test_throws ArgumentError RegularChunks(2,2,10)
    @test_throws ArgumentError RegularChunks(5,2,-1)
    b1 = IrregularChunks(; chunksizes=[3, 3, 4, 3, 3])
    @test b1[1] == 1:3
    @test b1[2] == 4:6
    @test b1[3] == 7:10
    @test b1[4] == 11:13
    @test b1[5] == 14:16
    @test length(b1) == 5
    @test size(b1) == (5,)
    @test_throws BoundsError b1[0]
    @test_throws BoundsError b1[6]
    @test subsetchunks(b1, 1:15) == IrregularChunks(; chunksizes=[3, 3, 4, 3, 2])
    @test subsetchunks(b1, 3:10) == IrregularChunks(; chunksizes=[1, 3, 4])
    gridc = GridChunks(a1, a2, b1)
    @test eltype(gridc) <: Tuple{UnitRange,UnitRange,UnitRange}
    @test gridc[1, 1, 1] == (1:3, 1:2, 1:3)
    @test gridc[2, 2, 2] == (4:8, 3:4, 4:6)
    @test_throws BoundsError gridc[4, 1, 1]
    @test size(gridc) == (3, 10, 5)
    @test DiskArrays.approx_chunksize(gridc) == (5, 2, 3)
    @test DiskArrays.grid_offset(gridc) == (2, 0, 0)
    @test DiskArrays.max_chunksize(gridc) == (5, 2, 4)
    @test_throws ArgumentError IrregularChunks([1,2,3])
    @test_throws ArgumentError IrregularChunks([0,5,4])
    # Make sure mixed Integer types work
    @test RegularChunks(Int32(5), 2, UInt32(10)) == RegularChunks(5, 2, 10)
end

@testset "SubsetChunks" begin
    r1 = RegularChunks(10, 2, 30)
    @test subsetchunks(r1, 1:30) == RegularChunks(10, 2, 30)
    @test subsetchunks(r1, 30:-1:1) == RegularChunks(10, 8, 30)
    @test subsetchunks(r1, 5:25) == RegularChunks(10, 6, 21)
    @test subsetchunks(r1, 25:-1:5) == RegularChunks(10, 3, 21)
    @test subsetchunks(r1, 1:2:30) == RegularChunks(5, 1, 15)
    @test subsetchunks(r1, 30:-2:1) == RegularChunks(5, 4, 15)
    @test subsetchunks(r1, 5:2:25) == RegularChunks(5, 3, 11)
    @test subsetchunks(r1, 25:-2:5) == RegularChunks(5, 1, 11)
    @test subsetchunks(r1, 5:15) == RegularChunks(7, 3, 11)
    @test subsetchunks(r1, 2:10) == RegularChunks(7, 0, 9)
    @test subsetchunks(r1, 15:-1:5) == RegularChunks(7, 0, 11)
    @test subsetchunks(r1, 10:-1:2) == RegularChunks(7, 5, 9)
    @test subsetchunks(r1, 9:14) == RegularChunks(6, 0, 6)
    @test subsetchunks(r1, 9:2:14) == RegularChunks(3, 0, 3)
    @test subsetchunks(r1, 14:-1:9) == RegularChunks(6, 0, 6)
    @test subsetchunks(r1, 14:-2:9) == RegularChunks(3, 0, 3)
    @test subsetchunks(r1, 1:3:30) == IrregularChunks(; chunksizes=[3, 3, 4])
    @test subsetchunks(r1, 28:-3:1) == IrregularChunks(; chunksizes=[4, 3, 3])
    @test subsetchunks(r1, [5, 6, 7, 19, 20, 21]) == [1:3, 4:6]
    @test subsetchunks(r1, [28, 27, 19, 17, 10, 7]) == [1:3, 4:5, 6:6]
    @test subsetchunks(r1, [1, 2, 3]) == [1:3]
    @test subsetchunks(r1, [1, 2, 3, 10, 11]) == [1:3, 4:5]
    @test_throws ArgumentError subsetchunks(r1, [3, 4, 5, 1])

    r2 = IrregularChunks(; chunksizes=[3, 3, 4, 3, 3, 4])
    @test subsetchunks(r2, 1:20) == r2
    @test subsetchunks(r2, 3:18) == IrregularChunks(; chunksizes=[1, 3, 4, 3, 3, 2])
    @test subsetchunks(r2, 5:10) == [1:2, 3:6]
    @test subsetchunks(r2, 4:8) == [1:3, 4:5]
    @test subsetchunks(r2, [2:9; 11:18]) == RegularChunks(3, 1, 16)
end

@testset "Unchunked DiskArrays" begin
    a = UnchunkedDiskArray(reshape(1:1000, (10, 20, 5)))
    v = view(a, 1:2, 1, 1:3)
    @test v == [1 201 401; 2 202 402]
end

@testset "Index interpretation" begin
    import DiskArrays: DimsDropper, Reshaper
    a = zeros(3, 3, 1)
    @test interpret_indices_disk(a, (:, 2, :)) ==
        ((Base.OneTo(3), 2:2, Base.OneTo(1)), DimsDropper{Tuple{Int}}((2,)))
    @test interpret_indices_disk(a, (1, 2, :)) ==
        ((1:1, 2:2, Base.OneTo(1)), DimsDropper{Tuple{Int,Int}}((1, 2)))
    @test interpret_indices_disk(a, (1, 2, 2, 1)) ==
        ((1:1, 2:2, 2:2), DimsDropper{Tuple{Int,Int,Int}}((1, 2, 3)))
    @test interpret_indices_disk(a, (1, 2, 2, 1)) ==
        ((1:1, 2:2, 2:2), DimsDropper{Tuple{Int,Int,Int}}((1, 2, 3)))
    @test interpret_indices_disk(a, (:, 1:2)) ==
        ((Base.OneTo(3), 1:2, 1:1), DimsDropper{Tuple{Int}}((3,)))
    @test interpret_indices_disk(a, (:,)) ==
        ((Base.OneTo(3), Base.OneTo(3), Base.OneTo(1)), DiskArrays.Reshaper{Int}(9))
end

@testset "AbstractDiskArray getindex" begin
    a = _DiskArray(reshape(1:20, 4, 5, 1))
    test_getindex(a)
end

@testset "AbstractDiskArray setindex" begin
    a = _DiskArray(zeros(Int, 4, 5, 1))
    test_setindex(a)
end

@testset "Zerodimensional" begin
    a = _DiskArray(zeros(Int))
    @test a[] == 0
    @test a[1] == 0
    a[] = 5
    @test a[] == 5
    a[1] = 6
    @test a[] == 6
end

@testset "Views" begin
    a = _DiskArray(zeros(Int, 4, 5, 1))
    test_view(a)
end

import Statistics: mean
@testset "Reductions" begin
    a = data -> _DiskArray(data; chunksize=(5, 4, 2))
    test_reductions(a)
end

@testset "Broadcast" begin
    a_disk1 = _DiskArray(rand(10, 9, 2); chunksize=(5, 3, 2))
    test_broadcast(a_disk1)
end

@testset "zip" begin
    a = rand(10, 9, 2)
    b = rand(10, 9, 2)
    da = _DiskArray(a; chunksize=(5, 3, 2))
    db = _DiskArray(b; chunksize=(2, 3, 1))
    z = zip(a, b)
    zd = zip(da, db)
    zdc = collect(zd)
    zc = collect(z)
    @test da.getindex_count[] == 6
    @test db.getindex_count[] == 6
    @test all(zd .== z)
    @test all(zdc .== zc)
    @test zip(a, da, a) isa DiskArrays.DiskZip
    @test zip(da, da, a) isa DiskArrays.DiskZip
    @test zip(da, da, da) isa DiskArrays.DiskZip
    @test zip(a, da, da) isa DiskArrays.DiskZip
    # Should we add moree dispatch to fix this?
    @test_broken zip(a, a, da) isa DiskArrays.DiskZip
    zd3_a = zip(a, da, a)
    zd3_b = zip(da, da, a)
    zd3_c = zip(da, a, a)
    za3 = zip(a, a, a)
    @test collect(zd3_a) == collect(zd3_b) == collect(zd3_c) == collect(za3)
    @test all(zd3_a .== zd3_b .== zd3_c .== za3)
    @test_throws DimensionMismatch zip(da, rand(2, 3, 1))
end

@testset "cat" begin
    da = _DiskArray(collect(reshape(1:24, 4, 6, 1)))
    a = view(da, :, 1:3, :)
    b = view(da, :, 4:6, :)
    ca = cat(a, b; dims=2)
    @test ca == da
    @test ca .* 2 == da .* 2

    @testset "cat on all dims" begin
        @test collect(cat(a, b; dims=1)) == cat(collect(a), collect(b); dims=1)
        @test collect(cat(a, b; dims=2)) == cat(collect(a), collect(b); dims=2)
        @test collect(cat(a, b; dims=3)) == cat(collect(a), collect(b); dims=3)
        @test collect(cat(a, b; dims=4)) == cat(collect(a), collect(b); dims=4)
        @test collect(cat(a, b; dims=5)) == cat(collect(a), collect(b); dims=5)
    end

    @testset "cat mixed arrays and disk arrays is still a ConcatDiskArray" begin
        @test cat(a, collect(b); dims=1) isa DiskArrays.ConcatDiskArray
        @test collect(cat(a, collect(b); dims=1)) == cat(collect(a), collect(b); dims=1)
        @test cat(collect(a), b; dims=1) isa DiskArrays.ConcatDiskArray
        @test collect(cat(collect(a), b; dims=1)) == cat(collect(a), collect(b); dims=1)
    end

    @testset "write concat" begin
        ca .= reshape(0:23, 4, 6)
        @test sum(ca) == sum(0:23)
    end

    @testset "cat mixed chunk size" begin
        a = _DiskArray(collect(1:10); chunksize=(3,))
        b = _DiskArray(collect(1:9); chunksize=(4,))
        c = _DiskArray(collect(1:7); chunksize=(3,))
        d = cat(a, b, c; dims=1)
        @test d == [1:10; 1:9; 1:7]
        @test DiskArrays.eachchunk(d) == [
            (1:3,)
            (4:6,)
            (7:9,)
            (10:10,)
            (11:14,)
            (15:18,)
            (19:19,)
            (20:22,)
            (23:25,)
            (26:26,)
        ]
        d .= 1:26
        @test d == 1:26
        @test c == 20:26
    end
end

@testset "Broadcast with length 1 and 0 final dim" begin
    a_disk1 = _DiskArray(rand(10, 9, 1); chunksize=(5, 3, 1))
    a_disk2 = _DiskArray(rand(1:10, 1, 9); chunksize=(1, 3))
    s = a_disk1 .+ a_disk2
    @test DiskArrays.eachchunk(s) isa DiskArrays.GridChunks{3}
    @test size(collect(s)) == (10, 9, 1)
    a_disk1 = _DiskArray(zeros(Int); chunksize=())
    r = ones(Int)
    r .= a_disk1
    @test r[] == 0
end

@testset "Getindex/Setindex with vectors" begin
    a = _DiskArray(reshape(1:20, 4, 5, 1); chunksize=(4, 1, 1))
    @test a[:, [1, 4], 1] == trueparent(a)[:, [1, 4], 1]
    @test getindex_count(a) == 2
    coords = CartesianIndex.([(1, 1, 1), (3, 1, 1), (2, 4, 1), (4, 4, 1)])
    @test a[coords] == trueparent(a)[coords]
    @test getindex_count(a) == 4

    aperm = permutedims(a, (2, 1, 3))
    coordsperm = (x -> CartesianIndex(x.I[[2, 1, 3]])).(coords)
    @test aperm[coordsperm] == a[coords]

    coords = CartesianIndex.([(1, 1), (3, 1), (2, 4), (4, 4)])
    @test a[coords, :] == trueparent(a)[coords, :]
    @test getindex_count(a) == 10

    @test a[3:4, [1, 4], 1] == trueparent(a)[3:4, [1, 4], 1]
    @test getindex_count(a) == 12

    aperm = permutedims(a, (2, 1, 3))
    coordsperm = (x -> CartesianIndex((x.I[[2, 1]]))).(coords)
    @test aperm[coordsperm, :] == a[coords, :]

    #With pre-allocated output array
    aout = zeros(Int, 4, 2)
    DiskArrays.disk_getindex_batch!(aout, a, [(1:4, 1, 1), (1:4, 4, 1)])
    @test aout == trueparent(a)[:, [1, 4], 1]

    #Index with range stride much larger than chunk size
    a = _DiskArray(reshape(1:100, 20, 5, 1); chunksize=(1, 5, 1))
    @test a[1:9:20, :, 1] == trueparent(a)[1:9:20, :, 1]
    @test getindex_count(a) == 3

    b = _DiskArray(zeros(4, 5, 1); chunksize=(4, 1, 1))
    b[[1, 4], [2, 5], 1] = ones(2, 2)
    @test setindex_count(b) == 2
    mask = falses(4, 5, 1)
    mask[2, 1] = true
    mask[3, 1] = true
    mask[1, 3] = true
    mask[4, 3] = true
    b[mask] = fill(2.0, 4)
    @test setindex_count(b) == 4
end

@testset "generator" begin
    a = collect(reshape(1:90, 10, 9))
    a_disk = _DiskArray(a; chunksize=(5, 3))
    @test [aa for aa in a_disk] == a
    #The array has 6 chunks so getindex_count should be 6
    @test getindex_count(a_disk) == 6
    # Filtered generators dont work yet
    @test_broken [aa for aa in a_disk if aa > 40] == [aa for aa in a if aa > 40]
    #Iterator interface tests
    g = Base.Generator(identity, a_disk)
    @test g isa DiskArrays.DiskGenerator
    @test size(g) == (10, 9)
    @test !isempty(g)
    @test length(g) == 90
    @test ndims(g) == 2
    @test keys(g) == CartesianIndices((10, 9))
end

@testset "Array methods" begin
    a = collect(reshape(1:90, 10, 9))
    a_disk = _DiskArray(a; chunksize=(5, 3))
    ei = eachindex(a_disk)
    @test ei isa DiskArrays.BlockedIndices
    @test length(ei) == 90
    @test eltype(ei) == CartesianIndex{2}
    @test collect(a_disk) == a
    @test Array(a_disk) == a
    @testset "copyto" begin
        x = zero(a)
        copyto!(x, a_disk)
        @test x == a
        copyto!(x, CartesianIndices((1:3, 1:2)), a_disk, CartesianIndices((8:10, 8:9)))
    end

    @test collect(reverse(a_disk)) == reverse(a)
    @test reverse(view(a_disk, :, 1)) == reverse(a[:, 1])
    @test reverse(view(a_disk, :, 1), 1) == reverse(a[:, 1], 1)
    @test collect(reverse(a_disk)) == reverse(a)
    @test collect(reverse(a_disk; dims=2)) == reverse(a; dims=2)
    @test replace(a_disk, 1 => 2) == replace(a, 1 => 2)
    @test rotr90(a_disk) == rotr90(a)
    @test rotl90(a_disk) == rotl90(a)
    @test rot180(a_disk) == rot180(a)
    @test extrema(a_disk) == extrema(a)
    @test mean(a_disk) == mean(a)
    @test mean(a_disk; dims=1) == mean(a; dims=1)
    @test std(a_disk) == std(a)
    @test median(a_disk) == median(a)
    @test median(a_disk; dims=1) == median(a; dims=1) # Works but very slow
    @test median(a_disk; dims=2) == median(a; dims=2) # Works but very slow
    @test vcat(a_disk, a_disk) == vcat(a, a)
    @test hcat(a_disk, a_disk) == hcat(a, a)
    @test cat(a_disk, a_disk; dims=3) == cat(a, a; dims=3)
    @test_broken circshift(a_disk, 2) == circshift(a, 2) # This one is super weird. The size changes.
end

@testset "Reshape" begin
    a = reshape(_DiskArray(reshape(1:20, 4, 5)), 4, 5, 1)
    test_getindex(a)
    a = reshape(_DiskArray(zeros(Int, 4, 5)), 4, 5, 1)
    test_setindex(a)
    a = reshape(_DiskArray(zeros(Int, 4, 5)), 4, 5, 1)
    test_view(a)
    a = data -> reshape(_DiskArray(data; chunksize=(5, 4, 2)), 10, 20, 2, 1)
    test_reductions(a)
    a = reshape(_DiskArray(reshape(1:20, 4, 5)), 4, 5, 1)
    @test ReshapedDiskArray(a.parent, a.keepdim, a.newsize) === a
    # Reshape with existing trailing 1s works
    a = reshape(_DiskArray(reshape(1:100, 5, 5, 2, 2, 1, 1)), 5, 5, 2, 2, 1, 1, 1)
    @test a[5, 5, 2, 2, 1, 1, 1] == 100
end

import Base.PermutedDimsArrays.invperm
@testset "Permutedims" begin
    p = (3, 1, 2)
    ip = invperm(p)
    a = permutedims(_DiskArray(permutedims(reshape(1:20, 4, 5, 1), ip)), p)
    test_getindex(a)
    a = permutedims(_DiskArray(zeros(Int, 5, 1, 4)), p)
    test_setindex(a)
    a = permutedims(_DiskArray(zeros(Int, 5, 1, 4)), p)
    test_view(a)
    a = data -> permutedims(_DiskArray(permutedims(data, ip); chunksize=(4, 2, 5)), p)
    test_reductions(a)
    a_disk1 = permutedims(_DiskArray(rand(9, 2, 10); chunksize=(3, 2, 5)), p)
    test_broadcast(a_disk1)
    @test PermutedDiskArray(a_disk1.a) === a_disk1
end

@testset "Unchunked String arrays" begin
    a = reshape(1:200000, 200, 1000)
    b = string.(a)
    c = collect(Union{Int,Missing}, a)

    DiskArrays.default_chunk_size[] = 100
    DiskArrays.fallback_element_size[] = 100
    @test DiskArrays.estimate_chunksize(a) == DiskArrays.GridChunks(a, (200, 1000))
    @test DiskArrays.eachchunk(a) == DiskArrays.GridChunks(a, (200, 1000))
    @test DiskArrays.estimate_chunksize(b) == DiskArrays.GridChunks(b, (200, 1000))
    @test DiskArrays.eachchunk(b) == DiskArrays.GridChunks(b, (200, 1000))
    @test DiskArrays.estimate_chunksize(c) == DiskArrays.GridChunks(c, (200, 1000))
    @test DiskArrays.eachchunk(c) == DiskArrays.GridChunks(c, (200, 1000))
    DiskArrays.default_chunk_size[] = 1
    @test DiskArrays.estimate_chunksize(a) == DiskArrays.GridChunks(a, (200, 625))
    @test DiskArrays.eachchunk(a) == DiskArrays.GridChunks(a, (200, 625))
    @test DiskArrays.estimate_chunksize(b) == DiskArrays.GridChunks(b, (200, 50))
    @test DiskArrays.eachchunk(b) == DiskArrays.GridChunks(b, (200, 50))
    @test DiskArrays.estimate_chunksize(c) == DiskArrays.GridChunks(c, (200, 625))
    @test DiskArrays.eachchunk(c) == DiskArrays.GridChunks(c, (200, 625))
    DiskArrays.fallback_element_size[] = 1000
    @test DiskArrays.estimate_chunksize(a) == DiskArrays.GridChunks(a, (200, 625))
    @test DiskArrays.eachchunk(a) == DiskArrays.GridChunks(a, (200, 625))
    @test DiskArrays.estimate_chunksize(b) == DiskArrays.GridChunks(b, (200, 5))
    @test DiskArrays.eachchunk(b) == DiskArrays.GridChunks(b, (200, 5))
    @test DiskArrays.estimate_chunksize(c) == DiskArrays.GridChunks(c, (200, 625))
    @test DiskArrays.eachchunk(c) == DiskArrays.GridChunks(c, (200, 625))
end

@testset "Mixed size chunks" begin
    a1 = _DiskArray(zeros(24, 16); chunksize=(1, 1))
    a2 = _DiskArray((2:25) * vec(1:16)'; chunksize=(1, 2))
    a3 = _DiskArray((3:26) * vec(1:16)'; chunksize=(3, 4))
    a4 = _DiskArray((4:27) * vec(1:16)'; chunksize=(6, 8))
    v1 = view(_DiskArray((1:30) * vec(1:21)'; chunksize=(5, 7)), 3:26, 2:17)
    v2 = view(_DiskArray((1:30) * vec(1:21)'; chunksize=(5, 7)), 4:27, 3:18)
    a1 .= a2
    @test Array(a1) == Array(a2)
    a1 .= a3
    @test all(Array(a1) .== Array(a3))
    a1 .= a4
    @test all(Array(a1) .== Array(a4))
    a4 .= a3
    @test all(Array(a4) .== Array(a3))
    a3 .= a2
    @test all(Array(a3) .== Array(a2))

    a1 .= v1
    @test all(Array(a1) .== (3:26) * vec(2:17)')
    a1 .= v2
    @test all(Array(a1) .== (4:27) * vec(3:18)')

    # TODO Chunks that don't align at all - need to work out 
    # how to choose the smallest chunks to read twice, and when
    # to just ignore the chunks and load the whole array.
    # a2 .= v1
    # @test all(Array(a2) .== (3:26) * vec(2:17)')
    # a2 .= v2
    # @test all(Array(a2) .== (4:27) * vec(3:18)')
end

struct TestArray{T,N} <: AbstractArray{T,N} end

@testset "All macros apply" begin
    DiskArrays.@implement_getindex TestArray
    DiskArrays.@implement_setindex TestArray
    DiskArrays.@implement_broadcast TestArray
    DiskArrays.@implement_iteration TestArray
    DiskArrays.@implement_mapreduce TestArray
    DiskArrays.@implement_reshape TestArray
    DiskArrays.@implement_array_methods TestArray
    DiskArrays.@implement_permutedims TestArray
    DiskArrays.@implement_subarray TestArray
    DiskArrays.@implement_batchgetindex TestArray
    DiskArrays.@implement_diskarray TestArray
end

# issue #123

mutable struct ResizableArray{T,N} <: AbstractArray{T,N}
    A::AbstractArray{T,N}
end

Base.size(RA::ResizableArray) = size(RA.A)
Base.getindex(RA::ResizableArray,inds...) = getindex(RA.A,inds...)
Base.checkbounds(::Type{Bool},RA::ResizableArray,inds...) = all(minimum.(inds) .> 0)
function Base.setindex!(RA::ResizableArray{T,N}, value, inds::Vararg{Int, N}) where {T,N}
    sz = max.(size(RA),inds)
    if sz != size(RA)
        # grow
        oldA = RA.A
        RA.A = Array{T,N}(undef,sz)
        RA.A[axes(oldA)...] = oldA
    end
    RA.A[inds...] = value
end

@testset "Resizable arrays" begin
    a = ResizableArray(Vector{Int}(undef,0))
    @test size(a) == (0,)
    a[1:5] = 1:5
    @test a == 1:5
    @test size(a) == (5,)

    b = ResizableArray(Vector{Int}(undef,0))
    b1 = _DiskArray(b,chunksize=(5,))
    @test size(b1) == (0,)
    b1[1:5] = 1:5
    @test b1 == 1:5
    @test size(b1) == (5,)
    @test setindex_count(b1) == 1

    c = ResizableArray(Matrix{Int}(undef,(0,0)))
    c1 = _DiskArray(c,chunksize=(5,5))
    @test size(c1) == (0,0)
    c1[1:5,1:5] = ones(Int,5,5)
    @test c1 == ones(Int,5,5)
    @test size(c1) == (5,5)
    @test setindex_count(c1) == 1
end
