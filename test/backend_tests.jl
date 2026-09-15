# Helper: create a materialized + disk array pair
make_arrays(data; chunksize=size(data)) = (
    materialized = data,
    disk = AccessCountDiskArray(data; chunksize=chunksize),
)

# ── Backend test helpers ─────────────────────────────────────────────────────

function test_specialized_reductions_default(data; chunksize=size(data))
    @testset "DefaultBackend" begin
        mat, da = make_arrays(data; chunksize)

        @testset "sum" begin
            @test sum(da) ≈ sum(mat)
            @test sum(da, dims=1) ≈ sum(mat, dims=1)
            @test sum(da, dims=(1, 2)) ≈ sum(mat, dims=(1, 2))
            @test sum(identity, da) ≈ sum(mat)
            @test sum(x -> 2x, da) ≈ sum(x -> 2x, mat)
            @test sum(x -> 2x, da, dims=1) ≈ sum(x -> 2x, mat, dims=1)
        end

        @testset "prod" begin
            @test prod(da) ≈ prod(mat)
            @test prod(da, dims=1) ≈ prod(mat, dims=1)
            @test prod(identity, da) ≈ prod(mat)
            @test prod(x -> 2x, da) ≈ prod(x -> 2x, mat)
            @test prod(x -> 2x, da, dims=1) ≈ prod(x -> 2x, mat, dims=1)
        end

        @testset "all" begin
            @test all(da .> -1.0) == all(mat .> -1.0)
            @test all(x -> x > -1.0, da) == all(mat .> -1.0)
            @test all(all(da .> -1.0, dims=1)) == all(all(mat .> -1.0, dims=1))
            @test all(all(x -> x > -1.0, da, dims=1)) == all(all(mat .> -1.0, dims=1))
        end

        @testset "any" begin
            @test any(da .< 0.0) == any(mat .< 0.0)
            @test any(x -> x < 0.0, da) == any(mat .< 0.0)
            @test any(any(da .< 0.0, dims=1)) == any(any(mat .< 0.0, dims=1))
            @test any(any(x -> x < 0.0, da, dims=1)) == any(any(mat .< 0.0, dims=1))
        end

        @testset "minimum" begin
            @test minimum(da) ≈ minimum(mat)
            @test minimum(da, dims=1) ≈ minimum(mat, dims=1)
            @test minimum(identity, da) ≈ minimum(mat)
            @test minimum(x -> abs(x), da) ≈ minimum(x -> abs(x), mat)
            @test minimum(x -> abs(x), da, dims=1) ≈ minimum(x -> abs(x), mat, dims=1)
        end

        @testset "maximum" begin
            @test maximum(da) ≈ maximum(mat)
            @test maximum(da, dims=1) ≈ maximum(mat, dims=1)
            @test maximum(identity, da) ≈ maximum(mat)
            @test maximum(x -> abs(x), da) ≈ maximum(x -> abs(x), mat)
            @test maximum(x -> abs(x), da, dims=1) ≈ maximum(x -> abs(x), mat, dims=1)
        end

        @testset "extrema" begin
            @test extrema(da) == extrema(mat)
            @test extrema(da, dims=1) == extrema(mat, dims=1)
            @test extrema(identity, da) == extrema(mat)
            @test extrema(x -> abs(x), da) == extrema(x -> abs(x), mat)
            @test extrema(x -> abs(x), da, dims=1) == extrema(x -> abs(x), mat, dims=1)
        end

        @testset "count" begin
            @test count(x -> x > 0, da) == count(x -> x > 0, mat)
            @test count(x -> x > 0, da, dims=1) == count(x -> x > 0, mat, dims=1)
        end

        @testset "unique" begin
            @test_broken unique(da) == unique(mat)
            @test_broken unique(identity, da) == unique(mat)
            @test_broken unique(x -> x > 0, da) == unique(x -> x > 0, mat)
        end
    end
end

function test_statistics_reductions_default(data; chunksize=size(data))
    @testset "DefaultBackend: Statistics" begin
        mat, da = make_arrays(data; chunksize)
        @testset "mean" begin
            @test mean(da) ≈ mean(mat)
            @test mean(da, dims=1) ≈ mean(mat, dims=1)
            @test mean(identity, da) ≈ mean(mat)
            @test mean(x -> 2x, da) ≈ mean(x -> 2x, mat)
            @test mean(x -> 2x, da, dims=1) ≈ mean(x -> 2x, mat, dims=1)
        end
        @testset "median" begin
            @test median(da) ≈ median(mat)
            @test median(da, dims=1) ≈ median(mat, dims=1)
            @test median(da, dims=2) ≈ median(mat, dims=2)
        end
    end
end

function test_mapreduce_default(data; chunksize=size(data))
    @testset "DefaultBackend: mapreduce" begin
        mat, da = make_arrays(data; chunksize)
        @testset "mapreduce (no dims, no init)" begin
            @test mapreduce(x -> 2x, +, da) ≈ mapreduce(x -> 2x, +, mat)
            @test mapreduce(*, -, da) ≈ mapreduce(*, -, mat)
        end
        @testset "mapreduce (dims=)" begin
            @test mapreduce(x -> 2x, +, da; dims=1) ≈ mapreduce(x -> 2x, +, mat; dims=1)
            @test mapreduce(x -> 2x, +, da; dims=(1, 2)) ≈ mapreduce(x -> 2x, +, mat; dims=(1, 2))
        end
        @testset "mapreduce (init)" begin
            @test mapreduce(identity, +, da; init=0) ≈ mapreduce(identity, +, mat; init=0)
            @test mapreduce(identity, *, da; init=1) ≈ mapreduce(identity, *, mat; init=1)
            @test mapreduce(identity, +, da; dims=1, init=0) ≈ mapreduce(identity, +, mat; dims=1, init=0)
        end
        @testset "mapreducedim!" begin
            R = zeros(size(da, 1), size(da, 2), 1)
            mapreducedim!(x -> 2x, +, R, da)
            @test R ≈ mapreducedim!(x -> 2x, +, similar(R, size(da, 1), size(da, 2), 1), mat)
            R2 = zeros(1, size(da, 2), size(da, 3))
            mapreducedim!(x -> x^2, +, R2, da)
            @test R2 ≈ mapreducedim!(x -> x^2, +, similar(R2), mat)
        end
        @testset "mapfoldl (no init)" begin
            @test mapfoldl(x -> 2x, +, da) ≈ mapfoldl(x -> 2x, +, mat)
        end
        @testset "mapfoldl (with init)" begin
            @test mapfoldl(identity, +, da; init=0) ≈ mapfoldl(identity, +, mat; init=0)
            @test mapfoldl(identity, *, da; init=1) ≈ mapfoldl(identity, *, mat; init=1)
        end
        @testset "Base.reduce" begin
            @test reduce(+, da) ≈ reduce(+, mat)
            @test reduce(*, da) ≈ reduce(*, mat)
            @test reduce(max, da) ≈ reduce(max, mat)
            @test reduce(min, da) ≈ reduce(min, mat)
        end
        @testset "Base.cumsum/cumprod" begin
            @test cumsum(da) ≈ cumsum(mat)
            @test cumsum(da, dims=1) ≈ cumsum(mat, dims=1)
            @test cumprod(da) ≈ cumprod(mat)
            @test cumprod(da, dims=1) ≈ cumprod(mat, dims=1)
        end
    end
end

# ── DAE versions ────────────────────────────────────────────────────────────

function test_specialized_reductions_dae(data; chunksize=size(data))
    @testset "DiskArrayEngineBackend" begin
        mat, da = make_arrays(data; chunksize)

        @testset "sum" begin
            @test Array(sum(da)) ≈ sum(mat)
            @test Array(sum(da, dims=1)) ≈ sum(mat, dims=1)
            @test Array(sum(da, dims=(1, 2))) ≈ sum(mat, dims=(1, 2))
            @test Array(sum(identity, da)) ≈ sum(mat)
            @test Array(sum(x -> 2x, da)) ≈ sum(x -> 2x, mat)
            @test Array(sum(x -> 2x, da, dims=1)) ≈ sum(x -> 2x, mat, dims=1)
        end

        @testset "prod" begin
            @test Array(prod(da)) ≈ prod(mat)
            @test Array(prod(da, dims=1)) ≈ prod(mat, dims=1)
            @test Array(prod(identity, da)) ≈ prod(mat)
            @test Array(prod(x -> 2x, da)) ≈ prod(x -> 2x, mat)
            @test Array(prod(x -> 2x, da, dims=1)) ≈ prod(x -> 2x, mat, dims=1)
        end

        @testset "all" begin
            @test all(Array(da .> -1.0)) == all(mat .> -1.0)
            @test all(all(da .> -1.0, dims=1)) == all(all(mat .> -1.0, dims=1))
            @test all(all(x -> x > -1.0, da, dims=1)) == all(all(mat .> -1.0, dims=1))
        end

        @testset "any" begin
            @test any(Array(da .< 0.0)) == any(mat .< 0.0)
            @test any(any(da .< 0.0, dims=1)) == any(any(mat .< 0.0, dims=1))
            @test any(any(x -> x < 0.0, da, dims=1)) == any(any(mat .< 0.0, dims=1))
        end

        @testset "minimum" begin
            @test Array(minimum(da)) ≈ minimum(mat)
            @test Array(minimum(da, dims=1)) ≈ minimum(mat, dims=1)
            @test Array(minimum(identity, da)) ≈ minimum(mat)
            @test Array(minimum(x -> abs(x), da)) ≈ minimum(x -> abs(x), mat)
            @test Array(minimum(x -> abs(x), da, dims=1)) ≈ minimum(x -> abs(x), mat, dims=1)
        end

        @testset "maximum" begin
            @test Array(maximum(da)) ≈ maximum(mat)
            @test Array(maximum(da, dims=1)) ≈ maximum(mat, dims=1)
            @test Array(maximum(identity, da)) ≈ maximum(mat)
            @test Array(maximum(x -> abs(x), da)) ≈ maximum(x -> abs(x), mat)
            @test Array(maximum(x -> abs(x), da, dims=1)) ≈ maximum(x -> abs(x), mat, dims=1)
        end

        @testset "extrema" begin
            @test Array(extrema(da)) == extrema(mat)
            @test Array(extrema(da, dims=1)) == extrema(mat, dims=1)
            @test Array(extrema(identity, da)) == extrema(mat)
            @test Array(extrema(x -> abs(x), da)) == extrema(x -> abs(x), mat)
            @test Array(extrema(x -> abs(x), da, dims=1)) == extrema(x -> abs(x), mat, dims=1)
        end

        @testset "count" begin
            @test Array(count(x -> x > 0, da)) == count(x -> x > 0, mat)
            @test Array(count(x -> x > 0, da, dims=1)) == count(x -> x > 0, mat, dims=1)
        end

        @testset "unique" begin
            @test_broken Array(unique(da)) == unique(mat)
            @test_broken Array(unique(identity, da)) == unique(mat)
            @test_broken Array(unique(x -> x > 0, da)) == unique(x -> x > 0, mat)
        end
    end
end

function test_statistics_reductions_dae(data; chunksize=size(data))
    @testset "DAE: Statistics" begin
        mat, da = make_arrays(data; chunksize)
        @testset "mean" begin
            @test Array(mean(da)) ≈ mean(mat)
            @test Array(mean(da, dims=1)) ≈ mean(mat, dims=1)
            @test Array(mean(identity, da)) ≈ mean(mat)
            @test Array(mean(x -> 2x, da)) ≈ mean(x -> 2x, mat)
            @test Array(mean(x -> 2x, da, dims=1)) ≈ mean(x -> 2x, mat, dims=1)
        end
        @testset "median" begin
            @test Array(median(da)) ≈ median(mat)
            @test Array(median(da, dims=1)) ≈ median(mat, dims=1)
            @test Array(median(da, dims=2)) ≈ median(mat, dims=2)
        end
    end
end

function test_mapreduce_dae(data; chunksize=size(data))
    @testset "DiskArrayEngineBackend: mapreduce" begin
        mat, da = make_arrays(data; chunksize)
        @testset "mapreduce (no dims, no init)" begin
            @test Array(mapreduce(x -> 2x, +, da)) ≈ mapreduce(x -> 2x, +, mat)
            @test Array(mapreduce(*, -, da)) ≈ mapreduce(*, -, mat)
        end
        @testset "mapreduce (dims=)" begin
            @test Array(mapreduce(x -> 2x, +, da; dims=1)) ≈ mapreduce(x -> 2x, +, mat; dims=1)
            @test Array(mapreduce(x -> 2x, +, da; dims=(1, 2))) ≈ mapreduce(x -> 2x, +, mat; dims=(1, 2))
        end
        @testset "mapreduce (init)" begin
            @test Array(mapreduce(identity, +, da; init=0)) ≈ mapreduce(identity, +, mat; init=0)
            @test Array(mapreduce(identity, *, da; init=1)) ≈ mapreduce(identity, *, mat; init=1)
            @test Array(mapreduce(identity, +, da; dims=1, init=0)) ≈ mapreduce(identity, +, mat; dims=1, init=0)
        end
        @testset "mapreducedim!" begin
            R = zeros(size(da, 1), size(da, 2), 1)
            mapreducedim!(x -> 2x, +, R, da)
            @test R ≈ mapreducedim!(x -> 2x, +, similar(R, size(da, 1), size(da, 2), 1), mat)
            R2 = zeros(1, size(da, 2), size(da, 3))
            mapreducedim!(x -> x^2, +, R2, da)
            @test R2 ≈ mapreducedim!(x -> x^2, +, similar(R2), mat)
        end
        @testset "mapfoldl (no init)" begin
            @test Array(mapfoldl(x -> 2x, +, da)) ≈ mapfoldl(x -> 2x, +, mat)
        end
        @testset "mapfoldl (with init)" begin
            @test Array(mapfoldl(identity, +, da; init=0)) ≈ mapfoldl(identity, +, mat; init=0)
            @test Array(mapfoldl(identity, *, da; init=1)) ≈ mapfoldl(identity, *, mat; init=1)
        end
    end
end

# ── Backend test suites ─────────────────────────────────────────────────────

@testset "Backend test suite: DefaultBackend specialized reductions" begin
    test_specialized_reductions_default(randn(5, 4, 2); chunksize=(3, 2, 2))
    test_specialized_reductions_default(randn(10); chunksize=(3,))
    test_specialized_reductions_default(randn(10, 20); chunksize=(4, 7))
    test_specialized_reductions_default(randn(3, 3, 3); chunksize=(2, 2, 2))
end

@testset "Backend test suite: DefaultBackend statistics" begin
    test_statistics_reductions_default(randn(5, 4, 2); chunksize=(3, 2, 2))
    test_statistics_reductions_default(randn(10); chunksize=(3,))
    test_statistics_reductions_default(randn(10, 20); chunksize=(4, 7))
end

@testset "Backend test suite: DefaultBackend mapreduce" begin
    test_mapreduce_default(randn(5, 4, 2); chunksize=(3, 2, 2))
    test_mapreduce_default(randn(10); chunksize=(3,))
    test_mapreduce_default(randn(10, 20); chunksize=(4, 7))
end

@testset "Backend test suite: DiskArrayEngine" begin
    @testset "DAE specialized reductions" begin
        @testset for data in (randn(5, 4, 2), randn(10), randn(10, 20), randn(3, 3, 3))
            test_specialized_reductions_dae(data; chunksize=ntuple(i -> max(1, size(data)[i] ÷ 2), ndims(data)))
        end
    end
    @testset "DAE statistics" begin
        @testset for data in (randn(5, 4, 2), randn(10), randn(10, 20), randn(3, 3, 3))
            test_statistics_reductions_dae(data; chunksize=ntuple(i -> max(1, size(data)[i] ÷ 2), ndims(data)))
        end
    end
    @testset "DAE mapreduce" begin
        @testset for data in (randn(5, 4, 2), randn(10), randn(10, 20), randn(3, 3, 3))
            test_mapreduce_dae(data; chunksize=ntuple(i -> max(1, size(data)[i] ÷ 2), ndims(data)))
        end
    end
end

# ── Edge cases ──────────────────────────────────────────────────────────────

@testset "Edge cases: Bool array" begin
    mat = rand(Bool, 10, 5)
    da = AccessCountDiskArray(mat; chunksize=(3, 2))
    @test sum(da) == sum(mat)
    @test count(da) == count(mat)
    @test count(da, dims=1) == count(mat, dims=1)
    @test all(da) == all(mat)
    @test any(da) == any(mat)
    @test extrema(da) == extrema(mat)
end

@testset "Edge cases: Integer array" begin
    mat = rand(1:100, 5, 4)
    da = AccessCountDiskArray(mat; chunksize=(3, 2))
    @test sum(da) == sum(mat)
    @test prod(da) == prod(mat)
    @test minimum(da) == minimum(mat)
    @test maximum(da) == maximum(mat)
    @test extrema(da) == extrema(mat)
    @test sum(da, dims=1) == sum(mat, dims=1)
end

@testset "Edge cases: Float32 array" begin
    mat = rand(Float32, 5, 4)
    da = AccessCountDiskArray(mat; chunksize=(3, 2))
    @test sum(da) ≈ sum(mat)
    @test mean(da) ≈ mean(mat)
    @test sum(da, dims=1) ≈ sum(mat, dims=1)
end

@testset "Edge cases: 0-dimensional array" begin
    mat = fill(42.0)
    da = UnchunkedDiskArray(mat)
    @test sum(da) == sum(mat)
    @test mean(da) == mean(mat)
end

@testset "Edge cases: single element array" begin
    mat = rand(5)
    da = AccessCountDiskArray(mat; chunksize=(5,))
    @test sum(da) == sum(mat)
    @test prod(da) == prod(mat)
    @test minimum(da) == minimum(mat)
    @test maximum(da) == maximum(mat)
    @test any(da .> 0) == any(mat .> 0)
    @test all(da .> 0) == all(mat .> 0)
end

@testset "Edge cases: small chunks, large array" begin
    mat = rand(100)
    da = AccessCountDiskArray(mat; chunksize=(1,))
    @test sum(da) == sum(mat)
    @test mean(da) == mean(mat)
    @test count(da) == count(mat)
end

@testset "Edge cases: all-same values" begin
    mat = fill(5.0, 10, 10)
    da = AccessCountDiskArray(mat; chunksize=(3, 3))
    @test sum(da) == sum(mat)
    @test prod(da) == prod(mat)
    @test minimum(da) == minimum(mat)
    @test maximum(da) == maximum(mat)
    @test mean(da) == mean(mat)
end

@testset "Edge cases: negative values" begin
    mat = randn(10, 10)
    da = AccessCountDiskArray(mat; chunksize=(3, 3))
    @test sum(da) == sum(mat)
    @test minimum(da) == minimum(mat)
    @test maximum(da) == maximum(mat)
    @test extrema(da) == extrema(mat)
end
