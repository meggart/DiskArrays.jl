using DiskArrays
using Test

@testset "Index interpretation" begin
  import DiskArrays: DimsDropper, Reshaper
  a = zeros(3,3,1)
  @test interpret_indices_disk(a, (:,2,:)) == ((Base.OneTo(3), 2:2, Base.OneTo(1)), DimsDropper{Tuple{Int64}}((2,)))
  @test interpret_indices_disk(a, (1,2,:)) == ((1:1, 2:2, Base.OneTo(1)), DimsDropper{Tuple{Int64,Int64}}((1, 2)))
  @test interpret_indices_disk(a, (1,2,2,1)) == ((1:1, 2:2, 2:2), DimsDropper{Tuple{Int64,Int64,Int64}}((1, 2, 3)))
  @test interpret_indices_disk(a, (1,2,2,1)) == ((1:1, 2:2, 2:2), DimsDropper{Tuple{Int64,Int64,Int64}}((1, 2, 3)))
  @test interpret_indices_disk(a, (:,1:2)) == ((Base.OneTo(3), 1:2, 1:1), DimsDropper{Tuple{Int64}}((3,)))
  @test interpret_indices_disk(a, (:,)) == ((Base.OneTo(3), Base.OneTo(3), Base.OneTo(1)), DiskArrays.Reshaper{Int64}(9))
end

@testset "AbstractDiskArray getindex" begin
  struct RangeArray{N} <: AbstractDiskArray{Int,N}
    s::NTuple{N,Int}
  end
  Base.size(s::RangeArray) = s.s
  Base.size(s::RangeArray,i) = s.s[i]
  RangeArray(s...) = RangeArray(s)
  function DiskArrays.readblock!(r::RangeArray, aout, inds::AbstractUnitRange...)
    ndims(r) == length(inds) || error("This will never happen")
    aout .= sum.(Iterators.product(inds...))
  end
  a = RangeArray(4,5,1)
  @test a[2,3,1] == 6
  @test a[2,3] == 6
  @test a[2,3,1,1] == 6
  @test a[:,1] == [3, 4, 5, 6]
  @test a[1:2, 1:2,1,1] == [3 4; 4 5]
  # Test bitmask indexing
  m = falses(4,5,1)
  m[2,:,1] .= true
  @test a[m] == [4,5,6,7,8]
end
