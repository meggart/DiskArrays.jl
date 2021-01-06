using DiskArrays
using DiskArrays: ReshapedDiskArray, PermutedDiskArray
using Test

#Define a data structure that can be used for testing
struct _DiskArray{T,N,A<:AbstractArray{T,N}} <: AbstractDiskArray{T,N}
  getindex_count::Ref{Int}
  setindex_count::Ref{Int}
  parent::A
  chunksize::NTuple{N,Int}
end
_DiskArray(a;chunksize=size(a)) = _DiskArray(Ref(0),Ref(0),a,chunksize)
Base.size(a::_DiskArray) = size(a.parent)
DiskArrays.haschunks(::_DiskArray) = DiskArrays.Chunked()
DiskArrays.eachchunk(a::_DiskArray) = DiskArrays.GridChunks(a,a.chunksize)
getindex_count(a::_DiskArray) = a.getindex_count[]
setindex_count(a::_DiskArray) = a.setindex_count[]
trueparent(a::_DiskArray) = a.parent
getindex_count(a::ReshapedDiskArray) = getindex_count(a.parent)
setindex_count(a::ReshapedDiskArray) = setindex_count(a.parent)
trueparent(a::ReshapedDiskArray) = trueparent(a.parent)
getindex_count(a::PermutedDiskArray) = getindex_count(a.a.parent)
setindex_count(a::PermutedDiskArray) = setindex_count(a.a.parent)
trueparent(a::PermutedDiskArray{T,N,<:PermutedDimsArray{T,N,perm,iperm}}) where {T,N,perm,iperm} = permutedims(trueparent(a.a.parent),perm)
function DiskArrays.readblock!(a::_DiskArray,aout,i::AbstractUnitRange...)
  ndims(a) == length(i) || error("Number of indices is not correct")
  all(r->isa(r,AbstractUnitRange),i) || error("Not all indices are unit ranges")
  #println("reading from indices ", join(string.(i)," "))
  a.getindex_count[] += 1
  aout .= a.parent[i...]
end
function DiskArrays.writeblock!(a::_DiskArray,v,i::AbstractUnitRange...)
  ndims(a) == length(i) || error("Number of indices is not correct")
  all(r->isa(r,AbstractUnitRange),i) || error("Not all indices are unit ranges")
  #println("Writing to indices ", join(string.(i)," "))
  a.setindex_count[] += 1
  view(a.parent,i...) .= v
end

function test_getindex(a)
  @test a[2,3,1] == 10
  @test a[2,3] == 10
  @test a[2,3,1,1] == 10
  @test a[:,1] == [1, 2, 3, 4]
  @test a[1:2, 1:2,1,1] == [1 5; 2 6]
  @test a[2:2:4,1:2:5] == [2 10 18; 4 12 20]
  @test a[end:-1:1,1,1] == [4,3,2,1]
  @test a[[1,3,4],[1,3],1] == [1 9; 3 11; 4 12]
  # Test bitmask indexing
  m = falses(4,5,1)
  m[2,:,1] .= true
  @test a[m] == [2,6,10,14,18]
  #Test that readblock was called exactly onces for every getindex
  @test getindex_count(a) == 9
end

function test_setindex(a)
  a[1,1,1] = 1
  a[1,2] = 2
  a[1,3,1,1] = 3
  a[2,:] = [1, 2, 3, 4, 5]
  a[3, 3:4,1,1] = [3,4]
  # Test bitmask indexing
  m = falses(4,5,1)
  m[4,:,1] .= true
  a[m] = [10,11,12,13,14]
  #Test that readblock was called exactly onces for every getindex
  @test setindex_count(a) == 6
  @test trueparent(a)[1,1:3,1] == [1,2,3]
  @test trueparent(a)[2,:,1] == [1,2,3,4,5]
  @test trueparent(a)[3,3:4,1] == [3,4]
  @test trueparent(a)[4,:,1] == [10,11,12,13,14]
  a[1:2:4,1:2:5,1] = [1 2 3; 5 6 7]
  @test trueparent(a)[1:2:4,1:2:5,1] == [1 2 3; 5 6 7]
  @test setindex_count(a) == 7
  a[[2,4],1:2,1] = [1 2; 5 6]
  @test trueparent(a)[[2,4],1:2,1] == [1 2; 5 6]
  @test setindex_count(a) == 8
end

function test_view(a)
  v = view(a,2:3,2:4,1)

  v[1:2,1] = [1,2]
  v[1:2,2:3] = [4 4; 4 4]
  @test v[1:2,1] == [1,2]
  @test v[1:2,2:3] == [4 4; 4 4]
  @test trueparent(a)[2:3,2] == [1,2]
  @test trueparent(a)[2:3,3:4] == [4 4; 4 4]
  @test getindex_count(a) == 2
  @test setindex_count(a) == 2
end

function test_reductions(af)
  data = rand(10,20,2)
  for f in (minimum,maximum,sum,
            (i,args...;kwargs...)->all(j->j>0.1,i,args...;kwargs...),
            (i,args...;kwargs...)->any(j->j<0.1,i,args...;kwargs...),
            (i,args...;kwargs...)->mapreduce(x->2*x,+,i,args...;kwargs...))
    a = af(data)
    @test isapprox(f(a),f(data))
    @test getindex_count(a) <= 10
    #And test reduction along dimensions
    a = _DiskArray(data,chunksize=(5,4,2))
    @test all(isapprox.(f(a,dims=2),f(data,dims=2)))
    #The minimum and maximum functions do some initialization, which will increase
    #the number of reads
    @test f in (minimum, maximum) || getindex_count(a) <= 12
    a = _DiskArray(data,chunksize=(5,4,2))
    @test all(isapprox.(f(a,dims=(1,3)),f(data,dims=(1,3))))
    @test f in (minimum, maximum) || getindex_count(a) <= 12
  end
end

function test_broadcast(a_disk1)
  a_disk2 = _DiskArray(rand(1:10,1,9), chunksize=(1,3))
  a_mem   = reshape(1:2,1,1,2);

  s = a_disk1 .+ a_disk2
  #Test lazy broadcasting
  @test s isa DiskArrays.BroadcastDiskArray
  @test getindex_count(a_disk1)==0
  @test setindex_count(a_disk1)==0
  @test getindex_count(a_disk2)==0
  @test setindex_count(a_disk2)==0
  @test size(s)==(10,9,2)
  @test eltype(s) == Float64
  #Lets merge another broadcast
  s2 = s ./ a_mem
  @test s isa DiskArrays.BroadcastDiskArray
  @test getindex_count(a_disk1)==0
  @test getindex_count(a_disk2)==0
  @test size(s)==(10,9,2)
  @test eltype(s) == Float64
  #And now do the computation with Array as a sink
  aout = zeros(10,9,2)
  aout .= s2
  #Test if the result is correct
  @test aout == (trueparent(a_disk1) .+ trueparent(a_disk2))./a_mem
  @test getindex_count(a_disk1)==6
  @test getindex_count(a_disk2)==6
  #Now use another DiskArray as the output
  aout = _DiskArray(zeros(10,9,2),chunksize=(5,3,2))
  aout .= s ./ a_mem
  @test trueparent(aout) == (trueparent(a_disk1) .+ trueparent(a_disk2))./a_mem
  @test setindex_count(aout)==6
  @test getindex_count(a_disk1)==12
  @test getindex_count(a_disk2)==12
  #Test reduction of broadcasted expression
  r = sum(s2, dims=(1,2))
  @test all(isapprox.(sum((trueparent(a_disk1) .+ trueparent(a_disk2))./a_mem,dims=(1,2)),r))
  @test getindex_count(a_disk1)==18
  @test getindex_count(a_disk2)==18
end

@testset "Index interpretation" begin
  import DiskArrays: DimsDropper, Reshaper
  a = zeros(3,3,1)
  @test interpret_indices_disk(a, (:,2,:)) == ((Base.OneTo(3), 2:2, Base.OneTo(1)), DimsDropper{Tuple{Int}}((2,)))
  @test interpret_indices_disk(a, (1,2,:)) == ((1:1, 2:2, Base.OneTo(1)), DimsDropper{Tuple{Int,Int}}((1, 2)))
  @test interpret_indices_disk(a, (1,2,2,1)) == ((1:1, 2:2, 2:2), DimsDropper{Tuple{Int,Int,Int}}((1, 2, 3)))
  @test interpret_indices_disk(a, (1,2,2,1)) == ((1:1, 2:2, 2:2), DimsDropper{Tuple{Int,Int,Int}}((1, 2, 3)))
  @test interpret_indices_disk(a, (:,1:2)) == ((Base.OneTo(3), 1:2, 1:1), DimsDropper{Tuple{Int}}((3,)))
  @test interpret_indices_disk(a, (:,)) == ((Base.OneTo(3), Base.OneTo(3), Base.OneTo(1)), DiskArrays.Reshaper{Int}(9))
end

@testset "AbstractDiskArray getindex" begin
  a = _DiskArray(reshape(1:20,4,5,1))
  test_getindex(a)
end


@testset "AbstractDiskArray setindex" begin
  a = _DiskArray(zeros(Int,4,5,1))
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
  a = _DiskArray(zeros(Int,4,5,1))
  test_view(a)
end

# The remaing tests only work for Julia >= 1.3
if VERSION >= v"1.3.0"
import Statistics: mean
@testset "Reductions" begin
  a = data -> _DiskArray(data,chunksize=(5,4,2))
  test_reductions(a)
end

@testset "Broadcast" begin
  a_disk1 = _DiskArray(rand(10,9,2), chunksize=(5,3,2))
  test_broadcast(a_disk1)
end

@testset "Reshape" begin
  a = reshape(_DiskArray(reshape(1:20,4,5)),4,5,1)
  test_getindex(a)
  a = reshape(_DiskArray(zeros(Int,4,5)),4,5,1)
  test_setindex(a)
  a = reshape(_DiskArray(zeros(Int,4,5)),4,5,1)
  test_view(a)
  a = data -> reshape(_DiskArray(data,chunksize=(5,4,2)),10,20,2,1)
  test_reductions(a)
end

import Base.PermutedDimsArrays.invperm
@testset "Permutedims" begin
  p = (3,1,2)
  ip = invperm(p)
  a = permutedims(_DiskArray(permutedims(reshape(1:20,4,5,1),ip)),p)
  test_getindex(a)
  a = permutedims(_DiskArray(zeros(Int,5,1,4)),p)
  test_setindex(a)
  a = permutedims(_DiskArray(zeros(Int,5,1,4)),p)
  test_view(a)
  a = data -> permutedims(_DiskArray(permutedims(data,ip),chunksize=(4,2,5)),p)
  test_reductions(a)
  a_disk1 = permutedims(_DiskArray(rand(9,2,10), chunksize=(3,2,5)),p)
  test_broadcast(a_disk1)
end

@testset "Unchunked String arrays" begin
  a = string.(reshape(1:100000,200,500));

  DiskArrays.default_chunk_size[] = 100
  DiskArrays.fallback_element_size[] = 100
  @test DiskArrays.estimate_chunksize(a) == (200,500)
  @test DiskArrays.eachchunk(a) == DiskArrays.GridChunks(a,(200,500))
  DiskArrays.default_chunk_size[] = 1
  @test DiskArrays.estimate_chunksize(a) == (200,50)
  @test DiskArrays.eachchunk(a) == DiskArrays.GridChunks(a,(200,50))
  DiskArrays.fallback_element_size[] = 200
  @test DiskArrays.estimate_chunksize(a) == (200,25)
  @test DiskArrays.eachchunk(a) == DiskArrays.GridChunks(a,(200,25))
end

end
