using DiskArrays
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
function DiskArrays.readblock!(a::_DiskArray,aout,i...)
  ndims(a) == length(i) || error("Number of indices is not correct")
  all(r->isa(r,AbstractUnitRange),i) || error("Not all indices are unit ranges")
  #println("reading from indices ", join(string.(i)," "))
  a.getindex_count[] += 1
  aout .= a.parent[i...]
end
function DiskArrays.writeblock!(a::_DiskArray,v,i...)
  ndims(a) == length(i) || error("Number of indices is not correct")
  all(r->isa(r,AbstractUnitRange),i) || error("Not all indices are unit ranges")
  #println("Writing to indices ", join(string.(i)," "))
  a.setindex_count[] += 1
  view(a.parent,i...) .= v
end

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
  a = _DiskArray(reshape(1:20,4,5,1))
  @test a[2,3,1] == 10
  @test a[2,3] == 10
  @test a[2,3,1,1] == 10
  @test a[:,1] == [1, 2, 3, 4]
  @test a[1:2, 1:2,1,1] == [1 5; 2 6]
  # Test bitmask indexing
  m = falses(4,5,1)
  m[2,:,1] .= true
  @test a[m] == [2,6,10,14,18]
  #Test that readblock was called exactly onces for every getindex
  @test a.getindex_count[] == 6
end

@testset "AbstractDiskArray setindex" begin
  a = _DiskArray(zeros(Int,4,5,1))
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
  @test a.setindex_count[] == 6
  @test a.parent[1,1:3,1] == [1,2,3]
  @test a.parent[2,:,1] == [1,2,3,4,5]
  @test a.parent[3,3:4,1] == [3,4]
  @test a.parent[4,:,1] == [10,11,12,13,14]
end

@testset "Views" begin
  a = _DiskArray(zeros(Int,4,5,1))
v = view(a,2:3,2:4,1)

v[1:2,1] = [1,2]
v[1:2,2:3] = [4 4; 4 4]
@test v[1:2,1] == [1,2]
@test v[1:2,2:3] == [4 4; 4 4]
@test a.parent[2:3,2] == [1,2]
@test a.parent[2:3,3:4] == [4 4; 4 4]
@test a.getindex_count[] == 2
@test a.setindex_count[] == 2
end

import Statistics: mean
@testset "Reductions" begin
  data = rand(10,20,2)
  for f in (minimum,maximum,sum,
            (i,args...;kwargs...)->all(j->j>0.1,i,args...;kwargs...),
            (i,args...;kwargs...)->any(j->j<0.1,i,args...;kwargs...),
            (i,args...;kwargs...)->mapreduce(x->2*x,+,i,args...;kwargs...))
    a = _DiskArray(data,chunksize=(5,4,2))
    @test isapprox(f(a),f(data))
    @test a.getindex_count[] <= 10
    #And test reduction along dimensions
    a = _DiskArray(data,chunksize=(5,4,2))
    @test all(isapprox.(f(a,dims=2),f(data,dims=2)))
    #The minimum and maximum functions do some initialization, which will increase
    #the number of reads
    @test f in (minimum, maximum) || a.getindex_count[] <= 12
    a = _DiskArray(data,chunksize=(5,4,2))
    @test all(isapprox.(f(a,dims=(1,3)),f(data,dims=(1,3))))
    @test f in (minimum, maximum) || a.getindex_count[] <= 12
  end
end

@testset "Broadcast" begin
  a_disk1 = _DiskArray(rand(10,9,2), chunksize=(5,3,2))
  a_disk2 = _DiskArray(rand(1:10,1,9), chunksize=(1,3))
  a_mem   = reshape(1:2,1,1,2);

  s = a_disk1 .+ a_disk2
  #Test lazy broadcasting
  @test s isa DiskArrays.BroadcastDiskArray
  @test a_disk1.getindex_count[]==0
  @test a_disk1.setindex_count[]==0
  @test a_disk2.getindex_count[]==0
  @test a_disk2.setindex_count[]==0
  @test size(s)==(10,9,2)
  @test eltype(s) == Float64
  #Lets merge another broadcast
  s2 = s ./ a_mem
  @test s isa DiskArrays.BroadcastDiskArray
  @test a_disk1.getindex_count[]==0
  @test a_disk2.getindex_count[]==0
  @test size(s)==(10,9,2)
  @test eltype(s) == Float64
  #And now do the computation with Array as a sink
  aout = zeros(10,9,2)
  aout .= s2
  #Test if the result is correct
  @test aout == (a_disk1.parent .+ a_disk2.parent)./a_mem
  @test a_disk1.getindex_count[]==6
  @test a_disk2.getindex_count[]==6
  #Now use another DiskArray as the output
  aout = _DiskArray(zeros(10,9,2),chunksize=(5,3,2))
  aout .= s ./ a_mem
  @test aout.parent == (a_disk1.parent .+ a_disk2.parent)./a_mem
  @test aout.setindex_count[]==6
  @test a_disk1.getindex_count[]==12
  @test a_disk2.getindex_count[]==12
  #Test reduction of broadcasted expression
  r = sum(s2, dims=(1,2))
  @test all(isapprox.(sum((a_disk1.parent .+ a_disk2.parent)./a_mem,dims=(1,2)),r))
  @test a_disk1.getindex_count[]==18
  @test a_disk2.getindex_count[]==18

end
