# DiskArrays.jl

![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://meggart.github.io/DiskArrays.jl/stable)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://meggart.github.io/DiskArrays.jl/dev)
[![CI](https://github.com/meggart/DiskArrays.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/meggart/DiskArrays.jl/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/meggart/DiskArrays.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/meggart/DiskArrays.jl/tree/main)

This package provides a collection of utilities for working with n-dimensional array-like data
structures that do have considerable overhead for single read operations. 
Most important examples are arrays that represent data on hard disk that are accessed through a C
library or that are compressed in chunks. 
It can be inadvisable to make these arrays a direct subtype of `AbstractArray` many functions working with AbstractArrays assume fast random access into single values (including basic things like `getindex`, `show`, `reduce`, etc...). 

Currently supported features are:

  - `getindex`/`setindex` with the same rules as base (trailing or singleton dimensions etc)
  - views into `DiskArrays`
  - a fallback `Base.show` method that does not call getindex repeatedly
  - implementations for `mapreduce` and `mapreducedim`, that respect the chunking of the underlying
  dataset. This greatly increases performance of higher-level reductions like `sum(a,dims=d)`
  - an iterator over the values of a DiskArray that caches a chunk of data and returns the values
  within. This allows efficient usage of e.g. `using DataStructures; counter(a)`
  - customization of `broadcast` when there is a `DiskArray` on the LHS. This at least makes things
  like `a.=5` possible and relatively fast


## AbstractDiskArray Interface definition

Package authors who want to use this library to make their disk-based array an `AbstractDiskArray` should at least
implement methods for the following functions:

````julia
Base.size(A::CustomDiskArray)
readblock!(A::CustomDiskArray{T,N},aout,r::Vararg{AbstractUnitRange,N})
writeblock!(A::CustomDiskArray{T,N},ain,r::Vararg{AbstractUnitRange,N})
```` 

Here `readblock!` will read a subset of array `A` in a hyper-rectangle defined by the unit ranges `r`. The results shall be written into `aout`. `writeblock!` should write the data given by `ain` into the (hyper-)rectangle of A defined by `r`
When defining the functions it can be safely assumed that `length(r) == ndims(A)` as well as `size(ain) == length.(r)`.
However, bounds checking is *not* performed by the DiskArray machinery and currently should be done by the implementation. 

If the data on disk has rectangular chunks as underlying storage units, you should addtionally implement the following
methods to optimize some operations like broadcast, reductions and sparse indexing:

````julia
DiskArrays.haschunks(A::CustomDiskArray) = DiskArrays.Chunked()
DiskArrays.eachchunk(A::CustomDiskArray) = DiskArrays.GridChunks(A, chunksize)
````

where `chunksize` is a int-tuple of chunk lengths. If the array does not have an internal chunking structure, one should
define

````julia
DiskArrays.haschunks(A::CustomDiskArray) = DiskArrays.Unchunked()
````

Implementing only these methods makes all kinds of strange indexing patterns work (Colons, StepRanges, Integer vectors,
Boolean masks, CartesianIndices, Arrays of CartesianIndex, and mixtures of all these) while making sure that as few
`readblock!` or `writeblock!` calls as possible are performed by reading a rectangular bounding box of the required
array values and re-arranging the resulting values into the output array. 

In addition, DiskArrays.jl provides a few optimizations for sparse indexing patterns to avoid reading and discarding 
too much unnecessary data from disk, for example for indices like `A[:,:,[1,1500]]`. 

# Example

Here we define a new array type that wraps a normal AbstractArray.
The only access method that we define is a
`readblock!` function where indices are strictly given as unit ranges along
every dimension of the array. This is a very common API used in libraries
like HDF5, NetCDF and Zarr. We also define a chunking, which will control
the way iteration and reductions are computed. In order to understand how exactly
data is accessed, we added the additional print statements in the `readblock!`
and `writeblock!` functions.


````julia
using DiskArrays

struct PseudoDiskArray{T,N,A<:AbstractArray{T,N}} <: AbstractDiskArray{T,N}
  parent::A
  chunksize::NTuple{N,Int}
end
PseudoDiskArray(a;chunksize=size(a)) = PseudoDiskArray(a,chunksize)
haschunks(a::PseudoDiskArray) = Chunked()
eachchunk(a::PseudoDiskArray) = GridChunks(a,a.chunksize)
Base.size(a::PseudoDiskArray) = size(a.parent)
function DiskArrays.readblock!(a::PseudoDiskArray,aout,i::AbstractUnitRange...)
  ndims(a) == length(i) || error("Number of indices is not correct")
  all(r->isa(r,AbstractUnitRange),i) || error("Not all indices are unit ranges")
  println("Reading at index ", join(string.(i)," "))
  aout .= a.parent[i...]
end
function DiskArrays.writeblock!(a::PseudoDiskArray,v,i::AbstractUnitRange...)
  ndims(a) == length(i) || error("Number of indices is not correct")
  all(r->isa(r,AbstractUnitRange),i) || error("Not all indices are unit ranges")
  println("Writing to indices ", join(string.(i)," "))
  view(a.parent,i...) .= v
end
a = PseudoDiskArray(rand(4,5,1))
````
````
Disk Array with size 10 x 9 x 1
````

Now all the Base indexing behaviors work for our array, while minimizing the
number of reads that have to be done:

````julia
a[:,3]
````
````
Reading at index Base.OneTo(10) 3:3 1:1

10-element Array{Float64,1}:
 0.8821177068878834
 0.6220977650963209
 0.22676949571723437
 0.3177934541451004
 0.08014908894614026
 0.9989838001681182
 0.5865160181790519
 0.27931778627456216
 0.449108677620097  
 0.22886146620923808
````

As can be seen from the read message, only a single call to `readblock` is performed,
which will map to a single call into the underlying C library.

````julia
mask = falses(4,5,1)
mask[3,2:4,1] .= true
a[mask]
````
````
3-element Array{Int64,1}:
 6
 7
 8
````

One can check in a similar way, that reductions respect the chunks defined by the data type:

````julia
sum(a,dims=(1,3))
````
````
Reading at index 1:5 1:3 1:1
Reading at index 6:10 1:3 1:1
Reading at index 1:5 4:6 1:1
Reading at index 6:10 4:6 1:1
Reading at index 1:5 7:9 1:1
Reading at index 6:10 7:9 1:1

1×9×1 Array{Float64,3}:
[:, :, 1] =
 6.33221  4.91877  3.98709  4.18658  …  6.01844  5.03799  3.91565  6.06882
 ````

When a DiskArray is on the LHS of a broadcasting expression, the results with be
written chunk by chunk:

````julia
va = view(a,5:10,5:8,1)
va .= 2.0
a[:,:,1]
````
````
Writing to indices 5:5 5:6 1:1
Writing to indices 6:10 5:6 1:1
Writing to indices 5:5 7:8 1:1
Writing to indices 6:10 7:8 1:1
Reading at index Base.OneTo(10) Base.OneTo(9) 1:1

10×9 Array{Float64,2}:
 0.929979   0.664717  0.617594  0.720272   …  0.564644  0.430036  0.791838
 0.392748   0.508902  0.941583  0.854843      0.682924  0.323496  0.389914
 0.761131   0.937071  0.805167  0.951293      0.630261  0.290144  0.534721
 0.332388   0.914568  0.497409  0.471007      0.470808  0.726594  0.97107
 0.251657   0.24236   0.866905  0.669599      2.0       2.0       0.427387
 0.388476   0.121011  0.738621  0.304039   …  2.0       2.0       0.687802
 0.991391   0.621701  0.210167  0.129159      2.0       2.0       0.733581
 0.371857   0.549601  0.289447  0.509249      2.0       2.0       0.920333
 0.76309    0.648815  0.632453  0.623295      2.0       2.0       0.387723
 0.0882056  0.842403  0.147516  0.0562536     2.0       2.0       0.107673
````

## Accessing strided Arrays

There are situations where one wants to read every other value along a certain axis or provide arbitrary strides. Some DiskArray backends may want to provide optimized methods to read these strided arrays. 
In this case a backend can define `readblock!(a,aout,r::OrdinalRange...)` and the respective `writeblock`
method which will overwrite the fallback behavior that would read the whol block of data and only return
the desired range.

## Arrays that do not implement eachchunk

There are arrays that live on disk but which are not split into rectangular chunks, so that the `haschunks` trait returns `Unchunked()`. In order to still enable broadcasting and reductions for these arrays, a chunk size will be estimated in a way that a certain memory limit per chunk is not exceeded. This memory limit defaults to 100MB and can be modified by changing `DiskArrays.default_chunk_size[]`. Then a chunk size is computed based on the element size of the array. However, there are cases where the size of the element type is undefined, e.g. for Strings or variable-length vectors. In these cases one can overload the `DiskArrays.element_size` function for certain container types which returns an approximate element size (in bytes). Otherwise the size of an element will simply be assumed to equal the value stored in `DiskArrays.fallback_element_size` which defaults to 100 bytes. 


[ci-img]: https://github.com/meggart/DiskArrays.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/meggart/DiskArrays.jl/actions?query=workflow%3ACI
[codecov-img]: http://codecov.io/github/meggart/DiskArrays.jl/coverage.svg?branch=main
[codecov-url]: (http://codecov.io/github/meggart/DiskArrays.jl?branch=main)
