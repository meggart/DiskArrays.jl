# DiskArrays.jl

![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.com/meggart/DiskArrays.jl.svg?branch=master)](https://travis-ci.com/meggart/DiskArrays.jl)
[![codecov.io](http://codecov.io/github/meggart/DiskArrays.jl/coverage.svg?branch=master)](http://codecov.io/github/meggart/DiskArrays.jl?branch=master)

This package is an attempt to collect utilities for working with n-dimensional array-like data
structures that do not have considerable overhead for single read operations. Most important
examples are arrays that represent data on hard disk that are accessed through a C
library or that are compressed in chunks. It can be inadvisable to make these arrays a subtype
of `AbstractArray` many functions working with AbstractArrays assume fast random access into single
values (including basic things like `getindex`, `show`, `reduce`, etc...). Currently supported features are:

  - `getindex`/`setindex` with the same rules as base (trailing or singleton dimensions etc)
  - views into `DiskArrays`
  - a fallback `Base.show` method that does not call getindex repeatedly
  - implementations for `mapreduce` and `mapreducedim`, that respect the chunking of the underlying
  dataset. This greatly increases performance of higher-level reductions like `sum(a,dims=d)`
  - an iterator over the values of a DiskArray that caches a chunk of data and returns the values
  within. This allows efficient usage of e.g. `using DataStructures; counter(a)`
  - customization of `broadcast` when there is a `DiskArray` on the LHS. This at least makes things
  like `a.=5` possible and relatively fast

There are basically two ways to use this package.
Either one makes the abstraction directly a subtype of `AbstractDiskArray` which requires
to implement a single `readblock!` method that reads a Cartesian range of data points.
The remaining `getindex` methods will come for free then. The second way is to use
the `interpret_indices_disk` function to get a translation of the user-supplied indices
into a set of ranges and then use these to read the data from disk.

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
function DiskArrays.readblock!(a::PseudoDiskArray,aout,i...)
  ndims(a) == length(i) || error("Number of indices is not correct")
  all(r->isa(r,AbstractUnitRange),i) || error("Not all indices are unit ranges")
  println("Reading at index ", join(string.(i)," "))
  aout .= a.parent[i...]
end
function DiskArrays.writeblock!(a::PseudoDiskArray,v,i...)
  ndims(a) == length(i) || error("Number of indices is not correct")
  all(r->isa(r,AbstractUnitRange),i) || error("Not all indices are unit ranges")
  println("Writing to indices ", join(string.(i)," "))
  view(a.parent,i...) .= v
end
a = RangeArray(4,5,1)
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
