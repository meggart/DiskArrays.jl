# DiskArrays.jl

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
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

There are basically two ways to use this package.
Either one makes the abstraction directly a subtype of `AbstractDiskArray` which requires
to implement a single `readblock!` method that reads a Cartesian range of data points.
The remaining `getindex` methods will come for free then. The second way is to use
the `interpret_indices_disk` function to get a translation of the user-supplied indices
into a set of ranges and then use these to read the data from disk.

# Example

Here we define a new array type. The only access method that we define is a
`readblock!` function where indices are strictly given as unit ranges along
every dimension of the array. This is a very common API used in libraries
like HDF5, NetCDF and Zarr.


````julia
using DiskArrays

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
````

Now all the Base indexing behaviors work for our array, while minimizing the
number of reads that have to be done:

````julia
a[:,3]
````
````
4-element Array{Int64,1}:
 5
 6
 7
 8
````

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
