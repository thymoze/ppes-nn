# Tensors

The `Tensor` class implements a n-dimensional data store, with support for a variety of math and logic operations, broadcasting and automatic differentiation on a subset of available operators.

This text gives a short overview over the different components, which make up a tensor.

## Tensor data

This component abstracts the data storage of the tensor aswell as the tensors "view" on the underlying data. Tensors can share their actual data, however each will have their own `TensorData` object. The `TensorData` class contains a pointer to the data, and the shape and strides which are used to index into the array. As such it also defines a bunch of operation to iterpret the underlying data differently by creating a new `TensorData` object with different shape and/or strides, like `permute()` (transposition) and `view()`.

There are two subclasses of `TensorData` which deal with the actual data storage. `VectorStorage` holds a `shared_ptr<vector<T>>`, i.e. the data is owned by the program and stored in RAM. `PointerStorage` on the other hand is built to support storing tensors in the flash memory of the Pico and only holds a `T*` to the memory address where the data is stored - it has no ownership of the data.

## Tensor Ops

This file handles mathematical and logic operations on tensors. The underlying observation is that most operations share some structural similiarities which `TensorOps` tries to abstract over. `TensorOps` only defines 3 basic operations:
* **map**: The operation is applied to each element of the tensor individually
* **zip**: The operation is applies to a pair of elements from two input tensors
* **reduce**: The operation groups together elements of the input tensor along a specified axis

With these, most actual operations can be implemented. It would even be possible to implement matrix multiplication through broadcasting and reduce. However to allow for more efficient, specialized implementations `matrix_multiply` is a separate 4th operation. (Due to type system constraints there is also `reduce_index` but that is really just a reduce, which also keeps track of the index on the side, e.g. for argmax)

Thanks to this abstration it is really easy to implement different versions of these operations. Each tensor has its own `TensorOps` object (wrapped in a `TensorBackend`) with which it will execute any function called with it. 
We define `SimpleOps` which - as the name suggests - includes straight-forward, largely unoptimized implementations of all basic operations and `MTOps` which implements a multi-threaded matrix-multiplication for x86 systems based on `std::async`. This would also be an easy extension point to implement operations to make use the multicore functionality of the Pico.

With the `DEFAULT_TENSOR_BACKEND` pre-processor variable it is possible to control the default backend for tensors.

## Tensor Functions

`tensor_functions` includes the implementation of all differentiable operations. Each function has a `forward()` and `backward()` method which implements the required operations for applying and differentiating this function.
These function make up the core of the autodiff functionality of tensors. Each tensor stores the last function applied to it, so that when `backward()` is called it is possible to calculate the derivative using the chain-rule propagating the gradient backward through the expression graph.
A deeper explanation is beyond the scope here but below are some resources further exploring the concept of automatic differentiation.


## Resources

https://minitorch.github.io/  
https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html  
https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html  
https://github.com/MikeInnes/diff-zoo
https://github.com/autodiff/autodiff  
https://github.com/flashlight/flashlight/  