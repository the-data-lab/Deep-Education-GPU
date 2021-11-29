# Deep-Education-GPU

If you don't find pybind11 directory or its files, please run this command.
`git submodule update --init --recursive`

See `https://git-scm.com/book/en/v2/Git-Tools-Submodules` for more information on how to work with sub-modules

## Compilation
```
cd kernel
make
cd ../dl_python_code/
cmake ../kernel/
make
```


## Required Packages

You need numpy, scipy, pytorch (with cuda enabled) to run the GCN test. To run it, first fill the skeloton codebase using the business logic in C++ and CUDA whereever necessary including the kernels and initialization codes. the run:
```
python3 GCN_pubmed.py
```

## About
This repository is now available for public use for teaching end to end workflow of deep learning. This implies that learners/researchers will learn (by doing) beyond what is generally available as tutorial on general-purpose deep learning framework. The aim is to learn how to write a new operator as part of deep learning layer, and how to use it inside a deep learning module using Python environment (Pytorch or Tensorflow). The second stage is to know more about the role of tensor in the computation graph (forward and backward computation) and implement the kernel of the new operator in an independent CUDA module.

We have already made the CPU version online here 'https://github.com/the-data-lab/Deep-Education'

At the end of the assignment, you will learn following things:

- How to introduce a new operator and deep learning layer in Pytorch (ask us if you want to use Tensorflow) using its semantics.
- What is backward computation or gradient computation in a deep lerning training, and how to implement it for a new operator, and make it part of computation graph so that it is automatically invoked during training, but not during inference.
- How to implement the business/core logic of the operator in CUDA (called kernel) in an independent CUDA module. One can implement the kernel using pytorch's plugin environment. But then you should realize that it is not simple to do that, specifically in a classroom teaching (or may be in research) due to steep learning curve. In this assignment, our approach is simple: writing kernel should be in an almost independent CUDA module, so that no steep learning may be needed. However, you must undertand the tensor data structure of the deep learning framework, and why this data structure is most important one to make the layer/operator part of computation graph. You should also undertand, why it is hard to pass this tensor object to an independent CUDA module.
