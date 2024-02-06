# PyTorch Quantization Aware Training

## Introduction

PyTorch quantization aware training example for ResNet.

## Usages

### Build Docker Image

# Note: skip the cache during a docker build and run every step in the Dockerfile. It results in a slower build but will ensure you run every step. Specifying --no-cache is helpful for debugging build issues. You can also use it to force a dependency to upgrade
```
$ docker build -f docker/pytorch.Dockerfile --no-cache --tag=pytorch:1.8.1 .
```

### Run Docker Container

```
$ docker run -it --rm --gpus device=0 --ipc=host -v $(pwd):/mnt pytorch:1.8.1
```

### Run ResNet

```
$ cd mnt
$ python cifar.py
```

## References

* [PyTorch Quantization Aware Training](https://leimao.github.io/blog/PyTorch-Quantization-Aware-Training/)
* [PyTorch Static Quantization](https://leimao.github.io/blog/PyTorch-Static-Quantization/)
* [PyTorch CIFAR10 Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
* [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
* [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
* [PyTorch Distributed Training](https://leimao.github.io/blog/PyTorch-Distributed-Training/)


## Issues
1) NVIDIA GeForce RTX 3060 with CUDA capability sm_86 is not compatible with the current PyTorch installation
Just use a conda with torch v2

2) File "/home/abel/anaconda3/envs/nn/lib/python3.11/site-packages/torch/ao/nn/quantized/modules/batchnorm.py", line 70, in forward
        # disabling this since this is not symbolically traceable
        # self._check_input_dim(input)
        return torch.ops.quantized.batch_norm2d(
               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
            input, self.weight, self.bias, self.running_mean,
            self.running_var, self.eps, self.scale, self.zero_point)
RuntimeError: Cannot input a tensor of dimension other than 0 as a scalar argument
