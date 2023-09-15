# Multi-device Pi approximation example using Monte Carlo method

## Sample purpose

## Key APIs and Concepts

### Kernel logic

## Application flow

### Overview

### Command queues and synchronization

## Used API surface

```c++
cl::Buffer
cl::BuildError
cl::CommandQueue::CommandQueue(cl::CommandQueue)
cl::Context
cl::copy()
cl::Device::Device()
cl::EnqueueArgs()
cl::Error
cl::Event
cl::Kernel
cl::KernelFunctor
cl::NDRange
cl::Platform::Platform()
cl::Platform::Platform(cl::Platform)
cl::Platform::get()
cl::Program::Program()
cl::Program::Program(cl::Program)
cl::sdk::comprehend()
cl::sdk::fill_with_random()
cl::sdk::get_context()
cl::sdk::options::MultiDevice
cl::sdk::parse()
cl::sdk::parse_cli()
cl::sdk::options::DeviceTriplet
cl::sdk::options::Diagnostic
cl::sdk::options::SingleDevice
cl::util::Error
cl::util::get_duration()
cl::util::opencl_c_version_contains()
cl::WaitForEvents()
```