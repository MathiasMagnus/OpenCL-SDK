# Multi-device Convolution Example

## Sample purpose
This example showcases how to set up a multi-device execution of a given kernel, being the latter a convolution kernel in this particular case.

## Key APIs and Concepts
The main idea behind this example is that a given kernel can be run simultaneously by two (or potentially more) devices, therefore reducing its execution time. One can essentially think of two strategies for this workflow:
1. each device computes its proportional part of the solution at its own speed and the results are combined on the host's side when finished, and
2. each device executes the kernel at its own speed but after each iteration there is P2P communication between the devices to share the partial results.

This example implements the first approach.

### Kernel logic
The kernel is a simple $3 \times 3$ convolution, meaning that the convolution over the input matrix is performed using a $3 \times 3$ mask matrix.

In this implementation of the convolution kernel we assume that the input matrix is padded with 0s, so no extra conditional logic is necessary to ensure that the mask is applied to out-of-bounds elements (e.g. when processing element $(0,0)$ of the output matrix).

## Application flow
### Overview
By default the application will select whichever two devices are first found in the first platform available. A command-line option is added though, so the user can specify which type of device is preferable to be used (e.g. "cpu" or "gpu").

A random input matrix and mask are generated and the workload is equally divided between both devices: one of them performs the convolution over the left half of the matrix and the other one does the same with the right half. When both devices finish the execution of the kernel, the results are fetched and combined by the host.

### Device selection
As mentioned above, by default the first two devices of any type found will be the ones to be used. If there is only one device available, a single-device convolution will be performed.

If the user specifies a type of device to be used, the program iterates over all the devices available from any platform and selects the first two devices of that type found. Note that they do not necessarily belong to the same platform.

### Kernel launch
The rest of the program does not differ much from the usual single-device kernel launch. The only difference is that each device will need a separate set of runtime objects to be created: context, program, command queue, kernel functor, input/mask/output buffers and so on. Note that the input buffers will also contain different data in them, because the first device selected performs the convolution over the left half of the matrix while the second device calculates the convolution over the right half of it.

The kernel is then enqueued to the command queues of each device, and two different events are used to wait for them to be finished. When the devices finish the computations the results are combined in a single host matrix and compared to the host-side results.

## Used API surface
### C
```c
CL_BLOCKING
CL_DEVICE_PLATFORM
CL_DEVICE_TYPE_ALL
CL_HPP_TARGET_OPENCL_VERSION
CL_INVALID_ARG_VALUE
CL_KERNEL_WORK_GROUP_SIZE
CL_MEM_COPY_HOST_PTR
CL_MEM_HOST_READ_ONLY
CL_MEM_READ_ONLY
CL_MEM_WRITE_ONLY
CL_PROFILING_COMMAND_END
CL_PROFILING_COMMAND_START
CL_QUEUE_PROFILING_ENABLE
CL_QUEUE_PROPERTIES
CL_SUCCESS
cl_command_queue
cl_command_queue_properties
cl_context
cl_device_type
cl_event
cl_float
cl_int
cl_kernel
cl_mem
cl_platform_id
cl_program
cl_sdk_fill_with_random_ints_range(pcg32_random_t*, cl_int*, size_t, cl_int, cl_int)
cl_sdk_options_DeviceTriplet
cl_sdk_options_Diagnostic
cl_sdk_options_MultiDevice
cl_uint
cl_ulong
cl_util_build_program(cl_program, cl_device_id, char*)
cl_util_get_device(cl_uint, cl_uint, cl_device_type, cl_int*)
cl_util_get_event_duration(cl_event, cl_profiling_info, cl_profiling_info, cl_int*)
cl_util_print_device_info*(cl_device_id)
cl_util_print_error(cl_int)
cl_util_read_text_file(char*const, size_t*const, cl_int*)
get_dev_type(char*)
clCreateBuffer(cl_context, cl_mem_flags, size_t, void*, cl_int*)
clCreateCommandQueue(cl_context, cl_device_id, cl_command_queue_properties, cl_int*)
clCreateCommandQueueWithProperties(cl_context, cl_device_id, cl_queue_properties*, cl_int*) -> OpenCL >= 2.0
clCreateContext(cl_context_properties*, cl_uint, cl_device_id*, void *(char*, void*,size_t, void*), void*, cl_int*)
clCreateKernel(cl_program, char*, cl_int*)
clGetKernelWorkGroupInfo(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void*, size_t*)
clCreateProgramWithSource(cl_context, cl_uint, char**, size_t*, cl_int*)
clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, cl_uint, size_t*, size_t*, size_t*, cl_uint, cl_event*, cl_event*)
clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, cl_event*, cl_event*)
clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*)
clGetDeviceInfo(cl_device_id, cl_device_info, size_t, void*, size_t*)
clGetPlatformIDs(cl_uint, cl_platform_id*, cl_uint*)
clReleaseCommandQueue(cl_command_queue)
clReleaseContext(cl_context)
clReleaseKernel(cl_kernel)
clReleaseMemObject(cl_mem)
clReleaseProgram(cl_program)
clSetKernelArg(cl_kernel, cl_uint, size_t, void *)
clWaitForEvents(cl_uint, cl_event*)
```

### C++
```c++
cl::Buffer::Buffer(const Context&, IteratorType, IteratorType, bool, bool=false, cl_int*=NULL)
cl::BuildError
cl::CommandQueue::CommandQueue(const cl::Context&, cl::QueueProperties, cl_int*=NULL)
cl::Context
cl::Device::Device()
cl::EnqueueArgs::EnqueueArgs(cl::CommandQueue&, cl::NDRange, cl::NDRange)
cl::Error
cl::Event
cl::Kernel
cl::KernelFunctor::KernelFunctor(const Program&, const string, cl_int*=NULL)
cl::NDRange::NDRange(size_t, size_t)
cl::NullRange
cl::Platform::Platform()
cl::Platform::Platform(cl::Platform)
cl::Platform::get(vector<cl::Platform>*)
cl::Program::Program()
cl::Program::Program(cl::Program)
cl::WaitForEvents(const vector<cl::Event>&)
cl::copy(const CommandQueue&, const cl::Buffer&, IteratorType, IteratorType)
cl::sdk::comprehend()
cl::sdk::fill_with_random()
cl::sdk::get_context(cl_uint, cl_uint, cl_device_type, cl_int*)
cl::sdk::options::MultiDevice
cl::sdk::parse()
cl::sdk::parse_cli()
cl::sdk::options::DeviceTriplet
cl::sdk::options::Diagnostic
cl::sdk::options::SingleDevice
cl::string::string(cl::string)
cl::util::Error
cl::util::get_duration(cl::Event&)
cl::util::opencl_c_version_contains(const cl::Device&, const cl::string&)
```
