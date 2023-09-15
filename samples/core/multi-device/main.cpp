/*
 * Copyright (c) 2023 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// OpenCL includes
#include <CL/Utils/Error.h>
#include <CL/Utils/Utils.hpp>
#include <CL/SDK/CLI.hpp>
#include <CL/SDK/Context.hpp>
#include <CL/SDK/Options.hpp>
#include <CL/SDK/Random.hpp>

// TCLAP includes
#include <tclap/CmdLine.h>

// Std library includes
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <tuple> // std::make_tuple
#include <valarray>

// Sample-specific options
struct ConvolutionOptions
{
    size_t x_dim;
    size_t y_dim;
};

// Add option to CLI-parsing SDK utility for input dimensions.
template <> auto cl::sdk::parse<ConvolutionOptions>()
{
    return std::make_tuple(std::make_shared<TCLAP::ValueArg<size_t>>(
                               "x", "x_dim", "x dimension of input", false,
                               4'000, "positive integral"),
                           std::make_shared<TCLAP::ValueArg<size_t>>(
                               "y", "y_dim", "y dimension of input", false,
                               4'000, "positive integral"));
}
template <>
ConvolutionOptions cl::sdk::comprehend<ConvolutionOptions>(
    std::shared_ptr<TCLAP::ValueArg<size_t>> x_dim_arg,
    std::shared_ptr<TCLAP::ValueArg<size_t>> y_dim_arg)
{
    return ConvolutionOptions{ x_dim_arg->getValue(), y_dim_arg->getValue() };
}

// Add option to CLI parsing SDK utility for device type.
template <> auto cl::sdk::parse<cl::sdk::options::MultiDevice>()
{
    std::vector<std::string> valid_dev_strings{ "all", "cpu", "gpu",
                                                "acc", "cus", "def" };
    valid_dev_constraint =
        std::make_unique<TCLAP::ValuesConstraint<std::string>>(
            valid_dev_strings);

    return std::make_shared<TCLAP::ValueArg<std::string>>(
        "t", "type", "Type of device to use", false, "def",
        valid_dev_constraint.get());
}
template <>
cl::sdk::options::MultiDevice
cl::sdk::comprehend<cl::sdk::options::MultiDevice>(
    std::shared_ptr<TCLAP::ValueArg<std::string>> type_arg)
{
    cl_device_type device_type = [](std::string in) -> cl_device_type {
        if (in == "all")
            return CL_DEVICE_TYPE_ALL;
        else if (in == "cpu")
            return CL_DEVICE_TYPE_CPU;
        else if (in == "gpu")
            return CL_DEVICE_TYPE_GPU;
        else if (in == "acc")
            return CL_DEVICE_TYPE_ACCELERATOR;
        else if (in == "cus")
            return CL_DEVICE_TYPE_CUSTOM;
        else if (in == "def")
            return CL_DEVICE_TYPE_DEFAULT;
        else
            throw std::logic_error{ "Unkown device type after CLI parse." };
    }(type_arg->getValue());

    cl::sdk::options::MultiDevice devices;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::sdk::options::MultiDevice devices;
    for (cl::Platform platform : platforms)
    {
        std::vector<cl::Device> platform_devices;
        platform.getDevices(device_type, &platform_devices);
        for (cl::Device device : platform_devices)
        {
            if (devices.triplets.size() < 2)
            {
                cl::sdk::options::DeviceTriplet{ platform, device,
                                                 device_type };
                devices.triplets.push_back({ platform, device, device_type });
            }
        }
    }
    if (devices.triplets.size() < 2)
    {
        std::cerr << "Error: Not enough OpenCL devices of type "
            < < < < " found" << std::endl;
        return NULL;
    }

    return devices;
}

// Host-side implementation of the convolution for verification. Padded input
// assumed.
constexpr uint mask_dim = 3;
const uint pad_width = mask_dim / 2;
void host_convolution(std::vector<cl_float> in, std::vector<cl_float>& out,
                      std::vector<cl_float> mask, size_t x_dim, size_t y_dim)
{
    const size_t pad_x_dim = x_dim + 2 * pad_width;
    for (size_t x = 0; x < x_dim; ++x)
    {
        for (size_t y = 0; y < y_dim; ++y)
        {
            float result = 0.0;
            for (size_t grid_column = x, mask_column = 0;
                 mask_column < mask_dim; ++grid_column, ++mask_column)
            {
                for (size_t grid_row = y, mask_row = 0; mask_row < mask_dim;
                     ++grid_row, ++mask_row)
                {
                    result += mask[mask_column + mask_row * mask_dim]
                        * in[grid_column + grid_row * pad_x_dim];
                }
            }
            out[x + y * x_dim] = result;
        }
    }
}

int main(int argc, char* argv[])
{
    try
    {
        // Check availability of OpenCL devices.
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        std::vector<cl::Device> devices;
        for (size_t i = 0; i < platforms.size(); ++i)
        {
            std::vector<cl::Device> platform_devices;
            platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &platform_devices);
            for (size_t j = 0; j < platform_devices.size(); ++j)
            {
                devices.push_back(platform_devices[j]);
            }
        }

        // Improvement: use only a specific type of device suitable for the
        // task.

        if (!devices.size())
        {
            std::cerr << "Error: No OpenCL devices available" << std::endl;
            return -1;
        }

        // Parse command-line options.
        auto opts = cl::sdk::parse_cli<cl::sdk::options::Diagnostic,
                                       cl::sdk::options::SingleDevice,
                                       // cl::sdk::options::SingleDevice,
                                       ConvolutionOptions>(argc, argv);
        const auto& diag_opts = std::get<0>(opts);
        const auto& dev1_opts = std::get<1>(opts);
        // const auto& dev2_opts = std::get<1>(opts);
        const auto& conv_opts = std::get<2>(opts);

        // Create runtime objects based on user preference or default.
        cl::Context context1 = cl::sdk::get_context(dev1_opts.triplet);
        // cl::Context context2 = cl::sdk::get_context(dev2_opts.triplet);
        cl::Context context2 = context1;
        cl::Device dev1 = context1.getInfo<CL_CONTEXT_DEVICES>().at(0);
        cl::Device dev2 = context2.getInfo<CL_CONTEXT_DEVICES>().at(0);
        cl::Platform platform1{
            dev1.getInfo<CL_DEVICE_PLATFORM>()
        }; // https://github.com/KhronosGroup/OpenCL-CLHPP/issues/150
        cl::Platform platform2{ dev2.getInfo<CL_DEVICE_PLATFORM>() };

        if (!diag_opts.quiet)
        {
            std::cout << "First selected device: "
                      << dev1.getInfo<CL_DEVICE_NAME>() << "\n"
                      << "Using " << platform1.getInfo<CL_PLATFORM_VENDOR>()
                      << " platform\n"
                      << std::endl;
            std::cout << "\nSecond selected device: "
                      << dev2.getInfo<CL_DEVICE_NAME>() << "\n"
                      << "Using " << platform2.getInfo<CL_PLATFORM_VENDOR>()
                      << " platform\n"
                      << std::endl;
        }

        // Query device and runtime capabilities.
        auto d1_highest_device_opencl_c_is_2_x =
            cl::util::opencl_c_version_contains(dev1, "2.");
        auto d1_highest_device_opencl_c_is_3_x =
            cl::util::opencl_c_version_contains(dev1, "3.");

        // Compile kernel.
        const char* kernel_location = "./convolution.cl";
        std::ifstream kernel_stream{ kernel_location };
        if (!kernel_stream.is_open())
            throw std::runtime_error{
                std::string{ "Cannot open kernel source: " } + kernel_location
            };

        cl::Program program1{ context1,
                              std::string{ std::istreambuf_iterator<char>{
                                               kernel_stream },
                                           std::istreambuf_iterator<char>{} } };
        cl::Program program2{ context2,
                              std::string{ std::istreambuf_iterator<char>{
                                               kernel_stream },
                                           std::istreambuf_iterator<char>{} } };

        // If no -cl-std option is specified then the highest 1.x version
        // supported by each device is used to compile the program. Therefore,
        // it's only necessary to add the -cl-std option for 2.0 and 3.0 OpenCL
        // versions.
        cl::string compiler_options =
            cl::string{ d1_highest_device_opencl_c_is_2_x ? "-cl-std=CL2.0 "
                                                          : "" }
            + cl::string{ d1_highest_device_opencl_c_is_3_x ? "-cl-std=CL3.0 "
                                                            : "" };
        program1.build(dev1, compiler_options.c_str());

        auto convolution =
            cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_ulong,
                              cl_ulong>(program1, "convolution_3x3");

        // Query maximum workgroup size (WGS) of kernel supported on each device
        // based on private mem (registers) constraints.
        cl::Kernel reduce_kernel = convolution.getKernel();
        auto wgs1 =
            convolution.getKernel().getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(
                dev1);
        auto wgs2 =
            convolution.getKernel().getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(
                dev2);

        // Initialize host-side storage.
        const size_t x_dim = conv_opts.x_dim;
        const size_t y_dim = conv_opts.y_dim;
        const size_t pad_x_dim = x_dim + 2 * pad_width;
        const size_t pad_y_dim = y_dim + 2 * pad_width;

        auto prng = [engine = std::default_random_engine{},
                     dist = std::uniform_real_distribution<cl_float>{
                         -1.0, 1.0 }]() mutable { return dist(engine); };

        // Initialize input matrix. The input will be padded to remove
        // conditional branches from the convolution kernel for determining
        // out-of-bounds.
        std::vector<cl_float> h_input_grid(pad_x_dim * pad_y_dim);
        if (diag_opts.verbose)
        {
            std::cout << "Generating " << x_dim * y_dim
                      << " random numbers for convolution input grid."
                      << std::endl;
        }
        cl::sdk::fill_with_random(prng, h_input_grid);

        // Fill with 0s the extra rows and columns added for padding the input
        // matrix.
        for (size_t j = 0; j < pad_x_dim; ++j)
        {
            for (size_t i = 0; i < pad_y_dim; ++i)
            {
                if (i == 0 || j == 0 || i == (pad_y_dim - 1)
                    || j == (pad_x_dim - 1))
                {
                    h_input_grid[j + i * pad_x_dim] = 0;
                }
            }
        }

        // Declare output matrix. Output will not be padded.
        std::vector<cl_float> h_output_grid(x_dim * y_dim);

        // Initialize convolution mask.
        std::vector<cl_float> h_mask(mask_dim * mask_dim);
        if (diag_opts.verbose)
        {
            std::cout << "Generating " << mask_dim * mask_dim
                      << " random numbers for convolution mask." << std::endl;
        }
        cl::sdk::fill_with_random(prng, h_mask);

        // Initialize device-side storage.
        const size_t grid_midpoint = pad_y_dim / 2;

        if (diag_opts.verbose)
        {
            std::cout << "Initializing device-side storage...";
            std::cout.flush();
        }

        // Initialize queues for command execution on each device.
        cl::CommandQueue queue1{ context1, dev1,
                                 cl::QueueProperties::Profiling };
        cl::CommandQueue queue2{ context2, dev2,
                                 cl::QueueProperties::Profiling };

        // First device performs the convolution in the upper half (middle
        // border included).
        cl::Buffer dev1_input_grid(
            queue1, h_input_grid.begin(),
            h_input_grid.begin() + pad_x_dim * (grid_midpoint + 1), false);

        // Second device performs the convolution in the lower half (middle
        // border included).
        cl::Buffer dev2_input_grid{ queue2,
                                    h_input_grid.begin()
                                        + pad_x_dim * (grid_midpoint - 1),
                                    h_input_grid.end(), false };

        cl::Buffer dev1_output_grid(queue1, h_output_grid.begin(),
                                    h_output_grid.end(), false);
        cl::Buffer dev2_output_grid(queue2, h_output_grid.begin(),
                                    h_output_grid.end(), false);

        cl::Buffer d1_mask{ queue1, h_mask.begin(), h_mask.end(), false };
        cl::Buffer d2_mask{ queue2, h_mask.begin(), h_mask.end(), false };

        // Launch kernels.
        if (diag_opts.verbose)
        {
            std::cout << " done.\nExecuting on device... ";
            std::cout.flush();
        }

        // Initialize global and local buffers for device execution.
        const cl::NDRange global{ pad_x_dim, pad_y_dim };
        const cl::NDRange local1{ wgs1, 1 };
        const cl::NDRange local2{ wgs2, 1 };

        std::vector<cl::Event> kernel_run;
        auto dev_start = std::chrono::high_resolution_clock::now();

        // Enqueue kernel calls and wait for them to finish.
        kernel_run.push_back(convolution(
            cl::EnqueueArgs{ queue1, global, local1 }, dev1_input_grid,
            dev1_output_grid, d1_mask, x_dim, y_dim));
        kernel_run.push_back(convolution(
            cl::EnqueueArgs{ queue2, global, local2 }, dev2_input_grid,
            dev2_output_grid, d2_mask, x_dim, y_dim));

        cl::WaitForEvents(kernel_run);
        auto dev_end = std::chrono::high_resolution_clock::now();

        // Compute reference host-side convolution.
        if (diag_opts.verbose)
        {
            std::cout << " done.\nExecuting on host... ";
            std::cout.flush();
        }
        auto host_start = std::chrono::high_resolution_clock::now();

        host_convolution(h_input_grid, h_output_grid, h_mask, x_dim, y_dim);

        auto host_end = std::chrono::high_resolution_clock::now();

        if (diag_opts.verbose) std::cout << "done." << std::endl;

        // Fetch and combine results from devices.
        std::vector<cl_float> concatenated_results(x_dim * y_dim);
        cl::copy(queue1, dev1_output_grid, concatenated_results.begin(),
                 concatenated_results.begin() + x_dim * (y_dim / 2));
        cl::copy(queue2, dev2_output_grid,
                 concatenated_results.begin() + x_dim * (y_dim / 2),
                 concatenated_results.end());

        // Validate device-side solution.
        cl_float deviation = 0.0;
        const cl_float tolerance = 1e-6;

        for (size_t i = 0; i < concatenated_results.size(); ++i)
        {
            deviation += std::fabs(concatenated_results[i] - h_output_grid[i]);
        }
        deviation /= concatenated_results.size();

        if (deviation > tolerance)
        {
            std::cerr << "Failed convolution! Normalized deviation "
                      << deviation
                      << " between host and device exceeds tolerance "
                      << tolerance << std::endl;
        }
        else
        {
            std::cout << "Successful convolution!" << std::endl;
        }

        if (!diag_opts.quiet)
        {
            std::cout << "Kernels execution time as measured by host: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(
                             dev_end - dev_start)
                             .count()
                      << " us." << std::endl;
            std::cout << "Kernels execution time as measured by devices: "
                      << std::endl;
            for (auto& pass : kernel_run)
                std::cout << "  - "
                          << cl::util::get_duration<CL_PROFILING_COMMAND_START,
                                                    CL_PROFILING_COMMAND_END,
                                                    std::chrono::microseconds>(
                                 pass)
                                 .count()
                          << " us." << std::endl;
            std::cout << "Reference execution as seen by host: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count()
                      << " us." << std::endl;
        }
    } catch (cl::BuildError& e)
    {
        std::cerr << "OpenCL build error: " << e.what() << std::endl;
        for (auto& build_log : e.getBuildLog())
        {
            std::cerr << "\tBuild log for device: "
                      << build_log.first.getInfo<CL_DEVICE_NAME>() << "\n"
                      << std::endl;
            std::cerr << build_log.second << "\n" << std::endl;
        }
        std::exit(e.err());
    } catch (cl::util::Error& e)
    {
        std::cerr << "OpenCL utils error: " << e.what() << std::endl;
        std::exit(e.err());
    } catch (cl::Error& e)
    {
        std::cerr << "OpenCL runtime error: " << e.what() << std::endl;
        std::exit(e.err());
    } catch (std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return 0;
}
