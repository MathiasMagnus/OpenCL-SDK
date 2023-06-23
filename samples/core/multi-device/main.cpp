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

// OpenCL SDK includes
#include <CL/Utils/Utils.hpp>
#include <CL/SDK/Context.hpp>
#include <CL/SDK/Options.hpp>
#include <CL/SDK/CLI.hpp>
#include <CL/SDK/Random.hpp>

#include <CL/Utils/Error.h>

// TCLAP includes
#include <tclap/CmdLine.h>

// STL includes
#include <iostream>
#include <valarray>
#include <random>
#include <algorithm>
#include <fstream>
#include <tuple> // std::make_tuple
#include <numeric> // std::accumulate

// Sample-specific option
struct ConvolutionOptions
{
    size_t x_dim;
    size_t y_dim;
};

// Add option to CLI parsing SDK utility
template <> auto cl::sdk::parse<ConvolutionOptions>()
{
    return std::make_tuple(std::make_shared<TCLAP::ValueArg<size_t>>(
                               "x", "x_dim", "x-dimension of input", false,
                               4'000, "positive integral"),
                           std::make_shared<TCLAP::ValueArg<size_t>>(
                               "y", "y_dim", "y-dimension of input", false,
                               4'000, "positive integral"));
}
template <>
ConvolutionOptions cl::sdk::comprehend<ConvolutionOptions>(
    std::shared_ptr<TCLAP::ValueArg<size_t>> x_dim_arg,
    std::shared_ptr<TCLAP::ValueArg<size_t>> y_dim_arg)
{
    return ConvolutionOptions{ x_dim_arg->getValue(), y_dim_arg->getValue() };
}

constexpr uint mask_dim = 3;

// host-side implementation of the convolution for verification
void host_convolution(std::vector<cl_float> in,
                      std::vector<cl_float> &out,
                      std::vector<cl_float> mask,
                      size_t x_dim,
                      size_t y_dim)
{

    for(size_t x = 1; x < x_dim - 1; ++x){
        for(size_t y = 1; y < y_dim - 1; ++y){
            float result = 0.0;
            for(size_t grid_column = x - 1, mask_column = 0;
                mask_column < mask_dim; ++mask_column, ++grid_column){
                for(size_t grid_row = y - 1, mask_row = 0;
                    mask_row < mask_dim; ++mask_row, ++grid_row){
                        result += in[grid_column + grid_row * x_dim] * mask[mask_column + mask_row * mask_dim];
                }
            }
            out[x + y * x_dim] = result;
        }
    }



}

// Add option to CLI parsing SDK utility
template <> auto cl::sdk::parse<cl::sdk::options::MultiDevice>()
{
    std::vector<std::string> valid_dev_strings{ "all", "cpu", "gpu",
                                                "acc", "cus", "def" };
    valid_dev_constraint =
        std::make_unique<TCLAP::ValuesConstraint<std::string>>(
            valid_dev_strings);

    return std::make_shared<TCLAP::ValueArg<std::string>>(
                               "t", "type", "Type of device to use", false,
                               "def", valid_dev_constraint.get());
}
template <>
cl::sdk::options::MultiDevice cl::sdk::comprehend<cl::sdk::options::MultiDevice>(
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
            throw std::logic_error{ "Unkown device type after cli parse." };
    }(type_arg->getValue());

    cl::sdk::options::MultiDevice devices;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    // TODO: get proper command line interface for multiple devices
    //for(cl::Platform platform : platforms){
    //    std::vector<cl::Device> platform_devices;
    //    platform.getDevices(device_type, &platform_devices);
    //    for(cl::Device device : platform_devices){
    //        if(devices.triplets.size() < 2){
    //            cl::sdk::options::DeviceTriplet {platform, device, device_type};
    //            devices.triplets.push_back({
    //                            platform,
    //                            device,
    //                            device_type
    //            });
    //        }
    //    }
    //}

    return devices;
}

int main(int argc, char* argv[])
{
    try
    {
        // Parse command-line options
        auto opts =
            cl::sdk::parse_cli<cl::sdk::options::Diagnostic,
                               cl::sdk::options::SingleDevice,
                               //cl::sdk::options::SingleDevice,
                               ConvolutionOptions>(
                argc, argv);
        const auto& diag_opts = std::get<0>(opts);
        const auto& dev1_opts = std::get<1>(opts);
        //const auto& dev2_opts = std::get<2>(opts);
        const auto& conv_opts = std::get<2>(opts);

        // Create runtime objects based on user preference or default
        cl::Context context1 = cl::sdk::get_context(dev1_opts.triplet);
        cl::Device device1 = context1.getInfo<CL_CONTEXT_DEVICES>().at(0);
        cl::CommandQueue queue1{ context1, device1,
                                cl::QueueProperties::Profiling };
        cl::Platform platform1{
            device1.getInfo<CL_DEVICE_PLATFORM>()
        }; // https://github.com/KhronosGroup/OpenCL-CLHPP/issues/150

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform plat;

        //for(iterate over platforms)
        //    for(iterate over devices)
        //        if(device_type is correct)
        //            add device to devices_to_be_used


        if(!diag_opts.quiet){
            std::cout << "Selected devices: " << std::endl;
        }

        if (!diag_opts.quiet){
            std::cout << "Selected platform: "
                      << platform1.getInfo<CL_PLATFORM_VENDOR>() << "\n"
                      << "Selected device: " << device1.getInfo<CL_DEVICE_NAME>()
                      << "\n"
                      << std::endl;
        }

        // Query device and runtime capabilities
        auto d1_highest_device_opencl_c_is_2_x =
            cl::util::opencl_c_version_contains(device1, "2.");
        auto d1_highest_device_opencl_c_is_3_x =
            cl::util::opencl_c_version_contains(device1, "3.");

        // Compile kernel
        const char* kernel_location = "./convolution.cl";
        std::ifstream kernel_stream{ kernel_location };
        if (!kernel_stream.is_open())
            throw std::runtime_error{
                std::string{ "Cannot open kernel source: " } + kernel_location
            };

        cl::Program d1_program{ context1,
                             std::string{std::istreambuf_iterator<char>{kernel_stream},
                                         std::istreambuf_iterator<char>{} }};
        cl::string compiler_options =
            cl::string{ d1_highest_device_opencl_c_is_2_x ? "-cl-std=CL2.0 "
                                                         : "" }
            + cl::string{ d1_highest_device_opencl_c_is_3_x ? "-cl-std=CL3.0 "
                                                         : "" };
        d1_program.build(device1, compiler_options.c_str());

        auto convolution =
            cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer,
                              cl_ulong, cl_ulong>(d1_program, "convolution_3x3");

        // Query maximum supported WGS of kernel on device based on private mem
        // (register) constraints
        cl::Kernel reduce_kernel = convolution.getKernel();
        auto wgs =
            convolution.getKernel().getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(
                device1);

        // Initialize host-side storage
        const size_t x_dim = conv_opts.x_dim;
        const size_t y_dim = conv_opts.y_dim;

        auto prng = [engine = std::default_random_engine{},
                     dist = std::uniform_real_distribution<cl_float>{
                         -1.0, 1.0 }]() mutable { return dist(engine); };

        std::vector<cl_float> h_input_grid(x_dim * y_dim);
        if (diag_opts.verbose){
            std::cout << "Generating " << x_dim * y_dim
                      << " random numbers for convolution input grid." << std::endl;
        }
        cl::sdk::fill_with_random(prng, h_input_grid);
        // copy input to output to conserve boundary conditions
        std::vector<cl_float> h_output_grid(h_input_grid);

        std::vector<cl_float> h_mask(mask_dim * mask_dim);
        if (diag_opts.verbose){
            std::cout << "Generating " << mask_dim * mask_dim
                      << " random numbers for convolution mask." << std::endl;
        }
        cl::sdk::fill_with_random(prng, h_mask);

        // Initialize device-side storage
        const size_t grid_midpoint = y_dim/2;

        if (diag_opts.verbose){
            std::cout << "Initializing device-side storage...";
            std::cout.flush();
        }
        //TODO: split work between two devices - i.e.:
        // - transfer half the grid + border to each device (easiest way is to split at a certain row, then the arising memory partitions are contiguous
        // - run kernel on each device with appropriate offset to work groups
        // - copy back results from each device to appropriate location on host-side result

        // device 1 performs the convolution in the upper half
        // end at grid_midpoint + 1 because the border needs to be included
        //cl::Buffer d1_input_grid{queue1, h_input_grid.begin(), h_input_grid.begin() + x_dim * (grid_midpoint + 1), false};
        cl::Buffer d1_input_grid(queue1, h_input_grid.begin(), h_input_grid.end(), false);

        // device 2 performs the convolution in the lower half
        // starts at grid_mid_point - 1 because the border needs to be included
        //cl::Buffer d2_input_grid{queue2, h_input_grid.begin() + x_dim * (grid_midpoint - 1), h_input_grid.end(), false};

        cl::Buffer d1_output_grid(queue1, h_input_grid.begin(), h_input_grid.end(), false);
        //cl::Buffer d2_output_grid{queue2}

        cl::Buffer d1_mask{queue1, h_mask.begin(), h_mask.end(), false};
        //cl::Buffer d2_mask{queue2, h_mask.begin(), h_mask.end(), false};

        // Launch kernels
        if (diag_opts.verbose)
        {
            std::cout << " done.\nExecuting on device... ";
            std::cout.flush();
        }
        const cl::NDRange global1 {x_dim, y_dim};
        const cl::NDRange offset1 {0, 0};
        const cl::NDRange local1 {wgs, 1};
        //cl::NDRange offset2 {0, y_dim / 2};

        std::vector<cl::Event> kernel_run;
        auto dev_start = std::chrono::high_resolution_clock::now();
        //actual kernel call
        kernel_run.push_back(convolution(
            cl::EnqueueArgs{ queue1, offset1, global1, local1 }, d1_input_grid, d1_output_grid, d1_mask, x_dim, y_dim));

        cl::WaitForEvents(kernel_run);
        auto dev_end = std::chrono::high_resolution_clock::now();

        // calculate reference dataset
        if (diag_opts.verbose){
            std::cout << " done.\nExecuting on host... ";
            std::cout.flush();
        }
        auto host_start = std::chrono::high_resolution_clock::now();

        host_convolution(h_input_grid, h_output_grid, h_mask, x_dim, y_dim);

        auto host_end = std::chrono::high_resolution_clock::now();
        if (diag_opts.verbose) std::cout << "done." << std::endl;

        // Fetch results
        std::vector<cl_float> concatenated_results(x_dim * y_dim);
        cl::copy(queue1, d1_output_grid, concatenated_results.begin(), concatenated_results.end());
        //cl::copy(queue2, d2_output_grid, concatenated_results.begin() + y_dim/2 * x_dim, concatenated_results.end());

        // Validate
        cl_float deviation = 0.0;
        for(size_t i = 0; i < concatenated_results.size(); ++i){
            deviation += std::fabs(concatenated_results[i] - h_output_grid[i]);
        }
        deviation /= concatenated_results.size();
        if(deviation > 1e-6){
            std::cerr << "Normalized deviation between host and device results: " << deviation << std::endl;
        }

        if (!diag_opts.quiet)
        {
            std::cout << "Total device execution as seen by host: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(
                             dev_end - dev_start).count()
                      << " us." << std::endl;
            // TODO: print times for different devices?
            std::cout << "Kernel execution time as measured by device: ";
            for (auto& pass : kernel_run)
                std::cout << cl::util::get_duration<
                                CL_PROFILING_COMMAND_START,
                                CL_PROFILING_COMMAND_END,
                                std::chrono::microseconds>(pass).count()
                          << " us." << std::endl;
            std::cout << "Reference execution as seen by host: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count()
                      << " us." << std::endl;
        }
    } catch (cl::util::Error& e)
    {
        std::cerr << "OpenCL utils error: " << e.what() << std::endl;
        std::exit(e.err());
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
