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

// OpenCL SDK includes.
#include <CL/SDK/CLI.h>
#include <CL/SDK/Context.h>
#include <CL/SDK/Options.h>
#include <CL/SDK/Random.h>

// OpenCL Utils includes.
#include <CL/Utils/Error.h>
#include <CL/Utils/Event.h>
#include <CL/Utils/Utils.h>

// Standard header includes.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Sample-specific options.
struct convolution_options
{
    size_t x_dim;
    size_t y_dim;
};

// Add option to CLI-parsing SDK utility for input dimensions.
cag_option ConvolutionOptions[] = { { .identifier = 'x',
                                      .access_letters = "x",
                                      .access_name = "x_dim",
                                      .value_name = "(positive integral)",
                                      .description = "x dimension of input" },

                                    { .identifier = 'y',
                                      .access_letters = "y",
                                      .access_name = "y_dim",
                                      .value_name = "(positive integral)",
                                      .description = "y dimension of input" } };

ParseState parse_ConvolutionOptions(const char identifier,
                                    cag_option_context* cag_context,
                                    struct convolution_options* opts)
{
    const char* value;

    switch (identifier)
    {
        case 'x':
            if ((value = cag_option_get_value(cag_context)))
            {
                opts->x_dim = strtoull(value, NULL, 0);
                return ParsedOK;
            }
            else
                return ParseError;
        case 'y':
            if ((value = cag_option_get_value(cag_context)))
            {
                opts->y_dim = strtoull(value, NULL, 0);
                return ParsedOK;
            }
            else
                return ParseError;
    }
    return NotParsed;
}

// Add option to CLI parsing SDK utility for multi-device type.
cag_option MultiDeviceOptions[] = { { .identifier = 't',
                                      .access_letters = "t",
                                      .access_name = "type",
                                      .value_name = "(all|cpu|gpu|acc|cus|def)",
                                      .description =
                                          "Type of device to use" } };

ParseState parse_MultiDeviceOptions(const char identifier,
                                    cag_option_context* cag_context,
                                    struct cl_sdk_options_MultiDevice* opts)
{
    const char* value;

    switch (identifier)
    {
        case 't':
            if ((value = cag_option_get_value(cag_context)))
            {
                // If the user selects a device type, query all devices and
                // filter the ones of that type.
                struct cl_sdk_options_DeviceTriplet* triplets;
                size_t number = 0;
                cl_device_type dev_type = get_dev_type(value);

                // Get platforms IDs.
                cl_uint num_platforms = 0;
                clGetPlatformIDs(0, NULL, &num_platforms);

                cl_platform_id* platforms = (cl_platform_id*)malloc(
                    num_platforms * sizeof(cl_platform_id));
                clGetPlatformIDs(num_platforms, platforms, NULL);

                // Calculate total number of triplets to add.
                for (cl_uint platform_id = 0; platform_id < num_platforms;
                     ++platform_id)
                {
                    // Get devices IDs.
                    cl_uint num_devices = 0;
                    clGetDeviceIDs(platforms[platform_id], dev_type, 0, NULL,
                                   &num_devices);
                    number += num_devices;
                }

                if (!number)
                {
                    fprintf(stderr,
                            "Error: No OpenCL devices of type %s available",
                            value);
                    return ParseError;
                }
                else if (number < 2)
                {
                    printf("Not enough OpenCL devices of type %s available for "
                           "multi-device. Using only one device.",
                           value);
                }

                // Register triplets {platform, device, device_type} for each
                // device in each platform.
                triplets = malloc(
                    number * sizeof(struct cl_sdk_options_DeviceTriplet));
                number = 0;
                for (cl_uint platform_id = 0; platform_id < num_platforms;
                     ++platform_id)
                {
                    cl_uint num_devices;
                    clGetDeviceIDs(platforms[platform_id], dev_type, 0, NULL,
                                   &num_devices);

                    // Register triplets.
                    for (cl_uint device_id = 0; device_id < num_devices;
                         ++device_id, ++number)
                    {
                        struct cl_sdk_options_DeviceTriplet triplet = {
                            platform_id, device_id, dev_type
                        };
                        triplets[number] = triplet;
                    }
                }

                opts->triplets = triplets;
                opts->number = number;

                return ParsedOK;
            }
            else
            {
                return ParseError;
            }
    }
    return NotParsed;
}

cl_int parse_options(int argc, char* argv[],
                     struct cl_sdk_options_Diagnostic* diag_opts,
                     struct cl_sdk_options_MultiDevice* devs_opts,
                     struct convolution_options* convolution_opts)
{
    cl_int error = CL_SUCCESS;
    struct cag_option *opts = NULL, *tmp = NULL;
    size_t n = 0;

    // Prepare options array.
    MEM_CHECK(opts = add_CLI_options(opts, &n, DiagnosticOptions,
                                     CAG_ARRAY_SIZE(DiagnosticOptions)),
              error, end);
    MEM_CHECK(tmp = add_CLI_options(opts, &n, MultiDeviceOptions,
                                    CAG_ARRAY_SIZE(MultiDeviceOptions)),
              error, end);
    opts = tmp;
    MEM_CHECK(tmp = add_CLI_options(opts, &n, ConvolutionOptions,
                                    CAG_ARRAY_SIZE(ConvolutionOptions)),
              error, end);
    opts = tmp;

    char identifier;
    cag_option_context cag_context;

    // Prepare the context and iterate over all options.
    cag_option_prepare(&cag_context, opts, n, argc, argv);
    while (cag_option_fetch(&cag_context))
    {
        ParseState state = NotParsed;
        identifier = cag_option_get(&cag_context);

        PARS_OPTIONS(parse_DiagnosticOptions(identifier, diag_opts), state);
        PARS_OPTIONS(
            parse_MultiDeviceOptions(identifier, &cag_context, devs_opts),
            state);
        PARS_OPTIONS(parse_ConvolutionOptions(identifier, &cag_context,
                                              convolution_opts),
                     state);

        if (identifier == 'h')
        {
            printf("Usage: multidevice [OPTION]...\n");
            printf("Option name and value should be separated by '=' or a "
                   "space\n");
            printf(
                "Demonstrates convolution calculation with two devices.\n\n");
            cag_option_print(opts, n, stdout);
            exit((state == ParseError) ? CL_INVALID_ARG_VALUE : CL_SUCCESS);
        }
    }

end:
    free(opts);
    return error;
}

// Host-side implementation of the convolution for verification. Padded input
// assumed.
void host_convolution(cl_float* in, cl_float* out, cl_float* mask, size_t x_dim,
                      size_t y_dim)
{
    const cl_uint mask_dim = 3;
    const cl_uint pad_width = mask_dim / 2;
    const size_t pad_x_dim = x_dim + 2 * pad_width;
    for (size_t x = 0; x < x_dim; ++x)
    {
        for (size_t y = 0; y < y_dim; ++y)
        {
            float result = 0.f;
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
    cl_int error = CL_SUCCESS;
    cl_int end_error = CL_SUCCESS;
    cl_platform_id platform1, platform2;
    cl_device_id dev1, dev2;
    cl_context context1, context2;
    cl_command_queue queue1, queue2 = NULL;

    cl_program program1, program2;

    // Parse command-line options.
    struct cl_sdk_options_Diagnostic diag_opts = { .quiet = false,
                                                   .verbose = false };

    // By default assume that there is only one device available.
    // dev_opts->number is set to 1 so that when calling to cl_util_get_device
    // for the second device there is no index out of range.
    struct cl_sdk_options_MultiDevice devs_opts = {
        .triplets =
            (struct cl_sdk_options_DeviceTriplet[]){
                { 0, 0, CL_DEVICE_TYPE_ALL }, { 0, 0, CL_DEVICE_TYPE_ALL } },
        .number = 1
    };
    struct convolution_options convolution_opts = { .x_dim = 4000,
                                                    .y_dim = 4000 };

    OCLERROR_RET(
        parse_options(argc, argv, &diag_opts, &devs_opts, &convolution_opts),
        error, end);

    // Create runtime objects based on user preference or default.
    OCLERROR_PAR(dev1 =
                     cl_util_get_device(devs_opts.triplets[0].plat_index,
                                        devs_opts.triplets[0].dev_index,
                                        devs_opts.triplets[0].dev_type, &error),
                 error, end);
    OCLERROR_PAR(dev2 = cl_util_get_device(
                     devs_opts.triplets[(devs_opts.number > 1)].plat_index,
                     devs_opts.triplets[(devs_opts.number > 1)].dev_index,
                     devs_opts.triplets[(devs_opts.number > 1)].dev_type,
                     &error),
                 error, end);
    OCLERROR_PAR(context1 = clCreateContext(NULL, 1, &dev1, NULL, NULL, &error),
                 error, end);
    OCLERROR_PAR(context2 = clCreateContext(NULL, 1, &dev2, NULL, NULL, &error),
                 error, end);
    OCLERROR_RET(clGetDeviceInfo(dev1, CL_DEVICE_PLATFORM,
                                 sizeof(cl_platform_id), &platform1, NULL),
                 error, cont);
    OCLERROR_RET(clGetDeviceInfo(dev2, CL_DEVICE_PLATFORM,
                                 sizeof(cl_platform_id), &platform2, NULL),
                 error, cont);

    if (!diag_opts.quiet)
    {
        cl_util_print_device_info(dev1);
        cl_util_print_device_info(dev2);
    }

    // Compile kernels.
    const char* kernel_location = "./convolution.cl";
    char *kernel = NULL, *tmp = NULL;
    size_t program_size = 0;
    OCLERROR_PAR(
        kernel = cl_util_read_text_file(kernel_location, &program_size, &error),
        error, cont);
    MEM_CHECK(tmp = (char*)realloc(kernel, program_size), error, ker);
    kernel = tmp;
    OCLERROR_PAR(program1 = clCreateProgramWithSource(
                     context1, 1, (const char**)&kernel, &program_size, &error),
                 error, ker);
    OCLERROR_PAR(program2 = clCreateProgramWithSource(
                     context2, 1, (const char**)&kernel, &program_size, &error),
                 error, ker);

    // If no -cl-std option is specified then the highest 1.x version
    // supported by each device is used to compile the program. Therefore,
    // it's only necessary to add the -cl-std option for 2.0 and 3.0 OpenCL
    // versions.
    char compiler_options[1023] = "";
#if CL_HPP_TARGET_OPENCL_VERSION >= 300
    strcat(compiler_options, "-cl-std=CL3.0 ");
#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
    strcat(compiler_options, "-cl-std=CL2.0 ");
#endif

    OCLERROR_RET(cl_util_build_program(program1, dev1, compiler_options), error,
                 prg);
    OCLERROR_RET(cl_util_build_program(program2, dev2, compiler_options), error,
                 prg);

    // Query maximum workgroup size (WGS) of kernel supported on each device
    // based on private mem (registers) constraints.
    size_t wgs1, wgs2;
    cl_kernel convolution1, convolution2;
    OCLERROR_PAR(convolution1 =
                     clCreateKernel(program1, "convolution_3x3", &error),
                 error, prg);
    OCLERROR_PAR(convolution2 =
                     clCreateKernel(program2, "convolution_3x3", &error),
                 error, prg);
    OCLERROR_RET(clGetKernelWorkGroupInfo(convolution1, dev1,
                                          CL_KERNEL_WORK_GROUP_SIZE,
                                          sizeof(size_t), &wgs1, NULL),
                 error, conv);
    OCLERROR_RET(clGetKernelWorkGroupInfo(convolution2, dev2,
                                          CL_KERNEL_WORK_GROUP_SIZE,
                                          sizeof(size_t), &wgs2, NULL),
                 error, conv);

    // Initialize host-side storage.
    const cl_uint mask_dim = 3;
    const cl_uint pad_width = mask_dim / 2;
    const size_t x_dim = convolution_opts.x_dim;
    const size_t y_dim = convolution_opts.y_dim;
    const size_t pad_x_dim = x_dim + 2 * pad_width;
    const size_t pad_y_dim = y_dim + 2 * pad_width;

    // Check that the WGSs can divide the global size (MacOS reports
    // CL_INVALID_WORK_GROUP_SIZE otherwise). If WGS is smaller than the x
    // dimension, then a NULL pointer will be used when calling
    // clEnqueueNDRangeKernel for enqueuing the kernels.
    if (pad_x_dim % wgs1 && pad_x_dim > wgs1)
    {
        size_t div = pad_x_dim / wgs1;
        wgs1 = sqrt(div * wgs1);
    }

    if (pad_x_dim % wgs2 && pad_x_dim > wgs2)
    {
        size_t div = pad_x_dim / wgs2;
        wgs2 = sqrt(div * wgs2);
    }

    // Random number generator.
    pcg32_random_t rng;
    pcg32_srandom_r(&rng, 11111, -2222);

    // Initialize input matrix. The input will be padded to remove
    // conditional branches from the convolution kernel for determining
    // out-of-bounds.
    cl_float* h_input_grid;
    MEM_CHECK(h_input_grid =
                  (cl_float*)malloc(sizeof(cl_float) * pad_x_dim * pad_y_dim),
              error, conv);
    if (diag_opts.verbose)
    {
        printf("Generating %zu random numbers for convolution input grid.\n",
               x_dim * y_dim);
    }
    cl_sdk_fill_with_random_ints_range(&rng, (cl_int*)h_input_grid,
                                       pad_x_dim * pad_y_dim, -1000, 1000);

    // Fill with 0s the extra rows and columns added for padding.
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
    cl_float* h_output_grid;
    MEM_CHECK(h_output_grid =
                  (cl_float*)malloc(sizeof(cl_float) * x_dim * y_dim),
              error, hinput);

    // Initialize convolution mask.
    cl_float* h_mask;
    MEM_CHECK(h_mask =
                  (cl_float*)malloc(sizeof(cl_float) * mask_dim * mask_dim),
              error, hinput);
    if (diag_opts.verbose)
    {
        printf("Generating %u random numbers for convolution mask.\n",
               mask_dim * mask_dim);
    }
    cl_sdk_fill_with_random_ints_range(&rng, (cl_int*)h_mask,
                                       mask_dim * mask_dim, -1000, 1000);

    /// Initialize device-side storage.
    const size_t grid_midpoint = y_dim / 2;
    const size_t pad_grid_midpoint = pad_y_dim / 2;

    if (diag_opts.verbose)
    {
        printf("Initializing device-side storage...");
    }

    // Initialize queues for command execution on each device.
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
    cl_command_queue_properties props[] = { CL_QUEUE_PROPERTIES,
                                            CL_QUEUE_PROFILING_ENABLE, 0 };
    OCLERROR_PAR(queue1 = clCreateCommandQueueWithProperties(context1, dev1,
                                                             props, &error),
                 error, hinput);
    OCLERROR_PAR(queue2 = clCreateCommandQueueWithProperties(context2, dev2,
                                                             props, &error),
                 error, que1);
#else
    OCLERROR_PAR(queue1 = clCreateCommandQueue(
                     context1, dev1, CL_QUEUE_PROFILING_ENABLE, &error),
                 error, hinput);
    OCLERROR_PAR(queue2 = clCreateCommandQueue(
                     context2, dev2, CL_QUEUE_PROFILING_ENABLE, &error),
                 error, que1);
#endif

    // First device performs the convolution in the upper half and second device
    // in the lower half.
    cl_mem dev1_input_grid;
    OCLERROR_PAR(dev1_input_grid = clCreateBuffer(
                     context1, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_float) * pad_x_dim * (pad_grid_midpoint + 1),
                     h_input_grid, &error),
                 error, que2);
    // Second device performs the convolution in the lower half (middle
    // border included).
    cl_mem dev2_input_grid;
    OCLERROR_PAR(dev2_input_grid = clCreateBuffer(
                     context2, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_float) * pad_x_dim * (pad_grid_midpoint + 1),
                     h_input_grid + pad_x_dim * (pad_grid_midpoint - 1),
                     &error),
                 error, bufin1);

    cl_mem dev1_output_grid;
    OCLERROR_PAR(dev1_output_grid = clCreateBuffer(
                     context1, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                     sizeof(cl_float) * x_dim * y_dim, NULL, &error),
                 error, bufin2);
    cl_mem dev2_output_grid;
    OCLERROR_PAR(dev2_output_grid = clCreateBuffer(
                     context2, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                     sizeof(cl_float) * x_dim * y_dim, NULL, &error),
                 error, bufout1);

    cl_mem dev1_mask;
    OCLERROR_PAR(dev1_mask = clCreateBuffer(
                     context1, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_float) * mask_dim * mask_dim, h_mask, &error),
                 error, bufout2);
    cl_mem dev2_mask;
    OCLERROR_PAR(dev2_mask = clCreateBuffer(
                     context2, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_float) * mask_dim * mask_dim, h_mask, &error),
                 error, bufmask1);

    // Set kernels arguments.
    OCLERROR_RET(
        clSetKernelArg(convolution1, 0, sizeof(cl_mem), &dev1_input_grid),
        error, bufmask2);
    OCLERROR_RET(
        clSetKernelArg(convolution1, 1, sizeof(cl_mem), &dev1_output_grid),
        error, bufmask2);
    OCLERROR_RET(clSetKernelArg(convolution1, 2, sizeof(cl_mem), &dev1_mask),
                 error, bufmask2);
    OCLERROR_RET(clSetKernelArg(convolution1, 3, sizeof(size_t), &x_dim), error,
                 bufmask2);
    OCLERROR_RET(
        clSetKernelArg(convolution1, 4, sizeof(size_t), &grid_midpoint), error,
        bufmask2);

    OCLERROR_RET(
        clSetKernelArg(convolution2, 0, sizeof(cl_mem), &dev2_input_grid),
        error, bufmask2);
    OCLERROR_RET(
        clSetKernelArg(convolution2, 1, sizeof(cl_mem), &dev2_output_grid),
        error, bufmask2);
    OCLERROR_RET(clSetKernelArg(convolution2, 2, sizeof(cl_mem), &dev2_mask),
                 error, bufmask2);
    OCLERROR_RET(clSetKernelArg(convolution2, 3, sizeof(size_t), &x_dim), error,
                 bufmask2);
    OCLERROR_RET(
        clSetKernelArg(convolution2, 4, sizeof(size_t), &grid_midpoint), error,
        bufmask2);

    // Launch kernels.
    if (diag_opts.verbose)
    {
        printf("done.\nExecuting on device... ");
        fflush(stdout);
    }

    // Initialize global and local buffers for device execution.
    const size_t* global = (size_t[]){ pad_x_dim, pad_y_dim };
    const size_t* local1 = (size_t[]){ wgs1, 1 };
    const size_t* local2 = (size_t[]){ wgs2, 1 };

    // Enqueue kernel calls and wait for them to finish.
    cl_event *dev1_kernel_run = NULL, *dev2_kernel_run = NULL;
    MEM_CHECK(dev1_kernel_run = (cl_event*)malloc(sizeof(cl_event)), error,
              bufmask2);
    MEM_CHECK(dev2_kernel_run = (cl_event*)malloc(sizeof(cl_event)), error,
              run1);
    GET_CURRENT_TIMER(dev_start)
    OCLERROR_RET(clEnqueueNDRangeKernel(queue1, convolution1, 2, NULL, global,
                                        (pad_x_dim < wgs1) ? NULL : local1, 0,
                                        NULL, dev1_kernel_run),
                 error, run2);
    OCLERROR_RET(clEnqueueNDRangeKernel(queue2, convolution2, 2, NULL, global,
                                        (pad_x_dim < wgs2) ? NULL : local2, 0,
                                        NULL, dev2_kernel_run),
                 error, run2);

    OCLERROR_RET(clWaitForEvents(1, dev1_kernel_run), error, run2);
    OCLERROR_RET(clWaitForEvents(1, dev2_kernel_run), error, run2);
    GET_CURRENT_TIMER(dev_end)
    cl_ulong dev_time;
    TIMER_DIFFERENCE(dev_time, dev_start, dev_end)

    // Compute reference host-side convolution.
    if (diag_opts.verbose)
    {
        printf("done.\nExecuting on host... ");
    }

    GET_CURRENT_TIMER(host_start)
    host_convolution(h_input_grid, h_output_grid, h_mask, x_dim, y_dim);
    GET_CURRENT_TIMER(host_end)
    cl_ulong host_time;
    TIMER_DIFFERENCE(host_time, host_start, host_end)

    if (diag_opts.verbose)
    {
        printf("done.\n");
    }

    // Fetch and combine results from devices.
    cl_float* concatenated_results;
    MEM_CHECK(concatenated_results =
                  (cl_float*)malloc(sizeof(cl_float) * x_dim * y_dim),
              error, run2);
    OCLERROR_RET(clEnqueueReadBuffer(queue1, dev1_output_grid, CL_BLOCKING, 0,
                                     sizeof(cl_float) * x_dim * grid_midpoint,
                                     concatenated_results, 0, NULL, NULL),
                 error, run2);
    OCLERROR_RET(
        clEnqueueReadBuffer(queue2, dev2_output_grid, CL_BLOCKING, 0,
                            sizeof(cl_float) * x_dim * grid_midpoint,
                            concatenated_results + x_dim * grid_midpoint, 0,
                            NULL, NULL),
        error, run2);

    // Validate device-side solution.
    cl_float deviation = 0.f;
    const cl_float tolerance = 1e-6;

    for (size_t i = 0; i < x_dim * y_dim; ++i)
    {
        deviation += fabs(concatenated_results[i] - h_output_grid[i]);
    }
    deviation /= (x_dim * y_dim);

    if (deviation > tolerance)
    {
        printf("Failed convolution! Normalized deviation %.6f between host and "
               "device exceeds tolerance %.6f\n",
               deviation, tolerance);
    }
    else
    {
        printf("Successful convolution!\n");
    }

    if (!diag_opts.quiet)
    {
        printf("Kernels execution time as seen by host: %llu us.\n",
               (unsigned long long)(dev_time + 500) / 1000);

        printf("Kernels execution time as measured by devices :\n");
        printf("\t%llu us.\n",
               (unsigned long long)(cl_util_get_event_duration(
                                        *dev1_kernel_run,
                                        CL_PROFILING_COMMAND_START,
                                        CL_PROFILING_COMMAND_END, &error)
                                    + 500)
                   / 1000);
        printf("\t%llu us.\n",
               (unsigned long long)(cl_util_get_event_duration(
                                        *dev2_kernel_run,
                                        CL_PROFILING_COMMAND_START,
                                        CL_PROFILING_COMMAND_END, &error)
                                    + 500)
                   / 1000);

        printf("Reference execution as seen by host: %llu us.\n",
               (unsigned long long)(host_time + 500) / 1000);
    }

run2:
    free(dev2_kernel_run);
run1:
    free(dev1_kernel_run);
bufmask2:
    OCLERROR_RET(clReleaseMemObject(dev2_mask), end_error, bufmask1);
bufmask1:
    OCLERROR_RET(clReleaseMemObject(dev1_mask), end_error, bufout2);
bufout2:
    OCLERROR_RET(clReleaseMemObject(dev2_output_grid), end_error, bufout1);
bufout1:
    OCLERROR_RET(clReleaseMemObject(dev1_output_grid), end_error, bufin2);
bufin2:
    OCLERROR_RET(clReleaseMemObject(dev2_input_grid), end_error, bufin1);
bufin1:
    OCLERROR_RET(clReleaseMemObject(dev1_input_grid), end_error, hinput);
que2:
    OCLERROR_RET(clReleaseCommandQueue(queue2), end_error, que1);
que1:
    OCLERROR_RET(clReleaseCommandQueue(queue1), end_error, cont);
hinput:
    free(h_input_grid);
conv:
    OCLERROR_RET(clReleaseKernel(convolution1), end_error, prg);
    OCLERROR_RET(clReleaseKernel(convolution2), end_error, prg);
prg:
    OCLERROR_RET(clReleaseProgram(program1), end_error, ker);
    OCLERROR_RET(clReleaseProgram(program2), end_error, ker);
ker:
    free(kernel);
cont:
    OCLERROR_RET(clReleaseContext(context1), end_error, end);
    OCLERROR_RET(clReleaseContext(context2), end_error, end);
end:
    if (error) cl_util_print_error(error);
    return error;
}
