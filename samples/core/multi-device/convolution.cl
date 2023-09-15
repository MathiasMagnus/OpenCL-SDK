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

kernel void convolution_3x3(const global float* in, global float* out,
                            const global float* mask, const unsigned long x_dim,
                            const unsigned long y_dim)
{
    const uint mask_dim = 3;
    const uint pad_width = mask_dim / 2;

    const size_t x = get_global_id(0);
    const size_t y = get_global_id(1);

    // Padded constants.
    const size_t pad_x_dim = x_dim + 2 * pad_width;
    const size_t pad_y_dim = y_dim + 2 * pad_width;

    // Check possible out of bounds.
    if (!(x < x_dim && y < y_dim))
    {
        return;
    }

    // Perform convolution. Fix one column at a time and iterate over each
    // element of it, as data is stored column-major.
    float result = 0.0f;
    #if __OPENCL_C_VERSION__ >= 200
    __attribute__((opencl_unroll_hint))
    #endif
    for (size_t grid_column = x, mask_column = 0; mask_column < mask_dim;
         ++grid_column, ++mask_column)
    {
        #if __OPENCL_C_VERSION__ >= 200
        __attribute__((opencl_unroll_hint))
        #endif
        for (size_t grid_row = y, mask_row = 0; mask_row < mask_dim;
             ++grid_row, ++mask_row)
        {
            result += mask[mask_column + mask_row * mask_dim]
                * in[grid_column + grid_row * pad_x_dim];
        }
    }

    // Write result to correspoding output cell.
    out[x + y * x_dim] = result;
}
