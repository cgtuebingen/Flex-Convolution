/* Copyright 2019 ComputerGraphics Tuebingen. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Authors: Fabian Groh, Patrick Wieschollek, Hendrik P.A. Lensch

#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

#include "../user_ops/kernels/flex_pool_kernel_gpu_impl.cuh"

}  // namespace

std::vector<at::Tensor> flexpool_cuda_forward(at::Tensor features,
                                              at::Tensor neighborhood) {
  const int B = features.size(0);
  const int D = features.size(1);
  const int N = features.size(2);

  const int K = neighborhood.size(1);

  auto output = at::zeros({B, D, N}, features.type());
  auto argmax = at::zeros({B, D, N}, neighborhood.type());

  const int threads = 32;
  dim3 block(threads, threads, 1);
  dim3 grid(up2(N, threads), up2(D, threads), B);

  AT_DISPATCH_FLOATING_TYPES(features.type(), "flexpool_forward_cuda", ([&] {
                               forward<scalar_t><<<grid, block>>>(
                                   B, N, K, D, features.data<scalar_t>(),
                                   neighborhood.data<int>(),
                                   output.data<scalar_t>(), argmax.data<int>(),
                                   std::numeric_limits<scalar_t>::lowest());
                             }));

  return {output, argmax};
}

std::vector<at::Tensor> flexpool_cuda_backward(at::Tensor features,
                                               at::Tensor neighborhood,
                                               at::Tensor topdiff,
                                               at::Tensor argmax) {
  const int B = features.size(0);
  const int D = features.size(1);
  const int N = features.size(2);

  const int K = neighborhood.size(1);

  auto bottom_diff = at::zeros({B, D, N}, features.type());

  const int threads = 32;
  dim3 block(threads, threads, 1);
  dim3 grid(up2(N, threads), up2(D, threads), B);

  AT_DISPATCH_FLOATING_TYPES(features.type(), "flexpool_backward_cuda", ([&] {
                               backward<scalar_t><<<grid, block>>>(
                                   B, N, K, D, features.data<scalar_t>(),
                                   neighborhood.data<int>(),
                                   topdiff.data<scalar_t>(), argmax.data<int>(),
                                   bottom_diff.data<scalar_t>());
                             }));

  return {bottom_diff};
}
