/* Copyright 2017 ComputerGraphics Tuebingen. All Rights Reserved.

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

#ifndef LIB_FLEX_POOL_KERNEL_GPU_IMPL_H_
#define LIB_FLEX_POOL_KERNEL_GPU_IMPL_H_

inline int up2(int len, int th) { return (len - 1) / th + 1; }

template <typename Dtype>
__global__ void forward(const int B, const int N, const int K, const int D,
                        const Dtype* features, const int* neighborhood,
                        Dtype* output, int* argmax, Dtype float_min_value) {
  // features: each feature description for each point [B, D, N].
  // neighborhood: all K nearest neighbors [B, K, N].
  // output: each feature description for each point [B, D, N].
  // argmax: global id in neighborhood who was winning the pooling [B, D, N].
  const int b = blockIdx.z;

  for (int d = blockIdx.y * blockDim.y + threadIdx.y; d < D;
       d += blockDim.y * gridDim.y) {
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < N;
         n += blockDim.x * gridDim.x) {
      Dtype best_value = float_min_value;
      int best_id = 0;

      const int current_flat = b * D * N + d * N + n;

      for (int k_ = 0; k_ < K; ++k_) {
        const int other_global_id = neighborhood[b * K * N + k_ * N + n];
        const Dtype v = features[b * D * N + d * N + other_global_id];

        if (best_value < v) {
          best_id = other_global_id;
          best_value = v;
        }
      }

      output[current_flat] = best_value;
      argmax[current_flat] = best_id;
    }
  }
}

template <typename Dtype>
__global__ void backward(const int B, const int N, const int K, const int D,

                         const Dtype* features, const int* neighborhood,
                         const Dtype* topdiff, const int* argmax,

                         Dtype* grad_features) {
  // features: each feature description for each point [B, D, N].
  // neighborhood: all K nearest neighbors [B, K, N].
  // gradients: topdiff[B, D, N].
  // argmax: argmax[B, D, N].
  // grad_features: gradient to each feature description for each point [B, D,
  // N].
  const int b = blockIdx.z;

  for (int d = blockIdx.y * blockDim.y + threadIdx.y; d < D;
       d += blockDim.y * gridDim.y) {
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < N;
         n += blockDim.x * gridDim.x) {
      const int top_id_flat = b * D * N + d * N + n;
      const int argmax_id = argmax[top_id_flat];
      const int bottom_id_flat = b * D * N + d * N + argmax_id;

      // TODO(patwie): scattered write, yeah :-(
      atomicAdd(&grad_features[bottom_id_flat], topdiff[top_id_flat]);
    }
  }
}

#endif  // LIB_FLEX_POOL_KERNEL_GPU_IMPL_H_
