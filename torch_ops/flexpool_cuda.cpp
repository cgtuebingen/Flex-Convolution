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

#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> flexpool_cuda_forward(at::Tensor features,
                                              at::Tensor neighborhood);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> flexpool_forward(at::Tensor features,
                                         at::Tensor neighborhood) {
  CHECK_INPUT(features);
  CHECK_INPUT(neighborhood);

  return flexpool_cuda_forward(features, neighborhood);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &flexpool_forward, "FlexPool forward (CUDA)");
}

#undef CHECK_CUDA
#undef CHECK_CONTIGUOUS
#undef CHECK_INPUT
