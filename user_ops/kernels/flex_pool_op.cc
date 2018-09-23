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
//Authors: Fabian Groh, Patrick Wieschollek, Hendrik P.A. Lensch

#include "flex_pool_op.h"

#include <stdio.h>
#include <type_traits>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

// Forward-Pass (CPU, GPU)
// --------------------------------------------------
template <typename Device, typename Dtype>
class FlexPoolOp : public OpKernel {
 public:
  explicit FlexPoolOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // printf("--> Compute CPU Version <--\n");
    const Tensor& features_ = ctx->input(0);
    const Tensor& neighborhood_ = ctx->input(1);

    const int B = features_.dim_size(0);
    const int D = features_.dim_size(1);
    const int N = features_.dim_size(2);

    Tensor* output_ = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({B, D, N}), &output_));

    Tensor* argmax_ = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(1, TensorShape({B, D, N}), &argmax_));

    ::tensorflow::functor::FlexPoolFunctor<Device, Dtype>()(
        ctx, features_, neighborhood_, output_, argmax_);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(FlexPoolOp);
};

// Backward-Pass (CPU, GPU)
// --------------------------------------------------
template <typename Device, typename Dtype>
class FlexPoolGradOp : public OpKernel {
 public:
  explicit FlexPoolGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // printf("--> Compute CPU Version <--\n");
    const Tensor& features_ = ctx->input(0);
    const Tensor& neighborhood_ = ctx->input(1);
    const Tensor& topdiff_ = ctx->input(2);
    const Tensor& argmax_ = ctx->input(3);

    // specify output shape
    Tensor* grad_features_ = nullptr;

    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, features_.shape(), &grad_features_));

    ::tensorflow::functor::FlexPoolGrad<Device, Dtype>()(
        ctx, features_, neighborhood_, topdiff_, argmax_, grad_features_);
  }
};

// Register the CPU kernels.
#define REGISTER_FLEXPOOL_OP_CPU(T)                                   \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("FlexPool").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
      FlexPoolOp<CPUDevice, T>)                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("FlexPoolGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      FlexPoolGradOp<CPUDevice, T>)

TF_CALL_float(REGISTER_FLEXPOOL_OP_CPU);
#undef REGISTER_FLEXPOOL_OP_CPU

// Register the GPU kernels.
#ifdef GOOGLE_CUDA

#define REGISTER_FLEXPOOL_OP_GPU(T)                                   \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("FlexPool").Device(DEVICE_GPU).TypeConstraint<T>("T"),     \
      FlexPoolOp<GPUDevice, T>)                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("FlexPoolGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      FlexPoolGradOp<GPUDevice, T>)

TF_CALL_float(REGISTER_FLEXPOOL_OP_GPU);
#undef REGISTER_FLEXPOOL_OP_GPU

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
