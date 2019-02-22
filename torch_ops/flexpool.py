# Copyright 2019 ComputerGraphics Tuebingen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tensorflow op performing flex convolution operation."""


from torch.autograd import Function
import torch

import flexpool_cuda

torch.manual_seed(42)


class FlexPoolFunction(Function):
  @staticmethod
  def forward(ctx, features, neighborhood):
    outputs = flexpool_cuda.forward(features, neighborhood)
    output, argmax = outputs[:2]
    ctx.save_for_backward(output, argmax)

    return output
