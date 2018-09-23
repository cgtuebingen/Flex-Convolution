#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 ComputerGraphics Tuebingen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Authors: Fabian Groh, Patrick Wieschollek, Hendrik P.A. Lensch


"""
Demonstration of using FlexConvolution Layer.
"""

import numpy as np
import tensorflow as tf
from layers import flex_convolution

B, Din, Dout, Dp, N, K = 1, 2, 4, 3, 10, 5

features = np.random.randn(B, Din, N).astype(np.float32)
positions = np.random.randn(B, Dp, N).astype(np.float32)
neighbors = np.random.randint(0, N, [B, K, N]).astype(np.int32)

features = tf.convert_to_tensor(features)
positions = tf.convert_to_tensor(positions)
neighbors = tf.convert_to_tensor(neighbors)

features2 = flex_convolution(features, positions, neighbors, Dout)
features3 = flex_convolution(features2, positions, neighbors, Dout, trainable=False)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(features2)
  sess.run(features3)

  print(tf.trainable_variables())
