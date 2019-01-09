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
Demonstration of using FlexConvolution, FlexPooling Layer.
"""

import numpy as np
import tensorflow as tf
from tabulate import tabulate
from layers import (flex_convolution,
                    flex_convolution_transpose,
                    flex_pooling,
                    knn_bruteforce)


B, Din, Dout, Dout2, Dp, N, K = 1, 2, 4, 8, 3, 10, 5

features = np.random.randn(B, Din, N).astype(np.float32)
positions = np.random.randn(B, Dp, N).astype(np.float32)

features = tf.convert_to_tensor(features, name='features')
positions = tf.convert_to_tensor(positions, name='positions')

net = [features]
# use our FlexConv similar to a traditional convolution layer

neighbors = knn_bruteforce(positions, k=5)
net.append(flex_convolution(net[-1],
                            positions,
                            neighbors,
                            Dout,
                            activation=tf.nn.relu))
# pool and sub-sampling are different operations
net.append(flex_pooling(net[-1], neighbors))

# when ordering the points beforehand sub-sampling is simply
features = net[-1][:, :, :N // 2]
positions = positions[:, :, :N // 2]
net.append(features)

neighbors = knn_bruteforce(positions, k=3)
# we didn't notice any improvements using the transposed version vs. pooling
net.append(flex_convolution_transpose(net[-1],
                                      positions,
                                      neighbors,
                                      Dout,
                                      activation=tf.nn.relu))
# of course any commonly used arguments work here as well
net.append(flex_convolution(net[-1],
                            positions,
                            neighbors,
                            Dout,
                            activation=tf.nn.relu,
                            trainable=False))

gradient_wrt_feature = tf.gradients(net[-1], net[0])

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  ouputs = sess.run(net[-1])
  grads = sess.run(gradient_wrt_feature)

  assert not np.isnan(ouputs).any()
  assert not np.isnan(grads).any()

  print(tabulate([[v.name, v.shape] for v in tf.trainable_variables()],
                 headers=["Name", "Shape"]))

  print(tabulate([[n.name, n.shape] for n in net], headers=["Name", "Shape"]))
