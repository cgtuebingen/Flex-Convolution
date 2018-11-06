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


import numpy as np
import tensorflow as tf

from misc import FakePointCloud
from __init__ import flex_convolution


"""
export LD_LIBRARY_PATH=/graphics/opt/opt_Ubuntu16.04/cuda/toolkit_9.0/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
"""


np.random.seed(42)
tf.set_random_seed(42)


case = FakePointCloud(B=8, N=4096, K=8, Din=64, Dout=64, Dp=3)
case.init_ops(dtype=np.float32)

forward_op = flex_convolution(case.features_op,
                              case.position_op,
                              case.neighborhood_op,
                              case.theta_op,
                              case.bias_op)

builder = tf.profiler.ProfileOptionBuilder
opts = builder(builder.time_and_memory()).order_by('micros').build()

with tf.contrib.tfprof.ProfileContext('./.profiling_outputs/flex_convolution') as pctx:

  with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    back_prop = tf.gradients(
        forward_op, [case.theta_op, case.bias_op, case.features_op])
    back_prop2 = tf.gradients(forward_op, [case.theta_op, case.bias_op,
                                           case.features_op, case.position_op])

    # warmup
    for i in range(2):
      actual = sess.run([forward_op, back_prop])

    # benchmark
    for i in range(10):
      pctx.trace_next_step()
      pctx.dump_next_step()
      _ = sess.run([forward_op])
      pctx.profiler.profile_operations(options=opts)

    for i in range(10):
      pctx.trace_next_step()
      pctx.dump_next_step()
      _ = sess.run([back_prop])
      pctx.profiler.profile_operations(options=opts)
