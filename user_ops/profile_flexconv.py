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

from PointTestCase import FakePointCloud, random_values
from __init__ import flex_conv

"""
export LD_LIBRARY_PATH=/graphics/opt/opt_Ubuntu16.04/cuda/toolkit_9.0/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
"""


np.random.seed(42)
tf.set_random_seed(42)

N = 8
TPC = FakePointCloud(8, 4096, N, 64, 64, 2, 1, N)


class PointTestCase(object):

    def __init__(self, data):
        self.position = random_values([data.B, data.Dp, data.N])
        self.features = random_values([data.B, data.Din, data.N])

        # make sure, each neighbor hood has no duplicates and first entry is point n
        # THIS IS IMPORTANT!!
        self.neighborhood = np.zeros((data.B, data.K, data.N), dtype=np.int32)
        for b in range(data.B):
            for n in range(data.N):
                x = np.arange(data.N)
                # does not support axis, hence the loop
                np.random.shuffle(x)
                offset = np.argwhere(x == n)[0][0]
                # roll array such that n is first entry
                x = np.roll(x, -offset)
                self.neighborhood[b, :, n] = x[:data.K].astype(np.int32)

        self.neighborhood_ds = np.zeros((data.B, data.K, data.N2), dtype=np.int32)
        for b in range(data.B):
            for n in range(data.N2):
                x = np.arange(data.N2)
                # does not support axis, hence the loop
                np.random.shuffle(x)
                offset = np.argwhere(x == n)[0][0]
                # roll array such that n is first entry
                x = np.roll(x, -offset)
                self.neighborhood[b, :, n] = x[:data.K].astype(np.int32)

        self.theta = random_values([data.Degree, data.Dp, data.Din, data.Dout])
        self.bias = random_values([data.Din, data.Dout])

    def init_ops(self):
        self.features_op = tf.Variable(self.features, name='f')
        self.position_op = tf.Variable(self.position, name='p')
        self.neighborhood_op = tf.Variable(self.neighborhood, name='n')
        self.neighborhood_ds_op = tf.Variable(self.neighborhood_ds, name='m')
        self.theta_op = tf.Variable(self.theta, name='t')
        self.bias_op = tf.Variable(self.bias, name='b')


TestCase = PointTestCase(data=TPC)
TestCase.init_ops()


forward_op = flex_conv(TestCase.features_op,
                       TestCase.theta_op, TestCase.bias_op, TestCase.neighborhood_op,
                       TestCase.position_op, degree=1)

builder = tf.profiler.ProfileOptionBuilder
opts = builder(builder.time_and_memory()).order_by('micros').build()

with tf.contrib.tfprof.ProfileContext('./.profiling_outputs/flex_conv') as pctx:

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        back_prop = tf.gradients(forward_op, [TestCase.theta_op, TestCase.bias_op, TestCase.features_op])
        back_prop2 = tf.gradients(forward_op, [TestCase.theta_op, TestCase.bias_op,
                                               TestCase.features_op, TestCase.position_op])

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
