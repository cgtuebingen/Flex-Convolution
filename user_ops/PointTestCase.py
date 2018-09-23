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

np.random.seed(42)
tf.set_random_seed(42)


class FakePointCloud(object):
    """docstring for FakePointCloud"""

    def __init__(self, B, N, K, Din, Dout, Dp, N2, scaling=1):
        super(FakePointCloud, self).__init__()
        assert K < N
        self.B = B
        self.N = N
        self.K = K
        self.Din = Din
        self.Dout = Dout
        self.Dp = Dp
        self.N2 = N2

    def expected_feature_shape(self):
        return [self.B, self.Din, self.N]

    def expected_output_shape(self):
        return [self.B, self.Dout, self.N]


def random_values(shape, human_readable=False):
    """Return random values within range [-10, 10] and precision 2
    """
    length = np.prod(shape)
    return np.arange(length).astype(np.float32).reshape(shape) / float(length)


def summary(numeric_grad, graph_grad, name, eps=0.001, max_outputs=20):
    a, b = numeric_grad.flatten(), graph_grad.flatten()
    print("summary: %s" % name)
    print("\ttheirs\t\tours\t\tabs-diff")
    for i in range(np.prod(numeric_grad.shape)):
        if np.abs(a[i] - b[i]) > eps and max_outputs > 0:
            print('%i\t%f\t%f\t%f' % (i, a[i], b[i], np.abs(a[i] - b[i])))
            max_outputs -= 1
    if max_outputs == 20:
        for i in range(max_outputs):
                print('%i\t%f\t%f\t%f' % (i, a[i], b[i], np.abs(a[i] - b[i])))
    # print( np.stack([numeric_grad, graph_grad], axis=-1)
    print("%s - abs-diff (sum): " % name, np.abs(
        graph_grad - numeric_grad).sum())
    print("%s - abs-diff (max): " % name, np.abs(
        graph_grad - numeric_grad).max())
    print("%s - abs-diff (mean): " % name, np.abs(
        graph_grad - numeric_grad).mean())


# TestPointCloud(B, N, K, Din, Dout, Dp, N2)
# TPC = FakePointCloud(2, 32, 16, 5, 6, 3, 16)
# TPC = FakePointCloud(2, 16, 8, 5, 6, 3, 8)
TPC = FakePointCloud(2, 32, 4, 2, 6, 3, 16)
# TPC = FakePointCloud(2, 64, 8, 1, 6, 3, 64)


class PointTestCase(tf.test.TestCase):

    def __init__(self, methodName="runTest", data=None):

        if data is None:
            data = TPC

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

        self.theta = random_values([1, data.Dp, data.Din, data.Dout])
        self.bias = random_values([data.Din, data.Dout])

        super(PointTestCase, self).__init__(methodName)

    def init_ops(self):
        # needs to be called in each method, otherwise graph is empty
        # probably tf.reset_graph between calls
        self.features_op = tf.convert_to_tensor(self.features)
        self.position_op = tf.convert_to_tensor(self.position)
        self.neighborhood_op = tf.convert_to_tensor(self.neighborhood)
        self.neighborhood_ds_op = tf.convert_to_tensor(self.neighborhood_ds)
        self.theta_op = tf.convert_to_tensor(self.theta)
        self.bias_op = tf.convert_to_tensor(self.bias)
