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


from PointTestCase import PointTestCase
import tensorflow as tf
import numpy as np

from __init__ import flex_pooling


class FlexPoolTest(PointTestCase):
    def __init__(self, methodName="runTest"):
        super(FlexPoolTest, self).__init__(methodName)

    def _forward(self, use_gpu=False):
        self.init_ops()
        with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu) as sess:
            actual_op, winner_op = flex_pooling(self.features_op, self.neighborhood_op)
            actual = sess.run(actual_op)
        return actual

    def test_forward(self):
        cpu = self._forward(use_gpu=False)
        gpu = self._forward(use_gpu=True)
        self.assertAllClose(cpu, gpu)

    def test_backward(self):
        cpu, winner_cpu = self._backward(use_gpu=False)
        gpu, winner_gpu = self._backward(use_gpu=True)
        self.assertAllClose(cpu, gpu)
        self.assertAllClose(cpu[winner_cpu == 0].sum(), 0)
        self.assertAllClose(gpu[winner_gpu == 0].sum(), 0)

    def _backward(self, use_gpu=False):
        self.init_ops()
        with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu) as sess:
            actual_op, winner_op = flex_pooling(
                self.features_op,
                self.neighborhood_op)

            graph_features_grad = tf.gradients(actual_op, [self.features_op])[0]

            dx, winner = sess.run([graph_features_grad, winner_op])
            return dx, winner

    def _simple_backward(self, use_gpu=False):
        # BN
        x = np.array([[[1], [2], [5], [3]]]).transpose(0, 2, 1)
        n = np.array([[[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2, ]]]).transpose(0, 2, 1)

        x = tf.convert_to_tensor(x.astype(np.float32))
        n = tf.convert_to_tensor(n.astype(np.int32))

        with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu) as sess:
            actual_op, winner_op = flex_pooling(x, n)
            graph_features_grad = tf.gradients(actual_op, [x])[0]
            return sess.run(graph_features_grad)

    def test_backward_simple(self):
        cpu = self._simple_backward(use_gpu=False)
        cpu[0, 0, 2] -= 4
        self.assertEqual(cpu.sum(), 0)

        gpu = self._simple_backward(use_gpu=True)
        gpu[0, 0, 2] -= 4
        self.assertEqual(gpu.sum(), 0)


if __name__ == '__main__':
    tf.test.main()
