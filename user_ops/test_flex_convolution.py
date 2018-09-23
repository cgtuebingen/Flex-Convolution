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


from PointTestCase import TPC, PointTestCase, summary
import tensorflow as tf

from __init__ import flex_convolution


class FlexConvTest(PointTestCase):
  def __init__(self, methodName="runTest"):
    super(FlexConvTest, self).__init__(methodName)

  def _forward(self, use_gpu=False, force_gpu=False):
    self.init_ops()
    with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu) as sess:
      actual_op = flex_convolution(self.features_op,
                                   self.position_op, self.neighborhood_op,
                                   self.theta_op, self.bias_op)
      actual = sess.run(actual_op)
    return actual

  def test_forward(self):
    cpu = self._forward(use_gpu=False)
    gpu = self._forward(use_gpu=True)
    self.assertAllClose(cpu, gpu, 1e-5, 1e-5)

  def _backward_features(self, use_gpu=False):
    self.init_ops()
    with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu):
      actual_op = flex_convolution(self.features_op,
                                   self.position_op, self.neighborhood_op,
                                   self.theta_op, self.bias_op)
      graph_features_grad, num_features_grad = tf.test.compute_gradient(
          [self.features_op], [self.features.shape], actual_op,
          TPC.expected_output_shape())[0]
      summary(num_features_grad, graph_features_grad, 'self.features')

      err = tf.test.compute_gradient_error([self.features_op],
                                           [self.features.shape],
                                           actual_op, TPC.expected_output_shape())
      self.assertLess(err, 1e-2)

  def _backward_bias(self, use_gpu=False):
    self.init_ops()
    with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu):
      actual_op = flex_convolution(self.features_op,
                                   self.position_op, self.neighborhood_op,
                                   self.theta_op, self.bias_op)

      graph_bias_grad, num_bias_grad = tf.test.compute_gradient(
          [self.bias_op], [self.bias.shape], actual_op,
          TPC.expected_output_shape())[0]
      summary(num_bias_grad, graph_bias_grad, 'self.bias')

      err = tf.test.compute_gradient_error([self.bias_op],
                                           [self.bias.shape], actual_op,
                                           TPC.expected_output_shape())
      self.assertLess(err, 1e-2)

  def _backward_theta(self, use_gpu=False):
    self.init_ops()
    with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu):
      actual_op = flex_convolution(self.features_op,
                                   self.position_op, self.neighborhood_op,
                                   self.theta_op, self.bias_op)

      graph_theta_grad, num_theta_grad = tf.test.compute_gradient(
          [self.theta_op], [self.theta.shape], actual_op,
          TPC.expected_output_shape())[0]
      summary(num_theta_grad, graph_theta_grad, 'self.theta')

      err = tf.test.compute_gradient_error([self.theta_op],
                                           [self.theta.shape], actual_op,
                                           TPC.expected_output_shape())
      self.assertLess(err, 1e-2)

  def test_backward_features(self):
    self._backward_features(use_gpu=False)
    self._backward_features(use_gpu=True)

  def test_backward_bias(self):
    self._backward_bias(use_gpu=False)
    self._backward_bias(use_gpu=True)

  def test_backward_theta(self):
    self._backward_theta(use_gpu=False)
    self._backward_theta(use_gpu=True)


if __name__ == '__main__':
  tf.test.main()
