# Copyright 2018 ComputerGraphics Tuebingen. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader

_flex_convolution_op_so = loader.load_op_library(
    resource_loader.get_path_to_datafile("flex_conv_op.so"))

_flex_pooling_op_so = loader.load_op_library(
    resource_loader.get_path_to_datafile("flex_pool_op.so"))

_flex_deconvolution_op_so = loader.load_op_library(
    resource_loader.get_path_to_datafile("flex_deconv_op.so"))


# undocumented version
flex_conv = _flex_convolution_op_so.flex_conv
flex_conv_grad = _flex_convolution_op_so.flex_conv_grad
flex_pool = _flex_pooling_op_so.flex_pool
flex_pool_grad = _flex_pooling_op_so.flex_pool_grad
flex_deconv = _flex_deconvolution_op_so.flex_deconv
flex_deconv_grad = _flex_deconvolution_op_so.flex_deconv_grad

# pylint: disable=redefined-builtin


def flex_convolution(features,
                     position,
                     neighborhood,
                     theta,
                     bias,
                     name=None):
  """Flex-Convolution computation.

  Computes a convolution over arbitrary neighborhoods with elements of
  arbitrary positions:

    output(c', l) = sum_{c} sum_{l'}  w(c, l, l') * f(c, l')

  Args:
    features: A `Tensor` of the format [B, Din, N].
    position: A `Tensor` of the format [B, Dp, N].
    neighborhood: A `Tensor` of the format [B, K, N] (tf.int32).
    theta: A `Tensor` of the format [1, Dp, Din, Dout].
    bias: A `Tensor` of the format [Din, Dout].
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of the format [B, Dout, N].
  """

  with ops.name_scope(name, "flex_convolution"):
    return flex_conv(features, theta, bias, neighborhood, position)


@ops.RegisterGradient("FlexConv")
def _FlexConvGrad(op, *grads):  # noqa
  features = ops.convert_to_tensor(op.inputs[0])
  theta = ops.convert_to_tensor(op.inputs[1])
  bias = ops.convert_to_tensor(op.inputs[2])
  neighborhood = ops.convert_to_tensor(op.inputs[3], dtype=tf.int32)
  positions = ops.convert_to_tensor(op.inputs[4])
  topdiff = ops.convert_to_tensor(grads[0])

  df, dt, db = flex_conv_grad(
      features, theta, bias, neighborhood, positions, topdiff)

  df = ops.convert_to_tensor(df, name='gradient_features')
  dt = ops.convert_to_tensor(dt, name='gradient_theta')
  db = ops.convert_to_tensor(db, name='gradient_bias')

  return [df, dt, db, None, None]


# pylint: disable=redefined-builtin
def flex_pooling(features,
                 neighborhood,
                 name=None):
  """Flex-Pooling computation.

  Computes a pooling over arbitrary neighborhoods:

    output(n) = max_l'  f(l')

  Args:
    features: A `Tensor` of the format [B, D, N].
    neighborhood: A `Tensor` of the format [B, K, N] (tf.int32).
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of the format [B, D, N] containing the max values.
    A `Tensor` of the format [B, D, N] containing the max indicies.
  """

  with ops.name_scope(name, "flex_pooling"):
    return flex_pool(features, neighborhood)


@ops.RegisterGradient("FlexPool")
def _FlexPoolGrad(op, *grads):  # noqa
  features = ops.convert_to_tensor(op.inputs[0])
  neighborhood = ops.convert_to_tensor(op.inputs[1])
  argmax = ops.convert_to_tensor(op.outputs[1])
  topdiff = ops.convert_to_tensor(grads[0])

  df = flex_pool_grad(features, neighborhood, topdiff, argmax)
  df = ops.convert_to_tensor(df, name='gradient_features')

  return [df, None]


# pylint: disable=redefined-builtin
def flex_convolution_transpose(features,
                               position,
                               neighborhood,
                               theta,
                               bias,
                               name=None):
  """Flex-Convolution computation.

  Computes a tranposed convolution over arbitrary neighborhoods with elements of
  arbitrary positions.

  Args:
    features: A `Tensor` of the format [B, Din, N].
    position: A `Tensor` of the format [B, Dp, N].
    neighborhood: A `Tensor` of the format [B, K, N] (tf.int32).
    theta: A `Tensor` of the format [1, Dp, Din, Dout].
    bias: A `Tensor` of the format [Din, Dout].
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of the format [B, Dout, N].
  """

  with ops.name_scope(name, "flex_convolution_transpose"):
    return flex_deconv(features, theta, bias, neighborhood, position)


@ops.RegisterGradient("FlexDeconv")
def _FlexDeconvGrad(op, *grads):  # noqa
  features = ops.convert_to_tensor(op.inputs[0])
  theta = ops.convert_to_tensor(op.inputs[1])
  bias = ops.convert_to_tensor(op.inputs[2])
  neighborhood = ops.convert_to_tensor(op.inputs[3], dtype=tf.int32)
  positions = ops.convert_to_tensor(op.inputs[4])
  topdiff = ops.convert_to_tensor(grads[0])

  df, dt, db = flex_deconv_grad(
      features, theta, bias, neighborhood, positions, topdiff)

  df = ops.convert_to_tensor(df, name='gradient_features')
  dt = ops.convert_to_tensor(dt, name='gradient_theta')
  db = ops.convert_to_tensor(db, name='gradient_bias')

  return [df, dt, db, None, None]
