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
Run 3D-MNIST classification using Flex-Convolutions, without any fancy parts.

Example-output:

 python mnist_3d.py --gpu 0 -fusion pooling

 [@base.py:282] Epoch 56 (global_step 210000) finished, time:23 minutes 52 seconds.
 [@saver.py:77] Model saved to train_log/conv_position_pooling/model-210000.
 100%|##############################|625/625[00:55<00:00,11.28it/s]
 [@monitor.py:459] accuracy: 0.94699
 [@monitor.py:459] learning_rate: 0.0001
 [@monitor.py:459] train_error: 0.053013
 [@monitor.py:459] validation_accuracy: 0.9327
 [@monitor.py:459] validation_cross_entropy_loss: 0.209
 [@group.py:48] Callbacks took 55.513 sec in total. InferenceRunner: 55.4 seconds
 [@base.py:272] Start Epoch 57 ...


 python mnist_3d.py --gpu 0 -fusion conv

 [@base.py:282] Epoch 40 (global_step 150000) finished, time:34 minutes 49 seconds.
 [@saver.py:77] Model saved to train_log/conv_position_conv/model-150000.
 100%|##############################|625/625[00:57<00:00,10.89it/s]
 [@monitor.py:459] accuracy: 0.89019
 [@monitor.py:459] learning_rate: 0.0001
 [@monitor.py:459] train_error: 0.10981
 [@monitor.py:459] validation_accuracy: 0.8748
 [@monitor.py:459] validation_cross_entropy_loss: 0.38396

This implementation is based on Tensorpack
- http://tensorpack.com/
- https://github.com/tensorpack/tensorpack/

"""

import os
import argparse
import tensorflow as tf
import numpy as np
import cv2

from tensorpack import *
from layers import flex_convolution, flex_pooling, knn_bruteforce


enable_argscope_for_module(tf.layers)

TOTAL_BATCH_SIZE = 16
BATCH_SIZE = 16
SHAPE = 28
CHANNELS = 3

USE_POOLING = False

PC = {'num': 1024, 'dp': 3}


class Digit2Cloud(RNGDataFlow):
  """ A very basic 2D-MNIST to 3D-MNIST sampler on a regular grid.
  """
  def __init__(self, incoming_df, num=1024):
    super(Digit2Cloud, self).__init__()
    self.incoming_df = incoming_df
    self.num = num

  def reset_state(self):
    super(Digit2Cloud, self).reset_state()
    self.incoming_df.reset_state()

  def __len__(self):
    return self.incoming_df.__len__()

  def map(self, dp, num=1024):
    digit = dp[0]

    # detect edges
    def auto_canny(image, sigma=0.33):
      v = np.median(image)
      lower = int(max(0, (1.0 - sigma) * v))
      upper = int(min(255, (1.0 + sigma) * v))
      edged = cv2.Canny(image, lower, upper)
      return edged

    digit = np.tile(np.expand_dims(digit, axis=-1), [1, 1, 3])
    digit = cv2.resize(digit, (32, 32))

    img = (255 * digit).astype(np.uint8)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    canny = auto_canny(blurred)
    edge_x, edge_y = np.nonzero(canny > 0)

    xlen = np.max(edge_x) - np.min(edge_x)
    ylen = np.max(edge_y) - np.min(edge_y)

    face_x, face_y = np.nonzero(img[:, :, 0] > 0)
    points = []

    depth = max(ylen, xlen)
    padding = (32 - depth) / 2.

    def z_dim(x):
      return x * depth + padding

    # start sampling (just extrude digits)
    for i in range(1024):
      choice = self.rng.randint(2 + 4)
      if choice > 1:
        idx = self.rng.randint(len(edge_x))
        z = self.rng.rand()
        points.append([edge_x[idx], edge_y[idx], z_dim(z)])
      else:
        idx = self.rng.randint(len(face_x))
        z = self.rng.randint(2)
        points.append([face_x[idx], face_y[idx], z_dim(z)])
    return [np.array(points).T, dp[1]]

  def __iter__(self):
    for dp in self.incoming_df:
      dp = self.map(dp, self.num)
      yield dp


class Model(ModelDesc):
  def inputs(self):
    """Inputs are
    - pointcloud [batch, dim_position, num_points]
    - label [batch]
    """
    return [tf.placeholder(tf.float32, (None, PC['dp'], PC['num']), 'positions'),
            tf.placeholder(tf.int32, (None,), 'label')]

  def build_graph(self, positions, label):

    positions = positions / 16. - 1
    # initial features are the position them self
    features = positions
    neighbors = knn_bruteforce(positions, K=16)

    x = features

    def subsample(x):
      # probably too simplistic, just kick out 3 of 4 points randomly
      # see our paper IDISS approach in the paper for better sub-sampling
      n = x.shape.as_list()[-1]
      return x[:, :, :n // 4]

    # similar to traditional networks
    for stage in range(4):
      if stage > 0:
        x = flex_pooling(x, neighbors)
        x = subsample(x)
        positions = subsample(positions)
        neighbors = knn_bruteforce(positions, K=16)

      x = flex_convolution(x, positions, neighbors, 64 *
                           (stage + 1), activation=tf.nn.relu)
      x = flex_convolution(x, positions, neighbors, 64 *
                           (stage + 1), activation=tf.nn.relu)

    if USE_POOLING:
      # either do max-pooling of all remaining points...
      x = tf.expand_dims(x, axis=-1)
      x = tf.layers.max_pooling2d(x, [1, 16], [1, 16])
    else:
      # ... or do a flex-conv in (0, 0, 0) with all points as neighbors
      positions = tf.concat([positions, positions[:, :, :1] * 0], axis=-1)
      x = tf.concat([x, x[:, :, :1] * 0], axis=-1)
      K = positions.shape.as_list()[-1]
      neighbors = knn_bruteforce(positions, K=K)
      x = flex_convolution(x, positions, neighbors, 1024, activation=tf.nn.relu)
      x = x[:, :, -1:]

    # from now on just the code part we copied from the Tensorpack framework
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, 512, activation=tf.nn.relu, name='fc0')
    logits = tf.layers.dense(x, 10, activation=tf.identity, name='fc1')

    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=label)
    cost = tf.reduce_mean(cost, name='cross_entropy_loss')

    correct = tf.cast(tf.nn.in_top_k(logits, label, 1),
                      tf.float32, name='correct')
    accuracy = tf.reduce_mean(correct, name='accuracy')

    train_error = tf.reduce_mean(1 - correct, name='train_error')
    summary.add_moving_summary(train_error, accuracy)
    return cost

  def optimizer(self):
    # nothing fancy here, just stick with the defaults
    return tf.train.AdamOptimizer(1e-4)


def get_data():
  df_train = dataset.Mnist('train')
  df_train = Digit2Cloud(df_train, num=PC['num'])
  df_train = PrefetchDataZMQ(df_train, 2)
  df_train = BatchData(df_train, BATCH_SIZE)

  df_val = dataset.Mnist('test')
  df_val = Digit2Cloud(df_val, num=PC['num'])
  df_val = PrefetchDataZMQ(df_val, 2)
  df_val = BatchData(df_val, BATCH_SIZE)
  return df_train, df_val


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
  parser.add_argument('--load', help='load model')
  parser.add_argument('--fusion', help='run sampling', default='',
                      choices=['pooling', 'conv'])
  args = parser.parse_args()

  if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

  logger.set_logger_dir('train_log/fusion_%s' % (args.fusion))

  dataset_train, dataset_test = get_data()
  steps_per_epoch = len(dataset_train)

  USE_POOLING = (args.fusion == 'pooling')

  # get the config which contains everything necessary in a training
  config = TrainConfig(
      model=Model(),
      # The input source for training. FeedInput is slow, this is just for demo purpose.
      # In practice it's best to use QueueInput or others. See tutorials for details.
      data=FeedInput(dataset_train),
      callbacks=[
          ModelSaver(),       # save the model after every epoch
          InferenceRunner(    # run inference(for validation) after every epoch
              dataset_test,   # the DataFlow instance used for validation
              ScalarStats(['cross_entropy_loss', 'accuracy'])),
      ],
      extra_callbacks=[
          MovingAverageSummary(),
          ProgressBar(['accuracy', 'cross_entropy_loss']),
          MergeAllSummaries(),
          RunUpdateOps()
      ],
      steps_per_epoch=steps_per_epoch,
      max_epoch=100,
  )
  launch_train_with_config(config, SimpleTrainer())
