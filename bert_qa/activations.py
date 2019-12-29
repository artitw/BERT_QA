# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
# Copyright 2019 The BERT-QA Authors. All Rights Reserved.
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
"""Gaussian error linear unit."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf



def gelu(x):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


"""Customized Swish activation."""


def simple_swish(features):
  """Computes the Swish activation function.
  The tf.nn.swish operation uses a custom gradient to reduce memory usage.
  Since saving custom gradients in SavedModel is currently not supported, and
  one would not be able to use an exported TF-Hub module for fine-tuning, we
  provide this wrapper that can allow to select whether to use the native
  TensorFlow swish operation, or whether to use a customized operation that
  has uses default TensorFlow gradient computation.
  Args:
    features: A `Tensor` representing preactivation values.
  Returns:
    The activation value.
  """
  features = tf.convert_to_tensor(features)
  return features * tf.nn.sigmoid(features)



def hard_swish(features):
  """Computes a hard version of the swish function.
  This operation can be used to reduce computational cost and improve
  quantization for edge devices.
  Args:
    features: A `Tensor` representing preactivation values.
  Returns:
    The activation value.
  """
  features = tf.convert_to_tensor(features)
  return features * tf.nn.relu6(features + tf.constant(3.)) * (1. / 6.)



def identity(features):
  """Computes the identity function.
  Useful for helping in quantization.
  Args:
    features: A `Tensor` representing preactivation values.
  Returns:
    The activation value.
  """
  features = tf.convert_to_tensor(features)
  return tf.identity(features)