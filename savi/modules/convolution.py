# Copyright 2022 Google LLC.
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

"""Convolutional module library."""

import functools
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from flax import linen as nn
import jax.numpy as jnp

Shape = Tuple[int]

DType = Any
Array = jnp.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]


class CNN(nn.Module):
  """Flexible CNN model with conv. and normalization layers."""

  features: Sequence[int]
  kernel_size: Sequence[Tuple[int, int]]
  strides: Sequence[Tuple[int, int]]
  layer_transpose: Sequence[bool]
  activation_fn: Callable[[Array], Array] = nn.relu
  norm_type: Optional[str] = None
  axis_name: Optional[str] = None  # Over which axis to aggregate batch stats.
  output_size: Optional[int] = None

  @nn.compact
  def __call__(
      self, inputs: Array, train: bool = False) -> Tuple[Dict[str, Array]]:
    num_layers = len(self.features)

    assert num_layers >= 1, "Need to have at least one layer."
    assert len(self.kernel_size) == num_layers, (
        "len(kernel_size) and len(features) must match.")
    assert len(self.strides) == num_layers, (
        "len(strides) and len(features) must match.")
    assert len(self.layer_transpose) == num_layers, (
        "len(layer_transpose) and len(features) must match.")

    if self.norm_type:
      assert self.norm_type in {"batch", "group", "instance", "layer"}, (
          f"{self.norm_type} is not a valid normalization module.")

    # Whether transpose conv or regular conv.
    conv_module = {False: nn.Conv, True: nn.ConvTranspose}

    if self.norm_type == "batch":
      norm_module = functools.partial(
          nn.BatchNorm, momentum=0.9, use_running_average=not train,
          axis_name=self.axis_name)
    elif self.norm_type == "group":
      norm_module = functools.partial(
          nn.GroupNorm, num_groups=32)
    elif self.norm_type == "layer":
      norm_module = nn.LayerNorm

    x = inputs
    for i in range(num_layers):
      x = conv_module[self.layer_transpose[i]](
          name=f"conv_{i}",
          features=self.features[i],
          kernel_size=self.kernel_size[i],
          strides=self.strides[i],
          use_bias=False if self.norm_type else True)(x)

      # Normalization layer.
      if self.norm_type:
        if self.norm_type == "instance":
          x = nn.GroupNorm(
              num_groups=self.features[i],
              name=f"{self.norm_type}_norm_{i}")(x)
        else:
          norm_module(name=f"{self.norm_type}_norm_{i}")(x)

      # Activation layer.
      x = self.activation_fn(x)

    # Final dense layer.
    if self.output_size:
      x = nn.Dense(self.output_size, name="output_layer", use_bias=True)(x)
    return x
