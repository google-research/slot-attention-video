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

"""Decoder module library."""
import functools
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from flax import linen as nn

import jax.numpy as jnp

from savi.lib import utils

Shape = Tuple[int]

DType = Any
Array = jnp.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]


class SpatialBroadcastDecoder(nn.Module):
  """Spatial broadcast decoder for a set of slots (per frame)."""

  resolution: Sequence[int]
  backbone: Callable[[], nn.Module]
  pos_emb: Callable[[], nn.Module]
  target_readout: Optional[Callable[[], nn.Module]] = None

  # Vmapped application of module, consumes time axis (axis=1).
  @functools.partial(utils.time_distributed, in_axes=(1, None))
  @nn.compact
  def __call__(self, slots: Array, train: bool = False) -> Array:

    batch_size, n_slots, n_features = slots.shape

    # Fold slot dim into batch dim.
    x = jnp.reshape(slots, (batch_size * n_slots, n_features))

    # Spatial broadcast with position embedding.
    x = utils.spatial_broadcast(x, self.resolution)
    x = self.pos_emb()(x)

    # bb_features.shape = (batch_size * n_slots, h, w, c)
    bb_features = self.backbone()(x, train=train)
    spatial_dims = bb_features.shape[-3:-1]

    alpha_logits = nn.Dense(
        features=1, use_bias=True, name="alpha_logits")(bb_features)
    alpha_logits = jnp.reshape(
        alpha_logits, (batch_size, n_slots) + spatial_dims + (-1,))

    alphas = nn.softmax(alpha_logits, axis=1)
    if not train:
      # Define intermediates for logging / visualization.
      self.sow("intermediates", "alphas", alphas)

    targets_dict = self.target_readout()(bb_features, train)

    preds_dict = dict()
    for target_key, channels in targets_dict.items():

      # channels.shape = (batch_size, n_slots, h, w, c)
      channels = jnp.reshape(
          channels, (batch_size, n_slots) + (spatial_dims) + (-1,))

      # masked_channels.shape = (batch_size, n_slots, h, w, c)
      masked_channels = channels * alphas

      # decoded_target.shape = (batch_size, h, w, c)
      decoded_target = jnp.sum(masked_channels, axis=1)  # Combine target.
      preds_dict[target_key] = decoded_target

      if not train:
      # Define intermediates for logging / visualization.
        self.sow("intermediates", f"{target_key}_slots", channels)
        self.sow("intermediates", f"{target_key}_masked", masked_channels)
        self.sow("intermediates", f"{target_key}_combined", decoded_target)

    preds_dict["segmentations"] = jnp.argmax(alpha_logits, axis=1)

    return preds_dict
