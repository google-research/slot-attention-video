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

"""Input pipeline for TFDS datasets."""

import functools
from typing import Dict, List, Tuple

from clu import deterministic_data
from clu import preprocess_spec

import jax
import jax.numpy as jnp
import ml_collections

from savi.lib import preprocessing
import tensorflow as tf
import tensorflow_datasets as tfds

Array = jnp.ndarray
PRNGKey = Array


def preprocess_example(features: Dict[str, tf.Tensor],
                       preprocess_strs: List[str]) -> Dict[str, tf.Tensor]:
  """Processes a single data example.

  Args:
    features: A dictionary containing the tensors of a single data example.
    preprocess_strs: List of strings, describing one preprocessing operation
      each, in clu.preprocess_spec format.

  Returns:
    Dictionary containing the preprocessed tensors of a single data example.
  """
  all_ops = preprocessing.all_ops()
  preprocess_fn = preprocess_spec.parse("|".join(preprocess_strs), all_ops)
  return preprocess_fn(features)  # pytype: disable=bad-return-type  # allow-recursive-types


def get_batch_dims(global_batch_size: int) -> List[int]:
  """Gets the first two axis sizes for data batches.

  Args:
    global_batch_size: Integer, the global batch size (across all devices).

  Returns:
    List of batch dimensions

  Raises:
    ValueError if the requested dimensions don't make sense with the
      number of devices.
  """
  num_local_devices = jax.local_device_count()
  if global_batch_size % jax.host_count() != 0:
    raise ValueError(f"Global batch size {global_batch_size} not evenly "
                     f"divisble with {jax.host_count()}.")
  per_host_batch_size = global_batch_size // jax.host_count()
  if per_host_batch_size % num_local_devices != 0:
    raise ValueError(f"Global batch size {global_batch_size} not evenly "
                     f"divisible with {jax.host_count()} hosts with a per host "
                     f"batch size of {per_host_batch_size} and "
                     f"{num_local_devices} local devices. ")
  return [num_local_devices, per_host_batch_size // num_local_devices]


def create_datasets(
    config: ml_collections.ConfigDict,
    data_rng: PRNGKey) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Create datasets for training and evaluation.

  For the same data_rng and config this will return the same datasets. The
  datasets only contain stateless operations.

  Args:
    config: Configuration to use.
    data_rng: JAX PRNGKey for dataset pipeline.

  Returns:
    A tuple with the training dataset and the evaluation dataset.
  """
  dataset_builder = tfds.builder(
      config.data.tfds_name, data_dir=config.data.data_dir)
  batch_dims = get_batch_dims(config.batch_size)

  train_preprocess_fn = functools.partial(
      preprocess_example, preprocess_strs=config.preproc_train)
  eval_preprocess_fn = functools.partial(
      preprocess_example, preprocess_strs=config.preproc_eval)

  train_split_name = config.get("train_split", "train")
  eval_split_name = config.get("validation_split", "validation")

  train_split = deterministic_data.get_read_instruction_for_host(
      train_split_name, dataset_info=dataset_builder.info)
  train_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=train_split,
      rng=data_rng,
      preprocess_fn=train_preprocess_fn,
      cache=False,
      shuffle_buffer_size=config.data.shuffle_buffer_size,
      batch_dims=batch_dims,
      num_epochs=None,
      shuffle=True)

  eval_split = deterministic_data.get_read_instruction_for_host(
      eval_split_name, dataset_info=dataset_builder.info, drop_remainder=False)
  eval_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=eval_split,
      rng=None,
      preprocess_fn=eval_preprocess_fn,
      cache=False,
      batch_dims=batch_dims,
      num_epochs=1,
      shuffle=False,
      pad_up_to_batches="auto")

  return train_ds, eval_ds
