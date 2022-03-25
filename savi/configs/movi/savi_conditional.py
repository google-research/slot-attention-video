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

"""Config for conditional SAVi.

By default, this config uses bounding box coordinates as conditioning signal.
Set `center_of_mass` to `True` to condition on center-of-mass coords instead.
"""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.seed = 42
  config.seed_data = True

  config.batch_size = 64
  config.num_train_steps = 100000

  # Adam optimizer config.
  config.learning_rate = 2e-4
  config.warmup_steps = 2500
  config.max_grad_norm = 0.05

  config.log_loss_every_steps = 50
  config.eval_every_steps = 1000
  config.checkpoint_every_steps = 5000

  config.train_metrics_spec = {
      "loss": "loss",
      "ari": "ari",
      "ari_nobg": "ari_nobg",
  }
  config.eval_metrics_spec = {
      "eval_loss": "loss",
      "eval_ari": "ari",
      "eval_ari_nobg": "ari_nobg",
  }

  config.data = ml_collections.ConfigDict({
      "tfds_name": "movi_a/128x128:1.0.0",
      "data_dir": "gs://kubric-public/tfds",
      "shuffle_buffer_size": config.batch_size * 8,
  })

  config.max_instances = 10
  config.num_slots = config.max_instances + 1  # Only used for metrics.
  config.logging_min_n_colors = config.max_instances

  config.preproc_train = [
      "video_from_tfds",
      f"sparse_to_dense_annotation(max_instances={config.max_instances})",
      "temporal_random_strided_window(length=6)",
      "resize_small(64)",
      "flow_to_rgb()"  # NOTE: This only uses the first two flow dimensions.
  ]

  config.preproc_eval = [
      "video_from_tfds",
      f"sparse_to_dense_annotation(max_instances={config.max_instances})",
      "temporal_crop_or_pad(length=24)",
      "resize_small(64)",
      "flow_to_rgb()"  # NOTE: This only uses the first two flow dimensions.
  ]

  # Evaluate on full video sequence by iterating over smaller chunks.
  config.eval_slice_size = 6
  config.eval_slice_keys = ["video", "segmentations", "flow", "boxes"]

  # Dictionary of targets and corresponding channels. Losses need to match.
  config.targets = {"flow": 3}
  config.losses = ml_collections.ConfigDict({
      f"recon_{target}": {"loss_type": "recon", "key": target}
      for target in config.targets})

  config.conditioning_key = "boxes"

  config.model = ml_collections.ConfigDict({
      "module": "savi.modules.SAVi",

      # Encoder.
      "encoder": ml_collections.ConfigDict({
          "module": "savi.modules.FrameEncoder",
          "reduction": "spatial_flatten",
          "backbone": ml_collections.ConfigDict({
              "module": "savi.modules.CNN",
              "features": [32, 32, 32, 32],
              "kernel_size": [(5, 5), (5, 5), (5, 5), (5, 5)],
              "strides": [(1, 1), (1, 1), (1, 1), (1, 1)],
              "layer_transpose": [False, False, False, False]
          }),
          "pos_emb": ml_collections.ConfigDict({
              "module": "savi.modules.PositionEmbedding",
              "embedding_type": "linear",
              "update_type": "project_add",
              "output_transform": ml_collections.ConfigDict({
                  "module": "savi.modules.MLP",
                  "hidden_size": 64,
                  "layernorm": "pre"
              }),
          }),
      }),

      # Corrector.
      "corrector": ml_collections.ConfigDict({
          "module": "savi.modules.SlotAttention",
          "num_iterations": 1,
          "qkv_size": 128,
      }),

      # Predictor.
      "predictor": ml_collections.ConfigDict({
          "module": "savi.modules.TransformerBlock",
          "num_heads": 4,
          "qkv_size": 128,
          "mlp_size": 256
      }),

      # Initializer.
      "initializer": ml_collections.ConfigDict({
          "module": "savi.modules.CoordinateEncoderStateInit",
          "prepend_background": True,
          "center_of_mass": False,
          "embedding_transform": ml_collections.ConfigDict({
              "module": "savi.modules.MLP",
              "hidden_size": 256,
              "output_size": 128,
              "layernorm": None
          }),
      }),

      # Decoder.
      "decoder": ml_collections.ConfigDict({
          "module":
              "savi.modules.SpatialBroadcastDecoder",
          "resolution": (8, 8),  # Update if data resolution or strides change.
          "backbone": ml_collections.ConfigDict({
              "module": "savi.modules.CNN",
              "features": [64, 64, 64, 64],
              "kernel_size": [(5, 5), (5, 5), (5, 5), (5, 5)],
              "strides": [(2, 2), (2, 2), (2, 2), (1, 1)],
              "layer_transpose": [True, True, True, False]
          }),
          "pos_emb": ml_collections.ConfigDict({
              "module": "savi.modules.PositionEmbedding",
              "embedding_type": "linear",
              "update_type": "project_add"
          }),
          "target_readout": ml_collections.ConfigDict({
              "module": "savi.modules.Readout",
              "keys": list(config.targets),
              "readout_modules": [
                  ml_collections.ConfigDict({
                      "module": "savi.modules.Dense",
                      "features": config.targets[k]
                  }) for k in config.targets
              ],
          }),
      }),
      "decode_corrected": True,
      "decode_predicted": False,  # Disable prediction decoder to save memory.
  })

  # Define which video-shaped variables to log/visualize.
  config.debug_var_video_paths = {
      "recon_masks": "SpatialBroadcastDecoder_0/alphas",
  }
  for k in config.targets:
    config.debug_var_video_paths.update({
        f"{k}_recon_slots": f"SpatialBroadcastDecoder_0/{k}_slots",
        f"{k}_recon_slots_masked": f"SpatialBroadcastDecoder_0/{k}_masked",
        f"{k}_recon": f"SpatialBroadcastDecoder_0/{k}_combined"})

  # Define which attention matrices to log/visualize.
  config.debug_var_attn_paths = {
      "corrector_attn": "SlotAttention_0/InvertedDotProductAttention_0/GeneralizedDotProductAttention_0/attn"
  }

  # Widths of attention matrices (for reshaping to image grid).
  config.debug_var_attn_widths = {
      "corrector_attn": 64,
  }

  return config


def get_hyper(h):
  """Get the hyperparamater sweep."""
  sweeps = []
  sweeps.append(h.sweep("config.seed", list(range(1))))
  return h.product(sweeps)
