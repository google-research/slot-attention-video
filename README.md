## Slot Attention for Video (SAVi)

This repository contains the code release for "Conditional Object-Centric
Learning from Video" (ICLR 2022).

Paper: https://arxiv.org/abs/2111.12594
Project website: https://slot-attention-video.github.io/


## Instructions

Get dependencies and run model training via
```sh
./run.sh
```

Or use
```sh
pip3 install -r requirements.txt
```
to install dependencies and
```sh
python -m slot_attention_video.main --config configs/movi/savi_conditional.py --workdir tmp/
```
to train a SAVi model on the [MOVi-A](https://github.com/google-research/kubric/blob/main/docs/datasets/movi/README.md) dataset.

The MOVi datasets are stored in a [Google Cloud Storage (GCS) bucket](https://console.cloud.google.com/storage/browser/kubric-public/tfds)
and can be downloaded to local disk prior to training for improved efficiency
(replace `data_dir` in the model config with the local path to the dataset).

## Bibtex

```
@inproceedings{kipf2022conditional,
    author = {Kipf, Thomas and Elsayed, Gamaleldin F. and Mahendran, Aravindh
              and Stone, Austin and Sabour, Sara and Heigold, Georg
              and Jonschkowski, Rico and Dosovitskiy, Alexey and Greff, Klaus},
    title = {{Conditional Object-Centric Learning from Video}},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year  = {2022}
}
```

## Disclaimer
This is not an official Google product.
