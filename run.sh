#!/bin/bash
set -e
set -x


virtualenv -p python3 .
source ./bin/activate

pip3 install -r requirements.txt

python -m savi.main --config configs/movi/savi_conditional_bbox.py --workdir tmp/
