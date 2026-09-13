#!/usr/bin/env bash
cd "$(dirname "$0")/.."
exec env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -u visual_dev/pretrain.py --coverage v4b --out pretrain_v4b --steps 6000 --n-train 3200 --n-dev 320
