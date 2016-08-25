#!/usr/bin/env sh
set -e

caffe-segnet/build/tools/caffe train \
    -gpu 1 \
    --solver=Models/segnet_solver.prototxt \
    2>&1 | tee log/segnet_CamVid.log \
    $@
