#!/usr/bin/env sh
set -e

caffe-segnet/build/tools/caffe train \
    -gpu 0 \
    --solver=Models/IOUsegnet_solver.prototxt \
    2>&1 | tee /home/zshen5/Log/IOUsegnet_CamVid.log \
    $@
