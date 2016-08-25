#!/usr/bin/env sh
set -e

caffe-segnet/build/tools/caffe train \
    --solver=Models/segnet_solver.prototxt \
    --snapshot=models/bvlc_reference_caffenet/caffenet_train_10000.solverstate.h5 \
    $@
