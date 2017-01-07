#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

LMDBDATA=data
DATA=data
TOOLS=~/caffe/build/tools

$TOOLS/compute_image_mean $LMDBDATA/caffe_train_lmdb \
  $DATA/caffe_mean.binaryproto

echo "Done."
