#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

# CAFFEIMAGEPATH is the txt file path
# DATA is the txt file path


CAFFEDATAPATH=data
DATA=data
TOOLS=~/caffe/build/tools

# TRAIN_DATA_PATH & VAL_DATA_ROOT is root path of your images path, so your train.txt file must do not contain
# this line again!!
# I recommend using the generate_words_image_path.py script generate, and replace the following part
# We can not change this two because caffe need it.
TRAIN_DATA_ROOT=/home/jfg/Documents/PythonSpace/caffe_tiny5/tiny5/train
VAL_DATA_ROOT=/home/jfg/Documents/PythonSpace/caffe_tiny5/tiny5/valid

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $CAFFEDATAPATH/caffe_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/valid.txt \
    $CAFFEDATAPATH/caffe_val_lmdb

echo "Done."
