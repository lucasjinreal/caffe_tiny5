# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
caffe_test.py
http://www.lewisjin.coding.me
~~~~~~~~~~~~~~~
This script implement by Jin Fagang.
: copyright: (c) 2017 Didi-Chuxing.
: license: Apache2.0, see LICENSE for more details.
"""
import numpy as np
import sys
import os
import cv2

caffe_root = '/home/jfg/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe


net_file = '/home/jfg/Documents/PythonSpace/caffe_tiny5/solver/caffenet_deploy.prototxt'
caffe_model = '/home/jfg/Documents/PythonSpace/caffe_tiny5/solver/model/caffenet_iter_4500.caffemodel'
mean_file = '/home/jfg/Documents/PythonSpace/caffe_tiny5/data/caffe_mean.binaryproto'
print('Params loaded!')

caffe.set_mode_gpu()
net = caffe.Net(net_file,
                caffe_model,
                caffe.TEST)

mean_blob = caffe.proto.caffe_pb2.BlobProto()
mean_blob.ParseFromString(open(mean_file, 'rb').read())
mean_npy = caffe.io.blobproto_to_array(mean_blob)
a = mean_npy[0, :, 0, 0]

print(net.blobs['data'].data.shape)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))

transformer.set_mean('data', a)
transformer.set_raw_scale('data', 255.0)
transformer.set_channel_swap('data', (2, 1, 0))

test_img = 'elephant.jpeg'
im = caffe.io.load_image(test_img)
net.blobs['data'].data[...] = transformer.preprocess('data', im)

predict = net.forward()
names = []
with open('words.txt', 'r+') as f:
    for l in f.readlines():
        names.append(l.split(' ')[1].strip())

print(names)
prob = net.blobs['prob'].data[0].flatten()
print('prob: ', prob)
print('class: ', names[np.argmax(prob)])

img = cv2.imread(test_img)
cv2.imshow('Image', img)
cv2.waitKey(0)
