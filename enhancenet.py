#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from nnlib import *

PER_CHANNEL_MEANS = np.array([0.47614917, 0.45001204, 0.40904046])
fns = sorted([fn for fn in os.listdir('input/')])
if not os.path.exists('output'):
    os.makedirs('output')
for fn in fns:
    fne = ''.join(fn.split('.')[:-1])
    if os.path.isfile('output/%s-EnhanceNet.png' % fne):
        print('skipping %s' % fn)
        continue
    imgs = loadimg('input/'+fn)
    if imgs is None:
        continue
    imgs = np.expand_dims(imgs, axis=0)
    imgsize = np.shape(imgs)[1:]
    print('processing %s' % fn)
    xs = tf.placeholder(tf.float32, [1, imgsize[0], imgsize[1], imgsize[2]])
    rblock = [resi, [[conv], [relu], [conv]]]
    ys_est = NN('generator',
                [xs,
                 [conv], [relu],
                 rblock, rblock, rblock, rblock, rblock,
                 rblock, rblock, rblock, rblock, rblock,
                 [upsample], [conv], [relu],
                 [upsample], [conv], [relu],
                 [conv], [relu],
                 [conv, 3]])
    ys_res = tf.image.resize_images(xs, [4*imgsize[0], 4*imgsize[1]],
                                    method=tf.image.ResizeMethod.BICUBIC)
    ys_est += ys_res + PER_CHANNEL_MEANS
    sess = tf.InteractiveSession()
    tf.train.Saver().restore(sess, os.getcwd()+'/weights')
    output = sess.run([ys_est, ys_res+PER_CHANNEL_MEANS],
                      feed_dict={xs: imgs-PER_CHANNEL_MEANS})
    saveimg(output[0][0], 'output/%s-EnhanceNet.png' % fne)
    saveimg(output[1][0], 'output/%s-Bicubic.png' % fne)
    sess.close()
    tf.reset_default_graph()
