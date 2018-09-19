#encoding=utf8
import caffe
import surgery, score
import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

vgg_weights = '/home/jhy/env/caffe/fcn/ilsvrc-nets/vgg16-fcn.caffemodel'                 #预训练模型地址
vgg_proto = '/home/jhy/env/caffe/fcn/ilsvrc-nets/VGG_ILSVRC_16_layers_deploy.prototxt'  #网络描述文件地址

# init
caffe.set_device(int(0))      #修改为实际的GPU设备号，如果只有一个GPU，则设置为0
caffe.set_mode_gpu()
solver = caffe.SGDSolver('solver.prototxt') #solver.prototxt:trainval.prototxt and test.prototxt
vgg_net=caffe.Net(vgg_proto,vgg_weights,caffe.TRAIN)  #用transplant的方式获取vgg16的权重
surgery.transplant(solver.net,vgg_net)                  
del vgg_net                                       

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
test = np.loadtxt('../data/dstl/test.txt', dtype=str)
for _ in range(10):#设置solver.prototxt:max_iter=50*2000,在solver.prototxt中修改没有用
    solver.step(15000) #每迭代15000次会输出score.py中loss,accuracy,mean等,获取测试集loss等参数。15000迭代刚好把所有图片训练了一次
    # N.B. metrics on the semantic labels are off b.c. of missing classes;
    # score manually from the histogram instead for proper evaluation
    score.seg_tests(solver, False, test, layer='score_sem', gt='sem')
    score.seg_tests(solver, False, test, layer='score_geo', gt='geo')
