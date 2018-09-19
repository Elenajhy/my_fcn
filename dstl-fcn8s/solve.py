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

weights = '../dstl-fcn16s/models/dstl_iter_150000.caffemodel'

# init
caffe.set_device(int(0))
caffe.set_mode_gpu()
solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
test = np.loadtxt('../data/dstl/test.txt', dtype=str)
for _ in range(10):
    solver.step(15000)
    # N.B. metrics on the semantic labels are off b.c. of missing classes;
    # score manually from the histogram instead for proper evaluation
    score.seg_tests(solver, False, test, layer='score_sem', gt='sem')
    score.seg_tests(solver, False, test, layer='score_geo', gt='geo')
