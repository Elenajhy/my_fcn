import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

num = int(sys.argv[1])
interval = int(sys.argv[2])
iterations = range(0, num*interval, interval)

loss = np.loadtxt('loss.txt', dtype=np.float32)
loss_sem = np.loadtxt("loss_sem.txt", dtype=np.float32)
loss_geo = np.loadtxt("loss_geo.txt", dtype=np.float32)

xlabel = 'Iterations'
ylabel = 'Normalized Loss'

fig = plt.figure(figsize=(15, 15))

ax = fig.add_subplot(3,1,1)
#ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_xlim(0, num*interval)
ax.set_title("Loss")
ax.plot(iterations, loss, linewidth=2, color='r', marker='o',
         markerfacecolor='blue', markersize=8)

ax = fig.add_subplot(3,1,2)
#ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_xlim(0, num*interval)
ax.set_title("Loss_sem")
ax.plot(iterations, loss_sem, linewidth=2, color='r', marker='o',
         markerfacecolor='blue', markersize=8)

ax = fig.add_subplot(3,1,3)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_xlim(0, num*interval)
ax.set_title("Loss_geo")
ax.plot(iterations, loss_geo, linewidth=2, color='r', marker='o',
         markerfacecolor='blue', markersize=8)

plt.savefig('loss.png')
