import numpy as np
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
from matplotlib import pyplot as plt

s = np.load('shallow_mnist.npy')
d = np.load('deep_mnist.npy')
x_s = len(s)
x_d = len(d)

fig = plt.figure(figsize=(7,7), dpi=80)
plt.subplot(2,1,1)

X_s = np.linspace(1,x_s,x_s,endpoint=True)
X_d = np.linspace(1,x_d,x_d,endpoint=True)
Y_s = [float(i) for i in s]
Y_d = [float(i) for i in d]

xmin, xmax = X_s.min(), X_s.max()
ymin, ymax = min(min(Y_s,Y_d)),max(max(Y_s,Y_d))
dx = (xmax - xmin) * 0.02
dy = (ymax - ymin) * 0.05
plt.xlim(xmin - dx, xmax + dx)
plt.ylim(ymin - dy, ymax + dy)

plt.grid()

ax = plt.gca()

xmajorLocator   = MultipleLocator(10)
xmajorFormatter = FormatStrFormatter('%1.1f')
xminorLocator   = MultipleLocator(2)

ymajorLocator   = MultipleLocator(0.01)
ymajorFormatter = FormatStrFormatter('%1.2f')
yminorLocator   = MultipleLocator(0.001)

ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_major_formatter(xmajorFormatter)
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_minor_locator(xminorLocator)

ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_major_formatter(ymajorFormatter)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
ax.xaxis.grid(True, which='major')
ax.yaxis.grid(True, which='minor')
plt.xlabel("epoch of training")
plt.ylabel("error rate")
plt.title("shallow and deep network in MNIST")
plt.plot(X_s, Y_s, color="red", linewidth=1, linestyle="-", label="shallow")
plt.plot(X_d, Y_d, color="blue", linewidth=1, linestyle="-", label="deep")
plt.legend(loc='best', frameon=False)
#--------------------------------------
s = np.load('shallow_cifar10.npy')
d = np.load('deep_cifar10.npy')
x_s = len(s)
x_d = len(d)

fig = plt.figure(figsize=(7,7), dpi=80)
plt.subplot(2,1,2)

X_s = np.linspace(1,x_s,x_s,endpoint=True)
X_d = np.linspace(1,x_d,x_d,endpoint=True)
Y_s = [float(i) for i in s]
Y_d = [float(i) for i in d]
#设置图片边界
xmin, xmax = X_s.min(), X_s.max()
ymin, ymax = min(min(Y_s,Y_d)),max(max(Y_s,Y_d))
dx = (xmax - xmin) * 0.02
dy = (ymax - ymin) * 0.05
plt.xlim(xmin - dx, xmax + dx)
plt.ylim(ymin - dy, ymax + dy)
#开启网格
plt.grid()

ax = plt.gca()
#设置刻度
xmajorLocator   = MultipleLocator(10)
xmajorFormatter = FormatStrFormatter('%1.1f')
xminorLocator   = MultipleLocator(2)

ymajorLocator   = MultipleLocator(0.1)
ymajorFormatter = FormatStrFormatter('%1.1f')
yminorLocator   = MultipleLocator(0.02)

ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_major_formatter(xmajorFormatter)
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_minor_locator(xminorLocator)

ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_major_formatter(ymajorFormatter)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
ax.xaxis.grid(True, which='major')
ax.yaxis.grid(True, which='minor')
plt.xlabel("epoch of training")
plt.ylabel("error rate")
plt.title("shallow and deep network in cifar10")
plt.plot(X_s, Y_s, color="red", linewidth=1, linestyle="-", label="shallow")
plt.plot(X_d, Y_d, color="blue", linewidth=1, linestyle="-", label="deep")

plt.legend(loc='best', frameon=False)
#自动调整label显示方式，如果太挤则倾斜显示
fig.autofmt_xdate()
plt.show