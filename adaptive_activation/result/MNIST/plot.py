import numpy as np
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
from matplotlib import pyplot as plt

s1 = np.load('sin/sin_test1.npy')
s2 = np.load('sin_adaptive_alpha/sin_alpha_test1.npy')
c1 = np.load('cos/cos_test1.npy')
c2 = np.load('cos_adaptive_alpha/cos_alpha_test1.npy')
x_s1 = len(s1)
x_s2 = len(s2)
x_c1 = len(c1)
x_c2 = len(c2)

fig = plt.figure(figsize=(8,5), dpi=80)
plt.subplot(1,1,1)

X_s1 = np.linspace(1,x_s1,x_s1,endpoint=True)
X_s2 = np.linspace(1,x_s2,x_s2,endpoint=True)
X_c1 = np.linspace(1,x_c1,x_c1,endpoint=True)
X_c2 = np.linspace(1,x_c2,x_c2,endpoint=True)

Y_s1 = [float(i) for i in s1]
Y_s2 = [float(i) for i in s2]
Y_c1 = [float(i) for i in c1]
Y_c2 = [float(i) for i in c2]
#设置图片边界
xmin, xmax = X_s1.min(), X_s1.max()
ymin, ymax = min(min(Y_s1,Y_s2,Y_c1,Y_c2)),max(max(Y_s1,Y_s2,Y_c1,Y_c2))
dx = (xmax - xmin) * 0.02
dy = (ymax - ymin) * 0.1
plt.xlim(xmin - dx, xmax + dx)
plt.ylim(ymin - dy, ymax + dy)
#开启网格
plt.grid()

ax = plt.gca()
#设置刻度
xmajorLocator   = MultipleLocator(10) #将x主刻度标签设置为20的倍数
xmajorFormatter = FormatStrFormatter('%1.1f') #设置x轴标签文本的格式
xminorLocator   = MultipleLocator(2) #将x轴次刻度标签设置为5的倍数

ymajorLocator   = MultipleLocator(1) #将y轴主刻度标签设置为0.5的倍数
ymajorFormatter = FormatStrFormatter('%1.1f') #设置y轴标签文本的格式
yminorLocator   = MultipleLocator(0.2) #将此y轴次刻度标签设置为0.1的倍数

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
plt.xlabel("test number")
plt.ylabel("error rate")
plt.title("CNN for different actavation function in MNIST")
plt.plot(X_s1, Y_s1, color="green", linewidth=1, linestyle="-", label="sin(x)")
plt.plot(X_s2, Y_s2, color="blue", linewidth=1, linestyle="-", label="sin(ax)")
plt.plot(X_c1, Y_c1, color="green", linewidth=1, linestyle="--", label="cos(x)")
plt.plot(X_c2, Y_c2, color="blue", linewidth=1, linestyle="--", label="cos(ax)")

plt.legend(loc='best', frameon=False)
#自动调整label显示方式，如果太挤则倾斜显示
fig.autofmt_xdate()
plt.show