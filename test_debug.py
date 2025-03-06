from numpy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
font = {'size': '16'}
plt.rc('font', **font)
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

abide = [69.861, 70.096, 70.503, 69.338, 69.845, 69.174,68.445,68.0]
adhd = [68.560, 68.251, 68.976, 68.485, 68.527, 68.009,67.415,67.0]


# x = ['3','4','5','6','7','8']
x = [3,4,5,6,7,9,15,25]
plt.figure() #figsize是图片的大小`fig = plt.figure(figsize = (7,5))
ax=plt.axes()

# plt.grid(zorder=0, linewidth = "0.5", linestyle = "-.")  #显示网格，zorder控制网格显示的前后, color='#738CBC'   , color='#DFD478'

ax.plot(x, abide, marker='^', linestyle = 'dotted', lw=1.5, label='ABIDE') #‘^’ : 正三角形
ax.plot(x, adhd, marker='s', linestyle = '-.', lw=1.5, label='ADHD') #‘s’ : 方块状
ax.set_ylim(60,75)
plt.legend(loc='upper right', borderaxespad=0.5)   #显示标签，并放在外侧
plt.xlabel(r'$p$',usetex=True) #设置y轴的标签
# plt.xlabel(r'$p$')
plt.ylabel('Accuracy(%)') #设置y轴的标签
plt.savefig("para.pdf",dpi=800,bbox_inches='tight') # 保存图片
plt.xticks(x, x)
plt.show()