import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

""""Basic ports"""
fig = plt.figure()  # create figure
ax = plt.axes()  # set axes, which includes calibrations labels and depicted elements

x = np.linspace(0, 10, 1000)
# ax.plot(x, np.sin(x))
plt.plot(x, np.sin(x))  # automatically creates fig and ax
# plt.plot(x, np.cos(x))  # add extra curve

""""Axis Limitation"""
# latter lim setting works
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)

# inverse limitation
# plt.xlim(10, 0)
# plt.ylim(1.2, -1.2)

# axis setting in 1 line
# plt.axis([-1, 11, -1.5, 1.5])

# specific axis setting
# plt.axis('tight')
# plt.axis('equal')

"""Labels and Titles"""
plt.title("A Sine Curve")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.show()

'''Color Appointment'''
# without color appointment, the color will be repeated in a specific loop by default
plt.plot(x, np.sin(x - 0), color='blue')  # 通过颜色名称指定
plt.plot(x, np.sin(x - 1), color='g')  # 通过颜色简写名称指定(rgbcmyk)
plt.plot(x, np.sin(x - 2), color='0.75')  # 介于0-1之间的灰阶值
plt.plot(x, np.sin(x - 3), color='#FFDD44')  # 16进制的RRGGBB值
plt.plot(x, np.sin(x - 4), color=(1.0, 0.2, 0.3))  # RGB元组的颜色值，每个值介于0-1
plt.plot(x, np.sin(x - 5), color='chartreuse')  # 能支持所有HTML颜色名称值
plt.show()

'''Line Appointment'''
# without color appointment, the color will be repeated in a specific loop by default
plt.plot(x, x + 0, linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 2, linestyle='dashdot')
plt.plot(x, x + 3, linestyle='dotted')

# 还可以用形象的符号代表线条风格
plt.plot(x, x + 4, linestyle='-')  # 实线
plt.plot(x, x + 5, linestyle='--')  # 虚线
plt.plot(x, x + 6, linestyle='-.')  # 长短点虚线
plt.plot(x, x + 7, linestyle=':')  # 点线
plt.show()

""""Legends and Unified appointment"""
plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.axis('equal')
plt.legend()
plt.show()

# Unified with ax
# previous ax has been covered by plt.plot
ax = plt.axes()
ax.plot(x, np.sin(x))
ax.set(xlim=(0, 10), ylim=(-2, 2),
       xlabel='x', ylabel='sin(x)',
       title='A Simple Plot')
plt.show()
