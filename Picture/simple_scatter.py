import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

"""Scatter with plt.plot"""
# plot copy points and thus is faster than scatter on large dataset
rng = np.random.RandomState(0)  # seed assertion makes sure the same random numbers of different run
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker,
             label="marker='{0}'".format(marker))
plt.legend(numpoints=1)
plt.xlim(-0.2, 1.2)
plt.show()

"""Specify Markers"""
x = np.linspace(0, 10, 30)
y = np.sin(x)
# plt.plot(x, y, 'o')
# plt.scatter(x, y, marker='o')  # same function as last line
plt.plot(x, y, '-p', color='gray',
         markersize=15, linewidth=4,
         markerfacecolor='white',
         markeredgecolor='gray',
         markeredgewidth=2)
plt.ylim(-1.2, 1.2)
plt.show()

"""Scatter with plt.scatter"""
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
            cmap='viridis')
plt.colorbar()  # 显示颜色对比条
plt.show()

"""An example of Iris"""
from sklearn.datasets import load_iris
iris = load_iris()
features = iris.data.T

# able to show at most 4d scattered points
plt.scatter(features[0], features[1], s=100*features[3],
            c=iris.target, alpha=0.2, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.colorbar()  # 显示颜色对比条
plt.show()
