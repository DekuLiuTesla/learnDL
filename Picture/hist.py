import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
plt.style.use('seaborn-white')

data = np.random.randn(1000)
# simple version
plt.hist(data)
plt.show()
# improved version
plt.hist(data, bins=30, density=True, alpha=0.5, histtype='stepfilled',
         color='steelblue',  edgecolor='none')
plt.show()

# use histtype & alpha for multiple datasets
x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)
kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)

plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs)
plt.show()

# Just calculate statistics
counts, bin_edges = np.histogram(data, bins=5)
print(counts)
print(bin_edges)

# 2d histogram
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T
plt.hist2d(x, y, bins=30, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')
plt.show()

# Just calculate 2d statistics
counts, x_edges, y_edges = np.histogram2d(x, y, bins=30)

# Use hexagonal bins
plt.hexbin(x, y, gridsize=30, cmap='Blues')
cb = plt.colorbar(label='count in bin')
plt.show()

""" Use histogram to illustrate KDE"""
# generate data and
data = np.vstack([x, y])  # x, y are horizontal vector in default
kde = gaussian_kde(data)

# calculate kde output
xgrid = np.linspace(-3.5, 3.5, 40)
ygrid = np.linspace(-6, 6, 40)
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))  # use ravel to collapse dimensions

# draw a picture with data
plt.imshow(Z.reshape(Xgrid.shape),
           origin='lower', aspect='auto',
           extent=[-3.5, 3.5, -6, 6],
           cmap='Blues')
cb = plt.colorbar()
cb.set_label("density")
plt.show()
