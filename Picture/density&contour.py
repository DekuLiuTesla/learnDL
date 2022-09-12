import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)


x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
# dashed contours are negative while solid contours are positive
plt.contour(X, Y, Z, colors='black')
plt.show()
# deeper red contours mean lower negative values
plt.contour(X, Y, Z, 20, cmap='RdGy')
plt.show()

# fill the contour graph and add a color bar,
# but it will lead to discrete color distribution
plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar()
plt.show()
# range of axis needs to be specified by extent in imshow
plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy')
# aspect='image' controls unit of x and y axis to be the same
plt.axis('equal')
plt.colorbar()
plt.show()

# overlap contour and image
contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
           cmap='RdGy', alpha=0.5)
plt.colorbar()
plt.show()
