import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

mean_1 = np.array([1, 0]).T
mean_2 = np.array([-1, 0]).T
var = np.identity(2)
P = np.array([0.5, 0.5])

# visualize pdf
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
positions = np.stack((X, Y), axis=-1)
p1 = st.multivariate_normal.pdf(positions, mean=mean_1, cov=var)
p2 = st.multivariate_normal.pdf(positions, mean=mean_2, cov=var)
plt.contourf(x, y, p1*P[0]+p2*P[1], 20)
plt.colorbar()
plt.show()

err_rate_list = []
for num in range(100, 1100, 100):
    num1 = np.floor(num / 2).astype(np.int32)
    num2 = num - num1

    #  决策边界显然为x=0
    x1 = np.random.multivariate_normal(mean_1, var, (num1,), 'raise')
    x2 = np.random.multivariate_normal(mean_2, var, (num2,), 'raise')
    num_correct = (x1[:, 0] >= 0).sum() + (x2[:, 0] <= 0).sum()
    err_rate = 1 - num_correct/num
    err_rate_list.append(err_rate)
    print(f"Error Rate at num={num}: {err_rate:.4f}", )

plt.scatter(x1[:, 0], x1[:, 1], c='r')
plt.scatter(x2[:, 0], x2[:, 1], c='g')
plt.show()

x = np.arange(100, 1100, 100)
e = np.array(err_rate_list)

plt.style.use('seaborn-whitegrid')
plt.plot(x, e)
plt.ylim(0, 0.5)
plt.xlabel('num')
plt.ylabel('error rate')
plt.show()
