import numpy as np
import matplotlib.pyplot as plt
import time


def BatchPerception(data_1, data_2, a, max_iter=50):
    assert data_1.shape == data_2.shape
    ones = np.ones_like(data_1, shape=(data_1.shape[0], 1))
    data_1 = np.concatenate((data_1, ones), axis=-1)
    data_2 = np.concatenate((data_2, ones), axis=-1)
    data_2 = -data_2
    data = np.concatenate((data_1, data_2))
    assert data_2.shape[-1] == a.shape[0]
    for i in range(max_iter):
        print(f"Iter {i}: a = {a}\n")
        pred = data @ a
        if (pred > 0).all():
            break
        sum_neg = np.sum(data[pred <= 0], axis=0)
        a += sum_neg
    return a


def Ho_Kashyap(data_1, data_2, a, max_iter=50, eps=1.0):
    assert data_1.shape == data_2.shape
    ones = np.ones_like(data_1, shape=(data_1.shape[0], 1))
    data_1 = np.concatenate((data_1, ones), axis=-1)
    data_2 = np.concatenate((data_2, ones), axis=-1)
    data_2 = -data_2
    data = np.concatenate((data_1, data_2))
    assert data_2.shape[-1] == a.shape[0]
    b = np.ones_like(data @ a)
    eta = 1.0
    for i in range(max_iter):
        error_vec = data @ a - b
        error = error_vec.sum()
        b += (error_vec + np.abs(error_vec)) * eta
        a = np.linalg.pinv(data) @ b
        print(f"Iter {i}: error = {error:.4f}\n")
        if np.abs(error) < eps:
            return a, b
    print('No solution found!\n')


if __name__ == '__main__':
    samples_1 = [
        [0.1, 6.8, -3.5, 2.0, 4.1, 3.1, -0.8, 0.9, 5.0, 3.9],
        [1.1, 7.1, -4.1, 2.7, 2.8, 5.0, -1.3, 1.2, 6.4, 4.0]
    ]
    samples_1 = np.array(samples_1).T

    samples_2 = [
        [7.1, -1.4, 4.5, 6.3, 4.2, 1.4, 2.4, 2.5, 8.4, 4.1],
        [4.2, -4.3, 0.0, 1.6, 1.9, -3.2, -4.0, -6.1, 3.7, -2.2]
    ]
    samples_2 = np.array(samples_2).T

    samples_3 = [
        [-3.0, 0.5, 2.9, -0.1, -4.0, -1.3, -3.4, -4.1, -5.1, 1.9],
        [-2.9, 8.7, 2.1, 5.2, 2.2, 3.7, 6.2, 3.4, 1.6, 5.1]
    ]
    samples_3 = np.array(samples_3).T

    samples_4 = [
        [-2.0, -8.9, -4.2, -8.5, -6.7, -0.5, -5.3, -8.7, -7.1, -8.0],
        [-8.4, 0.2, -7.7, -3.2, -4.0, -9.2, -6.7, -6.4, -9.7, -6.3]
    ]
    samples_4 = np.array(samples_4).T

    a = np.array([0, 0, 0], dtype=samples_1.dtype)
    print("Classify class 1 and class 2: \n")
    BatchPerception(samples_1, samples_2, a)
    print('-------------------------------------\n')

    print("Classify class 3 and class 2: \n")
    a = np.array([0, 0, 0], dtype=samples_1.dtype)
    BatchPerception(samples_3, samples_2, a)
    print('-------------------------------------\n')

    print("Classify class 1 and class 3 with HK: \n")
    a = np.array([0, 0, 0], dtype=samples_1.dtype)
    Ho_Kashyap(samples_1, samples_3, a)
    print('-------------------------------------\n')
    plt.scatter(samples_1[:, 0], samples_1[:, 1], label='w1')
    plt.scatter(samples_3[:, 0], samples_3[:, 1], label='w3')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    print("Classify class 2 and class 4 with HK: \n")
    a = np.array([0, 0, 0], dtype=samples_1.dtype)
    Ho_Kashyap(samples_2, samples_4, a)
    print('-------------------------------------\n')

    print("MSE for Multiclass: \n")
    train_num = 8
    test_num = samples_1.shape[0] - train_num

    X_train = np.concatenate((samples_1[:train_num, :], samples_2[:train_num, :],
                              samples_3[:train_num, :], samples_4[:train_num, :]), axis=0)
    Y_train = np.zeros((4, X_train.shape[0]))
    for i in range(4):
        Y_train[i, (i * 8):((i + 1) * 8)] = 1

    X_test = np.concatenate((samples_1[train_num:, :], samples_2[train_num:, :],
                             samples_3[train_num:, :], samples_4[train_num:, :]), axis=0)
    Y_test = np.zeros((4, X_test.shape[0]))
    for i in range(4):
        Y_test[i, (i * test_num):((i + 1) * test_num)] = 1

    ones = np.ones_like(X_train, shape=(X_train.shape[0], 1))
    X_train = np.concatenate((X_train, ones), axis=-1).T
    W = np.linalg.inv(X_train @ X_train.T +
                      0.01 * np.identity(X_train.shape[0])) @ X_train @ Y_train.T

    pred_train = W.T @ X_train
    pred_label = np.argmax(pred_train, axis=0)
    real_label = np.argmax(Y_train, axis=0)
    print(f"Training Correct Rate: {(pred_label==real_label).sum()/real_label.shape[0]}")

    ones = np.ones_like(X_test, shape=(X_test.shape[0], 1))
    X_test = np.concatenate((X_test, ones), axis=-1).T
    pred_test = W.T @ X_test
    pred_label = np.argmax(pred_test, axis=0)
    real_label = np.argmax(Y_test, axis=0)
    print(f"Testing Correct Rate: {(pred_label == real_label).sum() / real_label.shape[0]}")
