import numpy as np
import time


def partition(low, high, arr):
    """Partition arr[low, high].

        Args:
            low (int): Begin index for sorting.
            high (int): End index for sorting, the available indexes are [low, high]
            arr (np.ndarray): array to be sorted

        Return:
            None.
    """
    v = arr[low]
    i, j = low, high+1
    while 1:
        for i in range(i+1, high + 1):
            if arr[i] > v:
                break
        for j in range(j-1, low, -1):
            if arr[j] <= v:
                break
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]
        else:
            break
    if low == j - 1 and arr[low] <= arr[j]:
        return low
    else:
        arr[low] = arr[j]
        arr[j] = v
        return j


def quickSort(low, high, arr):
    """MergeSort for a array.

        Args:
            low (int): Begin index for sorting.
            high (int): End index for sorting, the available indexes are [low, high]
            arr (np.ndarray): array to be sorted

        Return:
            None.
    """
    if low < high:
        pi = partition(low, high, arr)
        quickSort(low, pi - 1, arr)
        quickSort(pi + 1, high, arr)


if __name__ == '__main__':
    length_arr_list = [10000, 30000, 50000, 80000, 100000, 200000]
    for length_arr in length_arr_list:
        test_arr = np.random.choice(length_arr, length_arr, replace=False)
        time_start = time.time()
        quickSort(0, length_arr - 1, test_arr)
        time_end = time.time()
        print(f'time cost when length_arr = {length_arr}: {time_end - time_start:4f} s')
