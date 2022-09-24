import numpy as np
import time


def mergeSort(low, high, seq):
    """MergeSort for a sequence.

        Args:
            low (int): Begin index for sorting.
            high (int): End index for sorting, the available indexes are [low, high]
            seq (np.ndarray): Sequence to be sorted

        Return:
            None.
    """
    if low >= high:
        return
    elif low == high - 1:
        if seq[low] > seq[high]:
            temp = seq[low]
            seq[low] = seq[high]
            seq[high] = temp
    else:
        mid = (low + high) // 2
        mergeSort(low, mid, seq)
        mergeSort(mid+1, high, seq)
        merge(low, mid, high, seq)


def merge(low, mid, high, seq):
    """Merge ordered sequences.

            Args:
                low (int): Begin index for sorting.
                mid (int): Middle index for sorting.
                high (int): End index for sorting, the available indexes are [low, high]
                seq (np.ndarray): Sequence to be sorted

            Return:
                None.
    """
    low_index, high_index = low, mid + 1
    temp_seq = []
    while (low_index <= mid) & (high_index <= high):
        if seq[low_index] <= seq[high_index]:
            temp_seq.append(seq[low_index])
            low_index += 1
        else:
            temp_seq.append(seq[high_index])
            high_index += 1
    if low_index <= mid:
        for i in range(mid-low_index+1):
            temp_seq.append(seq[i+low_index])
    elif high_index <= high:
        for i in range(high-high_index+1):
            temp_seq.append(seq[i+high_index])
    try:
        seq[low:high+1] = np.array(temp_seq)
    except:
        print('Error')


if __name__ == '__main__':
    length_seq_list = [10000, 30000, 50000, 80000, 100000, 200000]
    for length_seq in length_seq_list:
        test_seq = np.random.choice(length_seq, length_seq, replace=False)
        time_start = time.time()
        mergeSort(0, length_seq-1, test_seq)
        time_end = time.time()
        print(f'time cost when length_seq = {length_seq}: {time_end - time_start:4f} s')

