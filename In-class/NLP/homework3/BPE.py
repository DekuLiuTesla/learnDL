import re
import numpy as np
from zhon.hanzi import punctuation


def BPE(org_data, max_iter=500):
    data = org_data
    for idx in range(max_iter):
        dictionary = {}
        for i in range(len(data)-1):
            merged_char = data[i] + data[i+1]
            if merged_char in dictionary.keys():
                dictionary[merged_char] += 1
            else:
                dictionary[merged_char] = 1
        sorted_dict = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)

        print(f'Iter {idx+1}, Top subword: {sorted_dict[0][0]}, '
              f'Top freq: {sorted_dict[0][1]}, Dictionary Length: {len(sorted_dict)}, '
              f'Text Length: {len(data)}')
        if sorted_dict[0][1] <= 20:
            break

        i, data_new = 0, []
        while True:
            if i >= len(data)-1:
                break
            merged_words = data[i] + data[i + 1]
            if merged_words == sorted_dict[0][0]:
                data_new.append(sorted_dict[0][0])
                i += 2
            else:
                data_new.append(data[i])
                i += 1
        if i == len(data)-1:
            data_new.append(data[-1])
        data = data_new

    return data


if __name__ == '__main__':
    results_file = 'data/results_RMM.npy'
    txt_file = 'data/results_RMM.txt'
    results = np.load(results_file)
    results_sep = []
    with open(txt_file, 'w', encoding='utf-8') as f:
        for line in results:
            line_list = [x for x in line.split('/') if x]
            clean_line = ' '.join(line_list)
            for i in punctuation:
                clean_line = clean_line.replace(' '+i, '')
            f.writelines(clean_line)
            f.write('\n')
            results_sep.extend(clean_line.split(' '))
    f.close()
    bpe_result = BPE(results_sep)

    print('Done!')
