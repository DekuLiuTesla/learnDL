# coding=utf-8
import os
import math
from random import sample
from progressbar import *
from glob import glob
from os.path import join
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS

if __name__ == '__main__':
    cnt = 0.0
    dataset_size = 0
    sample_ratio = 1
    word_count = {}

    novel_name = 'DouLuoDaLu2'
    load_dir = 'texts/'+novel_name+'_renamed'
    txt_pathnames = sorted(
        glob(join(load_dir, '*.txt')))
    selected_txt_pathnames = sample(txt_pathnames, math.ceil(len(txt_pathnames)*sample_ratio))
    # calculate number of characters and size of dataset
    pbar = ProgressBar().start()

    for i, txt_path in enumerate(selected_txt_pathnames):
        dataset_size += os.path.getsize(txt_path)
        with open(txt_path, 'r') as f:
            text = f.read().replace('\n', '')
            for u_char in text:
                if u'\u4e00' <= u_char <= u'\u9fa5':
                    if u_char in word_count.keys():
                        word_count[u_char] += 1
                    else:
                        word_count[u_char] = 1
                cnt += 1
        pbar.update(int(((i + 1) / len(selected_txt_pathnames)) * 100))

    # Calculate frequency and entropy
    entropy = 0.0
    if cnt > 0:
        for key in word_count.keys():
            # get frequency
            word_count[key] /= cnt
            # get entropy
            entropy += -word_count[key] * math.log2(word_count[key])
        dataset_size_M = dataset_size/(2**20)
        sorted_word_count = sorted(word_count.items(), key=lambda item: item[1], reverse=True)
        print(f"\nEntropy of characters in {novel_name}: {entropy:.3f} bits/symbol")
        print(f"Select {len(selected_txt_pathnames)}/{len(txt_pathnames)} txt files")
        print(f"Size of Dataset: {dataset_size_M:.3f} M")
        num = 30
        print(f"Most frequent {num} characters: ")
        for i in range(num):
            print(sorted_word_count[i][0], end=' ')
