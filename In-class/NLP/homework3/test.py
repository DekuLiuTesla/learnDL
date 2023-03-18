import jieba
import numpy as np
import matplotlib.pyplot as plt
from MM import *


def test_model(split_words_function, testing_text, corpus, name='Model'):
    p_r_f1 = []
    results = []
    for index, line_i in enumerate(testing_text):
        target = '/'.join(jieba.lcut(line_i))+'/'
        output = split_words_function(line_i, corpus)
        p_r_f1.append(precision_recall_f1(output, target))
        results.append(output)

    a, b, c = zip(*p_r_f1)
    average_precision = sum(a) / len(a)
    average_recall = sum(b) / len(b)
    average_f1 = sum(c) / len(c)
    print('average precision:', average_precision)
    print('average recall', average_recall)
    print('average F1-score', average_f1)
    plt.figure(figsize=(16, 4))
    plt.subplot(121)
    plt.xticks(np.arange(0, 1.05, 0.1))
    plt.hist(a, bins=np.arange(0, 1.05, 0.05))
    plt.xlabel('precision')
    plt.ylabel('frequency')
    plt.title(name + ' precision')
    plt.subplot(122)
    plt.xticks(np.arange(0, 1.05, 0.1))
    plt.hist(b, bins=np.arange(0, 1.05, 0.05))
    plt.xlabel('recall')
    plt.ylabel('frequency')
    plt.title(name + ' recall')
    plt.show()
    return results, a, b, c


if __name__ == '__main__':
    corpus_file = 'data/ChineseCorpus199801.txt'
    corpus, testing_text = build_corpus_and_testing_text(corpus_file)
    print('corpus size:', len(corpus))
    print('testing data size:', len(testing_text), 'lines')

    test_file = ['data/news.2012.zh.shuffled.deduped', 'data/DouLuoDaLu.txt']
    encodings = ['utf-8', 'GBK']
    i = 0
    testing_text = []
    for index, line in enumerate(readFile(test_file[i], encoding=encodings[i])):
        testing_text.append(line)

    print('Result of FMM: ')
    results_FMM, a_FMM, b_FMM, c_FMM = test_model(split_words, testing_text, corpus, name='FMM')

    print('Done!')
