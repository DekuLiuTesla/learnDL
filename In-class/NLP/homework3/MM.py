# -*- coding:utf-8 -*-
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn

plt.style.use('seaborn-whitegrid')


def readFile(filename, encoding='mbcs'):
    '''
    read file return a generator, each element is one line

    Parameters
    ----------
        filename: str, filename
        encoding: str, encoding method

    Returns:
    ----------
        generator
    '''

    with open(filename, 'r', encoding=encoding) as f:
        for line in f.readlines():
            line = line.strip()  # strip used to remove characters at end of line like \n
            if not len(line):
                continue
            else:
                yield line


def generate_text(text):
    '''
    Parameters
    ----------
        text: str, ’/w  ９９/m  昆明/ns  世博会/n  组委会/j  秘书长/n  、/w  云南省/ns

    Returns
    ----------
        str, ’９９昆明世博会组委会秘书长、云南省
    '''
    # map apply specified function to each element in sequence
    return ''.join(map(lambda x: x[: x.rfind('/')], text.split('  ')))


def generate_target(text):
    '''
    Parameters
    ----------
        text: str, ’/w  ９９/m  昆明/ns  世博会/n  组委会/j  秘书长/n  、/w  云南省/ns

    Returns
    ----------
        str, ’/９９/昆明/世博会/组委会/秘书长/、/云南省
    '''
    # map apply specified function to each element in sequence
    return ''.join(map(lambda x: x[:x.rfind('/') + 1], text.split('  ')))


def precision_recall_f1(output, target):
    '''
    Parameters
    ----------
        output, str

        target, str

    Returns
    ----------
        precision, recall and f1, float
    '''

    def extract_index_pair(text):
        o = [(0, 0)]
        index = 0
        for i in text:
            if i != '/':
                index += 1
            else:
                o.append((o[-1][-1], index))
        else:
            o.append((o[-1][-1], index))
        o = set(o)
        o.remove((0, 0))
        return o

    o = extract_index_pair(output)
    t = extract_index_pair(target)

    def precision_score(o, t):
        count = 0
        for i in t:
            if i in o:
                count += 1
        return count / len(t)

    precision, recall = precision_score(o, t), precision_score(t, o)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def build_corpus_and_testing_text(filename, training_ratio=0.7):
    '''
    forward maximum matching

    Parameters
    ----------
    filename: str

    training_ratio: float, ratio of training data

    Returns
    ----------
    corpus: set

    testing_text: list, each element is a sentence that need to be preprocessed
    '''
    corpus = set()
    num_of_lines = 0
    for line in readFile(filename):
        num_of_lines += 1
    all_index = np.arange(num_of_lines)
    np.random.shuffle(all_index)
    training_lines = set(all_index[:int(training_ratio * num_of_lines)].tolist())

    testing_text = []
    for index, line in enumerate(readFile(filename)):
        if index not in training_lines:
            testing_text.append(line)
            continue
        for j, temp in enumerate(map(lambda x: x.split('/'), line.split('  '))):
            if len(temp) != 2 or j == 0:
                # Ignore don't care cases
                continue
            word, _ = temp
            if 'ｈｔｔｐ' in word or 'ｗｗｗ．' in word:
                continue
            corpus.add(word)
    return corpus, testing_text


def split_words(line, corpus_set):
    '''
    forward maximum matching

    Parameters
    ----------
    line: str, a chinese string

    corpus_set: set, corpus

    Returns
    ----------
    str, results
    '''
    n_line = len(line)
    start, end = 0, n_line
    result = []
    while start < n_line:
        n = n_line - start
        if n == 1:
            result.append(line[start:])
            return '/'.join(result) + '/'
        current_word = line[start: end]
        if current_word in corpus_set:
            result.append(current_word)
            start = end
            end = n_line
            continue
        else:
            if len(current_word) == 1:
                corpus_set.add(current_word)
                result.append(current_word)
                start = end
                end = n_line
                continue
            end -= 1
            continue
        start += 1
    return '/'.join(result) + '/'


def split_words_reverse(line, corpus_set):
    '''
    backward maximum matching

    Parameters
    ----------
    line: str, a chinese string

    corpus_set: set, corpus

    Returns
    ----------
    str, results
    '''
    n_line = len(line)
    start, end = 0, n_line
    result = []
    while end > 0:
        #         time.sleep(0.5)
        #         print(result)
        if (end - 0) == 1:
            result.append(line[start: end])
            return '/'.join(reversed(result)) + '/'
        current_word = line[start: end]
        if current_word in corpus_set:
            result.append(current_word)
            end = start
            start = 0
            continue
        else:
            if len(current_word) == 1:
                corpus_set.add(current_word)
                result.append(current_word)
                end = start
                start = 0
                continue
            start += 1
            continue
        end -= 1
    return '/'.join(reversed(result)) + '/'


def split_words_bidirection(line, corpus_set):
    '''
    backward maximum matching

    Parameters
    ----------
    line: str, a chinese string

    corpus_set: set, corpus

    Returns
    ----------
    str, results
    '''

    output_FMM = split_words(line, corpus_set)
    len_FMM = len(output_FMM.split('/'))
    output_BMM = split_words_reverse(line, corpus_set)
    len_BMM = len(output_BMM.split('/'))
    return output_FMM if len_FMM < len_BMM else output_BMM


def run(split_words_function, testing_text, corpus, name='Model'):
    p_r_f1 = []
    results = []
    for index, line_i in enumerate(testing_text):
        rectified_line_i = '  '.join(line_i.split()[1:])
        text = generate_text(rectified_line_i)
        target = generate_target(rectified_line_i)
        output = split_words_function(text, corpus)
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
    print('Result of FMM: ')
    results_FMM, a_FMM, b_FMM, c_FMM = run(split_words, testing_text, corpus, name='FMM')
    np.save('data/results_FMM.npy', results_FMM)
    print('Result of BMM: ')
    results_BMM, a_BMM, b_BMM, c_BMM = run(split_words_reverse, testing_text, corpus, name='BMM')
    np.save('data/results_BMM.npy', results_BMM)
    print('Result of RMM: ')
    results_RMM, a_RMM, b_RMM, c_RMM = run(split_words_bidirection, testing_text, corpus, name='RMM')
    np.save('data/results_RMM.npy', results_RMM)
