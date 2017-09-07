
import numpy as np
from collections import defaultdict as dd
from scipy import sparse as sp
import cnn_rnn
import sample

LABEL_INDEX = ['PRP$', 'VBG', 'VBD', 'VBN', 'HT', 'POS', "''", 'VBP', 'WDT', 'USR', 'JJ',\
    'WP', 'VBZ', 'DT', 'RT', 'NONE', 'RP', 'VPP', 'NN', 'TO', ')', '(', 'FW', ',', '.', 'CC',\
    'PRP', 'RB', 'TD', ':', 'NNS', 'NNP', 'VB', 'WRB', 'URL', 'LS', 'PDT', 'RBS', 'RBR', 'O',\
    'CD', 'EX', 'IN', 'MD', 'NNPS', 'JJS', 'JJR', 'SYM', 'UH']


MAX_LEN = 40
MAX_CHAR_LEN = 30

DIR = 'twitter/pos.'
TRAIN_DATA = DIR + 'train.txt'
DEV_DATA = DIR + 'test.txt'
TEST_DATA = DIR + 'dev.txt'

HASH_FILE = 'words.lst'
EMB_FILE = 'embeddings.txt'

USE_DEV = True
LABELING_RATE = 0.1

def process(word):
    word = word.lower()
    word = "".join(c if not c.isdigit() else '0' for c in word)
    return word

def create_word_index(filenames):
    word_index, word_cnt = {}, 1

    for filename in filenames:
        for line in open(filename):
            if line.strip() == '': continue
            word = line.strip().split()[0]
            word = process(word)
            if word in word_index: continue
            word_index[word] = word_cnt
            word_cnt += 1
    return word_index, word_cnt

def create_char_index(filenames):
    char_index, char_cnt = {}, 3

    for filename in filenames:
        for line in open(filename):
            if line.strip() == '': continue
            word = line.strip().split()[0]
            for c in word:
                if c not in char_index:
                    char_index[c] = char_cnt
                    char_cnt += 1
    return char_index, char_cnt

def cnt_line(filename):
    ret = 0
    flag = False
    for line in open(filename):
        if line.strip() == '':
            if flag:
                ret += 1
            flag = False
        else:
            flag = True
    if flag:
        ret += 1
    return ret

def read_data(filename, word_index):
    line_cnt = cnt_line(filename)
    x, y = np.zeros((line_cnt, MAX_LEN), dtype = np.int32), np.zeros((line_cnt, MAX_LEN), dtype = np.int32)
    mask = np.zeros((line_cnt, MAX_LEN), dtype = np.float32)
    i, j = 0, 0
    for line in open(filename):
        if line.strip() == '':
            i += 1
            j = 0
            continue
        inputs = line.strip().split()
        label = inputs[1]
        word = inputs[0]
        word = process(word)
        word_ind, label_ind = word_index[word], LABEL_INDEX.index(label)
        x[i, j] = word_ind
        y[i, j] = label_ind
        mask[i, j] = 1.0
        j += 1
    return x, y, mask

def read_char_data(filename, char_index):
    line_cnt = cnt_line(filename)
    x = np.zeros((line_cnt, MAX_LEN, MAX_CHAR_LEN), dtype = np.int32)
    mask = np.zeros((line_cnt, MAX_LEN, MAX_CHAR_LEN), dtype = np.float32)
    i, j = 0, 0
    for line in open(filename):
        if line.strip() == '':
            i += 1
            j = 0
            continue
        inputs = line.strip().split()
        label = inputs[1]
        word = inputs[0]
        for k, c in enumerate(word):
            if k + 1 >= MAX_CHAR_LEN: break
            x[i, j, k + 1] = char_index[c]
            mask[i, j, k + 1] = 1.0
        x[i, j, 0] = 1
        mask[i, j, 0] = 1.0
        if len(word) + 1 < MAX_CHAR_LEN:
            x[i, j, len(word) + 1] = 2
            mask[i, j, len(word) + 1] = 1.0
        j += 1
    return x, mask

def read_word2embedding():
    words = []
    for line in open(HASH_FILE):
        words.append(line.strip())
    word2embedding = {}
    for i, line in enumerate(open(EMB_FILE)):
        if words[i] in word2embedding: continue
        inputs = line.strip().split()
        word2embedding[words[i]] = np.array([float(e) for e in inputs], dtype = np.float32)
    return word2embedding

def evaluate(py, y_, m_, full = False):
    if len(py.shape) > 1:
        py = np.argmax(py, axis = 1)
    y, m = y_.flatten(), m_.flatten()
    acc = 1.0 * (np.array(y == py, dtype = np.int32) * m).sum() / m.sum()

    return acc, acc, acc, acc

if __name__ == '__main__':
    word_index, word_cnt = create_word_index([TRAIN_DATA, DEV_DATA, TEST_DATA])

    wx, y, m = read_data(TRAIN_DATA, word_index)
    if USE_DEV:
        dev_wx, dev_y, dev_m = read_data(TEST_DATA, word_index)
        wx, y, m = np.vstack((wx, dev_wx)), np.vstack((y, dev_y)), np.vstack((m, dev_m))
    twx, ty, tm = read_data(DEV_DATA, word_index)
    char_index, char_cnt= create_char_index([TRAIN_DATA, DEV_DATA, TEST_DATA])
    x, cm = read_char_data(TRAIN_DATA, char_index)
    if USE_DEV:
        dev_x, dev_cm = read_char_data(TEST_DATA, char_index)
        x, cm = np.vstack((x, dev_x)), np.vstack((cm, dev_cm))
    tx, tcm = read_char_data(DEV_DATA, char_index)
    model = cnn_rnn.cnn_rnn(char_cnt, len(LABEL_INDEX), word_cnt)
    if LABELING_RATE < 1.0:
        ind = sample.create_sample_index(LABELING_RATE, x.shape[0])
        x, y, m, wx, cm = sample.sample_arrays((x, y, m, wx, cm), ind)
    model.add_data(x, y, m, wx, cm, None, tx, ty, tm, twx, tcm, None)
    model.build()
    word2embedding = read_word2embedding()
    model.set_embedding(word2embedding, word_index)
    model.train(evaluate)

