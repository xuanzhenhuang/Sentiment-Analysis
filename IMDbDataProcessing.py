import csv
import os
import re
import numpy as np
import pandas as pd



# 获取当前工作目录的绝对路径
# current_directory = os.getcwd()
# print("当前工作目录的绝对路径:", current_directory)


def get_data():
    pos1, pos2 = os.listdir('IMDbDataset/aclImdb/test/pos'), os.listdir('IMDbDataset/aclImdb/train/pos')
    neg1, neg2 = os.listdir('IMDbDataset/aclImdb/test/neg'), os.listdir('IMDbDataset/aclImdb/train/neg')
    pos_all, neg_all = [], []
    for p1, n1 in zip(pos1, neg1):
        with open('IMDbDataset/aclImdb/test/pos/' + p1, encoding='utf8') as f:
            pos_all.append(f.read())
        with open('IMDbDataset/aclImdb/test/neg/' + n1, encoding='utf8') as f:
            neg_all.append(f.read())
    for p2, n2 in zip(pos2, neg2):
        with open('IMDbDataset/aclImdb/train/pos/' + p2, encoding='utf8') as f:
            pos_all.append(f.read())
        with open('IMDbDataset/aclImdb/train/neg/' + n2, encoding='utf8') as f:
            neg_all.append(f.read())
    datasets = np.array(pos_all + neg_all)
    labels = np.array([1] * 25000 + [0] * 25000)
    return datasets, labels


def shuffle_process():
    sentences, labels = get_data()
    # Shuffle
    shuffle_indexs = np.random.permutation(len(sentences))
    datasets = sentences[shuffle_indexs]
    labels = labels[shuffle_indexs]
    return datasets, labels


def save_process():
    datasets, labels = shuffle_process()
    sentences = []
    punc = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]'
    for sen in datasets:
        sen = sen.replace('\n', '')
        sen = sen.replace('<br /><br />', ' ')
        sen = re.sub(punc, '', sen)
        pre_text = preprocess_text(sen)
        sentences.append(pre_text)

    # Save
    df = pd.DataFrame({'labels': labels, 'sentences': sentences})
    df.to_csv("imdbdatasets2.csv", index=False,encoding="utf-8")


if __name__ == '__main__':
    model = save_process()







