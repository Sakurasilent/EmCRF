import torch
from config import *
import pandas as pd


#  获取词表对 word2index
def get_vocab2index():
    df = pd.read_csv(VOCAB_PATH, names=['word', 'index'], sep=',')
    # vocab_dict = {df['index'].to_list: df['word'].to_list}
    return dict(df.values)


def get_label2index():
    df = pd.read_csv(LABEL_PATH, names=['label', 'index'], sep=',')
    return dict(df.values)


# 按照换行切分数据 且word 与 label 对应
def get_word_label(path):
    words, labels = [], []
    with open(path,'r', encoding='utf-8')as f:
        words_item = []
        labels_item = []
        while True:
            cont = f.readline().replace('\n', '')
            # 为空
            if not cont:
                if not words_item and not labels_item:
                    break
                else:
                    words.append(words_item)
                    labels.append(labels_item)
                    words_item, labels_item = [], []
            # 不为空
            else:
                cont = cont.split(' ')
                words_item.append(cont[0])
                labels_item.append(cont[1])
    return words, labels
# 构建dataset


class Dataset(torch.utils.data.Dataset):
    """返回长度 和对应位置的内容"""
    def __init__(self, path):
        super(Dataset, self).__init__()
        self.words, self.labels = get_word_label(path)
        self.word2index = get_vocab2index()
        self.label2index = get_label2index()

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item):
        words = self.words[item]
        labels = self.labels[item]
        # word 2 index
        words = [self.word2index.get(l, self.word2index.get('<UNK>')) for l in words]
        # label2 index
        labels= [self.label2index.get(l, self.label2index.get('OTHER')) for l in labels]
        return words, labels


# 构建 collate_fn 切分方法
def collate_fn(data):
    # 按照列长排序
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    # 获得最大列长
    # max_len = len(data[0][0])
    max_len = 180
    # print('max_len-----',max_len)
    input = []
    target = []
    mask = []
    for item in data:
        # 填充长度
        pad_len = max_len - len(item[0])
        input.append(item[0] + [WORD_PAD_ID] * pad_len)
        target.append(item[1] + [LABEL_OTH_ID] * pad_len)
        mask.append([1] * len(item[0]) + [0] * pad_len)
    return torch.tensor(input).to(device), torch.tensor(target).to(device), torch.tensor(mask).bool().to(device)


def extract(label, text):
    i = 0
    res = []
    while i < len(label):
        if label[i] != 'O':
            prefix, name = label[i].split('-')
            start = end = i
            i += 1
            while i < len(label) and ((label[i] == 'M-' + name) or (label[i] == 'E-' + name)) :
                end = i
                i += 1
            res.append([name, text[start:end + 1]])
        else:
            i += 1
    return res


if __name__ == '__main__':
    id2label = dict((v, k) for k, v in get_label2index().items())
    print(id2label)
    exit()
    word2id = get_vocab2index()
    print(word2id)