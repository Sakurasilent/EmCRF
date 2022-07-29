import pandas as pd
from config import *


# 生成词表
def generate_vocab():
    df = pd.read_csv(TRAIN_PATH, usecols=[0], names=['word'], sep=' ', error_bad_lines=False, engine='python')
    vocab_list = [WORD_PAD, WORD_UNK] + df['word'].value_counts().keys().tolist()
    vocab_list = vocab_list[:VOCAB_SIZE]
    # VOCAB_SIZE = len(vocab_list)
    # print(VOCAB_SIZE)
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab = pd.DataFrame(list(vocab_dict.items()))
    print(vocab)
    exit()
    vocab.to_csv(VOCAB_PATH, header=None, index=None)


# 生成标签对
def generate_label():
    df = pd.read_csv(TRAIN_PATH, usecols=[1], names=['label'], sep=' ', error_bad_lines=False, engine='python')
    label_list = df['label'].value_counts().keys().to_list()+['OTHER']
    label_dict = {v: k for k, v in enumerate(label_list)}
    label = pd.DataFrame(list(label_dict.items()))
    label.to_csv(LABEL_PATH, header=None, index=None)


if __name__ == '__main__':
    generate_vocab()
    # generate_label()
