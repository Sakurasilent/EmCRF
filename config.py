import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 1e-4
WORD_PAD = '<PAD>'
WORD_PAD_ID = 0
WORD_UNK = '<UNK>'

LABEL_O_ID = 0
LABEL_OTH_ID = 26

VOCAB_SIZE = 3000
VOCAB_PATH = './output/vocab.txt'

TRAIN_PATH = './input/data/ResumeNER/dev.char.bmes'
LABEL_PATH = './output/label.txt'
# 词向量 100 维
EMBEDDING_DIM = 100
# LSTM 隐藏层输出维度
HIDDEN_SIZE = 128
MODEL_DIR = './output/model/'