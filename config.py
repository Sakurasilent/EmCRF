import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 1e-4
WORD_PAD = '<PAD>'
WORD_PAD_ID = 0
WORD_UNK = '<UNK>'
WORD_UNK_ID = 1
LABEL_O_ID = 0
LABEL_OTH_ID = 26

VOCAB_SIZE = 3000
VOCAB_PATH = './output/vocab.txt'

DEV_PATH = './input/data/ResumeNER/dev.char.bmes'
TRAIN_PATH = './input/data/ResumeNER/train.char.bmes'
TEST_PATH = './input/data/ResumeNER/test.char.bmes'
LABEL_PATH = './output/label.txt'
# 词向量 100 维
EMBEDDING_DIM = 100
# LSTM 隐藏层输出维度
HIDDEN_SIZE = 128
MODEL_DIR = './output/model/'
MODEL_CNN_DIR = './output/model/cnn_lstm_crf_model/'
MODEL_LSTM_DIR = './output/model/lstm_crf_model/'
MODEL_2CNN_LSTM_DIR = './output/model/cnn2_lstm_crf_model/'