import torch.nn as nn
from config import *
from torchcrf import CRF
import torchcrf
import pandas as pd


class LSTM_CRF(nn.Module):
    def __init__(self):
        super(LSTM_CRF, self).__init__()
        vocab_size = len(pd.read_csv(VOCAB_PATH))
        label_size = len(pd.read_csv(LABEL_PATH))
        self.embed = nn.Embedding(vocab_size+1, EMBEDDING_DIM, WORD_PAD_ID)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True, bidirectional=True)
        self.classify = nn.Linear(HIDDEN_SIZE * 2, label_size+1)
        self.crf = CRF(label_size+1, batch_first=True)

    def _get_lstm_feature(self, input):
        out = self.embed(input)
        out, _ = self.lstm(out)
        return self.classify(out)

    def forward(self, input, mask):
        out = self._get_lstm_feature(input)
        return self.crf.decode(out, mask)

    def loss_fn(self, input, target, mask):
        y_pred = self._get_lstm_feature(input)
        return -self.crf.forward(y_pred, target, mask)


class CNN_LSTM_CRF(nn.Module):
    def __init__(self):
        super(CNN_LSTM_CRF, self).__init__()
        vocab_size = len(pd.read_csv(VOCAB_PATH))
        label_size = len(pd.read_csv(LABEL_PATH))

        self.embed = nn.Embedding(vocab_size + 1, EMBEDDING_DIM, WORD_PAD_ID)
        # 一维卷积是竖着走的 所以 in_channels 是最大的词的个数 而不是 词向量维度
        # 这里先写死in_channels 后面 后面改

        self.conv1d = nn.Conv1d(in_channels=180,
                                out_channels=180,
                                kernel_size=3,
                                bias=True,
                                padding=1)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True, bidirectional=True)
        self.classify = nn.Linear(HIDDEN_SIZE * 2, label_size+1)
        self.crf = CRF(label_size+1, batch_first=True)

    # def conv1d_self(self, in_channels, out_channels, kernel=3, bias=True, padding = 1):
    #     self.conv1d = nn.Conv1d(in_channels,
    #                             out_channels,
    #                             kernel,
    #                             bias,
    #                             padding)
    #     return self.conv1d()

    def _get_cnn_lstm_feature(self, inputs):
        # self.conv1d = self.conv1d_self(
        #     in_channels=max_len,
        #     out_channels=max_len
        # )
        out = self.embed(inputs)
        out = self.conv1d(out)
        out, _ = self.lstm(out)
        return self.classify(out)

    def forward(self, inputs, mask):

        out = self._get_cnn_lstm_feature(inputs)
        return self.crf.decode(out, mask)

    def loss_fn(self, inputs, target, mask):
        y_pred = self._get_cnn_lstm_feature(inputs)
        return -self.crf.forward(y_pred, target, mask)


if __name__ == '__main__':
    print(LSTM_CRF())
    print(CNN_LSTM_CRF())

    # exit()
    conv1 = nn.Conv1d(in_channels=96,
                      out_channels=96,
                      kernel_size=3,
                      bias=True,
                      padding=1)
    # batch_size = 32   256*35
    input = torch.randn(16, 96, 100)
    out = conv1(input)
    print(out.size())

