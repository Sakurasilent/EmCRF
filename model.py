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
    '''一维卷积 卷的 是所有 每个词向量 所有 的 特征
    修改成二位卷积纸卷每个词以及周围词向量的特征'''
    def __init__(self):
        super(CNN_LSTM_CRF, self).__init__()
        vocab_size = len(pd.read_csv(VOCAB_PATH))
        label_size = len(pd.read_csv(LABEL_PATH))

        self.embed = nn.Embedding(vocab_size + 1, EMBEDDING_DIM, WORD_PAD_ID)
        # 一维卷积是竖着走的 所以 in_channels 是最大的词的个数 而不是 词向量维度
        # 这里先写死in_channels 后面 后面改 写死是应为 每个传入的batch的句子长度（词/字数）的最大长度是不一样的
        # 只有取epoch最大的才不会报错

        self.conv1d = nn.Conv1d(in_channels=180,
                                out_channels=180,
                                kernel_size=3,
                                bias=True,
                                padding=1,)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True, bidirectional=True)
        self.classify = nn.Linear(HIDDEN_SIZE * 2, label_size+1)
        self.crf = CRF(label_size+1, batch_first=True)

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

# 修改后的Cnn
class Mod_CNN_LSTM_CRF(nn.Module):
    def __init__(self):
        super(Mod_CNN_LSTM_CRF, self).__init__()
        vocab_size = len(pd.read_csv(VOCAB_PATH))
        label_size = len(pd.read_csv(LABEL_PATH))
        self.embed = nn.Embedding(vocab_size + 1, EMBEDDING_DIM, WORD_PAD_ID)
        # 改为二维卷积
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=EMBEDDING_DIM, kernel_size=(3, 100), padding=1),
            nn.ReLU(),
            nn.AvgPool2d((1, 3)),
        )
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True, bidirectional=True, num_layers=2)
        self.classfier = nn.Linear(HIDDEN_SIZE*2, label_size+1)
        self.crf = CRF(label_size + 1, batch_first=True)

    def forward(self, inputs, mask):
        out = self._get_feature(inputs)
        return self.crf.decode(out, mask)

    def _get_feature(self, inputs):
        out = self.embed(inputs)
        out = out.unsqueeze(1)
        out = self.conv(out)
        out = out.reshape(out.size()[0],out.size()[2],out.size()[1])
        out, _ = self.lstm(out)
        return self.classfier(out)

    def loss_fn(self, inputs, target, mask):
        y_pred = self._get_feature(inputs)
        return -self.crf.forward(y_pred, target, mask)


if __name__ == '__main__':
    # print(LSTM_CRF())
    # print(CNN_LSTM_CRF())
    print(Mod_CNN_LSTM_CRF())
    exit()
    conv1 = nn.Conv1d(in_channels=96,
                      out_channels=96,
                      kernel_size=3,
                      bias=True,
                      padding=1)

    conv2 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, 100), padding=1)
    # batch_size = 32   256*35
    print('conv2', conv2)
    input = torch.randn(16, 96, 100)
    out = conv1(input)
    print(out.size())
    input = input.unsqueeze(1)
    print(input.size())
    out2 = conv2(input)
    print(out2.size())
    avg = nn.AvgPool2d((1, 3))
    out2= avg(out2)
    print(len(out2), len(out2[0]), len(out2[1]), len(out2[2]))
    print(type(out2.size()[0]))
    print(out2.size())
    out2 = out2.reshape(16, 96, 100)
    print(out2.size())


