import torch

from utils import *
from model import *
from config import *

if __name__ == '__main__':
    text = ''
    word2id = get_vocab2index()
    input = torch.tensor([[word2id.get(w, WORD_UNK_ID) for w in text]]).to(device)
    # input = torch.LongTensor([[word2id.get(w, WORD_UNK_ID) for w in text] + [WORD_PAD_ID]*(180-len(text))]).to(device)
    # print(input,len(input),len(input[0]))
    # exit()
    # mask = torch.tensor([[1] * len(text) + [0]*(180-len(text))]).bool().to(device)
    mask = torch.tensor([[1] * len(text) ]).bool().to(device)
    # print(mask)
    # exit()
    model = torch.load('output/model/lstm_crf_model/model_608.pth')
    model.to(device)
    y_pred = model(input, mask)
    id2label = dict((v, k) for k, v in get_label2index().items())

    label = [id2label[l] for l in y_pred[0]]
    print(text)
    print(label)
    print(extract(label, text))