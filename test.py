import torch

from utils import *
from model import *
from config import *


if __name__ == '__main__':
    dataset = Dataset(TRAIN_PATH)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    with torch.no_grad():
        model = torch.load('output/model/cnn_lstm_crf_model/model_2966.pth')
        for b, (input, target, mask) in enumerate(loader):
            y_pred = model(input, mask)
            loss = model.loss_fn(input, target, mask)

            print('>> batch:', b, 'loss:', loss.item())