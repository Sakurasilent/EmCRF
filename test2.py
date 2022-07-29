import torch

from utils import *
from model import *
from config import *


# 计算 acc 实现在训练过程中调用该方法 掌握模型在测试集上的准确度
def calu_acc(model, path,batch_size): # 传入模型与验证数据集的路径 与验证书籍的batch_size
    with torch.no_grad():
        dataset = Dataset(DEV_PATH)
        loader = torch.utils.data.DataLoader(dataset, batch_size, collate_fn=collate_fn)
        y_true_list = []
        y_pred_list = []
        for b, (input, target, mask) in enumerate(loader):
            y_pred = model(input, mask)
            # 拼接返回值
            for lst in y_pred:
                y_pred_list += lst
            for y, m in zip(target, mask):
                y_true_list += y[m == True].tolist()

        # 整体准确率
        y_true_tensor = torch.tensor(y_true_list)
        y_pred_tensor = torch.tensor(y_pred_list)
        # print(y_true_tensor)
        # print(y_pred_tensor)
        accuracy = (y_true_tensor == y_pred_tensor).sum() / len(y_true_tensor)
        print('>> total:', len(y_true_tensor), 'accuracy:', accuracy.item())
        return accuracy.item()


if __name__ == '__main__':
    model = torch.load('output/model/cnn_lstm_crf_model/model_2966.pth')
    acc = calu_acc(model, TRAIN_PATH, 32)
    print(type(acc))