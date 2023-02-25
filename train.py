import torch.optim.lr_scheduler

from utils import *
from model import *
from config import *
from test2 import *

if __name__ == '__main__':
    dataset = Dataset(TRAIN_PATH)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=90,
        shuffle=True,
        collate_fn=collate_fn
    )
    model = LSTM_CRF()
    # model = CNN_LSTM_CRF()
    # model = Mod_CNN_LSTM_CRF()
    model.to(device)
    # 两组跳参数的方法
    # lambda1 = lambda epoch: epoch // 30
    # lambda2 = lambda epoch: epoch ** 0.95
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # print(LR)
    # print(optimizer.state_dict()['param_groups'][0]['lr'])
    # 这里的lr_lambda没搞明白什么意思
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    acc = 0.9
    for e in range(1000):
        for b, (input, target, mask) in enumerate(loader):
            y_pred = model(input, mask)
            loss = model.loss_fn(input, target, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if b % 10 == 0:
                print('>> epoch:', e, 'batch', b, 'loss:', loss.item())

        if e >= 0:
            acc_cur = calu_acc(model, DEV_PATH, 32)
            if acc_cur > acc:
                acc = acc_cur
                # print(acc)
                torch.save(model, MODEL_LSTM_DIR + f'model_{e}.pth')
                # torch.save(model, MODEL_2CNN_LSTM_DIR + f'model_{e}.pth')
        scheduler.step()
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        # torch.save(model, MODEL_DIR + f'model_{e}.pth')
