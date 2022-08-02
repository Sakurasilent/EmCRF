from utils import *
from model import *
from config import *
from test2 import *

if __name__ == '__main__':
    dataset = Dataset(TRAIN_PATH)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    # model = LSTM_CRF()
    # model = CNN_LSTM_CRF()
    model = Mod_CNN_LSTM_CRF()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    acc = 0.8
    for e in range(1000):
        for b, (input, target, mask) in enumerate(loader):
            y_pred = model(input, mask)
            loss = model.loss_fn(input, target, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if b % 10 == 0:
                print('>> epoch:', e, 'loss:', loss.item())
        if e >= 0:
            acc_cur = calu_acc(model, DEV_PATH, 32)
            if acc_cur > acc:
                acc = acc_cur
                # print(acc)
                torch.save(model, MODEL_LSTM_DIR + f'model_{e}.pth')
        # torch.save(model, MODEL_DIR + f'model_{e}.pth')
