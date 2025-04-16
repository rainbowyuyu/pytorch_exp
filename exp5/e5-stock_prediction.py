# rainbow_yu exp5 🐋✨
# 股票预测 LSTM

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch.optim.lr_scheduler as lr_scheduler
import os

from rnn_models import (
    RNN,
    GRU,
    LSTM,
)

# 读取IBM股票的数据
dates = pd.date_range('2010-01-02', '2017-10-11', freq='B')
df1 = pd.DataFrame(index=dates)
df_ibm = pd.read_csv("data/ibm.us.txt", parse_dates=True, index_col=0)
df_ibm = df1.join(df_ibm)
df_ibm[['Close']].plot(figsize=(15, 6))
df_ibm = df_ibm[['Close']]

df_ibm = df_ibm.ffill()
scaler = MinMaxScaler(feature_range=(-1, 1))
df_ibm['Close'] = scaler.fit_transform(df_ibm['Close'].values.reshape(-1, 1))


def load_data(stock, look_back):
    data_raw = stock.values  # 转化成numpy array的数据类型
    data = []  # 建立所有元素个数为look_back的序列(板书表示)
    for index in range(len(data_raw) - look_back):
        # 从data_raw中截取长度为look_back的数据
        data.append(data_raw[index: index + look_back])

    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size

    x_train = torch.Tensor(data[:train_set_size, :-1, :])
    y_train = torch.Tensor(data[:train_set_size, -1, :])
    x_test = torch.Tensor(data[train_set_size:, :-1])
    y_test = torch.Tensor(data[train_set_size:, -1, :])
    return [x_train, y_train, x_test, y_test]


look_back = 60
[x_train, y_train, x_test, y_test] = load_data(df_ibm, look_back)

model = LSTM(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1)
# model = RNN(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1)

loss_fn = nn.MSELoss()

# 设置动态学习率
dynamic_lr = False  # 设置为True以启用动态学习率

if dynamic_lr:
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)  # 初始学习率设为0.01
    scheduler = lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.1)  # 每10个epoch将学习率降低为原来的0.1倍
else:
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

# cfg
model_name = model.__class__.__name__
loss_name = loss_fn.__class__.__name__
opt_name = optimiser.__class__.__name__

folder_name = f'output/{model_name}_{loss_name}_{opt_name}_{look_back}_{dynamic_lr}'
os.makedirs(folder_name, exist_ok=True)

# 为生成的文件命名
file_name = f"output/{model_name}_{loss_name}_{opt_name}_{look_back}_{dynamic_lr}/cfg.csv"

# 准备保存的参数数据
parameters = {
    'Model Name': model_name,
    'Loss Function': loss_name,
    'Optimizer': opt_name,
    'Look Back': look_back,
    'Dynamic LR': dynamic_lr
}

# 将数据保存为csv
df_params = pd.DataFrame([parameters])
df_params.to_csv(file_name, index=False)

# cfg
num_epochs = 200
hist = np.zeros(num_epochs)

for t in range(num_epochs):
    y_train_pred = model(x_train)
    loss = loss_fn(y_train_pred, y_train)

    if t % 10 == 0 and t != 0:
        print("Epoch ", t, f"{loss_name}: ", loss.item())

    hist[t] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    # 如果启用了动态学习率，更新学习率
    if dynamic_lr:
        scheduler.step()  # 更新学习率

# 绘制损失曲线
plt.figure()
plt.plot(hist, label='Training loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'output/{model_name}_{loss_name}_{opt_name}_{look_back}_{dynamic_lr}/training_curve_with.png')  # 保存图片

# 可视化测试结果
y_test_pred = model(x_test)
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test.detach().numpy())

figure, axes = plt.subplots(figsize=(15, 6))
axes.xaxis_date()

axes.plot(df_ibm[len(df_ibm) - len(y_test):].index, y_test, color='red',
          label='Real IBM Stock Price')
axes.plot(df_ibm[len(df_ibm) - len(y_test):].index, y_test_pred, color='blue',
          label='Predicted IBM Stock Price')
plt.title('IBM Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('IBM Stock Price')
plt.legend()
plt.grid()
plt.savefig(f'output/{model_name}_{loss_name}_{opt_name}_{look_back}_{dynamic_lr}/prediction_plot.png')
