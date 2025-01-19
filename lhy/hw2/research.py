# %%
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# %%
columns=["wage per hour", "capital gains", "capital losses", "dividends from stocks"]
x_train = pd.read_csv("/Users/hhhhhh/work/code/machine-learning/lhy/hw2/data/X_train", index_col=0).drop(columns=columns)
y_train = pd.read_csv("/Users/hhhhhh/work/code/machine-learning/lhy/hw2/data/Y_train", index_col=0)
x_test = pd.read_csv("/Users/hhhhhh/work/code/machine-learning/lhy/hw2/data/X_test", index_col=0).drop(columns=columns)
# %%
# 先用pytorch的模型, 试一遍所有东西
# 先试试logistics regression
torch.manual_seed(3047)

class LR(nn.Module):
    def __init__(self, in_features):
        super(LR, self).__init__()
        self.linear = nn.Linear(in_features, 1, bias=True)

    def sigmoid(self, x):
        return 1. / (1 + torch.exp(-x))

    def predict(self, x):
        return (self(x) > 0.5).int()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

def zscore(x):
    val = ((x - np.mean(x)) / np.std(x))
    mask = x.max(axis=0) > 1
    newx = np.where(mask, val, x)
    return newx

t_x_train = torch.from_numpy(zscore(x_train.to_numpy(dtype=np.float32)))
t_y_train = torch.from_numpy(y_train.to_numpy(dtype=np.float32))


model = LR(t_x_train.size(1))
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
epoch = 100
batch_size = 10
N = t_x_train.size(0)
for i in range(epoch):
    order = torch.randperm(N)
    X = t_x_train[order]
    y = t_y_train[order]
    for n in range(N // batch_size):
        input = t_x_train[n * batch_size : (n + 1) * batch_size]
        label = t_y_train[n * batch_size : (n + 1) * batch_size]
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if n % 100 == 0:
            print(loss)
# %%
t_x_test = torch.from_numpy(zscore(x_test.to_numpy(dtype=np.float32)))
result = pd.DataFrame(model(t_x_test).detach().numpy().flatten(), columns=["label"])
(result > 0.5).astype(int).reset_index().to_csv("hw2_predict.csv",  header=["id","label"], index=False)
# %%
# TODO: 手动实现