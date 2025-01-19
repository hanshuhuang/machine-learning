# %%
import time
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch

# %%
train = pd.read_csv("lhy/hw1/train.csv", index_col=[0, 1]).fillna(0)
test = pd.read_csv("lhy/hw1/test.csv", header=None).replace('NR', 0).fillna(0)

# %%
# 作图探索, 目标是画出从某个标的0~23小时的图
"""
由图可以看出:
1. 有明显异常值, 可能需要去极值
2. 不同标的值差异很大, 需要做归一化
3. 某些标的的变动不明显, 可能一天都不怎么动, 比如2014/3/9 THC、CH4
"""
for feature in set(train.index.get_level_values("feature")):
    train.query("index.get_level_values('feature')==@feature").T.plot(legend=False, title=feature)

# %%
# 画出相关系数
# 计算相关性矩阵
pm25 = train.query("index.get_level_values('feature')=='PM2.5'").reset_index().drop(columns="feature").set_index("time")
for feature in set(train.index.get_level_values("feature")):
    others = train.query("index.get_level_values('feature')==@feature").reset_index().drop(columns="feature").set_index("time")
    print(feature, others.corrwith(pm25, axis=1).mean())

# %%
# 计算相关性矩阵, 目的是剔除掉相关性高的
# 现在我明确地要把 THC和CH4 丢出feature
# 然后 RAINFALL、NO相关系数很低, 可以待定
features = {}
for feature in set(train.index.get_level_values("feature")):
    features[feature] = train.query("index.get_level_values('feature')==@feature").reset_index().drop(columns="feature").set_index("time")

corrs = defaultdict(dict)
for k1 in features.keys():
    for k2 in features.keys():
        corrs[k1][k2] = features[k1].corrwith(features[k2], axis=1).mean()
pd.DataFrame(corrs)
# %%
# %%

# %%
# 进行回归训练
# 准备数据, 因为是知道9h, 预测第10小时, 我们先简单点, 每天的每小时都是一个切片
# 假设每天就取前9h

def zscore(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    x = (x - x_mean) / x_std
    return x

def get_feature_and_label(data):
    x_batch = []
    y_batch = []
    date_set = set()
    for date in list(data.index.get_level_values("time")):
        if date in date_set:
            continue
        date_set.add(date)
        i = 0
        pm25 = data.query("index.get_level_values('time')==@date").droplevel(0).loc["PM2.5"].iloc[10 + i]
        rawx = zscore(data.query("index.get_level_values('time')==@date").droplevel(0).iloc[:, i:9 + i].T.to_numpy().flatten())
        rawx[rawx < 0] = 0
        pm25 = max(0, pm25)
        x_batch.append(rawx)
        y_batch.append(pm25)

    # x_mean = np.mean(x_batch)
    # x_std = np.std(x_batch)
    # x_batch = (x_batch - x_mean) / x_std

    return x_batch, (y_batch)

x_train_batch, y_train_batch = get_feature_and_label(train)

# 获取测试集x_batch
x_test_batch = []
xpm25_mean = []
xpm25_std = []
xpm25_last = []
for i in range(240):
    raw_test_x = zscore(test[test[0] == f"id_{i}"].iloc[:, 2:].to_numpy(dtype=np.float64))
    raw_test_x[raw_test_x < 0] = 0
    x_test_batch.append(raw_test_x.flatten())
    x_pm25 = test[test[0] == f"id_{i}"].iloc[9, 2:].to_numpy(dtype=np.float64)
    xpm25_mean.append(np.mean(x_pm25))
    xpm25_std.append(np.std(x_pm25))
    xpm25_last.append(x_pm25[-1])


# %%
# 线形回归
class LinearRegression:
    def __init__(self, size:float, lr:float) -> None:
        np.random.seed(int(time.time()))
        self.size = size
        self.w = np.full((self.size, ), 1, dtype=np.float64)
        self.b = 1.0
        self.lr = lr

    def train(self, x: np.ndarray, y:np.ndarray):
        """
        训练就是传入x、y, 然后predict
        因为要使得loss最小, 所以对loss求导数

        偏导数w: 2 * (y - (wx + b)) * (-x)
        偏导数b: 2 * (y - (wx + b)) * (-1)
        """
        y_predict = self.predict(x)
        w_grad = 2 * (y - y_predict) * -x
        b_grad = 2 * (y - y_predict) * -1
        # print(f"{y=}")
        # print(f"{y_predict=}")
        # print(f"{w_grad=}")
        # print(f"{b_grad=}")
        self.w -= w_grad * self.lr
        self.b -= b_grad * self.lr
        assert not np.any(np.isnan(self.w))
        assert not np.any(np.isinf(self.w))
        assert not np.any(np.isnan(self.b))
        assert not np.any(np.isinf(self.b))

    def loss(self, x: np.ndarray, y: float) -> float:
        """
        (y - (wx + b))^2
        """
        assert len(x) == self.size
        y_predict = self.predict(x)
        # TODO: 添加正则项
        return (y - y_predict) ** 2

    def predict(self, x: np.ndarray) -> float:
        return np.sum(self.w * x) + self.b

# %%
class LinearRegressionV2:

    def __init__(self, alpha: float, wsize: int, factor:float) -> None:
        """
        多元线性回归
        """
        self.alpha = alpha
        self.wsize = wsize
        self.w = np.full((self.wsize, ), 1)
        self.b = -1
        self.factor = factor

    def __valid(self, x, y):
        assert len(x[0]) == self.wsize

    def loss(self, x: List[np.ndarray], y: np.ndarray) -> float:
        """
        mse as cost function
        sum((y - y1)^2) / 2m + factor * sum((self.w) ^ 2)
        """
        self.__valid(x, y)
        y_pred = np.array([self.predict(i) for i in x])
        # 除以2是因为偏导数解出来有2的系数
        return (((y_pred - y)**2).sum() + np.sum(self.w * self.w) * self.factor) / (2 * len(x))

    def train(self, x: List[np.ndarray], y: np.ndarray):
        grad = np.array([(self.predict(b) - y[i]) * b + self.w * self.factor for i, b in enumerate(x)])
        w_gradient = np.sum(grad, axis=0) / len(x)
        w_tmp = self.w - self.alpha * w_gradient
        assert np.all((~np.isnan(w_tmp)) & (~np.isinf(w_tmp)))
        b_gradient = np.sum([self.predict(b) - y for b in x]) / len(y)
        b_tmp = self.b - self.alpha * b_gradient
        assert np.all((~np.isnan(b_tmp)) & (~np.isinf(b_tmp)))
        self.w = w_tmp
        self.b = b_tmp

    def predict(self, x: np.ndarray) -> np.ndarray:
        res = (self.w * x).sum() + self.b
        return res
# %%
"""
y = w1*x + w2 * x^2 + b
"""
class LinearRegressionV3:

    def __init__(self, alpha: float, wsize: int, factor:float) -> None:
        """
        多元线性回归
        """
        self.alpha = alpha
        self.wsize = wsize
        self.w1 = np.full((self.wsize, ), 1)
        self.w2 = np.full((self.wsize, ), 1)
        self.b = -1
        self.factor = factor

    def __valid(self, x, y):
        assert len(x[0]) == self.wsize

    def loss(self, x: List[np.ndarray], y: np.ndarray) -> float:
        """
        mse as cost function
        sum((y - y1)^2) / 2m + factor * sum((self.w) ^ 2)
        (y - y1) * (-(x + 2w*x))
        """
        self.__valid(x, y)
        y_pred = np.array([self.predict(i) for i in x])
        # 除以2是因为偏导数解出来有2的系数
        return (((y_pred - y)**2).sum() + self.factor * np.sum(self.w1 * self.w1 + self.w2 * self.w2)) / (2 * len(x))

    def train(self, x: List[np.ndarray], y: np.ndarray):
        # w1
        grad = np.array([(self.predict(b) - y[i]) * b + self.factor * self.w1  for i, b in enumerate(x)])
        w1_gradient = np.sum(grad, axis=0) / len(x)
        w1_tmp = self.w1 - self.alpha * w1_gradient
        assert np.all((~np.isnan(w1_tmp)) & (~np.isinf(w1_tmp)))
        # b
        b_gradient = np.sum([self.predict(b) - y for b in x]) / len(y)
        b_tmp = self.b - self.alpha * b_gradient
        assert np.all((~np.isnan(b_tmp)) & (~np.isinf(b_tmp)))
        # w2
        w2_gradient = np.array([(self.predict(b) - y[i]) * b + 2 * self.w2 * b * self.factor  for i, b in enumerate(x)])
        w2_gradient = np.sum(w2_gradient, axis=0) / len(x)
        w2_temp = self.w2 - self.alpha * w2_gradient
        assert np.all((~np.isnan(w2_temp)) & (~np.isinf(w2_temp)))
        # update
        self.w1 = w1_tmp
        self.b = b_tmp

    def predict(self, x: np.ndarray) -> np.ndarray:
        res = (self.w1 * x + (self.w2 * (x*x))).sum() + self.b
        return res
# %%
class LinearRegressionV4:

    def __init__(self, alpha: float, wsize: int, factor:float) -> None:
        """
        多元线性回归
        """
        self.alpha = alpha
        self.wsize = wsize
        self.w = np.full((self.wsize, ), 1)
        self.b = -1
        self.factor = factor

    def __valid(self, x, y):
        assert len(x[0]) == self.wsize

    def loss(self, x: List[np.ndarray], y: np.ndarray) -> float:
        """
        mse as cost function
        sqrt(sum((y - y1)^2) / 2m) + factor * sum((self.w) ^ 2)
        1/(2 * np.sqrt( sum(y-y1)^2 / 2m ))*(1/m * (-x)) + 2*w*factor
        """
        self.__valid(x, y)
        y_pred = np.array([self.predict(i) for i in x])
        # 除以2是因为偏导数解出来有2的系数
        return np.sqrt(((y_pred - y)**2).sum()) / (2 * len(x)) + (np.sum(self.w * self.w) * self.factor) / (2 * len(x))

    def train(self, x: List[np.ndarray], y: np.ndarray):
        grad = np.array([(self.predict(b) - y[i]) * b + self.w * self.factor for i, b in enumerate(x)])
        w_gradient = np.sum(grad, axis=0) / len(x)
        w_tmp = self.w - self.alpha * w_gradient
        assert np.all((~np.isnan(w_tmp)) & (~np.isinf(w_tmp)))
        b_gradient = np.sum([self.predict(b) - y for b in x]) / len(y)
        b_tmp = self.b - self.alpha * b_gradient
        assert np.all((~np.isnan(b_tmp)) & (~np.isinf(b_tmp)))
        self.w = w_tmp
        self.b = b_tmp

    def predict(self, x: np.ndarray) -> np.ndarray:
        res = (self.w * x).sum() + self.b
        return res

# %%
model = LinearRegressionV2(wsize=x_train_batch[0].shape[0], alpha=0.00001, factor=0.01)
# batch = 200
for i in range(20000):
    # num = np.random.randint(low=1, high=len(x_train_batch))
    x_random_choose =x_train_batch
    y_random_choose = y_train_batch
    model.train(x_random_choose, y_random_choose)
    if i % 1000 == 0:
        print(i, model.loss(x_train_batch, y_train_batch))


# %%
# 在测试集做推理
result = []
for i, t in enumerate(x_test_batch):
    result.append(model.predict(t))

pd.DataFrame(result, index=[f"id_{i}" for i in range(240)]).reset_index().to_csv("predict.csv", header=["id","value"], index=False)

# %%
# TODO: 直接试试pytorch啥情况呢, 一个一个看自己为什么没别人做得好
# TODO: 试试np.lxxxx那个函数
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(x_train_batch[0].shape[0],1)

    def forward(self,x):
        out = self.linear(x)
        return out


class Linear_Model():
    def __init__(self):
        """
        Initialize the Linear Model
        """
        self.learning_rate = 0.001
        self.epoches = 100000
        self.loss_function = torch.nn.MSELoss(reduction="mean")
        self.create_model()

    def create_model(self):
        self.model = LinearRegression()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,  weight_decay=0.01)

    def train(self, x, y):
        for epoch in range(self.epoches):
            y_predict = self.model.forward(x)
            RMSE_loss = torch.sqrt(self.loss_function(y_predict, y))
            self.optimizer.zero_grad()
            RMSE_loss.backward()
            self.optimizer.step()
            if epoch % 500 == 0:
                print("epoch: {}, loss is: {}".format(epoch, RMSE_loss.item()))

    def test(self, x):
        return self.model.forward(x)

lm = Linear_Model()
lm.train(x=torch.from_numpy(np.array(x_train_batch).astype(np.float32)), y=torch.from_numpy(np.array([[y]for y in  y_train_batch]).astype(np.float32)))
result = lm.model.forward(torch.from_numpy(np.array(x_test_batch).astype(np.float32))).detach().numpy()
pd.DataFrame(result, index=[f"id_{i}" for i in range(240)]).reset_index().to_csv("predict.csv", header=["id","value"], index=False)
# %%
