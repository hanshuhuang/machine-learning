# %%
import time
from collections import defaultdict

import numpy as np
import pandas as pd

# %%
train = pd.read_csv("lhy/hw1/train.csv", index_col=[0, 1]).fillna(0)
test = pd.read_csv("lhy/hw1/test.csv", index_col=[0, 1]).fillna(0)

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
# 线形回归
class LinearRegression:
    def __init__(self, size:float, lr:float) -> None:
        np.random.seed(int(time.time()))
        self.size = size
        self.w = np.random.random_sample(size=size)
        self.b = np.random.random_sample(size=size)
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
        print(y_predict, w_grad, b_grad)
        print(self.w, self.b)
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
        return np.sum((y_predict - y) ** 2)

    def predict(self, x: np.ndarray) -> float:
        return self.w * x + self.b
# %%
# 进行回归训练
# 准备数据, 因为是知道9h, 预测第10小时, 我们先简单点, 每天的每小时都是一个切片
feature_set = [
    'AMB_TEMP',
    # 'CH4',
    'CO',
    'NMHC',
    'NO',
    'NO2',
    'NOx',
    'O3',
    'PM10',
    # 'PM2.5',
    # 'RAINFALL',
    'RH',
    'SO2',
    # 'THC',
    'WD_HR',
    'WIND_DIREC',
    'WIND_SPEED',
    'WS_HR'
]

# 预测值
label = "PM2.5"

# 每天24小时, 一共240天, 即5760个样本点
x_batch = []
y_batch = []
date_set = set()
for date in list(train.index.get_level_values("time")):
    if date in date_set:
        continue
    date_set.add(date)
    print(date)
    pm25 = train.query("index.get_level_values('time')==@date").droplevel(0).loc["PM2.5"].to_numpy()
    for i, t in enumerate(train.query("index.get_level_values('time')==@date").droplevel(0).loc[feature_set].T.to_numpy()):
        x_batch.append(t)
        y_batch.append(pm25[i])


# %%
# 训练模型
model = LinearRegression(size=len(feature_set), lr=0.01)
for i, x in enumerate(x_batch):
    print(i)
    y = y_batch[i]
    model.train(x, y)
    if i % 100 == 0:
        print(i, model.loss(x, y))
    break
# %%
