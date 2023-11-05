# %%
"""
- 随机生成一个 y = w * x + b的函数, 使用该函数生成一组数据
- 使用梯度下降的方法，在随机初始化 w, b权重后, 开始训练, 拟合w、b
"""
from typing import Tuple, List
import numpy as np
import time
import pandas as pd
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

global_w = None
global_b = None


def generate() -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray]:
    """
    return x_train, y_train, x_test, y_test
    """
    # 初始化w、b
    size = 10
    w = np.random.random(size)
    b = np.random.randint(10, 100)
    global global_b
    global_b = b
    # 构造数据
    batch = 1000
    x = np.array([np.random.random(size) * [np.random.randint(-50, 40) for i in range(size)] for _ in range(batch)])
    y = np.array([np.sum(x[i] * w) + b for i in range(len(x))])
    trainidx = int(0.7 * batch)
    global global_w
    global_w = w
    print(f"generate feature={len(w)} {w=} {b=} {batch=} {trainidx=}")
    return x[:trainidx], y[:trainidx], x[trainidx:], y[trainidx:]


# %%
class LinearRegression:

    def __init__(self, alpha: float, wsize: int) -> None:
        """
        多元线性回归
        """
        self.alpha = alpha
        self.wsize = wsize
        self.w = np.full((self.wsize, ), 100000)
        self.b = -10000

    def __valid(self, x, y):
        assert len(x[0]) == self.wsize

    def cost(self, x: List[np.ndarray], y: np.ndarray) -> float:
        """
        mse as cost function
        sum((y - y1)^2) / 2m
        """
        self.__valid(x, y)
        y_pred = np.array([self.predict(i) for i in x])
        # 除以2是因为偏导数解出来有2的系数
        return ((y_pred - y)**2).sum() / (2 * len(x))

    def train(self, x: List[np.ndarray], y: np.ndarray):
        grad = np.array([(self.predict(b) - y[i]) * b for i, b in enumerate(x)])
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
x_train, y_train, x_test, y_test = generate()
lr = LinearRegression(alpha=0.001, wsize=len(x_train[0]))
print(f"random init linear regression to {lr.w=} {lr.b=}")
# %%
i = 0
cost_of_train = []
cost_of_test = []
while True:
    lr.train(x_train, y_train)
    if i % 10 == 0:
        cost_of_train.append(lr.cost(x_train, y_train))
        cost_of_test.append(lr.cost(x_test, y_test))
        if len(cost_of_train) > 1000:
            cost_of_train = cost_of_train[-1000:]
        if len(cost_of_test) > 1000:
            cost_of_test = cost_of_test[-1000:]
        if i % 100:
            print(f" {lr.b=} {global_b=} {lr.w=} {global_w=} {cost_of_train[-1]=} {cost_of_test[-1]=}")
            # clear_output(wait=True)
            # plt.close()
            # pd.DataFrame({"train":cost_of_train, "test":cost_of_test}).plot()
            # plt.show()
            time.sleep(0.5)
    i += 1
# %%
