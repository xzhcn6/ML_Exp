
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
# 导入数据集
data = pd.read_csv('./Input/data.csv')
print(data)

# buses: 城镇公交车运营数量
buses = data['Bus']

# pdgp: 人均国民生产总值
pgdp = data['PGDP']

#Plot 为绘图函数，同学们可以利用这个函数建立画布和基本的网格
def Plot():
    plt.figure()
    plt.title('Data')
    plt.xlabel('Buses')
    plt.ylabel('PGDP(Yuan)')
    plt.grid(True)
    return plt

# 绘制pgdp与buses之间的关系
plt = Plot()
plt.plot(buses, pgdp, 'k.')
plt.show()
