import numpy as np
#from scipy.optimize import leastsq

def f(x1, x2, p):
    k0, k1, k2 = p
    return k0 +k1*x1 + k2*x2
def cost(p,x1,x2,y):
    return y-f(x1,x2, p)

def featureNormalize(X):
    """
    归一化处理(Z-Score):x = (x-mu)/sigma
    """
    X_norm = X
    mu = np.mean(X)
    sigma = np.std(X)
    X_norm = (X-mu)/sigma
    return X_norm

def gradientDescent(rooms, price, area):
    """
    数据差别较大，先做归一化处理
    """
    x2 = featureNormalize(rooms)
    x1 = featureNormalize(area)
    y = featureNormalize(price)

    diff = [0, 0, 0]
    theta = [1, 1, 1]

    epochs = 1000000#最大迭代周期
    lr = 0.001#学习系数
    epsilon = 0.00001  # 暂停阈值
    error0 = error1 = 0
    m = len(price)
    count = 0
    while True:
        count += 1
        diff[0] = diff[1] = diff[2] = 0
        for j in range(m):
            diff[0] += ((theta[0] + theta[1] * x1[j] + theta[2] * x2[j]) - y[j]) * 1
            diff[1] += diff[0] * x1[j]
            diff[2] += diff[0] * x2[j]

        theta[0] = theta[0] - lr * (1 / m) * diff[0]
        theta[1] = theta[1] - lr * (1 / m) * diff[1]
        theta[2] = theta[2] - lr * (1 / m) * diff[2]


        error1 = 0
        for e in range(m):
            error1 += 1 / 2 * (y[e] - (theta[0] + theta[1] * x1[e] + theta[2] * x2[e])) ** 2
        if abs(error1 - error0) < epsilon:
            break
        else:
            error0 = error1
        if count > epochs:
            break
    print(count)
    return theta




area = [2104, 1600, 2400, 1416, 3000]
price = [400, 330, 369, 342, 540]
rooms = [3, 3, 3, 2, 4]

x1=np.array([2104,1600,2400,1416,3000])

x2=np.array([3,3,3,2,4])

y=np.array([400,330,369,232,540])


theta = gradientDescent(rooms, price, area)

a1 = (20000-np.mean(area))/np.std(area)
r1 = (30-np.mean(rooms))/np.std(rooms)
p1 = theta[0]+theta[1]*a1+theta[2]*r1

p1 = p1*np.std(price) + np.mean(price)

print(p1)
