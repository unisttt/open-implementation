import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 数値微分
def numerical_deff(f, x):
    h = 1e-4    # 0.0001 = 10^-4
    return ((f+x) - (f-x) / 2*h)

# 数値偏微分
def numerical_gradient(f, x):
    h = 1e-4    # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)の計算
        x[idx] = tmp_val -h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val    # 値を元に戻す
    
    return grad

def func1(x, y):
    return x**2 + y**2

def plt_show3D():
    x = np.arange(-3.0, 3.0, 0.1)
    y = np.arange(-3.0, 3.0, 0.1)

    X, Y = np.meshgrid(x, y)
    print(X.shape)
    Z = func1(X, Y)
    print(Z.shape)
    fig = plt.figure()  # 2d
    ax = Axes3D(fig)    # 3d
    ax.set_label('X')
    ax.set_label("y")
    ax.set_label("f(x, y")

    ax.plot_wireframe(X, Y, Z)
    #plt.show()

if __name__ == "__main__":
    plt_show3D()