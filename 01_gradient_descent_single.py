# coding: utf-8
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

# --- 線形回帰(単特徴) ---
# ある都市に出店するか否かの判断する為
# 人口(x)から利益(y)を予測する
# データセットはそれぞれ10000で割ったfloat。x/=10000 y/=10000


def plot_data(X, y):
    plot.plot(X, y, 'rx', markersize=10)
    plot.ylabel('Profit in $10,000s')
    plot.xlabel('Population of City in 10,000s')
    plot.show()


def compute_cost(X, y, theta):
    m = y.size
    costs = (X.dot(theta) - y) ** 2
    return costs.sum() / (2.0 * m)


def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        h = X.dot(theta)
        errors = h - y
        delta = X.T.dot(errors)
        theta -= (alpha / m) * delta
        J_history[i] = compute_cost(X, y, theta)
    return (theta, J_history)


if __name__ == '__main__':
    # データのロード、変数の初期化
    data1 = np.loadtxt('./assets/data1.txt', delimiter=',')
    X = data1[:,0]
    y = data1[:,1]
    m = y.size
    theta = np.zeros(2)

    # トレーニングデータの可視化
    plot_data(X, y)

    # インターセプト項の挿入、ループ回数、学習率
    X = np.vstack((np.ones(m), X)).T
    iterations = 1500
    alpha = 0.01

    # 最急降下法
    (theta, J_history) = gradient_descent(X, y, theta, alpha, iterations)

    # 最急降下法の可視化
    plot.plot(X[:,1], y, 'rx', markersize=10)
    plot.ylabel('Profit in $10,000s')
    plot.xlabel('Population of City in 10,000s')
    plot.plot(X[:,1], X.dot(theta), '-')
    plot.show()

    # 予測値の出力
    population = 3.5
    predict = np.array([1, population]).dot(theta)
    print("人口{population}人の都市で出店した場合の利益予測値は{predict}です。".format(population=population*10000, predict=predict*10000))
    population = 7.0
    predict = np.array([1, population]).dot(theta)
    print("人口{population}人の都市で出店した場合の利益予測値は{predict}です。".format(population=population*10000, predict=predict*10000))

    # J(theta_0, theta_1)の可視化
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((theta0_vals.size, theta1_vals.size))
    for i in range(theta0_vals.size):
        for j in range(theta1_vals.size):
            t = np.array([theta0_vals[i], theta1_vals[j]])
            J_vals[i, j] = compute_cost(X, y, t)
    J_vals = J_vals.transpose()

    # 等高線図
    plot.contour(theta0_vals, theta1_vals, J_vals, levels=np.logspace(-2, 3, 20))
    plot.plot(theta[0], theta[1], 'rx')
    plot.show()

    # 3D surface
    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')
    t0, t1 = np.meshgrid(theta0_vals, theta1_vals)
    ax.plot_surface(t0, t1, J_vals)
    plot.show()
