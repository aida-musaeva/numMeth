import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import math

def f(x):
    return (x ** 2) * np.cos(2 * x) + 1

def h(x):
    return np.abs(x)*((x ** 2) * np.cos(2 * x) + 1)

def pLagrange(X, Y):
    x = symbols('x')
    L = 0
    for j in range(np.prod(X.shape)):
        l = 1
        l_j = 1
        for i in range(np.prod(X.shape)):
            if i == j:
                l *= 1
                l_j *= 1
            else:
                l *= (x - X[i])
                l_j *= (X[j] - X[i])
        L += Y[j] * l / l_j
    return collect(expand(L), x)

def makeData(func, flag = 'equidistant', n = 10, a = -10, b = 10):
    if flag == 'equidistant':
        X = np.arange(a, b + 1, (b - a) / n)
        Y = np.array(func(X))
        return X, Y
    elif flag == 'minError':
        X = np.array([((b/2-a/2)*math.cos((2*i+1)*math.pi/(2*n+2)) + (b+a)/2) for i in range(n+1)])
        Y = np.array(func(X))
        return X, Y
    print('Incorrect flag')
    return -1

def plotCompare(func, L1, L2):
    xnew = np.arange(-10, 11, 0.01)
    fig1, ax = plt.subplots()
    if func.__name__ == "h":
        ax.set_ylim([-2500, 2500])
    else:
        ax.set_ylim([-250, 250])
    line1, = ax.plot(xnew, func(xnew),
                     label=func.__name__+ "(x)")

    line2, = ax.plot(xnew, [L1.subs(x, xnew[i]) for i in range(np.prod(xnew.shape))],
                     label='Lagrange polynomial (eq)')
    line3 = ax.plot(xnew, [L2.subs(x, xnew[i]) for i in range(np.prod(xnew.shape))],
                    label='Lagrange polynomial (minErr)')
    ax.legend(loc='upper left')
    plt.show()

x = symbols('x')
dot1 = makeData(h, 'equidistant', 14)
L1 = pLagrange(dot1[0], dot1[1])
print(L1)
dot2 = makeData(h, 'minError', 14)
L2 = pLagrange(dot2[0], dot2[1])
print(L2)
plotCompare(h, L1, L2)

dot3 = makeData(f, 'equidistant', 14)
dot4 = makeData(f, 'minError', 14)
plotCompare(f, pLagrange(dot3[0], dot3[1]), pLagrange(dot4[0],dot4[1]))





