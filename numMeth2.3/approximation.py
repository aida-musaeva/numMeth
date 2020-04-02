import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import sympy
import math


x = Symbol('x')

def f(x):
    return (x + 3) * np.cos(x)

def leastSquares(func):
    x = Symbol('x')
    def Gauss(A, b):
        n = A.shape[0]
        for i in range(n):
            b[i] /= A[i][i]
            A[i] /= A[i][i]
            for j in range(i + 1, n):
                b[j] -= A[j][i] * b[i]
                A[j] -= A[j][i] * A[i]

        for i in range(n - 1, -1, -1):
            for j in range(i - 1, -1, -1):
                b[j] -= b[i] * A[j][i]
        return b

    def phi(dot):
        return Array([1, x, x ** 2, x ** 3]).subs(x, dot)

    X = np.linspace(-1, 1, 5)
    Q = np.array([[(phi(i))] for i in X]).reshape(5, 4)
    a = Gauss(Q.transpose().dot(Q), Q.transpose().dot(func(X)))
    return a.dot(phi(x))


def approxLegendre(func):
    Q = Array([1, x, (3 * x ** 2 - 1) / 2, (5 * x ** 3 - 3 * x) / 2])
    I1 = Array([integrate(func * Q[i]) for i in range(4)])
    I1 = I1.subs(x, 1) - I1.subs(x, -1)
    I1 = np.array([I1[i].evalf() for i in range(4)])
    I2 = Array([[integrate(Q[i] ** 2, x)] for i in range(4)])
    I2 = np.array(I2.subs(x, 1) - I2.subs(x, -1))
    C = np.array([I1[i] / I2[i] for i in range(4)])
    return C.dot(Q.reshape(4, 1))


def plotCompare(func, leastSq, pLegendre):
    xnew = np.arange(-1, 1.01, 0.01)
    fig1, ax = plt.subplots()
    line1, = ax.plot(xnew, func(xnew),
                     label=func.__name__+ "(x)")

    line2, = ax.plot(xnew, [leastSq.subs(x, xnew[i]) for i in range(np.prod(xnew.shape))],
                     label='lSq Polynomial (x)')
    line3 = ax.plot(xnew, [pLegendre.subs(x, xnew[i]) for i in range(np.prod(xnew.shape))],
                    label='approx by pLegendre (x)')

    ax.legend(loc='upper left')
    plt.show()


lSq = leastSquares(f)
print(lSq)
func = (x + 3) * sympy.cos(x)
pLeg = approxLegendre(func)
print(pLeg)
plotCompare(f, lSq, pLeg)