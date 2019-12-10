import numpy as np
import sympy
from sympy import *
x = Symbol('x')
X = np.array([1, 5, 2])
Y = np.array([13, 2, 3])
dY = np.array([10, 2, -3])
d2Y = np.array([23, 130])
omega = collect(expand((x-1) * (x-2) * (x-5)), x)
def HermitPolynomial(X,dY, d2Y):
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

    def P4():

        P4 = np.array([(dY[i] - pLagrange(X, Y).diff(x).subs(x, X[i])) / omega.diff(x).subs(x, X[i]) for i in range(3)])
        dP4 = np.array([(d2Y[i] - pLagrange(X, Y).diff(x).diff(x).subs(x, X[i]) - omega.diff(x).diff(x).subs(x, X[i]) *
                         P4[i]) / (2 * omega.diff(x)) for i in range(2)])

        def P1():
            P1 = np.array(
                [(dP4[i] - pLagrange(X, P4).diff(x).subs(x, X[i])) / omega.diff(x).subs(x, X[i]) for i in range(2)])
            coeff = Array([[-1255/13824 ], [-23755/13824]])
            P1 = coeff[0] * x + coeff[1]
            print(simplify(P1))
            return P1

        res = pLagrange(X, P4) + omega * P1()
        return res

    H = pLagrange(X, Y) + omega * P4()
    return simplify(H)

H = HermitPolynomial(X, dY, d2Y)
print(H)