import numpy as np
import math
import pylab
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sympy import *
x, y, z  = symbols('x y z')
f = 2 * x ** 2 + 4.1 * y ** 2 + 5.1 * z ** 2 + x * y - y * z + x * z + x - 2 * y + 3 * z + 11


def grad(f,dot):
    x, y, z = symbols('x y z')
    a = np.array([x, y, z])
    grad0 = np.array([f.diff(i) for i in a])
    return np.array([grad0[i].subs([(x,dot[0]), (y,dot[1]), (z,dot[2])]) for i in range(3)])


def norm(a):
    return math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)


def steepestGradDesc(f, x_prev=np.array([0, 0, 0]), eps = 0.000001):
    A = np.array([[4, 1, 1], [1, 8.2, -1], [1, -1, 10.2]])
    x, y, z = symbols('x y z')
    a = np.array([x, y, z])
    phi_prev = - ((grad(f,x_prev).transpose()).dot(grad (f,x_prev)))/((grad(f,x_prev).transpose()).dot(A.dot((grad(f,x_prev)))))
    x_next = x_prev + phi_prev * grad(f,x_prev)
    i = 0
    while (norm(x_next-x_prev)>eps):
        x_prev = x_next
        phi_prev = - ((grad(f, x_prev).transpose()).dot(grad(f, x_prev))) / ((grad(f, x_prev).transpose()).dot(A.dot((grad(f, x_prev)))))
        x_next = x_prev + phi_prev * grad(f, x_next)
        print(x_next)
        i+=1
    return(x_next,i)

def steepestCoordinateDesc(f, x_prev=np.array([0,0,0]), eps = 0.000001):
    A = np.array([[4, 1, 1], [1, 8.2, -1], [1, -1, 10.2]])
    x, y, z  = symbols('x y z')
    a = np.array([x, y, z])
    def e(i):
        return np.array([int(j == i) for j in range(3)]).transpose()
    phi_prev = - (e(0).dot(grad(f,x_prev)))/(e(0).transpose().dot(A.dot(e(0).transpose())))
    x_next = x_prev + phi_prev * e(0).transpose()
    iter = 1
    i = 0
    while (norm(x_next-x_prev)>eps):
        x_prev = x_next
        i+=1
        i%=3
        phi_prev = - (e(i).dot(grad(f,x_prev)))/(e(i).transpose().dot(A.dot(e(i).transpose())))
        x_next = x_prev + phi_prev * e(i).transpose()
        print(x_next)
        iter+=1
    return(x_next, iter)


xOpt, it = steepestCoordinateDesc(f)
print(xOpt, it)