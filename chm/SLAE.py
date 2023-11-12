#1
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as li
from scipy.linalg import lu
# A = np.array([[22.8, -1.4, 13.2, -3.6, 52.5, -0.02625],
#              [24.15, 2.45, -25.2, -2.7, -22.5, 0.02625],
#              [82.5, -3.5, 16.8, -13.23, 0.5, -0.13125],
#              [390, 0.7, -10.4, 0.9, -6.25, -0.18375],
#              [7.5, -51.3, 9.36, 3.6, 0.8, -0.021],
#              [0.62, 2.1, -21.6, -3.69, 12.5, 12.235]])
b = np.array([5, -12, 75, -7, 9, 5])
np.set_printoptions(precision=2, suppress=True, linewidth=120)
# print('A= ', A, 'b= ', b, sep='\n')
# print('det(A)= ', li.det(A))
#2
# del A
# A = np.array([[75, -1.4, 13.2, -3.6, 52.5, -0.02625],
#              [24.15, 82.45, -25.2, -2.7, -22.5, 0.02625],
#              [82.5, -3.5, 116.8, -13.23, 0.5, -0.13125],
#              [390, 0.7, -10.4, 410.9, -6.25, -0.18375],
#              [7.5, -51.3, 9.36, 3.6, 80.8, -0.021],
#              [0.62, 2.1, -21.6, -3.69, 12.5, 52.235]])
A = np.array([[10, 3, 2],
              [1, -10, 7],
              [3, 4, -10]])
# print(f'new A = \n{A}', f'det A = {li.det(A)}', sep='\n')
D = np.diag(np.diag(A))
A_0 = A - D
B = np.dot(-1*li.inv(D), A_0)
# print(f'Norm(B) = {li.norm(B, ord=np.inf)} < 1')
#3
#Kramar
# X = []
# for i in range(6):
#     j = A.copy()
#     j[:, i] = b
#     X.append(li.det(j)/li.det(A))
# print(f'X = {X}')
# def NAK(ep):
#     # X = jacob(ep, init)[1]
#     N = np.dot(A, X) - b
#     return (sum([abs(N[i]) for i in range(6)])/6)
# print('Kрамер = ', '{0:.20f}'.format(NAK(0.01)))
#InverseMatrix Method

# del X
X = list(np.dot(li.inv(A), b))
# print(f'X = {X}')
def NAI(ep):
    N = np.dot(A, X) - b
    return (sum([abs(N[i]) for i in range(6)])/6)
print('Об.М-ця = ', '{0:.20f}'.format(NAI(0.01)))
#LU-Decomposition
p, l, u = lu(A)
# print('PA = ', np.dot(p, A))
# print('LU = ', np.dot(l, u))
y = np.dot(li.inv(l), np.dot(p, b))
# print('Розв\'язок Ly = Pb: ', list(y))
del X
X = np.dot(li.inv(u), y)
# print('Остаточний результат: ', list(X))
def NAL(ep):
    N = np.dot(A, X) - b
    return (sum([abs(N[i]) for i in range(6)])/6)
print('LU-Розклад = ', '{0:.20f}'.format(NAL(0.01)))
del X
#4
g = np.dot(li.inv(D), b)
def jacob(ep, init):
    epk = 10
    X = init.copy()
    k = 0
    while ep < epk:
        y = np.dot(B, X)+g
        epk = max([abs(X[i] - y[i]) for i in range(len(X))])
        X = y
        k += 1
    return [k, X, epk]
k, X, epsk = jacob(0.1, g)
# print(f'Розв\'язок: {list(X)}, \nКількість ітерацій: {k}, \nДійсне розходження: {epsk}')
# print(list(X), k, epsk, sep='\n')
def testJacob(g):
    for i in range(1, 10):
        k, X, epsk = jacob(10**(-i), g)
        print(f'Розв\'язок = {list(X)}, К-ть ітерацій = {k}, Дійсне розходження = {epsk}')
# testJacob(g)
Zero = [0, 0, 0, 0, 0, 0]
# testJacob(Zero)
def NAJ(ep, init):
    X = jacob(ep, init)[1]
    N = np.dot(A, X) - b
    return (sum([abs(N[i]) for i in range(6)])/6)
print(f'Якобі = {NAJ(0.01, Zero)}') #нев'язок метода Якобі
#Zeidel
def Zeidel(ep, init):
    epk = 10
    k = 0
    X = init.copy()
    y = 0
    while ep < epk:
        epk = 0
        for i in range(len(X)):
            s = 0
            for j in range(len(X)):
                s += B[i, j] * X[j]
            y = g[i] + s
            epk = max(abs(y - X[i]), epk)
            X[i] = y
        k += 1
    return [k, X, epk]
k, X, epsk = Zeidel(0.1, g)
print(f'Розв\'язок: {X}, К-ть ітерацій: {k}, Дійсне розходження: {epsk}')
def testZeidel(g):
    for i in range(1, 10):
        k, X, epsk = Zeidel(10**(-i), g)
        print(f'Розв\'язок: {X}, К-ть ітерацій: {k}, Дійсне розходження: {epsk}')
# testZeidel(g)
# testZeidel(Zero)
def NAZ(ep, init):
    X = Zeidel(ep, init)[1]
    N = np.dot(A, X) - b
    return (sum([abs(N[i]) for i in range(6)])/6)
print(f'Зейделя = {NAZ(0.01, Zero)}')
y1Zeidel = [Zeidel(10**(-i), g)[0] for i in range(1, 11)]
y2Zeidel = [Zeidel(10**(-i), Zero)[0] for i in range(1, 11)]
def Graph(y1, y2):
    x = [i for i in range(1, 11)]
    plt.plot(x, y1, label = 'X=g')
    plt.plot(x, y2, label = 'X=Zero')
    plt.xlabel('10^-точність')
    plt.ylabel("к-ть ітерацій")
    plt.legend()
    plt.show()
# Graph(y1Zeidel, y2Zeidel)
y1Jacobi = [jacob(10**(-i), g)[0] for i in range(1, 11)]
y2Jacobi = [jacob(10**(-i), Zero)[0] for i in range(1, 11)]
Graph(y1Jacobi, y2Jacobi)
#5
B = np.dot(A.transpose(), A)
f = np.dot(A.transpose(), b)
# print(f'B = {B} \n f = {list(f)}')
def ZR(QR, fr, om, eps):
    m = len(QR)
    QD = np.diag(QR.diagonal())
    QR0 = QR - QD
    QR1 = np.tril(QR, k = -1)
    RB = np.identity(m) - om * np.dot(li.inv(QD+om*QR1), QR)
    g = om * np.dot(li.inv(QD + om*QR1), fr)
    epk = 10
    x = g
    k = 0
    while epk > eps:
        y = np.dot(RB, x) + g
        epk = max([abs(y[i] - x[i]) for i in range(len(x))])
        x = y
        k += 1
    return [x, epk, k]
s = ZR(B, f, 1, 0.1)
# print(f'Розв\'язок: {list(s)[0]},\nДійсне розходження: {s[1]},\nК-ть ітерацій: {s[2]}')
# print(list(s[0]), '\n', s[2])
def testRelax():
    i = 1
    while i < 2:
        for j in range(10):
            z = ZR(B, f, i, 10**(-j))
        i += 0.1
        print(f'Розв\'язок: {list(z)[0]},\nДійсне розходження: {z[1]}, К-ть ітерацій: {z[2]}, omega: {i}')
# for j in range(10):
    # print(f'eps = {10 ** (-j)}')
# testRelax()
def NAR(QR, fr, omega, ep):
    X = ZR(QR, fr, omega, ep)[0]
    N = np.dot(A, X) - b
    return (sum([abs(N[i]) for i in range(6)]) / 6)
print(f'Релаксації = {NAR(B, f, 1.5, 0.01)}')