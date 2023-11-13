from numpy import *
import sympy as sp

#1
a, b, c = sp.symbols('a, b, c')
f = sp.Function('f')(a, b, c)

dicti = {a: 0.125, b: 0.124, c: 2.12}
delta = {a: 0.005, b: 0.003, c: 0.015}

f = ((a + b) * (a**2 - b) * sp.ln(a + c))/(a * b)**2

dfda = sp.diff(f, a)
print("Похідна по a: ", dfda)

res1 = dfda.subs(dicti)
print("Значення похідної по a: ", res1)

dfdb = sp.diff(f, b)
print('\nПохідна по b: ', dfdb)

res2 = abs(dfdb.subs(dicti))
print('Значення похідної по b: ', res2)

dfdc = sp.diff(f, c)
print('\nПохідна по c: ', dfdc)

res3 = abs(dfdc.subs(dicti))
print('Значення похідної по c: ', res3)

deltaf = (abs(res1) * delta[a]) + (abs(res2) * delta[b]) + (abs(res3) * delta[c])
print('\nАбсолютна похибка: ', deltaf)

f_value = f.subs(dicti)
# print(f_value)
inf = abs(f_value - deltaf)
print("\nНижня межа: ", inf)

sup = abs(f_value + deltaf)
print("\nВерхня межа: ", sup)

rel = abs(deltaf/f_value)
print('\nВідносна похибка: ', rel)
#2

g = sp.Function('g')(a, b, c)
# x1, x2, x3 = sp.symbols('x1 x2 x3')
values = {a: 1, b: 2, c: 3}
delta_val = 0.01
g = 2*a*sp.sin(a*b*(c**2))/(1+sp.exp(c*b + a))
g_val = g.subs(values)
# print(g_val)
dgda = sp.diff(g, a)
print('\nПохідна по а: ', dgda)

dgdb = sp.diff(g, b)
print('\nПохідна по b: ', dgdb)

dgdc = sp.diff(g, c)
print('\nПохідна по с: ', dgdc)

res4 = dgda.subs(values)
print('Значення похідної по а: ', res4)

res5 = abs(dgdb.subs(values))
print('\nЗначення похідної по b: ', res5)

res6 = abs(dgdc.subs(values))
print('\nЗначення похідної по с: ', res6)

deltag = (abs(res4) * delta_val) + (abs(res5) * delta_val) + (abs(res6) * delta_val)
print('\nАбсолютна похибка: ', round(deltag, 5))

inf1 = abs(g_val - deltag)
print('\nНижня межа: ', inf1)

sup1 = abs(g_val + deltag)
print('\nВерхня межа: ', sup1)

rel1 = abs(deltag/g_val)
print('\nВідносна похибка: ', round(rel1, 5))

v = 0.02*sp.sin(18)/(1 + sp.exp(7)) - 0.12*sp.exp(7)*sp.sin(18)/(1 + sp.exp(7))**2 + 0.78*sp.cos(18)/(1 + sp.exp(7))