import standard_form
import simplex
import re
import numpy as np


def line():
    print('*'*60)


def get_array(exp):
    array = []
    match = re.findall(r'(\(.*?\))', exp)
    for value in match:
        value = re.sub(r' ', '', value[1: -1]).split(',')
        array.append(value)
    return np.array(array)


def get_equation(array, fobj):
    equation = []
    for v, a in enumerate(array.T):
        if fobj:
            i, j = a
            my_str = '{}{}1 + {}{}2'.format(i, ['x', 'y', 't'][v], j, ['x', 'y', 't'][v])
        else:
            i, j, k = a
            my_str = '{}{}1 + {}{}2 = {}'.format(i, ['x', 'y', 't'][v], j, ['x', 'y', 't'][v], k)
        my_str = my_str.replace('+ -', '-').replace('-', ' - ')
        my_str = re.sub(r'[ ][ ]+', ' ', my_str).strip()
        if my_str.startswith('-'):
            my_str = my_str[0] + my_str[2:]
        equation.append(my_str)
    return equation


def print_level(R, Z):
    print('>> Fungsi Tujuan: Maksimasi Z =', obj)
    print('>> Kendala      :', cons[0])
    for c in cons[1:]:
        print('                :', c)
    print('                :', ', '.join(sorted(re.findall(r'[xyt]\d+', obj))) + ' >= 0\n')
    print('>> Menggunakan simpleks diperoleh:')
    my_str = []
    for v in re.findall(r'[xyt]\d+', obj):
        my_str.append('{} = {}'.format(v, R[v]))
    my_str.append('dan Z = {}'.format(Z))
    print('   ' + ', '.join(my_str), '\n')


objectives = '(-1, 2, 3)X1 + (2, 3, 4)X2'
constrains = ['(0, 1, 2)X1 + (1, 2, 3)X2 = (2, 10, 24)',
              '(1, 2, 3)X1 + (0, 1, 2)X2 = (1, 8, 21)']

objectives = '(1, 6, 9)X1 + (2, 3, 8)X2'
constrains = ['(2, 3, 4)X1 + (1, 2, 3)X2 = (6, 16, 30)',
              '(-1, 1, 2)X1 + (1, 3, 4)X2 = (1, 17, 30)']

'''objectives = '(1, 6, 9)X1 + (2, 2, 8)X2'
constrains = ['(0, 1, 1)X1 + (2, 2, 3)X2 = (4, 7, 14)',
              '(2, 2, 3)X1 + (-1, 4, 4)X2 = (-4, 14, 22)',
              '(2, 3, 4)X1 + (1, 2, 3)X2 = (-12, -3, 6)']'''

show = True

print('MASALAH AWAL PROGRAM LINEAR FUZZY:')
line()
print('Fungsi Tujuan: Maksimasi Z =', objectives)
print('Kendala      :', constrains[0])
for constrain in constrains[1:]:
    print('             :', constrain)
print('             : X1, X2 >= 0\n')
print('Misalkan     : X1 = (x1, y1, t1)')
print('             : X2 = (x2, y2, t2)')
print('             : Z = (z1, z2, z3)\n')

array_obj = get_array(objectives)
array_cns = np.array([get_array(i) for i in constrains])

eq_obj = get_equation(array_obj, True)
eq_cns = [get_equation(i, False) for i in array_cns]

print('MIDDLE LEVEL PROBLEM:')
line()
obj = eq_obj[1]
cons = [i[1] for i in eq_cns]
standard, variable = standard_form.metric(obj, cons, show=show)
Y, z2 = simplex.simplex(standard, variable, show=show)
y1, y2 = Y['y1'], Y['y2']
print_level(Y, z2)

print('UPPER LEVEL PROBLEM:')
line()
obj = eq_obj[2]
cons = [obj + ' >= ' + z2] + [i[2] for i in eq_cns]
standard, variable = standard_form.metric(obj, cons, show=show)
T, z3 = simplex.simplex(standard, variable, show=show)
t1, t2 = T['t1'], T['t2']
print_level(T, z3)

print('LOWER LEVEL PROBLEM:')
line()
obj = eq_obj[0]
cons = [obj + ' <= ' + z2] + [i[0] for i in eq_cns]
standard, variable = standard_form.metric(obj, cons, show=show)
X, z1 = simplex.simplex(standard, variable, show=show)
x1, x2 = X['x1'], X['x2']
print_level(X, z1)

print('SOLUSI OPTIMAL FUZZY:')
line()
print('X1 = ({}, {}, {}), X2 = ({}, {}, {}), dan Z = ({}, {}, {})'.format(x1, y1, t1, x2, y2, t2, z1, z2, z3))
