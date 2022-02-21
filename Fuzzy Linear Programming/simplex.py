import pandas as pd
import numpy as np


def line():
    print('-' * 60)


def integer(x):
    return x == int(x)


def getvalue(exp):
    try:
        a, b = exp.split('/')
    except ValueError:
        a, b = exp, '1'
    return a, b


def string_op(exp1, exp2, op):
    a, b = getvalue(exp1)
    x, y = getvalue(exp2)

    if op == '/':
        x, y = y, x
        op = '*'

    if op in '+ -'.split():
        eq1 = eval('({} * {}) {} ({} * {})'.format(a, y, op, x, b))
    else:
        eq1 = eval('{} * {}'.format(a, x))
    eq2 = eval('{} * {}'.format(b, y))

    if eq2 == 0:
        return "float('inf')"

    if integer(eq1 / eq2):
        return str(int(eq1 / eq2))

    if eq1 / eq2 == eq1:
        return str(eq1)

    for prime in [2, 3, 5, 7, 11]:
        while integer(eq1 / prime) and integer(eq2 / prime):
            eq1 /= prime
            eq2 /= prime
    if integer(eq1):
        eq1 = int(eq1)
    if integer(eq2):
        eq2 = int(eq2)
    eq = '{}/{}'.format(eq1, eq2)
    if eval(eq) > 0:
        return eq.replace('-', '')
    eq = eq.replace('-', '')
    return '-' + eq


def array_str(arr1, arr2, op):
    return [string_op(i, j, op) for i, j in zip(arr1, arr2)]


def frame(data, col, row):
    df = pd.DataFrame(data)
    df.columns = col
    df.index = row
    print(df, '\n')


def simplex(metric, variable, show=True):
    artificial = [i for i in variable if i.startswith('R')]
    basis = [i for i in variable if i.startswith(('S', 'R'))]
    length = len(variable)

    if len(artificial) != 0:
        if show:
            print('MERUBAH VARIABEL ARTIFICIAL MENJADI NOL')
            line()
        for art in artificial:
            col = variable.index(art)
            row = list(metric[:, col]).index(str(1))
            num = string_op(metric[0, col], str(-1), '*')
            metric[0] = array_str(metric[0], array_str([num] * length, metric[row], '*'), '+')

        if show:
            frame(metric, variable, ['Z'] + basis)

    iteration = 0
    while min([eval(i) for i in metric[0]]) < 0:
        if show:
            print('ITERASI {}'.format(iteration + 1))
            line()

        col_key = np.argmin([eval(i) for i in metric[0][:-1]])
        ratio = array_str(metric[1:, -1], metric[1:, col_key], '/')
        show_ratio = np.array(['-'] + [i if i != "float('inf')" else '-' for i in ratio])[:, np.newaxis]
        row_key = np.argmin([eval(i) if eval(i) > 0 else float('inf') for i in ratio])

        if show:
            frame(np.hstack((metric, show_ratio)), variable + ['rasio'], ['Z'] + basis)

        if show:
            print('Variabel Masuk :', variable[col_key])
            print('Variabel Keluar:', basis[row_key], '\n')

        basis[row_key] = variable[col_key]

        metric[row_key + 1] = array_str(metric[row_key + 1], [metric[row_key + 1, col_key]] * length, '/')
        for row in range(len(metric)):
            if row == row_key + 1:
                continue
            nums = string_op(metric[row, col_key], str(-1), '*')
            metric[row] = array_str(metric[row], array_str([nums] * length, metric[row_key + 1], '*'), '+')

        iteration += 1

    if show:
        frame(metric, variable, ['Z'] + basis)

    bass = {}
    for i, bas in enumerate(basis):
        if not bas.startswith(('x', 'y', 't')):
            continue
        if show:
            print('{} = {}'.format(bas, metric[i + 1, -1]))
        bass[bas] = metric[i + 1, -1]
    if show:
        print('Z = {}'.format(metric[0, -1]), '\n')
    return bass, metric[0, -1]
