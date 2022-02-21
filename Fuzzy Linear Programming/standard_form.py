import re
import numpy as np
import pandas as pd


def line():
    print('-' * 60)


def normal(expression, s=1, S=1, a=1):
    standard = []
    for exp in expression:
        exp = re.sub(r'[ ][ ]+', ' ', exp)
        exp = re.sub(r' x', ' 1x', exp)
        exp = re.sub(r'^x', '1x', exp)
        exp = re.sub(r'[\-]x', '-1x', exp)

        for op in '+-':
            exp = exp.replace(op + ' ', op)

        if not exp.startswith('-'):
            exp = '+' + exp

        match = re.findall(r' [<>=][=]* ', exp)

        if not match:
            standard.append(exp)
            continue

        operator = match[0]

        left, right = exp.split(operator)
        if operator in [' >= ', ' > ']:
            left += ' -1s{} +1R{}'.format(s, a)
            s += 1
            a += 1
        elif operator in [' <= ', ' < ']:
            left += ' +1S{}'.format(S)
            S += 1
        else:
            left += ' +1R{}'.format(a)
            a += 1

        standard.append('{}{}{}'.format(left, ' = ', right))
    return standard


def objective_null(exp):
    exps = []
    for obj in exp.split():
        if obj[0].startswith('+'):
            obj = obj.replace('+', '-')
        else:
            obj = obj.replace('-', '+')
        exps.append(obj)
    return '+1Z ' + ' '.join(exps) + ' = 0'


def keys(mys):
    if mys.startswith(('x', 'y', 't')):
        return eval(mys[1])
    elif mys.startswith('s'):
        return 10 + eval(mys[1])
    elif mys.startswith('S'):
        return 100 + eval(mys[1])
    return 1000 + eval(mys[1])


def remone(match):
    obj = match.group(0)
    return obj[0] + obj[2:]


def expand(match):
    obj = match.group(0)
    return obj + ' '


def equation(eq, M):
    eq = re.sub(r'[+\-]1[a-zA-Z]', remone, eq)
    eq = re.sub(r'[+\-]', expand, eq)
    eq = re.sub(str(M), 'M', eq)
    if eq.startswith('+'):
        return eq[2:]
    return eq[0]+eq[2:]


def metric(objective, constrains, M=1000, show=True):
    if show:
        print('INISIAL MASALAH')
        line()

        print('>> Fungsi Tujuan: Maksimasi Z =', objective)
        print('>> Kendala      :', constrains[0])
        for i in constrains[1:]:
            print('                :', i)
        print('')

        print('BENTUK STANDAR')
        line()

    standard_constrains = normal(constrains)

    variable = [re.findall(r'[a-zA-Z]\d', i) for i in standard_constrains]
    variable = ['Z'] + sorted(set(np.concatenate(variable)), key=keys)
    artificial = [i for i in variable if i.startswith('R')]

    objective += ' ' + ' '.join(['- {}{}'.format(M, i) for i in artificial])
    standard_objective = normal([objective])[0]

    if show:
        print('>> Fungsi Tujuan: Maksimasi Z =', equation(standard_objective, M))
        print('>> Kendala      :', equation(standard_constrains[0], M))
        for i in standard_constrains[1:]:
            print('                :', equation(i, M))

        print('')
        print('>> Keterangan   : ')
        if any([i.startswith('s') for i in variable]):
            print('   s = Variabel Surplus')
        if any([i.startswith('S') for i in variable]):
            print('   S = Variabel Slack')
        if any([i.startswith('R') for i in variable]):
            print('   R = Variabel Artificial')
        print('')

        print('TABEL SIMPLEKS AWAL')
        line()

    standard_objective = objective_null(standard_objective)

    standard = []
    for enu, func in enumerate([standard_objective] + standard_constrains):
        temp = []
        for var in variable:
            match = re.findall(r'[+\-]\d+' + var, func)
            if match:
                coef = match[0].replace(var, '')
                temp.append(eval(coef))
            else:
                temp.append(0)
        operator = re.findall(r' = ', func)[0]
        right = func.split(operator)[-1]
        standard.append(temp + [right])
    standard = np.array(standard)

    if show:
        df = pd.DataFrame(np.where(standard == str(M), 'M', standard))
        df.columns = variable + ['rhs']
        df.index = ['Z'] + [i for i in variable if i.startswith(('S', 'R'))]
        print(df, '\n')
    return standard, variable + ['rhs']
