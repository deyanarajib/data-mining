import numpy as np
import pandas as pd
import re


def integer(x):
    return x == int(x)


def getvalue(exp):
    try:
        a, b = exp.split('/')
    except ValueError:
        a, b = exp, '1'
    return a, b


def pecahan(exp1, exp2, op):
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

    if integer(eq1/eq2):
        return str(int(eq1/eq2))

    if eq1/eq2 == eq1:
        return str(eq1)

    for prime in [2, 3, 5, 7, 11]:
        while integer(eq1/prime) and integer(eq2/prime):
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
    return '-'+eq


def arrpec(arr1, arr2, op):
    return [pecahan(i, j, op) for i, j in zip(arr1, arr2)]


def sumpec(arr):
    if len(arr) == 0:
        return '0'
    jml = arr[0]
    for arr in arr[1:]:
        jml = pecahan(jml, arr, '+')
    return jml


def line():
    print('-'*70)


def sumM(arr):
    coef = []
    for i in [j for j in arr if 'M' in j]:
        if i == 'M':
            coef.append('1')
        elif i == '-M':
            coef.append('-1')
        else:
            coef.append(i.replace('M', ''))

    sumnums = sumpec([i for i in arr if 'M' not in i])
    sumcoef = sumpec(coef)

    if eval(sumcoef) == 0:
        if eval(sumnums) > 0:
            return '+'+sumnums
        elif eval(sumnums) < 0:
            return sumnums
        else:
            return '0'
    else:
        if eval(sumnums) > 0:
            return sumcoef+'M +'+sumnums
        elif eval(sumnums) < 0:
            return sumcoef+'M '+sumnums
        else:
            return sumcoef+'M'


def getZC():
    zc = []
    for i, c in enumerate(np.transpose([c2 + [NK[j]] for j, c2 in enumerate(C)])):
        temp = []
        for j, hrow in enumerate(headrow):
            if c[j] == '0' or hrow == '0':
                continue
            try:
                r = pecahan(c[j], hrow, '*')
            except NameError:
                if c[j] == '1':
                    r = hrow
                elif c[j] == '-1':
                    r = '-' + hrow
                else:
                    r = re.sub(r'M', c[j] + 'M', hrow)
                r = r.replace('--', '')
            temp.append(r)
        try:
            hcol = pecahan(headcol[i], '-1', '*')
            sums = sumM(temp + [hcol])
        except NameError:
            hcol = '-' + headcol[i]
            hcol = hcol.replace('--', '')
            sums = sumM(temp + [hcol])
        except IndexError:
            sums = sumM(temp)

        zc.append(sums)
    return zc


def printdf():
    frame = np.vstack((varall + ['NK'], [c + [NK[i]] for i, c in enumerate(C)], ZC))
    frame = np.hstack((np.array([''] + vslack + ['ZC'])[:, np.newaxis], frame))

    df = pd.DataFrame(frame)
    df.columns = [''] + headcol + ['']
    df.index = [''] + headrow + ['']
    print(df, '\n')


data = pd.read_excel('data.xlsx', sheet_name='Sheet4').get_values()

Z = [str(i) for i in data[0][:-2]]
C = [[str(j) for j in i] for i in data[1:, :-2]]

OP = data[1:, -2]
NK = [str(int(i)) if integer(i) else str(i) for i in data[1:, -1]]

headcol = [i for i in Z]

count = 0
for i, c in enumerate(C):
    if OP[i] == '>=':
        count += 1
        C[i].append('-1')
    else:
        C[i].append('0')

for i in range(count):
    headcol.append('M')

varall = ['x{}'.format(i + 1) for i in range(len(Z) + len(C) + count)]
vslack = varall[len(Z) + count:]

identity = np.identity(len(vslack))
for i in range(len(C)):
    for j in identity[i]:
        C[i].append(str(int(j)))

for i in range(len(vslack)):
    if OP[i] == '<=':
        headcol.append('0')
    else:
        headcol.append('-M')

headrow = headcol[-len(vslack):]

ZC = getZC()

max10 = 10^10


iters = 0
while True:
    print(ZC)
    col = np.argmin([eval(i.replace('M', '*max10')) for i in ZC[:-1]])
    row = np.argmin([eval(pecahan(i, j, '/')) if j != '0' else float('inf') for i, j in zip(NK, np.transpose(C)[col])])

    printdf()

    if np.min([eval(i.replace('M', '*max10')) for i in ZC[:-1]]) < 0:
        if iters > 0:
            print('   Masih Terdapat Nilai Negatif di Z, lanjut ke iterasi berikutnya\n')
        iters += 1
    else:
        print('   Semua Nilai Z Sudah >= 0, program berhenti\n')
        break

    print('>> ITERASI {}'.format(iters))
    line()

    print('   Variabel Masuk :', varall[col])
    print('   Variabel Keluar:', vslack[row], '\n')

    vslack[row] = varall[col]
    headrow[row] = headcol[col]

    num = C[row][col]

    C[row] = arrpec(C[row], [str(num)]*len(C[row]), '/')
    NK[row] = pecahan(NK[row], str(num), '/')

    print(np.array([[i] + c for i, c in zip(headrow, C)]), '\n')
    for i in range(len(C)):
        if i == row:
            continue
        cicol = pecahan(C[i][col], '-1', '*')
        C[i] = arrpec(C[i], arrpec(C[row], [cicol]*len(C[row]), '*'), '+')
        NK[i] = pecahan(NK[i], pecahan(NK[row], cicol, '*'), '+')

    ZC = getZC()

    print(np.array([[i] + c for i, c in zip(headrow, C)]))
    muah = input('')

