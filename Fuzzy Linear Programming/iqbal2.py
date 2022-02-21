import numpy as np
import re
from SIMPLEX import SIMPLEX

Z = '(-1,2,3)x1 + (2,3,4)x2'

K = ['(0,1,2)x1 + (1,2,3)x2 = (2,10,24)',
     '(1,2,3)x1 + (0,1,2)x2 = (1,8,21)']


def getarreq(eq):
    val, var = eq.split(')')
    val = val[1:].split(',')
    temp = []
    for v in val:
        temp.append('{}{}'.format(v, var))
    return temp


def tostr(arreq, ft=False):
    temp = [arreq[0]]
    for i, v in enumerate(arreq[1:]):
        if i == len(arreq) - 2 and not ft:
            temp.append('= '+v)
        else:
            if v[0].startswith('-'):
                temp.append('= ' + v)
            else:
                temp.append('+ ' + v)
    return ' '.join(temp)


Zeq = []
for z in re.split(r' [+\-] ', Z):
    Zeq.append(getarreq(z))
Zeq = np.array(Zeq)

Keq = []
for k in K:
    k, r = re.split(r' = ', k)
    temp = []
    for kk in re.split(r' [+\-] ', k):
        temp.append(getarreq(kk))
    Keq.append(temp + [r[1:-1].split(',')])
Keq = np.array(Keq)


for i in range(3):
    Z = tostr(Zeq[:, i], True)
    K = [tostr(k[:, i]) for k in Keq]
    SIMPLEX(Z, K)
    x = input('')

