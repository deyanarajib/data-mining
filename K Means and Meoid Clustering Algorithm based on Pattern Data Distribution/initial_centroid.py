import numpy as np, itertools
from scipy.stats import rankdata

def count_link(A,B):
    return np.dot(A,B)

def neighbors_and_link(disbank,k):
    nplus = k if k <= 7 else 2
    simbank = np.abs(disbank-np.max(disbank))
    teta    = np.average(simbank)
    neghbrs = np.where(simbank >= teta,1,0)
    neghsum = neghbrs.sum(0)
    idxcand = np.argsort(neghsum)[-(k+nplus):][::-1]
    linkcnd = []; simicnd = []; combine = []
    for i,j in itertools.combinations(idxcand,2):
        linkcnd.append(count_link(neghbrs[i],neghbrs[j]))
        simicnd.append(simbank[i][j])
        combine.append((i,j))
    ranklnk = rankdata(linkcnd,method='dense')
    ranksim = rankdata(simicnd,method='dense')
    ranksum = ranklnk+ranksim
    rankcom = []
    for i in itertools.combinations(idxcand,k):
        temp = 0
        for j in itertools.combinations(i,2):
            temp += ranksum[combine.index(j)]
        rankcom.append([[temp],list(i)])
    init = sorted(rankcom[np.argmin(np.transpose(rankcom)[0])][1])
    return init

def method_2(disbank,k):
    n = len(disbank)
    
    v = np.zeros(n)
    for j in range(n):
        sum_dij = 0
        for i in range(n):
            sum_dil = 0
            for l in range(n):
                sum_dil += disbank[i,l]
            sum_dij += (disbank[i,j]/sum_dil)
        v[j] = sum_dij
        
    init = sorted(np.argsort(v)[:k])
    return init
