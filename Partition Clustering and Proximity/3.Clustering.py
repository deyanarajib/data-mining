print('----------------------------------------------------------------------')
print('|| PARTITION BASED CLUSTERING USING COSINE, JACCARD AND CORRELATION ||')
print('----------------------------------------------------------------------')

# METHODS
# ----------------------------------------------------------------------------
from nltk.tokenize import word_tokenize as T
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform, pdist
from scipy.stats import rankdata, pearsonr
import pandas as pd, numpy as np, itertools, time


def cosine(A, B):
    return pdist([A, B], metric='cosine')[0]


def correlation(A, B):
    return pdist([A, B], metric='correlation')[0]


def jaccard(A, B):
    return 1 - sum(np.minimum(A, B)) / sum(np.maximum(A, B))


def link(A, B):
    return np.dot(A, B)


def FindInitCent(data, k):
    nplus = k
    disbank = pairwise_distances(data, metric=meas)
    simbank = 1 - disbank
    neghbrs = np.where(simbank >= teta, 1, 0)
    neghsum = neghbrs.sum(0)
    idxcand = np.argsort(neghsum, kind='mergesort')[-(k + nplus):][::-1]
    linkcnd = [];
    simicnd = [];
    combine = []
    for i, j in itertools.combinations(idxcand, 2):
        linkcnd.append(link(neghbrs[i], neghbrs[j]))
        simicnd.append(simbank[i][j])
        combine.append((i, j))
    ranklnk = rankdata(linkcnd, method='dense')
    ranksim = rankdata(simicnd, method='dense')
    ranksum = ranklnk + ranksim
    rankcom = []
    for i in itertools.combinations(idxcand, k):
        temp = 0
        for j in itertools.combinations(i, 2):
            temp += ranksum[combine.index(j)]
        rankcom.append([[temp], list(i)])
    return rankcom[np.argmin(np.transpose(rankcom)[0])][1], simbank, neghbrs


def FindCluster(data, cent, ocent):
    if sim != 'CORRELATION':
        lmax = len(data)
        ncent = []
        for i in cent:
            ncent.append([1 - pdist([i, j], metric=meas)[0] for j in data])
        ncent = np.where(np.array(ncent) >= teta, 1, 0)

    cluster = []
    for enu1, i in enumerate(data):
        simi = []
        for enu2, j in enumerate(cent):
            if sim == 'CORRELATION':
                simi.append(1 - meas(i, j))
            else:
                simi.append(alfa * (link(ncent[enu2], ocent[enu1]) / lmax) + (1 - alfa) * (1 - meas(i, j)))
        cluster.append(np.argmax(simi))
    return cluster


def FindCentroid(data, clus, k):
    newc = []
    for i in range(k):
        idx = np.isin(clus, i)
        newc.append(np.mean(data[idx], 0))
    return np.array(newc)


def FindSplit(data, cent, clus):
    avg = []
    for i, j in enumerate(cent):
        temp = []
        for k in data[np.isin(clus, i)]:
            temp.append(meas(j, k))
        avg.append(np.average(temp))
    return avg


def getSim(sim):
    sim = str.lower(sim)
    if sim == 'cosine':
        return eval(sim)
    elif sim == 'correlation':
        return eval(sim)
    return eval(sim)


def getMeth(data, k, meth):
    if meth == 'K_MEANS':
        return eval(meth)(data, k, [], True, [])
    elif meth == 'BKMEANS':
        return eval(meth)(np.arange(len(data)), np.zeros(len(data)), 0, [])
    return eval(meth)(data, k)


def ExpressionText(cent, clus, dicti):
    N = 10
    idx = [np.argsort(i)[-(N + 10):][::-1] for i in cent]
    val = [i[np.argsort(i)[-(N + 10):][::-1]] for i in cent]
    for i, j in itertools.combinations(np.arange(len(idx)), 2):
        inter = list(set(idx[i]) & set(idx[j]))
        if len(inter) == 0:
            continue
        for k in inter:
            idx1 = list(idx[i]).index(k)
            idx2 = list(idx[j]).index(k)
            if val[i][idx1] < val[j][idx2]:
                idx[i] = np.delete(idx[i], idx1)
            elif val[i][idx1] > val[j][idx2]:
                idx[j] = np.delete(idx[j], idx2)
    text = []
    for enu, i in enumerate(idx):
        print('      CLUSTER', enu + 1, ':', clus.count(enu), 'Doc >> ', end='')
        text.append(', '.join(np.array(dicti)[i[0:N]]))
        print(text[enu])
    return text


def PrintResult(var):
    var = np.array(var)
    X = [[i] for i in np.average(var, 1)]
    Y = np.hstack((np.average(var, 0), 0))
    Z = np.hstack((var, X))
    Z = np.vstack((Z, Y))
    n1 = ['   ' + i[0:3] for i in name1] + ['   AVG']
    n2 = name2 + ['AVERAGE']
    df = pd.DataFrame({n2[0]: [round(i, 6) for i in Z[0]],
                       n2[1]: [round(i, 6) for i in Z[1]],
                       n2[2]: [round(i, 6) for i in Z[2]],
                       n2[3]: [round(i, 6) for i in Z[3]]},
                      index=n1)
    print(df)
    indx1 = np.argmax(np.average(var, 1))
    indx2 = np.argmax(np.average(var, 0))
    print('   BEST USING', name2[indx1], name1[indx2], '\n')
    return np.array(var)


# READ DATA
# --------------------------------------------------------------------------
with open('DataWeight.csv', newline='') as f:
    reader = f.readlines()
reader = [x.replace(',', ' ') for x in reader]
tfidf = []
for row in reader:
    tfidf.append([float(j) for j in T(row)])
tfidf = np.array(tfidf)

with open('Dictionary.csv') as f:
    dictionary = f.readlines()
dictionary = [x.strip() for x in dictionary]

with open('SetSyns.csv') as f:
    reader = f.readlines()
reader = [x.replace(',', ' ') for x in reader]
setsyn = []
for row in reader:
    setsyn.append(' '.join([word for word in T(row)]))

for enu, i in enumerate(dictionary):
    if i.startswith('syn'):
        idx = int(i[3::]) - 1
        dictionary[enu] = '-'.join([x for x in T(setsyn[idx])])

nclas = [20, 19, 102, 73, 38, 8, 26]
count, clas = 0, []
for i in nclas:
    for j in range(i):
        clas.append(count)
    count += 1


# CLUSTERING METHODS
# --------------------------------------------------------------------------
def K_MEANS(data, k, initc, opt, negh):
    if opt:
        initc, temp, negh = FindInitCent(data, k)
    cent1 = data[initc]
    clus1 = FindCluster(data, cent1, negh)
    convergent = False
    while not convergent:
        cent2 = FindCentroid(data, clus1, k)
        clus2 = FindCluster(data, cent2, negh)
        if clus1 == clus2:
            convergent = True
        else:
            clus1 = clus2
    if not opt:
        return clus2, FindSplit(data, cent2, clus2)
    return clus2


def BKMEANS(idx, citer, k, nglob):
    initc, temp, negh = FindInitCent(tfidf[idx], 2)
    subc, avg = K_MEANS(tfidf[idx], 2, initc, False, negh)
    for i, j in enumerate(idx):
        citer[j] = subc[i] + k
    nglob = nglob + avg
    indxmax = np.argmax(nglob)
    if k + 2 == K:
        return [int(i) for i in list(citer)]
    citer = np.where(citer == indxmax, -1, np.where(citer > indxmax, citer - 1, citer))
    idxsplt = np.arange(len(citer))[np.isin(citer, -1)]
    k += 1
    nglob.remove(nglob[indxmax])
    return BKMEANS(idxsplt, citer, k, nglob)


def KMEDOID(data, k):
    M, simbank, negh = FindInitCent(data, k)

    if sim != 'CORRELATION':
        lmax = len(data)
        lnkbank = pairwise_distances(negh, metric=link)
        simbank = alfa * (lnkbank / lmax) + (1 - alfa) * simbank

    O = [i for i in range(len(data)) if i not in M]
    costmax = sum(np.max(simbank[M], 0))
    convergent = False
    while not convergent:
        Msave = [i for i in M]
        for i in range(len(O)):
            for j in range(len(M)):
                O[i], M[j] = M[j], O[i]
                costiter = sum(np.max(simbank[M], 0))
                if costiter > costmax:
                    costmax = costiter
                else:
                    O[i], M[j] = M[j], O[i]
        if len(set(Msave) - set(M)) == 0:
            clus = FindCluster(data, data[M], negh)
            convergent = True
    return clus


def F_Purity(clus, clas, k):
    n = len(clus)
    FM, PR = [], []
    for i in range(k):
        niF = clas.count(i)
        njP = clus.count(i)
        idF1 = np.arange(n)[np.isin(clas, i)]
        idP1 = np.arange(n)[np.isin(clus, i)]
        Fij, nijP = [], []
        for j in range(k):
            njF = clus.count(j)
            idF2 = np.arange(n)[np.isin(clus, j)]
            idP2 = np.arange(n)[np.isin(clas, j)]
            nijF = len(np.intersect1d(idF1, idF2))
            nijP.append(len(np.intersect1d(idP1, idP2)))
            if nijF == 0:
                Fij.append(0)
            else:
                Pij = nijF / njF
                Rij = nijF / niF
                Fij.append((2 * Pij * Rij) / (Pij + Rij))
        Pj = (1 / njP) * max(nijP)
        PR.append((njP / n) * Pj)
        FM.append((niF / n) * max(Fij))
    return round(sum(FM), 6), round(sum(PR), 6)


# MAIN PROGRAM
# --------------------------------------------------------------------------
K = 7
alfa = 0.9
name1 = ['COSINE', 'CORRELATION', 'JACCARD']
name2 = ['K_MEANS', 'BKMEANS', 'KMEDOID']

teta = 0.0005
TN = 0.1
accur = 0.0005

print('EXPERIMENT USING TETA =', teta, '-', TN, 'WITH', accur, 'ACCURACY')
print('AVERAGE PAIRWISE SIMILARITY:')
for i in name1:
    meas = getSim(i)
    print('>> ', str.upper(i), ':', np.average(1 - pairwise_distances(tfidf, metric=meas)))

FMPure, N = [], 0
while round(teta, 6) <= TN:
    print('\n>> FOR TETA =', teta)
    FP = []
    for sim in name1:
        meas = getSim(sim)
        for meth in name2:
            # print('   <> USING',sim,meth)
            # print('      Average Similarity:',np.average(1-pairwise_distances(tfidf,metric=meas)))
            clus = getMeth(tfidf, K, meth)
            cent = FindCentroid(tfidf, clus, K)
            FP = np.append(FP, F_Purity(clus, clas, K))
            # ExpressionText(cent,clus,dictionary)

    FP = np.array(FP).reshape(9, 2)
    print('\n   RESULT')
    print('   ------------------------------------------')
    print('   F-MEASURE')
    temp = PrintResult(np.transpose(FP.reshape(3, 3, 2))[0])
    print('   PURITY')
    temp = PrintResult(np.transpose(FP.reshape(3, 3, 2))[1])
    FMPure = np.append(FMPure, FP)

    teta += accur;
    N += 1

FMPure = np.reshape(FMPure, (N, 9, 2))
Rerata = np.average(FMPure, 0).reshape(3, 3, 2)

print('\nAVERAGE', N, 'TRIALS')
print('-------------------------------------------')
print('F-MEASURE')
Fdata = PrintResult(np.transpose(Rerata)[0])
print('PURITY')
Pdata = PrintResult(np.transpose(Rerata)[1])

df = pd.DataFrame(np.vstack((Fdata, Pdata)))
df.to_csv('FMPURITY.csv', index=False, header=False)
