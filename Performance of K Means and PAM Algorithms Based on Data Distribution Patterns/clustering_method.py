from scipy.spatial.distance import pdist
import numpy as np, pandas as pd

#K-MEANS
def printcent(cent,z):
    mystr = 'Centroid' if z == 0 else 'Medoid'
    for x,i in enumerate(cent):
        print('   '+mystr+' '+str(x+1)+':',i)
    print('')

def printclus(clus):
    col = 10
    for i in range(max(clus)+1):
        indx = list((np.arange(len(clus))[np.isin(clus,i)])+1)
        adds = col-(len(indx)%col)
        indx = indx+(['-']*adds)
        x = int(np.ceil(len(indx)/col))
        indx = np.asarray(indx).reshape(x,col)
        df = pd.DataFrame(indx)
        df.columns = ['']*col
        df.index   = ['   ']*x

        print('   Nomor Dokumen di Cluster '+str(i+1)+':')
        print(df,'\n')
    
def line():
    print('-'*70)
    
def euclid(a,b):
    return pdist([a,b])[0]

def find_cluster(data,cent):
    n = len(data)
    k = len(cent)
    dist = np.zeros((n,k))
    for i in range(n):
        for j in range(k):
            dist[i,j] = euclid(data[i],cent[j])
    return np.argmin(dist,1)

def find_centroid(data,clus):
    k = max(clus)+1
    newc = []
    for i in range(k):
        idx = np.isin(clus,i)
        newc.append(np.mean(data[idx],0))
    return np.array(newc)

def kmeans(initc,data):
    print('K-MEANS CLUSTERING'); line()

    print('>> INISIAL CENTROID:')
    cent1 = data[initc]
    printcent(cent1,0)

    print('>> INISIAL CLUSTER:')
    clus1 = find_cluster(data,cent1)
    printclus(clus1)

    convergent = False; I = 1
    while not convergent:
        print('ITERASI',I); line()

        print('>> CENTROID SEKARANG:')
        cent2 = find_centroid(data,clus1)
        printcent(cent2,0)

        print('>> CLUSTER SEKARANG')
        clus2 = find_cluster(data,cent2)
        printclus(clus2)
        
        if all([i==j for i,j in zip(clus1,clus2)]):
            print('   Cluster Sebelumnya = Cluster Sekarang, Program Berhenti\n')
            convergent = True
        else:
            print('   Cluster Sebelumnya != Cluster Sekarang, Lanjut ke Iterasi Berikutnya\n')
            clus1 = clus2
            I += 1
    return clus2

def kmedoid_non_pam(initmed,d):
    print('K-MEDOID NON-PAM CLUSTERING'); line()
    
    n = len(d)
    k = len(initmed)

    print('>> INISIAL MEDOID')
    meds1 = np.copy(initmed)
    printcent(meds1)

    print('>> INISIAL CLUSTER')
    clus1 = np.argmin(d[meds1],0)
    printclus(clus1)

    print('>> INISIAL COST')
    cost1 = sum(np.min(d[meds1],0))
    print('  ',cost1,'\n')

    decrease = True; I = 1
    while decrease:
        print('ITERASI',I); line()

        meds2 = np.int16(np.zeros(k))
        for i in range(k):
            indx = np.arange(n)[clus1==i]
            bank = d[indx][:,indx]
            sums = np.sum(bank,1)
            meds2[i] = indx[np.argmin(sums)]

        print('>> MEDOID SEKARANG:')
        meds2 = sorted(meds2)
        printcent(meds2)

        print('>> CLUSTER SEKARANG')
        clus2 = np.argmin(d[meds2],0)
        printclus(clus2)

        print('>> COST SEKARANG')
        cost2 = sum(np.min(d[meds2],0))
        print('  ',cost2,'\n')

        if cost2 < cost1:
            print('   Cost Sekarang < Cost Sebelumnya, Lanjut ke Iterasi Berikutnya\n')
            meds1 = meds2
            clus1 = clus2
            cost1 = cost2
            I += 1
        else:
            print('   Cost Sekarang >= Cost Sebelumnya, Program Berhenti\n')
            decrease = False
            
    return clus1,meds1

def find_cost(P,M,Op,Om,d):
    J = [v for v in P if v != Op]
    Cjmp = 0
    for Oj in J:
        indx_sorted = np.argsort(d[Oj][M])[0:2]
        closer_1st,closer_2nd = [M[v] for v in indx_sorted]
        if closer_1st == Om:
            if d[Oj,closer_2nd] <= d[Oj,Op]:
                Cjmp += d[Oj,closer_2nd] - d[Oj,Om]
            else:
                Cjmp += d[Oj,Op] - d[Oj,Om]
        else:
            if d[Oj,closer_1st] <= d[Oj,Op]:
                Cjmp += 0
            else:
                Cjmp += d[Oj,Op] - d[Oj,closer_1st]
    return Cjmp

def kmedoid_pam(initmed,d):
    print('PAM CLUSTERING'); line()
    
    n = len(d)
    k = len(initmed)

    print('>> INISIAL MEDOID')
    medoid = np.copy(initmed)
    printcent(medoid,1)
    
    nonmed = [i for i in range(n) if i not in medoid]

    repeat = True; I = 1
    while repeat:
        print('ITERASI',I); line()
        TCmp = []
        for m in medoid:
            for o in nonmed:
                Cjmp = find_cost(nonmed,medoid,o,m,d)
                TCmp.append(Cjmp)

        TCmp = np.asarray(TCmp).reshape(k,n-k)
        
        mintcmp = np.min(TCmp)
        print('>> MIN TCMP:',mintcmp)
                
        if np.min(TCmp) < 0:
            print('   Karena MIN TCmp Negatif, Lanjut ke Iterasi Selanjutnya\n')
            a,b = [i[0] for i in np.where(TCmp==np.min(TCmp))]

            print('>> MEDOID SEKARANG')
            medoid[a],nonmed[b] = nonmed[b],medoid[a]
            printcent(medoid,1)
            
            I += 1
        else:
            print('   Karena MIN TCmp Positif, Program Berhenti\n')
            repeat = False

    print('>> CLUSTER AKHIR:')
    clus = np.argmin(d[medoid],0)
    printclus(clus)
    
    return clus,medoid
                
        
