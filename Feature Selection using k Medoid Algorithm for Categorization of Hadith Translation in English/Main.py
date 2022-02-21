from nltk.tokenize import word_tokenize as wt
from nltk.corpus import stopwords
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import rankdata
import numpy as np, string, itertools, pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

#READ DATA
with open('bukhari (Autosaved).csv',encoding='utf-8')as f:
    read = f.readlines()
read = [i.strip() for i in read]

def getData(a,read,jml):
    x = []
    for i,data in enumerate(read[a::]):
        try:
            int(data[1])
            continue
        except:
            if data[0] == ';':
                continue
            else:
                x.append(data)
        if len(x) == jml:
            break
    return x


#Ganti
jml = 100

iman  = getData(24,read,jml)
ilmu  = getData(190,read,jml)
wudlu = getData(435,read,jml)
mandi = getData(797,read,jml)
haid  = getData(958,read,jml)

labels = ['iman','ilmu','wudlu','mandi','haid']
#----------------------------------------------

for name in labels:
    data = eval(name)
    for i,doc in enumerate(data):
        f = open(name+' ('+str(i+1)+').txt','w')
        f.write(doc+'\n')
        f.close()

print('1. PREPROCESSING & WEIGHTING\n')

punct = list(string.punctuation)

L = len(labels)
P = 50

raws = []
for i in labels:
    for j in range(P):
        f = open(i+' ('+str(j+1)+').txt').read()
        raws.append(f.strip())

docs = []
for doc_i in raws:
    temp = []
    for j in wt(doc_i.lower()):
        j = stemmer.stem(j)
        if j in punct:
            continue
        temp.append(j)
    docs.append(temp)

N = len(docs)

dictio = sorted(set(np.concatenate(docs)))

TF = []
for doc_i in docs:
    temp = []
    for word in dictio:
        temp.append(doc_i.count(word))
    TF.append(np.asarray(temp)/len(doc_i))
TF = np.asarray(TF)

DF = []
for word in dictio:
    count = 0
    for doc_i in docs:
        if word in doc_i:
            count += 1
    DF.append(count)
DF = np.asarray(DF)

IDF = np.log10(N/DF)

TFIDF = TF*IDF

label = np.hstack([[i]*P for i in range(5)])

print('   *KETERANGAN DATASET')
print('   >> Label Dataset:')
maxl = max(max([len(j) for j in labels]),14)
for i in labels:
    space = ' '*(maxl-len(i))
    print('      - '+i+space+':',P,'Dokumen')
print('     ','-'*30,'+\n   >> Total'+' '*(maxl-3)+':',N,'Dokumen')
print('   >> Data Latih'+' '*(maxl-8)+':',N,'Dokumen')
print('   >> Data Uji'+' '*(maxl-6)+':',int(N/2),'Dokumen')
print('   >> Jumlah Feature'+' '*(maxl-12)+':',len(dictio),'Feature\n')


def FindInit(data,k,meas):
    D = pairwise_distances(data)
    npls = k; N = npls+k
    SIMS = 1-(D/np.max(D)) if meas == 'euclidean' else 1-D
    teta = np.average(SIMS)
    negh = np.where(SIMS >= teta,1,0)
    idxc = np.argsort(negh.sum(0),kind='mergesort')
    idxc = idxc[::-1][0:N]
    link,simi = [],[]
    for i,j in itertools.combinations(idxc,2):
        link.append(np.dot(negh[i],negh[j]))
        simi.append(SIMS[i,j])
    rlnk = rankdata(link,method='dense')
    rsim = rankdata(simi,method='dense')
    rsum = squareform(rlnk+rsim)
    rcom,cand,temp = [],[],np.arange(N)
    for i in itertools.combinations(temp,k):
        sect = np.intersect1d(temp,i)
        rank = rsum[sect][:,sect]
        rcom.append(np.sum(rank)/2)
        cand.append([idxc[v] for v in i])
    return sorted(cand[np.argmin(rcom)])

def FindClus(data,cent,meas):
    clus = []
    for i in data:
        temp=[]
        for j in cent:
            temp.append(pdist([i,j],metric=meas)[0])
        clus.append(np.argmin(temp))
    return clus

def FindCent(data,clus,k):
    cent = []
    for i in range(k):
        cent.append(np.mean(data[np.isin(clus,i)],0))
    return np.asarray(cent)

def K_Means(data,k,meas,initc):
    cent1 = data[initc]
    clus1 = FindClus(data,cent1,meas)
    while True:
        cent2 = FindCent(data,clus1,k)
        clus2 = FindClus(data,cent2,meas)
        if clus1 == clus2:
            break
        else:
            clus1 = clus2
    return cent2

def FindCost(P,M,Op,Om,d):
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

def KMedoid(data,k,meas,initmed):
    n = len(data)
    d = pairwise_distances(data,metric=meas)
    medoid = np.copy(initmed)
    nonmed = [i for i in range(n) if i not in medoid]

    while True:
        TCmp = []
        for m in medoid:
            for o in nonmed:
                Cjmp = FindCost(nonmed,medoid,o,m,d)
                TCmp.append(Cjmp)
        TCmp = np.asarray(TCmp).reshape(k,n-k)
        if np.min(TCmp) < 0:
            #swap m with o
            a,b = [i[0] for i in np.where(TCmp==np.min(TCmp))]
            medoid[a],nonmed[b] = nonmed[b],medoid[a]
        else:
            break
    return data[medoid]

def MicroMacroF(clas,currclas,datacurr):
    expclas  = [clas[i] for i in currclas]
    dataclas = [clas[i] for i in datacurr]
    TP,FP,FN = [],[],[]
    precis,recall = [],[]
    for i in clas:
        idx = np.arange(len(dataclas))[np.isin(dataclas,i)]
        x,y = 0,0
        for j in idx:
            if expclas[j] == i:
                x += 1
            else:
                y += 1
        z = 0
        for j in np.arange(len(expclas))[np.isin(expclas,i)]:
            if j not in idx:
                z += 1
        TP.append(x); FP.append(y); FN.append(z)
        precis.append((x/(x+y))*100 if x+y != 0 else 0)
        recall.append((x/(x+z))*100 if x+z != 0 else 0)
    MicroP = (sum(TP)/(sum(TP)+sum(FP)))*100
    MicroR = (sum(TP)/(sum(TP)+sum(FN)))*100
    MicroF = 2*((MicroP*MicroR)/(MicroP+MicroR))
    MacroP = np.average(precis)
    MacroR = np.average(recall)
    MacroF = 2*((MacroP*MacroR)/(MacroP+MacroR))
    return MicroF, MacroF

print('2. FEATURE SELECTION PADA DATA LATIH')

K = 5
V = 5
allmeas = ['Cosine','Euclidean']
methods = ['KMeans','KMedoid']

print('   >> Banyaknya Label              (L):',L)
print('   >> Jumlah Cluster / Label       (K):',K)
print('   >> Feature yang Diambil / Label (V):',V)
print('   >> Maksimum Feature yang Diambil   : L.V.K =',L*V*K,'Feature')
print('   >> Metode Clustering               :',' dan '.join(methods))
print('   >> Dissimilarity Measure           :',' dan '.join(allmeas),'\n')
print('   >> Hasil Feature Selection:')

idxv1,idxv2 = [],[]
for meas in allmeas:
    meas = meas.lower()
    temp1,temp2 = [],[]
    for i in range(K):
        data  = TFIDF[label==i]
        initc = FindInit(data,K,meas)
        cent1 = K_Means(data,K,meas,initc)
        cent2 = KMedoid(data,K,meas,initc)
        for j in cent1:
            idxi = np.argsort(j)[::-1][0:V]
            for v in idxi:
                if v not in temp1:
                    temp1.append(v)
        for j in cent2:
            idxi = np.argsort(j)[::-1][0:V]
            for v in idxi:
                if v not in temp2:
                    temp2.append(v)
    idxv1.append(temp1)
    idxv2.append(temp2)


for i,idxv in enumerate([idxv1,idxv2]):
    for j,indx in enumerate(idxv):
        space = ' '*(30-len(allmeas[j]+' + '+methods[i]))
        print('      - '+allmeas[j]+' + '+methods[i]+space+': '+str(len(indx))+' Feature')
        
x = input('\n   Tampilkan Feature? [1.Ya/2.Tidak]: ')
if x == '1':
    print('')
    for i,idxv in enumerate([idxv1,idxv2]):
        for j,indx in enumerate(idxv):
            print('   >> '+(allmeas[j]+' '+methods[i]).upper()+' FEATURE:')
            n = len(indx)
            temp = sorted([dictio[v] for v in indx])
            cols = 4
            if n%4 == 0:
                mods = 0
            else:
                mods = 4-(n%4)
            for v in range(mods):
                temp.append('-')
            rows = int(len(temp)/4)
            temp = np.asarray(temp).reshape(rows,cols)
            df = pd.DataFrame(temp)
            df.columns = ['']*cols
            df.index   = ['   ']*rows
            print(df,'\n')

print('')
print('3. CLASSIFICATION PADA DATA UJI')

Z = int(P/2)
pick = list(range(0,Z))+list(range(Z*2,Z*3))+list(range(Z*4,Z*5))+list(range(Z*6,Z*7))+list(range(Z*8,Z*9))
labs = np.hstack([i]*Z for i in range(K))

types = ['Tanpa F.S.',
         'K-Means F.S.',
         'K-Medoid F.S.']

KNEAR = 7

print('   >> Klasifikasi Method:')
print('      - Nearest Centroid    (NCR)')
print('      - K-Nearest Neighbors (KNN) K =',KNEAR,'\n')

def NearestCentroid(data,clas,k,meas):
    cent = []
    for i in range(k):
        cent.append(np.mean(data[np.isin(clas,i)],0))
    label = []
    for i in data:
        temp=[]
        for j in cent:
            temp.append(pdist([i,j],metric=meas)[0])
        label.append(np.argmin(temp))
    return label

def KNearestNeighbors(data,k,meas,clas):
    D = pairwise_distances(data,metric=meas)
    knear = []
    for i,d in enumerate(D):
        count = 0
        while True:
            res = [clas[j] for j in np.argsort(d) if j != i][0:k+count]
            cnt = [res.count(j) for j in range(k)]
            if cnt.count(max(cnt)) > 1:
                count += 1
            else:
                break
        knear.append(np.argmax(cnt))
    return knear
    
for i,meas in enumerate(allmeas):
    print('   >> Menggunakan',meas,'Distances')
    meas = meas.lower()
    allmic,allmac=[],[]
    for j,data in enumerate([TFIDF[pick],TFIDF[pick][:,idxv1[i]],TFIDF[pick][:,idxv2[i]]]):
        clas1 = NearestCentroid(data,labs,K,meas)
        clas2 = KNearestNeighbors(data,KNEAR,meas,labs)
        micro1,macro1 = MicroMacroF(labels,clas1,labs)
        micro2,macro2 = MicroMacroF(labels,clas2,labs)
        allmic.append([round(micro1,2),round(micro2,2)])
        allmac.append([round(macro1,2),round(macro2,2)])
    df = pd.DataFrame({'MicroF NCR':np.asarray(allmic)[:,0],
                       'MicroF KNN':np.asarray(allmic)[:,1],
                       'MacroF NCR':np.asarray(allmac)[:,0],
                       'MacroF KNN':np.asarray(allmac)[:,1]},index=['      '+j for j in types])
    print(df,'\n')





