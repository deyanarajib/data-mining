baca= open('coba.txt',encoding='utf-8').readlines()

allisi,alljudul = [],[]
for i in baca:
    try:
        judul,isi = i.strip().split('\t')
    except:
        continue
    judul = str.lower(judul)
    for j in ['iman','shalat','zakat','puasa','haji']:
        if j in judul:
            judul = j
            break
    allisi.append(isi)
    alljudul.append(judul)

for i in ['iman','shalat','zakat','puasa','haji']:
    count = 0
    for x,j in enumerate(alljudul):
        if i == j:
            count += 1
            f = open((i)+str(count)+'.txt','w',encoding='utf-8')
            f.write(allisi[x])
            f.close()
#label = ['haji','iman','zakat','shalat','puasa']
#rawdata = []
#for i in label:
 #   for j in range(100):
  #      x = open((i)+' ('+str(count)+').txt','r', encoding='utf-8').read()
   #     rawdata.append(x.replace('\n',' '))

#PRE-PROCESSING
from nltk.tokenize import word_tokenize as token
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string, math, numpy as np

stop_word = stopwords.words('Arabic')+list(string.punctuation)
stemmer = SnowballStemmer('Arabic')

doc=[]
for i in rawdata:
    temp=[]
    for j in token(i):
        word = stemmer.stem(str.lower(j))
        if word not in stop_word and len(word) > 2 and not word.startswith(tuple(string.punctuation)+tuple([str(k) for k in range(10)])+tuple('¿')):
            temp.append(word)
    doc.append(temp)

dictionary=[]
for i in doc:
    for j in i:
        if j not in dictionary:
            dictionary.append(j)
            
tf=[]
for i in doc:
    temp=[]
    for j in dictionary:
        temp.append(i.count(j))
    tf.append(temp)
idf=[]
for i in dictionary:
    count=0
    for j in doc:
        if i in j:
            count+=1
    idf.append(math.log10(len(doc)/(1+count)))
tfidf = np.array([np.array(i)*np.array(idf) for i in tf])

trainset = np.vstack([tfidf[0:50],tfidf[100:150],tfidf[200:250],tfidf[300:350],tfidf[400:450]])
smpleset = np.vstack([tfidf[50:100],tfidf[150:200],tfidf[250:300],tfidf[350:400],tfidf[450:500]])
explabel = list(np.hstack([[i]*int((len(doc)/len(label))/2) for i in label]))

#CLASSIFICATION
import itertools
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import rankdata

def MicroMacroF(clas,expclas,dataclas):
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
    return round(MicroF,3),round(MacroF,3)

def NearestCentroid(train,sample,clas,expclas):
    centroid = []
    for i in clas:
        idx = np.isin(expclas,i)
        centroid.append(np.mean(train[idx],0))
    dataclas=[]
    for i in sample:
        dist = []
        for j in centroid:
            dist.append(pdist([i,j],metric=meas)[0])
        dataclas.append(clas[np.argmin(dist)])
    return dataclas

def KNearestNeighbors(train,sample,clas,expclas):
    K = 5
    dataclas=[]
    for i in sample:
        dist = []
        for j in train:
            dist.append(pdist([i,j],metric=meas)[0])
        Knearest = []
        for j in range(K):
            Knearest.append(expclas[np.argmin(dist)])
            dist[np.argmin(dist)] = 9999
            
        repeat = True
        while repeat:
            countclas = [Knearest.count(j) for j in clas]
            if countclas.count(max(countclas)) == 1:
                dataclas.append(clas[np.argmax(countclas)])
                repeat = False
            else:
                Knearest.append(expclas[np.argmin(dist)])
                dist[np.argmin(dist)] = 9999
            
    return dataclas

def CountLink(A,B):
    return np.dot(A,B)

def FindInitCent(data,k):
    nplus = k
    disbank = pairwise_distances(data,metric=meas)
    simbank = 1-(disbank/max(np.hstack(disbank))) if meas == 'euclidean' else 1-disbank
    neghbrs = np.where(simbank >= np.average(simbank),1,0)
    neghsum = neghbrs.sum(0)
    idxcand = np.argsort(neghsum)[-(k+nplus):][::-1]
    linkcnd = []; simicnd = []; combine = []
    for i,j in itertools.combinations(idxcand,2):
        linkcnd.append(CountLink(neghbrs[i],neghbrs[j]))
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
    rankinit = min(np.transpose(rankcom)[0])
    return rankcom[np.argmin(np.transpose(rankcom)[0])][1]

def FindCluster(data,cent):
    cluster=[]
    for a,i in enumerate(data):
        temp=[]
        for b,j in enumerate(cent):
            temp.append(pdist([i,j],metric=meas)[0])
        cluster.append(np.argmin(temp))
    return cluster

def Euclid(A,B):
    return np.linalg.norm(A-B)

def KMeans(data):
    K = 5
    initc = FindInitCent(data,K)
    cent1 = data[initc]
    clus1 = FindCluster(data,cent1)
    convergent = False
    while not convergent:
        cent2=[]
        for i in range(K):
            idx = np.isin(clus1,i)
            cent2.append(np.mean(data[idx],0))
        clus2 = FindCluster(data,cent2)
        if clus1 == clus2:
            convergent = True
        else:
            clus1 = clus2
    return cent2

print('1. ORIGINAL CLASSIFIER')
print('   >> Nearest Centroid')
meas = 'euclidean'
NC = NearestCentroid(np.where(trainset!=0,1,0),np.where(smpleset!=0,1,0),label,explabel)
micro,macro = MicroMacroF(label,explabel,NC)
print('      Micro-F:',micro,'Macro-F:',macro)
print('   >> K-Nearest Neighbors')
KNN = KNearestNeighbors(np.where(trainset!=0,1,0),np.where(smpleset!=0,1,0),label,explabel)
micro,macro = MicroMacroF(label,explabel,KNN)
print('      Micro-F:',micro,'Macro-F:',macro,'\n')

for enu,meas in enumerate(['cosine','euclidean']):
    print(str(enu+2)+'.',str.upper(meas),'DISTANCES + KMEANS FEATURE SELECTION')
    idxfeat=[]
    for i in label:
        idx = np.isin(explabel,i)
        cent = KMeans(trainset[idx])
        v = 20
        idxclasfeat=[]
        for centx in cent:
            while len(idxclasfeat) < len(cent)*v:
                if np.argmax(centx) not in idxclasfeat:
                    idxclasfeat.append(np.argmax(centx))
                centx[np.argmax(centx)] = 0
        idxfeat.append(idxclasfeat)

    #feature=[]
    #for i in idxfeat:
    #    temp=[]
    #    for j in i:
    #        temp.append(dictionary[i])
    #    feature.append(temp)

    traincopy = np.copy(trainset)
    samplecopy = np.copy(smpleset)
    removedfeature = sorted(list(set(np.arange(len(smpleset)))-set(np.hstack(idxfeat))))
    for i in list(reversed(removedfeature)):
        traincopy = np.delete(traincopy,i,1)
        samplecopy = np.delete(samplecopy,i,1)
    
    print('   >> Nearest Centroid')
    NCKMF = NearestCentroid(traincopy,samplecopy,label,explabel)
    micro,macro = MicroMacroF(label,explabel,NCKMF)
    print('      Micro-F:',micro,'Macro-F:',macro)
    print('   >> K-Nearest Neighbors')
    KNNKMF = KNearestNeighbors(traincopy,samplecopy,label,explabel)
    micro,macro = MicroMacroF(label,explabel,KNNKMF)
    print('      Micro-F:',micro,'Macro-F:',macro)
    print('')
