from nltk.tokenize import word_tokenize as wt
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np, time, pandas as pd, itertools

stemmer = SnowballStemmer('arabic')
swords = list(stopwords.words('arabic'))+['حدث','عبد','{','}']
clas = ['iman','shalat','zakat','puasa','haji']

#FUNCTION
#---------------------------------------------------------------------------------------
def line(x):
    print('='*x+'\n')

def getRawData(opt):
    rawdata,labels = [],[]
    for x,i in enumerate(clas):
        temp = []
        for j in range(150):
            name = opt+'_'+i+str(j+1)+'.txt'
            try:
                f = open(name,encoding='utf-8').read()
                labels.append(x)
            except:
                continue
            temp.append(f)
        rawdata.append(temp)
    return rawdata,labels

def getData(rawdata):
    data,dictionary = [],[]
    for label in rawdata:
        for doc in label:
            #Tokenizing
            doc = wt(doc)
            temp = []
            for word in doc:
                #Stemming
                word = stemmer.stem(word)
                #Remove Stopword
                if word not in swords:
                    if word not in dictionary:
                        dictionary.append(word)
                    temp.append(word)
            data.append(temp)
    return np.asarray(data),dictionary

def F_Purity(clus,clas,k):
    n = len(clus)
    FM,PR = [],[]
    for i in range(k):
        niF = clas.count(i)
        njP = clus.count(i)
        idF1 = np.arange(n)[np.isin(clas,i)]
        idP1 = np.arange(n)[np.isin(clus,i)]
        Fij,nijP = [],[]
        for j in range(k):
            njF = clus.count(j)
            idF2 = np.arange(n)[np.isin(clus,j)]
            idP2 = np.arange(n)[np.isin(clas,j)]
            nijF = len(np.intersect1d(idF1,idF2))
            nijP.append(len(np.intersect1d(idP1,idP2)))
            if nijF == 0:
                Fij.append(0)
            else:
                Pij = nijF/njF
                Rij = nijF/niF
                Fij.append((2*Pij*Rij)/(Pij+Rij))
        Pj = (1/njP)*max(nijP)
        PR.append((njP/n)*Pj)
        FM.append((niF/n)*max(Fij))
    return sum(FM),sum(PR)

def Accuracy(result,ori):
    count = 0
    for x,i in enumerate(result):
        if ori[x] == i:
            count += 1
    return count/len(result)

def PilihKata(k,n,data,label,clas):

    terms,value,Ns = [],[],[]
    for i in range(k):
        idx = np.arange(n)[np.isin(label,i)]
        Z = np.concatenate(data[idx]).tolist()
        setZ = sorted(set(Z))
        temp = []
        for j in setZ:
            temp.append(Z.count(j))
        Ns.append(len(Z))
        terms.append([setZ[j] for j in np.argsort(temp)[::-1]])
        value.append([temp[j] for j in np.argsort(temp)[::-1]])

    for a,i in enumerate(terms):
        for b,j in enumerate(terms):
            if a == b:
                continue
            sets = np.intersect1d(i,j)
            for h in sets:
                if value[a][i.index(h)] > value[b][j.index(h)]:
                    terms[b].remove(h)
                elif value[a][i.index(h)] < value[b][j.index(h)]:
                    terms[a].remove(h)
                else:
                    terms[a].remove(h)
                    terms[b].remove(h)

    persen = 1/100
    choice = [terms[i][0:int(persen*Ns[i])] for i in range(len(terms))]
    return choice

def tostr(labels):
    return ' '.join([str(i) for i in labels])

def NaiveBayes(k,n,d_latih,l_latih,d_uji,l_uji,choice,clas,PC):
    print('   HASIL NAIVE BAYES:')

    start = time.time()

    PW = []
    for i in range(k):
        x = d_latih[np.isin(l_latih,i)]
        flat = np.concatenate(x).tolist()
        temp = []
        for j in choice:
            temp.append((flat.count(j)+1)/(len(flat)+len(choice)))
        PW.append(temp)
    PW = np.asarray(PW)

    label_hasil = []
    for doc in d_uji:
        temp = []
        for x in range(k):
            jumlah = PC[x]
            for y,kata in enumerate(choice):
                c_kata = doc.count(kata)
                if c_kata == 0:
                    continue
                jumlah = jumlah * (c_kata*PW[x,y])
            temp.append(jumlah)
        label_hasil.append(np.argmax(temp))

    finish = time.time()

    F,P = F_Purity(label_hasil,l_uji,k)
    A = Accuracy(label_hasil,l_uji)
    T = finish-start
    
    print('   <> F-Measure:',F)
    print('   <> Purity   :',P)
    print('   <> Accuracy :',A)
    print('   <> Times    :',T,'seconds\n')
    return [F,P,A,T,tostr(label_hasil)]

def guKernel(a,b,h):
    return (np.exp(a-b)**2/(2*h**2))/((2*np.pi)**0.5)

def KernelNaiveBayes(k,n,v,d_latih,l_latih,d_uji,l_uji,dict_latih,choice,clas,PC):
    print('   HASIL KERNEL NAIVE BAYES:')

    start = time.time()
    
    TF = []
    for i in d_latih:
        temp = []
        for j in dict_latih:
            temp.append(i.count(j)/len(i))
        TF.append(temp)
    TF = np.asarray(TF)

    DF = []
    for i in dict_latih:
        count = 0
        for j in d_latih:
            if i in j:
                count += 1
        DF.append(count)
    DF = np.asarray(DF)

    IDF = np.asarray([np.log10(n/i) for i in DF])

    TFIDF = np.asarray([IDF*i for i in TF])
    indx = [dict_latih.index(i) for i in choice]

    h = 1

    PW2 = []
    for j in range(len(clas)):
        Pxc = []
        X = TFIDF[np.isin(l_latih,j)][:,indx]
        for i in range(len(choice)):
            Nc  = len(X)
            temp = 0
            for v in range(Nc):
                temp += guKernel(X[v,i],X[:,i].sum(),h)
            Pxc.append(temp/(Nc*h))
        PW2.append(Pxc)
    PW2 = np.asarray(PW2)

    label_hasil = []
    for doc in d_uji:
        temp = []
        for x in range(k):
            jumlah = PC[x]
            for y,kata in enumerate(choice):
                c_kata = doc.count(kata)
                if c_kata == 0:
                    continue
                jumlah = jumlah * (c_kata*PW2[x,y])
            temp.append(jumlah)
        label_hasil.append(np.argmax(temp))

    finish = time.time()

    F,P = F_Purity(label_hasil,l_uji,k)
    A = Accuracy(label_hasil,l_uji)
    T = finish-start
    
    print('   <> F-Measure:',F)
    print('   <> Purity   :',P)
    print('   <> Accuracy :',A)
    print('   <> Times    :',T,'seconds\n')
    
    return [F,P,A,T,tostr(label_hasil)]


#BACA DATA
#---------------------------------------------------------------------------------------
raw_data_latih,label_latih = getRawData('latih') #bukhari + muslim
flat_data_latih,dict_latih = getData(raw_data_latih)

#BAGI DATA TIAP KELAS
#---------------------------------------------------------------------------------------
d_all,l_all = [],[]
for i in range(len(clas)):
    x = np.arange(len(label_latih))[np.isin(label_latih,i)]
    y = round(len(x)/3)
    tmp1,tmp2 = [],[]
    for j in range(3):
        tmp1.append(flat_data_latih[x[j*y:(j+1)*y]])
        tmp2.append(np.asarray(label_latih)[x[j*y:(j+1)*y]])
    d_all.append(tmp1)
    l_all.append(tmp2)

a = [True,False]
b = [list(i) for i in itertools.product(a,a,a) if sum(i) == 1] #aaa karena dibagi 3 data
c = list(itertools.product(b,b,b,b,b))

nameclas = []
for i in clas:
    for j in range(3):
        nameclas.append((i+'_'+str(j+1)).upper())
nameclas = np.asarray(nameclas)

#KLASIFIKASI PADA MASING-MASING KOMBINASI DATA UJI & DATA LATIH
#---------------------------------------------------------------------------------------
RESULTS,MYSTR,CHOICES = [],[],[]
for a,b,c,d,e in c:
    mystr = ', '.join(nameclas[np.concatenate([a,b,c,d,e])])
    MYSTR.append(mystr)
    print('>> DATA UJI  :',mystr)
    print('>> DATA LATIH: Selainnya')
    
    data_latih,data_uji   = [],[]
    label_latih,label_uji = [],[]
    
    for x,i in enumerate([a,b,c,d,e]):
        j = np.isin(i,False)
        
        for v in np.concatenate(np.asarray(d_all)[x][i]):
            data_uji.append(v)
        for v in np.concatenate(np.asarray(l_all)[x][i]):
            label_uji.append(v)

        for v in np.concatenate(np.asarray(d_all)[x][j]):
            data_latih.append(v)
        for v in np.concatenate(np.asarray(l_all)[x][j]):
            label_latih.append(v)

    data_latih = np.asarray(data_latih)
    data_uji   = np.asarray(data_uji)
    
    dict_latih = sorted(set(np.concatenate(data_latih)))

    N = len(data_latih)
    K = len(clas)
    V = len(dict_latih)
 
    choice = PilihKata(K,N,data_latih,label_latih,clas)
    CHOICES.append([' '.join(i) for i in choice])
    
    choice = sorted(set(np.concatenate(choice)))

    PC = []
    for i in range(K):
        x = data_latih[np.isin(label_latih,i)]
        PC.append(len(x)/N)

    X = NaiveBayes(K,N,data_latih,label_latih,data_uji,label_uji,choice,clas,PC)
    Y = KernelNaiveBayes(K,N,V,data_latih,label_latih,data_uji,label_uji,dict_latih,choice,clas,PC)

    RESULTS.append(np.concatenate((X,Y)))
    line(60)

#SIMPAN DATA
#---------------------------------------------------------------------------------------
df = pd.DataFrame(RESULTS)
df.columns = ['F_NB','P_NB','A_NB','T_NB','CLAS_NB',
              'F_KNB','P_KNB','A_KNB','T_KNB','CLAS_KNB']

df.index = MYSTR
df.to_csv('RESULTS.csv')

f = open('TERM YANG DIGUNAKAN.txt','w',encoding='utf-8')
for x,i in enumerate(CHOICES):
    for y,j in enumerate(clas):
        j = j.upper()
        f.write(str(x+1)+'\t\t'+j+'\t\t'+i[y]+'\n')
    f.write('\n')
f.close()
