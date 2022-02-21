import warnings
warnings.filterwarnings('ignore')

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
import re, numpy as np, os, pandas as pd, time

def goplot(clus,k,T):
    for i in range(k):
        plt.scatter(X[clus==i],Y[clus==i])
    plt.title(T+' K='+str(k))
    plt.legend(['C'+str(i+1) for i in range(k)])
    plt.show()

path = './dataset/'

print('PILIH DATASET')
for i,file in enumerate(os.listdir(path)):
    print(str(i+1)+'. '+file.replace('.xlsx',''),end=', ')
    data = pd.read_excel(path+file)
    data = np.float64(data.values)
    print(len(data),'data')
opt = input('input opsi: '); opt = int(opt)-1

for i,file in enumerate(os.listdir(path)):
    if i != opt: continue
    print('\nDATASET',file.replace('.xlsx','').upper(),'\n')
    data = pd.read_excel(path+file)
    data = np.float64(data.values)
X = data[:,0]
Y = data[:,1]

#CLUSTERING
from sklearn.metrics.pairwise import pairwise_distances
from initial_centroid import neighbors_and_link
from clustering_method import kmeans, kmedoid_pam, kmedoid_non_pam
from sklearn.metrics import davies_bouldin_score, silhouette_score

K = input('Input K: '); K = int(K); print('')
distbank = pairwise_distances(data)
initcent = neighbors_and_link(distbank,K)

print('METODE CLUSTERING')
print('1. K-MEANS')
print('2. PAM')
opt = ' '
while opt not in '12':
    opt = input('input opsi: ')
print('')

if opt == '1':
    #K-MEANS
    start = time.time()
    clus_kmn = kmeans(initcent,data)
    finish = time.time()-start
    db = davies_bouldin_score(data,clus_kmn)
    ss = silhouette_score(data,clus_kmn)
    print('DAVIES BOULDIN INDEX:',db)
    print('SILHOUETTE SCORE    :',ss)
    print('RUNTIME             :',finish,'detik')
    goplot(clus_kmn,K,'K-MEANS')
else:
    #K-MEDOID PAM
    start = time.time()
    clus_kd2,mkd2 = kmedoid_pam(initcent,distbank)
    finish = time.time()-start
    db = davies_bouldin_score(data,clus_kd2)
    ss = silhouette_score(data,clus_kd2)
    print('DAVIES BOULDIN INDEX:',db)
    print('SILHOUETTE SCORE    :',ss)
    print('RUNTIME             :',finish,'detik')
    goplot(clus_kd2,K,'PAM')
