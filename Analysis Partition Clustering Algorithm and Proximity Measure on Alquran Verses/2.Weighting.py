print('--------------------------------')
print('|| WEIGHTING SURAT AL-BAQARAH ||')
print('--------------------------------')

from nltk.tokenize import word_tokenize as T
import numpy as np, pandas as pd

with open('DataPrePro.csv') as f:
    reader = f.readlines()
reader = [i.strip().replace(',',' ') for i in reader]
data = [[i for i in T(row)] for row in reader]
dics = list(set(np.concatenate(data)))

TF = [] #Term Frequency
for i in data:
    TF.append([i.count(j)/len(i) for j in dics])

DF = [] #Document Frequency
for i in dics:
    count = 0
    for j in data:
        if i in j:
            count += 1
    DF.append(count)

IDF = [np.log10(len(data)/i) for i in DF] #Inverse D.F.

#Weight
TFIDF = np.array([np.array(i)*np.array(IDF) for i in TF])

df1 = pd.DataFrame(TFIDF)
df1.to_csv('DataWeight.csv',index=False,header=False)
df2 = pd.DataFrame(dics)
df2.to_csv('Dictionary.csv',index=False,header=False)
