print('-------------------------------------')
print('|| PRE-PROCESSING SURAT AL-BAQARAH ||')
print('-------------------------------------')

#READ DATA
#---------------------------------------------------------------------------
with open('ahmedali.txt',newline='') as f:
    allayat = f.readlines()
allayat = [x.strip() for x in allayat]
raw = []
for ayat in allayat:
   if ayat.startswith('2|'):
      raw.append(ayat[len(str(len(raw)+1))+3::])

#MODUL YANG DIGUNAKAN 
#---------------------------------------------------------------------------
from nltk.tokenize import word_tokenize as T
from nltk import pos_tag as PT
from nltk.corpus import stopwords
from string import punctuation as PC
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as W
import numpy as np, pandas as pd

#FUNCTON
#---------------------------------------------------------------------------
def pos(x):
   if x == 'VERB': return 'v'
   elif x == 'NOUN': return 'n'
   elif x == 'ADJ': return 'a'
   elif x == 'ADJ_SAT': return 's'
   elif x == 'ADV': return 'r'
   return 'unknown'

def getXYZ(dicti,opt):
   x = [i for i,j in dicti]
   y = [j for i,j in dicti]
   z = set(y)
   print1,print2,count = [],[],0
   for i in z:
      if y.count(i) > 1:
         count += 1
         name = 'syn'+str(count) if opt else i
         idx = np.isin(y,i)
         print1.append(name)
         print2.append(' '.join([j for j in np.array(x)[idx]]))
   return print1,print2

def NTerm(D):
   if D == []:
      return '-'
   x = np.concatenate(D)
   if len(D[0][0]) == 2:
      y = x[:,0]
      return len(set(y))
   return len(set(x))

#PRE-PROCESSING
#---------------------------------------------------------------------------
punc = list(PC)+["``",'--',"''"]
pron = ['whosoever','wherever','whereby','whomsoever','whichever','whether','whoever','yet'
        'whole','whims','whatsoever','whose','wheresoever','whenever','whatever',"'s","'re"]
proc = ['RAW Data','Tokenizing','Case Folding','Stop Words Removal','Lemmatizing','Synonymizing']
SW = list(stopwords.words('english'))+pron
L = WordNetLemmatizer()

d0 = [T(doc) for doc in raw]
d1,d2,d3,d4,d5,d6 = [],[],[],[],[],[]

dictio = []
for x,D in enumerate(raw):
   #Tokenizing
   d1.append([i for i in T(D) if i not in punc])

   #POS Tagging
   taguniv = PT(d1[x],tagset='universal')
   d2.append([(i,pos(j)) for i,j in taguniv])

   #Case Folding
   d3.append([(str.lower(i),j) for i,j in d2[x]])

   #Stop Words Removal
   d4.append([(i,j) for i,j in d3[x] if i not in SW and len(i) > 2])
   for i,j in d4[x]:
      if (i,j) not in dictio:
         dictio.append((i,j))

#Fixing Error Tag
alltrm = [i for i,j in dictio]
errtag = [i for i in set(alltrm) if alltrm.count(i) > 1]
fixtag = []
for word in errtag:
   try:
      fixtag.append((word,W.synsets(word)[0].pos()))
   except:
      fixtag.append((word,'unknown'))
allfix = [i for i,j in fixtag]
d4fix = []
for doc in d4:
   temp = []
   for i,j in doc:
      if i in allfix:
         j = fixtag[allfix.index(i)][1]
      if j == 'unknown':
         continue
      temp.append((i,j))
   d4fix.append(temp)

print('CHOOSE PREPROCESSING SCHEME')
print('1.Using Synonymizing')
print('2.Without Synonymizing\n')
opt = input('Option: '); usesyn = True if opt == '1' else False

#Lemmatizing
dictlema = []
for doc in d4fix:
   temp = []
   for i,j in doc:
      lema = L.lemmatize(i,j)
      if (i,lema) not in dictlema:
         dictlema.append((i,lema))
      temp.append((lema,j) if usesyn else lema)
   d5.append(temp)

P1,P2 = getXYZ(dictlema,False)
df = pd.DataFrame({'LEMMA':P1,
                   'SET WORD':P2})
print('\n',df,'\n')

if usesyn:
   #Synonymizing
   dictdeff = []
   for doc in d5:
      for i,j in doc:
         try:
            deff = W.synsets(i,j)[0].definition()
         except:
            deff = i
         if (i,deff) not in dictdeff:
            dictdeff.append((i,deff))
         
   P1,P2 = getXYZ(dictdeff,True)
   P3 = [['syn'+str(i+1)]*len(T(P2[i])) for i in range(len(P2))]

   df = pd.DataFrame({'CODE SYNONYM':P1,
                      'SET LEMNA'   :P2})
   print(df,'\n')

   exp1 = T(' '.join(P2))
   exp2 = np.concatenate(P3).tolist()

   for doc in d5:
      temp = []
      for i,j in doc:
         if i in exp1:
            i = exp2[exp1.index(i)]
         temp.append(i)
      d6.append(temp)

   df = pd.DataFrame(P2)
   df.to_csv('SetSyns.csv',index=False,header=False)

df = pd.DataFrame({'METHOD'    : proc,
                   'TOTAL TERM': [NTerm(i) for i in [d0,d1,d3,d4,d5,d6]]})
print(df,'\n')

datasave = d6 if usesyn else d5
df = pd.DataFrame(datasave)
df.to_csv('DataPrePro.csv',index=False,header=False)
