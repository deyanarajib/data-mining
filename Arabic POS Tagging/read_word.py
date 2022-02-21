import re, numpy as np
from extended_buckwalter import transliterate as to_arabic, transliterate2 as to_arabic2, getletter

noun_varians = 'N PN ADJ IMPN PRON DEM REL T LOC'

def map_tags(tags):
    if tags in noun_varians:
        return 'N'
    if tags == 'V':
        return tags
    return 'P'

class struct:
    def __init__(self,form,tags,feat):
        self.form = form
        self.tags = tags
        self.feat = feat

def getkey(x):
    y = ':'.join([str(i) for i in x])
    y = y+':1'*(4-len(x))
    return y

f = open('quranic-corpus-morphology-0.4.txt').read()

get_data  = {}
locations = []
for i in f.splitlines()[57:]:
    locs, form, tags, feat = i.split('\t')
    form = '<blnk>' if form == '' else form
    locs = re.sub('[()]','',locs)
    feat = feat.split('|')[0]

    locations.append(locs.split(':'))
    get_data[locs] = struct(form,tags,feat)
locations = np.int64(locations)

A,B,C,D = [max(locations[:,i])+1 for i in range(4)]

header = """
LOCATIONS
TAGS
MAP_TAGS
WORD
WORD_FORM
REMOVE_PREF/SUFF
PREFIX
SUFFIX
""".split()

space  = [9,9,8,21,24,16,12,0]

header = ' | '.join([i+' '*(j-len(i)) for i,j in zip(header,space)])
header = '='*len(header)+'\n'+header+'\n'+'='*len(header)+'\n'

f = open('properties.txt','w')
f.write(header)

for a in range(1,A):
    
    if getkey([a]) not in get_data:
        break
    
    for b in range(1,B):
        
        if getkey([a,b]) not in get_data:
            break
        
        for c in range(1,C):

            ckey = getkey([a,b,c])
            if ckey not in get_data:
                break
            ckey = ckey.split(':')[:-1]
            
            prefix, suffix, postag, posmap, word, forms  = [], [], [], [], [], []
            for d in range(1,D):
                
                if getkey([a,b,c,d]) not in get_data:
                    break
                
                dkey = getkey([a,b,c,d])
                form = get_data[dkey].form
                tags = get_data[dkey].tags
                feat = get_data[dkey].feat

                if feat == 'PREFIX':
                    prefix.append(form)
                if feat == 'SUFFIX':
                    suffix.append(form)
                if feat == 'STEM':
                    forms.append(form)
                    postag.append(tags)
                    if map_tags(tags) not in posmap:
                        posmap.append(map_tags(tags))
                word.append(form)

            mystring = []
            for x,i in enumerate([ckey,postag,posmap,word,word,forms,prefix,suffix]):
                if x == 0: char = ':'
                elif x == 3: char = ''
                else: char = ' '
                
                temp = char.join(i)
                temp = '-' if temp == '' else temp
                temp = temp+' '*(space[x]-len(temp))

                mystring.append(temp)

            f.write(' | '.join(mystring)+'\n')

f.close()

data = []
a = 1
while True:
    if ':'.join([str(z) for z in [a,1,1,1]]) not in get_data:
        break
        
    b = 1
    while True:
        
        if ':'.join([str(z) for z in [a,b,1,1]]) not in get_data:
            break
        
        c = 1
        while True:

            if ':'.join([str(z) for z in [a,b,c,1]]) not in get_data:
                break
            
            d = 1
            form, tags, feat = [], [], []
            while True:

                key = ':'.join([str(z) for z in [a,b,c,d]])
                if key not in get_data:
                    break
                form.append(get_data[key].form)
                tags.append(get_data[key].tags)
                feat.append(get_data[key].feat)
                d += 1

            data.append([''.join(form),' '.join(tags),' '.join(feat)])
            c += 1
        b += 1
    a += 1

locations = sorted(set([':'.join(i.split(':')[:-1]) for i in locations]))

feature = {}
for key in locations:
    
    form, tags, feat = [], [], []

    j = 1
    while key+':'+str(j) in get_data:
        form.append(get_data[key+':'+str(j)].form)
        tags.append(get_data[key+':'+str(j)].tags)
        feat.append(get_data[key+':'+str(j)].feat)
        j += 1
    
    feature[key] = struct(' '.join(form),' '.join(tags),' '.join(feat))


maxsurah, maxayahs, maxwords = 0, 0, 0
for i in locations:
    a,b,c = [int(j) for j in i.split(':')]
    maxsurah = a if a > maxsurah else maxsurah
    maxayahs = b if b > maxayahs else maxayahs
    maxwords = c if c > maxwords else maxwords

data = open('transliterasi.txt','w')
feat = open('features.txt','w')
for i in range(1,maxsurah+1):
    for j in range(1,maxayahs+1):
        
        ayahs = []
        for v in range(1,maxwords+1):
            
            key = ':'.join([str(z) for z in [i,j,v]])
            
            if key not in feature: break

            token = ''.join(feature[key].form.split())
            ayahs.append(token)

            if 'PREFIX' in feature[key].feat:
                indx = feature[key].feat.split().index('PREFIX')
                pref = feature[key].form.split()[indx]
                #tags = feature[key].tags.split()
                #tags = ' '.join([tags[w] for w in range(len(tags)) if w != indx])
            else:
                pref = ''
            pref = pref+' '*(7-len(pref))

            if 'SUFFIX' in feature[key].feat:
                indx = feature[key].feat.split().index('SUFFIX')
                suff = feature[key].form.split()[indx]
                #tags = feature[key].tags.split()
                #tags = ' '.join([tags[w] for w in range(len(tags)) if w != indx])
            else:
                suff = ''
            suff = suff+' '*(7-len(suff))

            token = token+' '*(22-len(token))
            tags  = feature[key].tags
            feat.write(' | '.join([token,pref,suff,tags])+'\n')
            
        if ayahs == []:
            break
        ayahs = ' '.join(ayahs)
        data.write(str(i+1)+' | '+str(j+1)+' | '+ayahs+'\n')
data.close()
feat.close()
