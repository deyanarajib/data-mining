iman, shalat, zakat, puasa, haji = [],[],[],[],[]
for i in ['iman','puasa','haji','shalat','zakat']:
    temp = []
    for j in range(1,3):
        f = open(i+'('+str(j)+')'+'.txt','r',encoding='utf-8')
        temp.append(f)
        import x
    eval(i).append(temp)
