
import clustData
import computeKVs
import numpy as np
import csv
import math
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

#%% ############################# 1. Input params #############################
def cdsmote(file):
    classdecomp = 'Kmeans' # 'FCmeans', 'FCmeansOptimised' and 'DBSCAN' also available 
    oversampler = 'SMOTE' #'ADASYN' also available
    threshold = 10 # if samples in positive class are apart from average by more than this value, apply oversampling (Sec 3.2 paper)

    n_clusters = 2 # used in options "all" and majority
    k_type = 'majority' # Indicates how to calculate k values for class decomposition

    ## Load dataset
    with open(file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        num = 1
        data = []
        target = []
        for row in reader:
            if num==1:
                num +=1
                continue
            else:
                data.append(list(map(float,row[0:len(row)-1])))
                target.append(row[-1])
    del row, reader, f
        
    ## Find majority and minority classes
    majority_class = max(set(target), key=target.count)
    minority_class = min(set(target), key=target.count)

    histo = [['Class','Number of Samples']]
    for i, label1 in enumerate(sorted(list(set(target)))):
        cont = 0
        for j, label2 in enumerate(target):
            if label1 == label2:
                cont+=1
        histo.append([label1,cont])
    histo.append(['Total Samples', len(target)])

    ## Load as a panda
    histo_panda = pd.DataFrame.from_records(histo[1:-1], columns=histo[0])
        
    #%% ######################### 3. Class decomposition #########################

    ## Calculate k vector (for class decomposition)

    if k_type.lower() == 'majority':
        k = computeKVs.majority(data, target, n_clusters)
    elif k_type.lower() == 'majorityimbalanceratio':
        n_clusters = math.ceil(fileMatch[4])
        k = computeKVs.majority(data, target, n_clusters)
    else:
        print('Invalid k values option for kmeansSMOTE')
        sys.exit()

    ## Cluster the data
    if classdecomp.lower()=='kmeans':
        target_cd = clustData.Kmeans(data, target, k)
    elif classdecomp.lower()=='fcmeans':
        target_cd = clustData.FCmeans(data, target, k)
    elif classdecomp.lower()=='fcmeansoptimised':
        target_cd = clustData.FCmeansOptimised(data, target, k, max_nclusters = 10)   
    elif classdecomp.lower()=='dbscan':        
        target_cd = clustData.DBSCAN(data, target, k, eps=0.5, min_samples=5)
    else:
        print('Invalid class decomposition algorithm.')
        sys.exit()
        
    # Plot distribution after cd
    histo = [['Class','Number of Samples']]
    for i, label1 in enumerate(sorted(list(set(target_cd)))):
        cont = 0
        for j, label2 in enumerate(target_cd):
            if label1 == label2:
                cont+=1
        histo.append([label1,cont])
    histo.append(['Total Samples', len(target_cd)])
    ## Load as a panda
    histo_panda = pd.DataFrame.from_records(histo[1:-1], columns=histo[0])
    indexesUnique = list(set(target_cd))
    indexesUnique.sort()
    newclassdist_count = []
    for newclass in indexesUnique:
        newclassdist_count.append(target_cd.count(newclass))
    average = sum(newclassdist_count)/len(newclassdist_count)

        
    #%% ############################ 4. Oversampling #############################
    ## 1. see if the positive is far from the average (larger than the threshold)
    if abs(average-newclassdist_count[-1])>threshold:
        print('Oversampling original minority class...')
        ## 2. calculate new majority class for smote (closest to the average)
        c = np.inf
        newmaj = 0
        for i,j in enumerate(newclassdist_count[0:-1]):
            if abs(j-average)<c:
                c = abs(j-average)
                newmaj = i
        majority_class_new = majority_class+'_c'+str(newmaj)
        minority_class_new = minority_class+'_c0'
        ## 3. Create the dataset that only contains the new majority and minority classes
        data_majmin = []
        target_majmin = []
        for m, label in enumerate(target_cd):
            if label == majority_class_new or label == minority_class_new:
                data_majmin.append(data[m])
                target_majmin.append(label)
        if oversampler.lower() == 'smote':
            sm = SMOTE()
            data_over, target_over = sm.fit_resample(data_majmin, target_majmin) 
        elif oversampler.lower() == 'adasyn':
            ada = ADASYN()
            data_over, target_over = ada.fit_resample(data_majmin, target_majmin)
        else:
            print('Invalid oversampling algorithm.')
            sys.exit() 
        ## 4. combine this with the remaning classes
        data_cdsmote = data_over.copy()
        target_cdsmote = target_over.copy()
        for m, label in enumerate(target_cd):
            if label != minority_class_new and label != majority_class_new:
                data_cdsmote.append(data[m])
                target_cdsmote.append(label)
        a=pd.DataFrame(np.array(data_cdsmote))
        a = pd.concat([a,pd.DataFrame({'label':np.array(target_cdsmote)})],axis = 1)
        a.iloc[a[a['label']=='0_c0'].index,-1]=0
        a.iloc[a[a['label']=='0_c1'].index,-1]=0
        a.iloc[a[a['label']=='1_c1'].index,-1]=1
        a.iloc[a[a['label']=='1_c0'].index,-1]=1
        return a.iloc[:,:-1].values,a.iloc[:,-1].values

                
            

