import numpy as np
import pandas as pa
import random

def file_read(filename):
    file = pa.read_csv(filename)
    return file[file.iloc[:,-1].values==0].iloc[:,:-1].values,file[file.iloc[:,-1].values==1].iloc[:,:-1].values,file[file.iloc[:,-1].values==0].iloc[:,-1].values,file[file.iloc[:,-1].values==1].iloc[:,-1].values
# 基因变异
def gene_saltation(sample0,sample1,lamda):
    return sample0*lamda+sample1 *(1 - lamda)
def gene_cross(new_sample,sample1):
    result = random.choices(gene_list, weights=[1 - gene_saltation_precent, gene_saltation_precent], k=lth)
    # print(new_sample)
    for i in range(lth):
        if(result[i]==1):
            new_sample[i]=sample1[i]
    return new_sample
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power((vecA - vecB), 2)))
def check_krum(sample0,sample1):
    center1 = np.sum(sample1,axis = 0)
    krum_list = []
    for i in sample0:
        krum_list.append(distEclud(i,center1))
    return np.min(krum_list),center1
def check_label(krum,sample1,new):
    if(distEclud(sample1,new)>krum):
        return True
    else:
        return False
def sample_choose(sample0,sample1,num):
    new_list = []
    while num>0:
        # print(gene_saltation_precent)
        result = random.choices(gene_list,weights=[1-gene_saltation_precent,gene_saltation_precent],k = lth)
        # print(result)
        lambda_list = np.random.random_sample([1, lth])[0,:]
        lambda_list = np.around(lambda_list, decimals=features_len)
        lambda_list = lambda_list*result
        for i in range(lth):
            if(result[i]==0):
                lambda_list[i]=0.5
        new_gene = gene_saltation(random.choice(sample0),random.choice(sample1),lambda_list)
        new_gene = gene_cross(new_gene,random.choice(sample1))
        new_list.append(new_gene)
        num = num - 1
    return np.array(new_list)
def sample_choose_cross(sample0,sample1,num,krum):
    new_list = []
    while num>0:
        result = random.choices(gene_list,weights=[1-gene_saltation_precent,gene_saltation_precent],k = lth)
        lambda_list = np.random.random_sample([1, lth])[0,:]
        lambda_list = np.around(lambda_list, decimals=features_len)
        lambda_list = lambda_list*result
        for i in range(lth):
            if(result[i]==0):
                lambda_list[i]=0.5
        new_gene = gene_saltation(random.choice(sample0),random.choice(sample1),lambda_list)
        new_gene = gene_cross(new_gene,random.choice(sample1))
        if(check_label(krum,center,new_gene)):
            new_list.append(new_gene)
            num = num - 1
    return np.array(new_list)
def three_series(data = 'ecoli1.csv',new_num=1000):
    global need_num,features_len,gene_list,gene_saltation_precent,lth,krum,center
    feature0,feature1,label0,label1 = file_read(data)
    krum,center= check_krum(feature0,feature1)
    lth = feature0.shape[1]
    gene_saltation_precent = float(1/feature1.shape[1])
    gene_list = [0,1]
    features_len = len(str(feature0[0,0]).split(".")[1])
    need_num = len(feature0)-len(feature1)
    keep_data = sample_choose(feature0,feature1,new_num)
    keep_data1 = sample_choose_cross(keep_data,feature1,need_num,krum)
    keep_data1 = np.around(keep_data1, decimals=features_len)
    enddata = np.concatenate((keep_data1, np.ones((need_num,1))), axis=1)
    enddata0 = np.concatenate((feature0, label0.reshape(-1,1)), axis=1)
    enddata1 = np.concatenate((feature1, label1.reshape(-1,1)), axis=1)
    enddata = np.concatenate((enddata, enddata0), axis=0)
    return np.concatenate((enddata, enddata1), axis=0)
def three_series_train(x_train,y_train,new_num=1000):
    global need_num,features_len,gene_list,gene_saltation_precent,lth,krum,center
    # feature0,feature1,label0,label1 = file_read(data)
    feature = pa.DataFrame(np.concatenate((x_train, y_train.reshape(-1,1)), axis=1))
    feature0 = feature[feature.iloc[:,-1]==0].iloc[:,:-1].values
    feature1 = feature[feature.iloc[:,-1]==1].iloc[:,:-1].values
    label0 = feature[feature.iloc[:,-1]==0].iloc[:,-1].values
    label1 = feature[feature.iloc[:,-1]==1].iloc[:,-1].values
    krum,center= check_krum(feature0,feature1)
    lth = feature0.shape[1]
    gene_saltation_precent = float(1/feature1.shape[1])
    gene_list = [0,1]
    if (len(str(feature0[0,0]).split("."))==2):
        features_len = len(str(feature0[0,0]).split(".")[1])
    else:
        features_len = 2
    need_num = len(feature0)-len(feature1)
    keep_data = sample_choose(feature0,feature1,new_num)
    keep_data1 = sample_choose_cross(keep_data,feature1,need_num,krum)
    keep_data1 = np.around(keep_data1, decimals=features_len)
    enddata = np.concatenate((keep_data1, np.ones((need_num,1))), axis=1)
    enddata0 = np.concatenate((feature0, label0.reshape(-1,1)), axis=1)
    enddata1 = np.concatenate((feature1, label1.reshape(-1,1)), axis=1)
    enddata = np.concatenate((enddata, enddata0), axis=0)
    return np.concatenate((enddata, enddata1), axis=0)[:,:-1],np.concatenate((enddata, enddata1), axis=0)[:,-1].astype('int')

# if __name__ == '__main__':
#     new_num = 1000
#     data = 'data/vehicle0.csv'
#     feature0,feature1,label0,label1 = file_read(data)
#     krum,center= check_krum(feature0,feature1)
#     print(krum)
#     lth = feature0.shape[1]
#     gene_saltation_precent = float(1/feature1.shape[1])
#     gene_list = [0,1]
#     if (len(str(feature0[0,0]).split("."))==2):
#         features_len = len(str(feature0[0,0]).split(".")[1])
#     else:
#         features_len = 2
#     need_num = len(feature0)-len(feature1)
#     keep_data = sample_choose(feature0,feature1,new_num)
#     keep_data1 = sample_choose_cross(keep_data,feature1,need_num,krum)
#     keep_data1 = np.around(keep_data1, decimals=features_len)
# print("123".split("."))


