import warnings

from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTEENN,SMOTETomek
from sklearn import svm
from sklearn import tree
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler,EditedNearestNeighbours,NeighbourhoodCleaningRule ,OneSidedSelection,ClusterCentroids
from imblearn.over_sampling import SMOTE, ADASYN,SMOTEN,SVMSMOTE,SMOTENC,BorderlineSMOTE,KMeansSMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from Three_series import three_series_train
import math
import pandas as pa
from sklearn import metrics
warnings.filterwarnings("ignore")
def Gmean(tru,pre):
    s_t =f_t =0
    # print(tru)
    tu = pa.DataFrame(tru,columns = ['1'])
    f = tu['1'].value_counts().values[0]
    s = tu['1'].value_counts().values[1]
    for i in range(len(pre)):
        if(pre[i]==1 and tru[i]==1):
            f_t = f_t+1
        elif(pre[i]==0 and tru[i]==0):
            s_t = s_t+1
    print(s_t)
    print(f_t)
    return math.sqrt(float(s_t*1.0*f_t/s/f))
def read_data(filename):
    data =pa.read_csv(filename)
    data:pa.DataFrame = data.sample(frac=1.0)
    return data.iloc[:,:-1].values,data.iloc[:,-1].values
def SMOTEdata(x_train,y_train):
    return SMOTE().fit_resample(x_train,y_train)
def ADASYNdata(x_train,y_train):
    return ADASYN().fit_resample(x_train, y_train)
def SMOTENdata(x_train,y_train):
    return SMOTEN().fit_resample(x_train,y_train)
def RandomOverSamplerdata(x_train,y_train):
    return RandomOverSampler().fit_resample(x_train,y_train)
def BorderlineSMOTEdata(x_train,y_train):
    return BorderlineSMOTE().fit_resample(x_train,y_train)
# def KMeansSMOTEdata(x_train,y_train):
#     return KMeansSMOTE(cluster_balance_threshold=0.01).fit_resample(x_train,y_train)
# def SMOTENCdata(x_train,y_train):
#     return SMOTENC(random_state=42,categorical_features=['0','1','2','3','4','5','6']).fit_resample(x_train,y_train)
def SVMSMOTEdata(x_train,y_train):
    return SVMSMOTE().fit_resample(x_train,y_train)
def ClusterCentroidsdata(x_train,y_train):
    return ClusterCentroids().fit_resample(x_train,y_train)
def EditedNearestNeighboursdata(x_train,y_train):
    return EditedNearestNeighbours().fit_resample(x_train,y_train)
def OneSidedSelectiondata(x_train,y_train):
    return OneSidedSelection().fit_resample(x_train,y_train)
def NeighbourhoodCleaningRuledata(x_train,y_train):
    return NeighbourhoodCleaningRule().fit_resample(x_train,y_train)
def SMOTEENNdata(x_train,y_train):
    return SMOTEENN().fit_resample(x_train,y_train)
def SMOTETomekdata(x_train,y_train):
    return SMOTETomek().fit_resample(x_train,y_train)
def RandomUnderSamplerdata(x_train,y_train):
    return RandomUnderSampler().fit_resample(x_train,y_train)
def SMOTENCdata(x_train, y_train):
    return SMOTENC(categorical_features=[0, 1]).fit_resample(x_train, y_train)
def TomekLinksdata(x_train, y_train):
    return TomekLinks().fit_resample(x_train, y_train)
def calssifier_sample(index1,index2,X,Y,clf):
    # 三系
    us_precision = us_gmean = us_recall = us_f1 = us_roc = us_mcc = us_kappa = 0
    # smote
    smote_precision = smote_gmean = smote_recall = smote_f1 = smote_roc = smote_mcc = smote_kappa = 0
    # ADASYNdata
    adasyn_precision = adasyn_gmean = adasyn_recall = adasyn_f1 = adasyn_roc = adasyn_mcc = adasyn_kappa = 0
    # SMOTEN
    smoten_precision = smoten_gmean = smoten_recall = smoten_f1 = smoten_roc = smoten_mcc = smoten_kappa = 0
    #RandomOverSampler
    randomover_precision = randomover_gmean = randomover_recall = randomover_f1 = randomover_roc = randomover_mcc = randomover_kappa = 0
    #BorderlineSMOTE
    BorderlineSMOTE_precision = BorderlineSMOTE_gmean = BorderlineSMOTE_recall = BorderlineSMOTE_f1 = BorderlineSMOTE_roc = BorderlineSMOTE_mcc = BorderlineSMOTE_kappa = 0
    #SVMSMOTE
    SVMSMOTE_precision = SVMSMOTE_gmean = SVMSMOTE_recall = SVMSMOTE_f1 = SVMSMOTE_roc = SVMSMOTE_mcc = SVMSMOTE_kappa = 0
    # ClusterCentroids
    ClusterCentroids_precision = ClusterCentroids_gmean = ClusterCentroids_recall = ClusterCentroids_f1 = ClusterCentroids_roc = ClusterCentroids_mcc = ClusterCentroids_kappa = 0
    #EditedNearestNeighbours
    EditedNearestNeighbours_precision = EditedNearestNeighbours_gmean = EditedNearestNeighbours_recall = EditedNearestNeighbours_f1 = EditedNearestNeighbours_roc = EditedNearestNeighbours_mcc = EditedNearestNeighbours_kappa = 0
    # OneSidedSelection
    OneSidedSelection_precision = OneSidedSelection_gmean = OneSidedSelection_recall = OneSidedSelection_f1 = OneSidedSelection_roc = OneSidedSelection_mcc = OneSidedSelection_kappa = 0
    #SMOTEENN
    SMOTEENN_precision = SMOTEENN_gmean = SMOTEENN_recall = SMOTEENN_f1 = SMOTEENN_roc = SMOTEENN_mcc = SMOTEENN_kappa = 0
    #SMOTETomek
    SMOTETomek_precision = SMOTETomek_gmean = SMOTETomek_recall = SMOTETomek_f1 = SMOTETomek_roc = SMOTETomek_mcc = SMOTETomek_kappa = 0
    # RandomUnderSampler
    RandomUnderSampler_precision = RandomUnderSampler_gmean = RandomUnderSampler_recall = RandomUnderSampler_f1 = RandomUnderSampler_roc = RandomUnderSampler_mcc = RandomUnderSampler_kappa = 0
    #SMOTENC
    SMOTENC_precision = SMOTENC_gmean = SMOTENC_recall = SMOTENC_f1 = SMOTENC_roc = SMOTENC_mcc = SMOTENC_kappa = 0
    #TomekLink
    TomekLink_precision = TomekLink_gmean = TomekLink_recall = TomekLink_f1 = TomekLink_roc = TomekLink_mcc = TomekLink_kappa = 0
    for train_index,test_index in zip(index1,index2):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # 三系
        three_X_train, three_Y_train= three_series_train(X_train,Y_train,1000)
        # smote
        smote_X_train, smote_Y_train= SMOTEdata(X_train,Y_train)
        # adasyn
        adasyn_X_train, adasyn_Y_train= ADASYNdata(X_train,Y_train)
        # smoten
        smoten_X_train, smoten_Y_train= SMOTENdata(X_train,Y_train)
        # RandomOverSamplerdata
        randomover_X_train, randomover_Y_train= RandomOverSamplerdata(X_train,Y_train)
        #BorderlineSMOTE
        BorderlineSMOTE_X_train, BorderlineSMOTE_Y_train= BorderlineSMOTEdata(X_train,Y_train)
        #SVMSMOTE
        SVMSMOTE_X_train, SVMSMOTE_Y_train= SVMSMOTEdata(X_train,Y_train)
        #ClusterCentroids
        ClusterCentroids_X_train, ClusterCentroids_Y_train= ClusterCentroidsdata(X_train,Y_train)
        #EditedNearestNeighbours
        EditedNearestNeighbours_X_train, EditedNearestNeighbours_Y_train= EditedNearestNeighboursdata(X_train,Y_train)
        #OneSidedSelection
        OneSidedSelection_X_train, OneSidedSelection_Y_train= OneSidedSelectiondata(X_train,Y_train)
        #SMOTEENN
        SMOTEENN_X_train, SMOTEENN_Y_train= SMOTEENNdata(X_train,Y_train)
        #SMOTETomek
        SMOTETomek_X_train, SMOTETomek_Y_train= SMOTETomekdata(X_train,Y_train)
        #RandomUnderSampler
        RandomUnderSampler_X_train, RandomUnderSampler_Y_train= RandomUnderSamplerdata(X_train,Y_train)
        #SMOTENC
        SMOTENC_X_train, SMOTENC_Y_train= SMOTENCdata(X_train,Y_train)
        #omekLinks
        TomekLink_X_train, TomekLink_Y_train= TomekLinksdata(X_train,Y_train)

        # smote
        clf.fit(smote_X_train,smote_Y_train)
        smote_predict = clf.predict(X_test)
        smote_precision1=0
        smote_precision = smote_precision + metrics.classification_report(Y_test, smote_predict, output_dict=True)['0']['precision']
        smote_precision1 = smote_precision1 + metrics.classification_report(Y_test, smote_predict, output_dict=True)['1']['precision']
        smote_gmean = smote_gmean + Gmean(Y_test, smote_predict)
        smote_recall = smote_recall + metrics.classification_report(Y_test, smote_predict, output_dict=True)['0']['recall']
        smote_f1 = smote_f1 + metrics.classification_report(Y_test, smote_predict, output_dict=True)['0']['f1-score']
        smote_roc = smote_roc + metrics.roc_auc_score(Y_test, smote_predict)
        smote_mcc = smote_mcc + metrics.matthews_corrcoef(Y_test, smote_predict)
        smote_kappa = smote_kappa + metrics.cohen_kappa_score(Y_test, smote_predict)
        #adasyn
        clf.fit(adasyn_X_train,adasyn_Y_train)
        adasyn_predict = clf.predict(X_test)

        adasyn_precision = adasyn_precision + metrics.classification_report(Y_test, adasyn_predict, output_dict=True)['0']['precision']
        adasyn_gmean = adasyn_gmean + Gmean(Y_test, adasyn_predict)
        adasyn_recall = adasyn_recall + metrics.classification_report(Y_test, adasyn_predict, output_dict=True)['0']['recall']
        adasyn_f1 = adasyn_f1 + metrics.classification_report(Y_test, adasyn_predict, output_dict=True)['0']['f1-score']
        adasyn_roc = adasyn_roc + metrics.roc_auc_score(Y_test, adasyn_predict)
        adasyn_mcc = adasyn_mcc + metrics.matthews_corrcoef(Y_test, adasyn_predict)
        adasyn_kappa = adasyn_kappa + metrics.cohen_kappa_score(Y_test, adasyn_predict)
        # smoten
        clf.fit(smoten_X_train,smoten_Y_train)
        smoten_predict = clf.predict(X_test)

        smoten_precision = smoten_precision + metrics.classification_report(Y_test, smoten_predict, output_dict=True)['0']['precision']
        smoten_gmean = smoten_gmean + Gmean(Y_test, smoten_predict)
        smoten_recall = smoten_recall + metrics.classification_report(Y_test, smoten_predict, output_dict=True)['0']['recall']
        smoten_f1 = smoten_f1 + metrics.classification_report(Y_test, smoten_predict, output_dict=True)['0']['f1-score']
        smoten_roc = smoten_roc + metrics.roc_auc_score(Y_test, smoten_predict)
        smoten_mcc = smoten_mcc + metrics.matthews_corrcoef(Y_test, smoten_predict)
        smoten_kappa = smoten_kappa + metrics.cohen_kappa_score(Y_test, smoten_predict)
        #randomover
        clf.fit(randomover_X_train,randomover_Y_train)
        randomover_predict = clf.predict(X_test)

        randomover_precision = randomover_precision + metrics.classification_report(Y_test, randomover_predict, output_dict=True)['0']['precision']
        randomover_gmean = randomover_gmean + Gmean(Y_test, randomover_predict)
        randomover_recall = randomover_recall + metrics.classification_report(Y_test, randomover_predict, output_dict=True)['0']['recall']
        randomover_f1 = randomover_f1 + metrics.classification_report(Y_test, randomover_predict, output_dict=True)['0']['f1-score']
        randomover_roc = randomover_roc + metrics.roc_auc_score(Y_test, randomover_predict)
        randomover_mcc = randomover_mcc + metrics.matthews_corrcoef(Y_test, randomover_predict)
        randomover_kappa = randomover_kappa + metrics.cohen_kappa_score(Y_test, randomover_predict)
        # BorderlineSMOTE
        clf.fit(smote_X_train,BorderlineSMOTE_Y_train)
        BorderlineSMOTE_predict = clf.predict(X_test)

        BorderlineSMOTE_precision = BorderlineSMOTE_precision + metrics.classification_report(Y_test, BorderlineSMOTE_predict, output_dict=True)['0']['precision']
        BorderlineSMOTE_gmean = BorderlineSMOTE_gmean + Gmean(Y_test, BorderlineSMOTE_predict)
        BorderlineSMOTE_recall = BorderlineSMOTE_recall + metrics.classification_report(Y_test, BorderlineSMOTE_predict, output_dict=True)['0']['recall']
        BorderlineSMOTE_f1 = BorderlineSMOTE_f1 + metrics.classification_report(Y_test, BorderlineSMOTE_predict, output_dict=True)['0']['f1-score']
        BorderlineSMOTE_roc = BorderlineSMOTE_roc + metrics.roc_auc_score(Y_test, BorderlineSMOTE_predict)
        BorderlineSMOTE_mcc = BorderlineSMOTE_mcc + metrics.matthews_corrcoef(Y_test, BorderlineSMOTE_predict)
        BorderlineSMOTE_kappa = BorderlineSMOTE_kappa + metrics.cohen_kappa_score(Y_test, BorderlineSMOTE_predict)
        # SVMSMOTE
        clf.fit(SVMSMOTE_X_train,SVMSMOTE_Y_train)
        SVMSMOTE_predict = clf.predict(X_test)

        SVMSMOTE_precision = SVMSMOTE_precision + metrics.classification_report(Y_test, SVMSMOTE_predict, output_dict=True)['0']['precision']
        SVMSMOTE_gmean = SVMSMOTE_gmean + Gmean(Y_test, SVMSMOTE_predict)
        SVMSMOTE_recall = SVMSMOTE_recall + metrics.classification_report(Y_test, SVMSMOTE_predict, output_dict=True)['0']['recall']
        SVMSMOTE_f1 = SVMSMOTE_f1 + metrics.classification_report(Y_test, SVMSMOTE_predict, output_dict=True)['0']['f1-score']
        SVMSMOTE_roc = SVMSMOTE_roc + metrics.roc_auc_score(Y_test, SVMSMOTE_predict)
        SVMSMOTE_mcc = SVMSMOTE_mcc + metrics.matthews_corrcoef(Y_test, SVMSMOTE_predict)
        SVMSMOTE_kappa = SVMSMOTE_kappa + metrics.cohen_kappa_score(Y_test, SVMSMOTE_predict)
        # ClusterCentroids
        clf.fit(ClusterCentroids_X_train,ClusterCentroids_Y_train)
        ClusterCentroids_predict = clf.predict(X_test)

        ClusterCentroids_precision = ClusterCentroids_precision + metrics.classification_report(Y_test, ClusterCentroids_predict, output_dict=True)['0']['precision']
        ClusterCentroids_gmean = ClusterCentroids_gmean + Gmean(Y_test, ClusterCentroids_predict)
        ClusterCentroids_recall = ClusterCentroids_recall + metrics.classification_report(Y_test, ClusterCentroids_predict, output_dict=True)['0']['recall']
        ClusterCentroids_f1 = ClusterCentroids_f1 + metrics.classification_report(Y_test, ClusterCentroids_predict, output_dict=True)['0']['f1-score']
        ClusterCentroids_roc = ClusterCentroids_roc + metrics.roc_auc_score(Y_test, ClusterCentroids_predict)
        ClusterCentroids_mcc = ClusterCentroids_mcc + metrics.matthews_corrcoef(Y_test, ClusterCentroids_predict)
        ClusterCentroids_kappa = ClusterCentroids_kappa + metrics.cohen_kappa_score(Y_test, ClusterCentroids_predict)
        # EditedNearestNeighbours
        clf.fit(EditedNearestNeighbours_X_train,EditedNearestNeighbours_Y_train)
        EditedNearestNeighbours_predict = clf.predict(X_test)

        EditedNearestNeighbours_precision = EditedNearestNeighbours_precision + metrics.classification_report(Y_test, EditedNearestNeighbours_predict, output_dict=True)['0']['precision']
        EditedNearestNeighbours_gmean = EditedNearestNeighbours_gmean + Gmean(Y_test, EditedNearestNeighbours_predict)
        EditedNearestNeighbours_recall = EditedNearestNeighbours_recall + metrics.classification_report(Y_test, EditedNearestNeighbours_predict, output_dict=True)['0']['recall']
        EditedNearestNeighbours_f1 = EditedNearestNeighbours_f1 + metrics.classification_report(Y_test, EditedNearestNeighbours_predict, output_dict=True)['0']['f1-score']
        EditedNearestNeighbours_roc = EditedNearestNeighbours_roc + metrics.roc_auc_score(Y_test, EditedNearestNeighbours_predict)
        EditedNearestNeighbours_mcc = EditedNearestNeighbours_mcc + metrics.matthews_corrcoef(Y_test, EditedNearestNeighbours_predict)
        EditedNearestNeighbours_kappa = EditedNearestNeighbours_kappa + metrics.cohen_kappa_score(Y_test, EditedNearestNeighbours_predict)
        # OneSidedSelection
        clf.fit(OneSidedSelection_X_train,OneSidedSelection_Y_train)
        OneSidedSelection_predict = clf.predict(X_test)

        OneSidedSelection_precision = OneSidedSelection_precision + metrics.classification_report(Y_test, OneSidedSelection_predict, output_dict=True)['0']['precision']
        OneSidedSelection_gmean = OneSidedSelection_gmean + Gmean(Y_test, OneSidedSelection_predict)
        OneSidedSelection_recall = OneSidedSelection_recall + metrics.classification_report(Y_test, OneSidedSelection_predict, output_dict=True)['0']['recall']
        OneSidedSelection_f1 = OneSidedSelection_f1 + metrics.classification_report(Y_test, OneSidedSelection_predict, output_dict=True)['0']['f1-score']
        OneSidedSelection_roc = OneSidedSelection_roc + metrics.roc_auc_score(Y_test, OneSidedSelection_predict)
        OneSidedSelection_mcc = OneSidedSelection_mcc + metrics.matthews_corrcoef(Y_test, OneSidedSelection_predict)
        OneSidedSelection_kappa = OneSidedSelection_kappa + metrics.cohen_kappa_score(Y_test, OneSidedSelection_predict)
        #SMOTEENN
        clf.fit(SMOTEENN_X_train,SMOTEENN_Y_train)
        SMOTEENN_predict = clf.predict(X_test)

        SMOTEENN_precision = SMOTEENN_precision + metrics.classification_report(Y_test, SMOTEENN_predict, output_dict=True)['0']['precision']
        SMOTEENN_gmean = SMOTEENN_gmean + Gmean(Y_test, SMOTEENN_predict)
        SMOTEENN_recall = SMOTEENN_recall + metrics.classification_report(Y_test, SMOTEENN_predict, output_dict=True)['0']['recall']
        SMOTEENN_f1 = SMOTEENN_f1 + metrics.classification_report(Y_test, SMOTEENN_predict, output_dict=True)['0']['f1-score']
        SMOTEENN_roc = SMOTEENN_roc + metrics.roc_auc_score(Y_test, SMOTEENN_predict)
        SMOTEENN_mcc = SMOTEENN_mcc + metrics.matthews_corrcoef(Y_test, SMOTEENN_predict)
        SMOTEENN_kappa = SMOTEENN_kappa + metrics.cohen_kappa_score(Y_test, SMOTEENN_predict)
        # SMOTETomek
        clf.fit(SMOTETomek_X_train,SMOTETomek_Y_train)
        SMOTETomek_predict = clf.predict(X_test)

        SMOTETomek_precision = SMOTETomek_precision + metrics.classification_report(Y_test, SMOTETomek_predict, output_dict=True)['0']['precision']
        SMOTETomek_gmean = SMOTETomek_gmean + Gmean(Y_test, SMOTETomek_predict)
        SMOTETomek_recall = SMOTETomek_recall + metrics.classification_report(Y_test, SMOTETomek_predict, output_dict=True)['0']['recall']
        SMOTETomek_f1 = SMOTETomek_f1 + metrics.classification_report(Y_test, SMOTETomek_predict, output_dict=True)['0']['f1-score']
        SMOTETomek_roc = SMOTETomek_roc + metrics.roc_auc_score(Y_test, SMOTETomek_predict)
        SMOTETomek_mcc = SMOTETomek_mcc + metrics.matthews_corrcoef(Y_test, SMOTETomek_predict)
        SMOTETomek_kappa = SMOTETomek_kappa + metrics.cohen_kappa_score(Y_test, SMOTETomek_predict)
        # RandomUnderSampler
        clf.fit(RandomUnderSampler_X_train,RandomUnderSampler_Y_train)
        RandomUnderSampler_predict = clf.predict(X_test)

        RandomUnderSampler_precision = RandomUnderSampler_precision + metrics.classification_report(Y_test, RandomUnderSampler_predict, output_dict=True)['0']['precision']
        RandomUnderSampler_gmean = RandomUnderSampler_gmean + Gmean(Y_test, RandomUnderSampler_predict)
        RandomUnderSampler_recall = RandomUnderSampler_recall + metrics.classification_report(Y_test, RandomUnderSampler_predict, output_dict=True)['0']['recall']
        RandomUnderSampler_f1 = RandomUnderSampler_f1 + metrics.classification_report(Y_test, RandomUnderSampler_predict, output_dict=True)['0']['f1-score']
        RandomUnderSampler_roc = RandomUnderSampler_roc + metrics.roc_auc_score(Y_test, RandomUnderSampler_predict)
        RandomUnderSampler_mcc = RandomUnderSampler_mcc + metrics.matthews_corrcoef(Y_test, RandomUnderSampler_predict)
        RandomUnderSampler_kappa = RandomUnderSampler_kappa + metrics.cohen_kappa_score(Y_test, RandomUnderSampler_predict)
        # SMOTENC
        clf.fit(SMOTENC_X_train,SMOTENC_Y_train)
        SMOTENC_predict = clf.predict(X_test)

        SMOTENC_precision = SMOTENC_precision + metrics.classification_report(Y_test, SMOTENC_predict, output_dict=True)['0']['precision']
        SMOTENC_gmean = SMOTENC_gmean + Gmean(Y_test, SMOTENC_predict)
        SMOTENC_recall = SMOTENC_recall + metrics.classification_report(Y_test, SMOTENC_predict, output_dict=True)['0']['recall']
        SMOTENC_f1 = SMOTENC_f1 + metrics.classification_report(Y_test, SMOTENC_predict, output_dict=True)['0']['f1-score']
        SMOTENC_roc = SMOTENC_roc + metrics.roc_auc_score(Y_test, SMOTENC_predict)
        SMOTENC_mcc = SMOTENC_mcc + metrics.matthews_corrcoef(Y_test, SMOTENC_predict)
        SMOTENC_kappa = SMOTENC_kappa + metrics.cohen_kappa_score(Y_test, SMOTENC_predict)
        # TomekLink
        clf.fit(TomekLink_X_train,TomekLink_Y_train)
        TomekLink_predict = clf.predict(X_test)

        TomekLink_precision = TomekLink_precision + metrics.classification_report(Y_test, TomekLink_predict, output_dict=True)['0']['precision']
        TomekLink_gmean = TomekLink_gmean + Gmean(Y_test, TomekLink_predict)
        TomekLink_recall = TomekLink_recall + metrics.classification_report(Y_test, TomekLink_predict, output_dict=True)['0']['recall']
        TomekLink_f1 = TomekLink_f1 + metrics.classification_report(Y_test, TomekLink_predict, output_dict=True)['0']['f1-score']
        TomekLink_roc = TomekLink_roc + metrics.roc_auc_score(Y_test, TomekLink_predict)
        TomekLink_mcc = TomekLink_mcc + metrics.matthews_corrcoef(Y_test, TomekLink_predict)
        TomekLink_kappa = TomekLink_kappa + metrics.cohen_kappa_score(Y_test, TomekLink_predict)
        # 三系
        clf.fit(three_X_train, three_Y_train)
        endpredict = clf.predict(X_test)

        us_precision = us_precision + metrics.classification_report(Y_test, endpredict, output_dict=True)['0']['precision']
        us_gmean = us_gmean + Gmean(Y_test, endpredict)
        us_recall = us_recall + metrics.classification_report(Y_test, endpredict, output_dict=True)['0']['recall']
        us_f1 = us_f1 + metrics.classification_report(Y_test, endpredict, output_dict=True)['0']['f1-score']
        us_roc = us_roc + metrics.roc_auc_score(Y_test, endpredict)
        us_mcc = us_mcc + metrics.matthews_corrcoef(Y_test, endpredict)
        us_kappa = us_kappa + metrics.cohen_kappa_score(Y_test, endpredict)
    print('=============three series================')
    print('precision:',us_precision/10)
    print('Gmean:',us_gmean/10)
    print('us_recall:',us_recall/10)
    print('f1:',us_f1/10)
    print('roc:',us_roc/10)
    print('mcc:',us_mcc/10)
    print('kappa:',us_kappa/10)

    print('=============smote================')
    print('precision:',smote_precision/10)
    print("151561",smote_precision1/10)
    print('Gmean:',smote_gmean/10)
    print('us_recall:',smote_recall/10)
    print('f1:',smote_f1/10)
    print('roc:',smote_roc/10)
    print('mcc:',smote_mcc/10)
    print('kappa:',smote_kappa/10)

    print('=============adasyn================')
    print('precision:',adasyn_precision/10)
    print('Gmean:',adasyn_gmean/10)
    print('us_recall:',adasyn_recall/10)
    print('f1:',adasyn_f1/10)
    print('roc:',adasyn_roc/10)
    print('mcc:',adasyn_mcc/10)
    print('kappa:',adasyn_kappa/10)

    print('=============SMOTEN================')
    print('precision:',smoten_precision/10)
    print('Gmean:',smoten_gmean/10)
    print('us_recall:',smoten_recall/10)
    print('f1:',smoten_f1/10)
    print('roc:',smoten_roc/10)
    print('mcc:',smoten_mcc/10)
    print('kappa:',smoten_kappa/10)

    print('=============RandomOverSampler================')
    print('precision:',randomover_precision/10)
    print('Gmean:',randomover_gmean/10)
    print('us_recall:',randomover_recall/10)
    print('f1:',randomover_f1/10)
    print('roc:',randomover_roc/10)
    print('mcc:',randomover_mcc/10)
    print('kappa:',randomover_kappa/10)

    print('=============BorderlineSMOTE================')
    print('precision:',BorderlineSMOTE_precision/10)
    print('Gmean:',BorderlineSMOTE_gmean/10)
    print('us_recall:',BorderlineSMOTE_recall/10)
    print('f1:',BorderlineSMOTE_f1/10)
    print('roc:',BorderlineSMOTE_roc/10)
    print('mcc:',BorderlineSMOTE_mcc/10)
    print('kappa:',BorderlineSMOTE_kappa/10)

    print('=============SVMSMOTE================')
    print('precision:',SVMSMOTE_precision/10)
    print('Gmean:',SVMSMOTE_gmean/10)
    print('us_recall:',SVMSMOTE_recall/10)
    print('f1:',SVMSMOTE_f1/10)
    print('roc:',SVMSMOTE_roc/10)
    print('mcc:',SVMSMOTE_mcc/10)
    print('kappa:',SVMSMOTE_kappa/10)

    print('=============ClusterCentroids================')
    print('precision:',ClusterCentroids_precision/10)
    print('Gmean:',ClusterCentroids_gmean/10)
    print('us_recall:',ClusterCentroids_recall/10)
    print('f1:',ClusterCentroids_f1/10)
    print('roc:',ClusterCentroids_roc/10)
    print('mcc:',ClusterCentroids_mcc/10)
    print('kappa:',ClusterCentroids_kappa/10)

    print('=============EditedNearestNeighbours================')
    print('precision:',EditedNearestNeighbours_precision/10)
    print('Gmean:',EditedNearestNeighbours_gmean/10)
    print('us_recall:',EditedNearestNeighbours_recall/10)
    print('f1:',EditedNearestNeighbours_f1/10)
    print('roc:',EditedNearestNeighbours_roc/10)
    print('mcc:',EditedNearestNeighbours_mcc/10)
    print('kappa:',EditedNearestNeighbours_kappa/10)

    print('=============OneSidedSelection================')
    print('precision:',OneSidedSelection_precision/10)
    print('Gmean:',OneSidedSelection_gmean/10)
    print('us_recall:',OneSidedSelection_recall/10)
    print('f1:',OneSidedSelection_f1/10)
    print('roc:',OneSidedSelection_roc/10)
    print('mcc:',OneSidedSelection_mcc/10)
    print('kappa:',OneSidedSelection_kappa/10)

    print('=============SMOTEENN================')
    print('precision:',SMOTEENN_precision/10)
    print('Gmean:',SMOTEENN_gmean/10)
    print('us_recall:',SMOTEENN_recall/10)
    print('f1:',SMOTEENN_f1/10)
    print('roc:',SMOTEENN_roc/10)
    print('mcc:',SMOTEENN_mcc/10)
    print('kappa:',SMOTEENN_kappa/10)

    print('=============SMOTETomek================')
    print('precision:',SMOTETomek_precision/10)
    print('Gmean:',SMOTETomek_gmean/10)
    print('us_recall:',SMOTETomek_recall/10)
    print('f1:',SMOTETomek_f1/10)
    print('roc:',SMOTETomek_roc/10)
    print('mcc:',SMOTETomek_mcc/10)
    print('kappa:',SMOTETomek_kappa/10)

    print('=============RandomUnderSampler================')
    print('precision:',RandomUnderSampler_precision/10)
    print('Gmean:',RandomUnderSampler_gmean/10)
    print('us_recall:',RandomUnderSampler_recall/10)
    print('f1:',RandomUnderSampler_f1/10)
    print('roc:',RandomUnderSampler_roc/10)
    print('mcc:',RandomUnderSampler_mcc/10)
    print('kappa:',RandomUnderSampler_kappa/10)

    print('=============SMOTENC================')
    print('precision:',SMOTENC_precision/10)
    print('Gmean:',SMOTENC_gmean/10)
    print('us_recall:',SMOTENC_recall/10)
    print('f1:',SMOTENC_f1/10)
    print('roc:',SMOTENC_roc/10)
    print('mcc:',SMOTENC_mcc/10)
    print('kappa:',SMOTENC_kappa/10)

    print('=============TomekLink================')
    print('precision:',TomekLink_precision/10)
    print('Gmean:',TomekLink_gmean/10)
    print('us_recall:',TomekLink_recall/10)
    print('f1:',TomekLink_f1/10)
    print('roc:',TomekLink_roc/10)
    print('mcc:',TomekLink_mcc/10)
    print('kappa:',TomekLink_kappa/10)

if __name__ == '__main__':
    cat_file = 'data/ecoli1.csv'
    X,Y = read_data(cat_file)

    train = []
    test = []
    us_precision = us_gmean = us_recall = us_f1 = us_roc = us_mcc = us_kappa = 0
    # smote
    smote_precision = smote_gmean = smote_recall = smote_f1 = smote_roc = smote_mcc = smote_kappa = 0
    # ADASYNdata
    adasyn_precision = adasyn_gmean = adasyn_recall = adasyn_f1 = adasyn_roc = adasyn_mcc = adasyn_kappa = 0
    # SMOTEN
    smoten_precision = smoten_gmean = smoten_recall = smoten_f1 = smoten_roc = smoten_mcc = smoten_kappa = 0
    #RandomOverSampler
    randomover_precision = randomover_gmean = randomover_recall = randomover_f1 = randomover_roc = randomover_mcc = randomover_kappa = 0
    #BorderlineSMOTE
    BorderlineSMOTE_precision = BorderlineSMOTE_gmean = BorderlineSMOTE_recall = BorderlineSMOTE_f1 = BorderlineSMOTE_roc = BorderlineSMOTE_mcc = BorderlineSMOTE_kappa = 0
    #SVMSMOTE
    SVMSMOTE_precision = SVMSMOTE_gmean = SVMSMOTE_recall = SVMSMOTE_f1 = SVMSMOTE_roc = SVMSMOTE_mcc = SVMSMOTE_kappa = 0
    # ClusterCentroids
    ClusterCentroids_precision = ClusterCentroids_gmean = ClusterCentroids_recall = ClusterCentroids_f1 = ClusterCentroids_roc = ClusterCentroids_mcc = ClusterCentroids_kappa = 0
    #EditedNearestNeighbours
    EditedNearestNeighbours_precision = EditedNearestNeighbours_gmean = EditedNearestNeighbours_recall = EditedNearestNeighbours_f1 = EditedNearestNeighbours_roc = EditedNearestNeighbours_mcc = EditedNearestNeighbours_kappa = 0
    # OneSidedSelection
    OneSidedSelection_precision = OneSidedSelection_gmean = OneSidedSelection_recall = OneSidedSelection_f1 = OneSidedSelection_roc = OneSidedSelection_mcc = OneSidedSelection_kappa = 0
    #SMOTEENN
    SMOTEENN_precision = SMOTEENN_gmean = SMOTEENN_recall = SMOTEENN_f1 = SMOTEENN_roc = SMOTEENN_mcc = SMOTEENN_kappa = 0
    #SMOTETomek
    SMOTETomek_precision = SMOTETomek_gmean = SMOTETomek_recall = SMOTETomek_f1 = SMOTETomek_roc = SMOTETomek_mcc = SMOTETomek_kappa = 0
    # RandomUnderSampler
    RandomUnderSampler_precision = RandomUnderSampler_gmean = RandomUnderSampler_recall = RandomUnderSampler_f1 = RandomUnderSampler_roc = RandomUnderSampler_mcc = RandomUnderSampler_kappa = 0
    #SMOTENC
    SMOTENC_precision = SMOTENC_gmean = SMOTENC_recall = SMOTENC_f1 = SMOTENC_roc = SMOTENC_mcc = SMOTENC_kappa = 0
    #TomekLink
    TomekLink_precision = TomekLink_gmean = TomekLink_recall = TomekLink_f1 = TomekLink_roc = TomekLink_mcc = TomekLink_kappa = 0
    skf = StratifiedKFold(n_splits=10,shuffle=True)
    for train_index, test_index in skf.split(X, Y):
    # clf1 = MLPClassifier(max_iter=500)
    # clf2 = svm.SVC(max_iter=500)
    # clf3 = tree.DecisionTreeClassifier(criterion = 'entropy')
    # clf4 = tree.DecisionTreeClassifier(criterion = 'gini')
    # print('#######################################  MLP分类器  ######################################')
    # calssifier_sample(train,test,X, Y,clf1)
    # print('#######################################  SVM分类器  ######################################')
    # calssifier_sample(train,test,X, Y,clf2)
    # print('#######################################  ID3分类器  ######################################')
    # calssifier_sample(train,test,X, Y,clf3)
    # print('#######################################  cart分类器  ######################################')
    # calssifier_sample(train,test,X, Y,clf4)

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clf = tree.DecisionTreeClassifier(criterion = 'gini')
        # clf.fit(X_train,Y_train)
        # smote_predict = clf.predict(X_test)
        # smote_precision = smote_precision + metrics.classification_report(Y_test, smote_predict, output_dict=True)['0']['precision']
        # smote_gmean = smote_gmean + Gmean(Y_test, smote_predict)
        # smote_recall = smote_recall + metrics.classification_report(Y_test, smote_predict, output_dict=True)['0']['recall']
        # smote_f1 = smote_f1 + metrics.classification_report(Y_test, smote_predict, output_dict=True)['0']['f1-score']
        # smote_roc = smote_roc + metrics.roc_auc_score(Y_test, smote_predict)
        # smote_mcc = smote_mcc + metrics.matthews_corrcoef(Y_test, smote_predict)
        # smote_kappa = smote_kappa + metrics.cohen_kappa_score(Y_test, smote_predict)

        # 三系
        three_X_train, three_Y_train= three_series_train(X_train,Y_train,1000)
        # smote
        smote_X_train, smote_Y_train= SMOTEdata(X_train,Y_train)
        # adasyn
        adasyn_X_train, adasyn_Y_train= ADASYNdata(X_train,Y_train)
        # smoten
        smoten_X_train, smoten_Y_train= SMOTENdata(X_train,Y_train)
        # RandomOverSamplerdata
        randomover_X_train, randomover_Y_train= RandomOverSamplerdata(X_train,Y_train)
        #BorderlineSMOTE
        BorderlineSMOTE_X_train, BorderlineSMOTE_Y_train= BorderlineSMOTEdata(X_train,Y_train)
        #SVMSMOTE
        SVMSMOTE_X_train, SVMSMOTE_Y_train= SVMSMOTEdata(X_train,Y_train)
        #ClusterCentroids
        ClusterCentroids_X_train, ClusterCentroids_Y_train= ClusterCentroidsdata(X_train,Y_train)
        #EditedNearestNeighbours
        EditedNearestNeighbours_X_train, EditedNearestNeighbours_Y_train= EditedNearestNeighboursdata(X_train,Y_train)
        #OneSidedSelection
        OneSidedSelection_X_train, OneSidedSelection_Y_train= OneSidedSelectiondata(X_train,Y_train)
        #SMOTEENN
        SMOTEENN_X_train, SMOTEENN_Y_train= SMOTEENNdata(X_train,Y_train)
        #SMOTETomek
        SMOTETomek_X_train, SMOTETomek_Y_train= SMOTETomekdata(X_train,Y_train)
        #RandomUnderSampler
        RandomUnderSampler_X_train, RandomUnderSampler_Y_train= RandomUnderSamplerdata(X_train,Y_train)
        #SMOTENC
        SMOTENC_X_train, SMOTENC_Y_train= SMOTENCdata(X_train,Y_train)
        #omekLinks
        TomekLink_X_train, TomekLink_Y_train= TomekLinksdata(X_train,Y_train)


        # 三系
        clf.fit(three_X_train, three_Y_train)
        endpredict = clf.predict(X_test)
        us_precision = us_precision + metrics.classification_report(Y_test, endpredict, output_dict=True)['0']['precision']
        us_gmean = us_gmean + Gmean(Y_test, endpredict)
        print('G-mean:',Gmean(Y_test, endpredict))
        us_recall = us_recall + metrics.classification_report(Y_test, endpredict, output_dict=True)['0']['recall']
        us_f1 = us_f1 + metrics.classification_report(Y_test, endpredict, output_dict=True)['0']['f1-score']
        us_roc = us_roc + metrics.roc_auc_score(Y_test, endpredict)
        us_mcc = us_mcc + metrics.matthews_corrcoef(Y_test, endpredict)
        us_kappa = us_kappa + metrics.cohen_kappa_score(Y_test, endpredict)
        # smote
        clf.fit(smote_X_train, smote_Y_train)
        smote_predict = clf.predict(X_test)

        smote_precision = smote_precision + metrics.classification_report(Y_test, smote_predict, output_dict=True)['0']['precision']
        smote_gmean = smote_gmean + Gmean(Y_test, smote_predict)
        smote_recall = smote_recall + metrics.classification_report(Y_test, smote_predict, output_dict=True)['0']['recall']
        smote_f1 = smote_f1 + metrics.classification_report(Y_test, smote_predict, output_dict=True)['0']['f1-score']
        smote_roc = smote_roc + metrics.roc_auc_score(Y_test, smote_predict)
        smote_mcc = smote_mcc + metrics.matthews_corrcoef(Y_test, smote_predict)
        smote_kappa = smote_kappa + metrics.cohen_kappa_score(Y_test, smote_predict)
        #adasyn
        clf.fit(adasyn_X_train,adasyn_Y_train)
        adasyn_predict = clf.predict(X_test)

        adasyn_precision = adasyn_precision + metrics.classification_report(Y_test, adasyn_predict, output_dict=True)['0']['precision']
        adasyn_gmean = adasyn_gmean + Gmean(Y_test, adasyn_predict)
        adasyn_recall = adasyn_recall + metrics.classification_report(Y_test, adasyn_predict, output_dict=True)['0']['recall']
        adasyn_f1 = adasyn_f1 + metrics.classification_report(Y_test, adasyn_predict, output_dict=True)['0']['f1-score']
        adasyn_roc = adasyn_roc + metrics.roc_auc_score(Y_test, adasyn_predict)
        adasyn_mcc = adasyn_mcc + metrics.matthews_corrcoef(Y_test, adasyn_predict)
        adasyn_kappa = adasyn_kappa + metrics.cohen_kappa_score(Y_test, adasyn_predict)
        # smoten
        clf.fit(smoten_X_train,smoten_Y_train)
        smoten_predict = clf.predict(X_test)

        smoten_precision = smoten_precision + metrics.classification_report(Y_test, smoten_predict, output_dict=True)['0']['precision']
        smoten_gmean = smoten_gmean + Gmean(Y_test, smoten_predict)
        smoten_recall = smoten_recall + metrics.classification_report(Y_test, smoten_predict, output_dict=True)['0']['recall']
        smoten_f1 = smoten_f1 + metrics.classification_report(Y_test, smoten_predict, output_dict=True)['0']['f1-score']
        smoten_roc = smoten_roc + metrics.roc_auc_score(Y_test, smoten_predict)
        smoten_mcc = smoten_mcc + metrics.matthews_corrcoef(Y_test, smoten_predict)
        smoten_kappa = smoten_kappa + metrics.cohen_kappa_score(Y_test, smoten_predict)
        #randomover
        clf.fit(randomover_X_train,randomover_Y_train)
        randomover_predict = clf.predict(X_test)

        randomover_precision = randomover_precision + metrics.classification_report(Y_test, randomover_predict, output_dict=True)['0']['precision']
        randomover_gmean = randomover_gmean + Gmean(Y_test, randomover_predict)
        randomover_recall = randomover_recall + metrics.classification_report(Y_test, randomover_predict, output_dict=True)['0']['recall']
        randomover_f1 = randomover_f1 + metrics.classification_report(Y_test, randomover_predict, output_dict=True)['0']['f1-score']
        randomover_roc = randomover_roc + metrics.roc_auc_score(Y_test, randomover_predict)
        randomover_mcc = randomover_mcc + metrics.matthews_corrcoef(Y_test, randomover_predict)
        randomover_kappa = randomover_kappa + metrics.cohen_kappa_score(Y_test, randomover_predict)
        # BorderlineSMOTE
        clf.fit(smote_X_train,BorderlineSMOTE_Y_train)
        BorderlineSMOTE_predict = clf.predict(X_test)

        BorderlineSMOTE_precision = BorderlineSMOTE_precision + metrics.classification_report(Y_test, BorderlineSMOTE_predict, output_dict=True)['0']['precision']
        BorderlineSMOTE_gmean = BorderlineSMOTE_gmean + Gmean(Y_test, BorderlineSMOTE_predict)
        BorderlineSMOTE_recall = BorderlineSMOTE_recall + metrics.classification_report(Y_test, BorderlineSMOTE_predict, output_dict=True)['0']['recall']
        BorderlineSMOTE_f1 = BorderlineSMOTE_f1 + metrics.classification_report(Y_test, BorderlineSMOTE_predict, output_dict=True)['0']['f1-score']
        BorderlineSMOTE_roc = BorderlineSMOTE_roc + metrics.roc_auc_score(Y_test, BorderlineSMOTE_predict)
        BorderlineSMOTE_mcc = BorderlineSMOTE_mcc + metrics.matthews_corrcoef(Y_test, BorderlineSMOTE_predict)
        BorderlineSMOTE_kappa = BorderlineSMOTE_kappa + metrics.cohen_kappa_score(Y_test, BorderlineSMOTE_predict)
        # SVMSMOTE
        clf.fit(SVMSMOTE_X_train,SVMSMOTE_Y_train)
        SVMSMOTE_predict = clf.predict(X_test)

        SVMSMOTE_precision = SVMSMOTE_precision + metrics.classification_report(Y_test, SVMSMOTE_predict, output_dict=True)['0']['precision']
        SVMSMOTE_gmean = SVMSMOTE_gmean + Gmean(Y_test, SVMSMOTE_predict)
        SVMSMOTE_recall = SVMSMOTE_recall + metrics.classification_report(Y_test, SVMSMOTE_predict, output_dict=True)['0']['recall']
        SVMSMOTE_f1 = SVMSMOTE_f1 + metrics.classification_report(Y_test, SVMSMOTE_predict, output_dict=True)['0']['f1-score']
        SVMSMOTE_roc = SVMSMOTE_roc + metrics.roc_auc_score(Y_test, SVMSMOTE_predict)
        SVMSMOTE_mcc = SVMSMOTE_mcc + metrics.matthews_corrcoef(Y_test, SVMSMOTE_predict)
        SVMSMOTE_kappa = SVMSMOTE_kappa + metrics.cohen_kappa_score(Y_test, SVMSMOTE_predict)
        # ClusterCentroids
        clf.fit(ClusterCentroids_X_train,ClusterCentroids_Y_train)
        ClusterCentroids_predict = clf.predict(X_test)

        ClusterCentroids_precision = ClusterCentroids_precision + metrics.classification_report(Y_test, ClusterCentroids_predict, output_dict=True)['0']['precision']
        ClusterCentroids_gmean = ClusterCentroids_gmean + Gmean(Y_test, ClusterCentroids_predict)
        ClusterCentroids_recall = ClusterCentroids_recall + metrics.classification_report(Y_test, ClusterCentroids_predict, output_dict=True)['0']['recall']
        ClusterCentroids_f1 = ClusterCentroids_f1 + metrics.classification_report(Y_test, ClusterCentroids_predict, output_dict=True)['0']['f1-score']
        ClusterCentroids_roc = ClusterCentroids_roc + metrics.roc_auc_score(Y_test, ClusterCentroids_predict)
        ClusterCentroids_mcc = ClusterCentroids_mcc + metrics.matthews_corrcoef(Y_test, ClusterCentroids_predict)
        ClusterCentroids_kappa = ClusterCentroids_kappa + metrics.cohen_kappa_score(Y_test, ClusterCentroids_predict)
        # EditedNearestNeighbours
        clf.fit(EditedNearestNeighbours_X_train,EditedNearestNeighbours_Y_train)
        EditedNearestNeighbours_predict = clf.predict(X_test)

        EditedNearestNeighbours_precision = EditedNearestNeighbours_precision + metrics.classification_report(Y_test, EditedNearestNeighbours_predict, output_dict=True)['0']['precision']
        EditedNearestNeighbours_gmean = EditedNearestNeighbours_gmean + Gmean(Y_test, EditedNearestNeighbours_predict)
        EditedNearestNeighbours_recall = EditedNearestNeighbours_recall + metrics.classification_report(Y_test, EditedNearestNeighbours_predict, output_dict=True)['0']['recall']
        EditedNearestNeighbours_f1 = EditedNearestNeighbours_f1 + metrics.classification_report(Y_test, EditedNearestNeighbours_predict, output_dict=True)['0']['f1-score']
        EditedNearestNeighbours_roc = EditedNearestNeighbours_roc + metrics.roc_auc_score(Y_test, EditedNearestNeighbours_predict)
        EditedNearestNeighbours_mcc = EditedNearestNeighbours_mcc + metrics.matthews_corrcoef(Y_test, EditedNearestNeighbours_predict)
        EditedNearestNeighbours_kappa = EditedNearestNeighbours_kappa + metrics.cohen_kappa_score(Y_test, EditedNearestNeighbours_predict)
        # OneSidedSelection
        clf.fit(OneSidedSelection_X_train,OneSidedSelection_Y_train)
        OneSidedSelection_predict = clf.predict(X_test)

        OneSidedSelection_precision = OneSidedSelection_precision + metrics.classification_report(Y_test, OneSidedSelection_predict, output_dict=True)['0']['precision']
        OneSidedSelection_gmean = OneSidedSelection_gmean + Gmean(Y_test, OneSidedSelection_predict)
        OneSidedSelection_recall = OneSidedSelection_recall + metrics.classification_report(Y_test, OneSidedSelection_predict, output_dict=True)['0']['recall']
        OneSidedSelection_f1 = OneSidedSelection_f1 + metrics.classification_report(Y_test, OneSidedSelection_predict, output_dict=True)['0']['f1-score']
        OneSidedSelection_roc = OneSidedSelection_roc + metrics.roc_auc_score(Y_test, OneSidedSelection_predict)
        OneSidedSelection_mcc = OneSidedSelection_mcc + metrics.matthews_corrcoef(Y_test, OneSidedSelection_predict)
        OneSidedSelection_kappa = OneSidedSelection_kappa + metrics.cohen_kappa_score(Y_test, OneSidedSelection_predict)
        #SMOTEENN
        clf.fit(SMOTEENN_X_train,SMOTEENN_Y_train)
        SMOTEENN_predict = clf.predict(X_test)

        SMOTEENN_precision = SMOTEENN_precision + metrics.classification_report(Y_test, SMOTEENN_predict, output_dict=True)['0']['precision']
        SMOTEENN_gmean = SMOTEENN_gmean + Gmean(Y_test, SMOTEENN_predict)
        SMOTEENN_recall = SMOTEENN_recall + metrics.classification_report(Y_test, SMOTEENN_predict, output_dict=True)['0']['recall']
        SMOTEENN_f1 = SMOTEENN_f1 + metrics.classification_report(Y_test, SMOTEENN_predict, output_dict=True)['0']['f1-score']
        SMOTEENN_roc = SMOTEENN_roc + metrics.roc_auc_score(Y_test, SMOTEENN_predict)
        SMOTEENN_mcc = SMOTEENN_mcc + metrics.matthews_corrcoef(Y_test, SMOTEENN_predict)
        SMOTEENN_kappa = SMOTEENN_kappa + metrics.cohen_kappa_score(Y_test, SMOTEENN_predict)
        # SMOTETomek
        clf.fit(SMOTETomek_X_train,SMOTETomek_Y_train)
        SMOTETomek_predict = clf.predict(X_test)

        SMOTETomek_precision = SMOTETomek_precision + metrics.classification_report(Y_test, SMOTETomek_predict, output_dict=True)['0']['precision']
        SMOTETomek_gmean = SMOTETomek_gmean + Gmean(Y_test, SMOTETomek_predict)
        SMOTETomek_recall = SMOTETomek_recall + metrics.classification_report(Y_test, SMOTETomek_predict, output_dict=True)['0']['recall']
        SMOTETomek_f1 = SMOTETomek_f1 + metrics.classification_report(Y_test, SMOTETomek_predict, output_dict=True)['0']['f1-score']
        SMOTETomek_roc = SMOTETomek_roc + metrics.roc_auc_score(Y_test, SMOTETomek_predict)
        SMOTETomek_mcc = SMOTETomek_mcc + metrics.matthews_corrcoef(Y_test, SMOTETomek_predict)
        SMOTETomek_kappa = SMOTETomek_kappa + metrics.cohen_kappa_score(Y_test, SMOTETomek_predict)
        # RandomUnderSampler
        clf.fit(RandomUnderSampler_X_train,RandomUnderSampler_Y_train)
        RandomUnderSampler_predict = clf.predict(X_test)

        RandomUnderSampler_precision = RandomUnderSampler_precision + metrics.classification_report(Y_test, RandomUnderSampler_predict, output_dict=True)['0']['precision']
        RandomUnderSampler_gmean = RandomUnderSampler_gmean + Gmean(Y_test, RandomUnderSampler_predict)
        RandomUnderSampler_recall = RandomUnderSampler_recall + metrics.classification_report(Y_test, RandomUnderSampler_predict, output_dict=True)['0']['recall']
        RandomUnderSampler_f1 = RandomUnderSampler_f1 + metrics.classification_report(Y_test, RandomUnderSampler_predict, output_dict=True)['0']['f1-score']
        RandomUnderSampler_roc = RandomUnderSampler_roc + metrics.roc_auc_score(Y_test, RandomUnderSampler_predict)
        RandomUnderSampler_mcc = RandomUnderSampler_mcc + metrics.matthews_corrcoef(Y_test, RandomUnderSampler_predict)
        RandomUnderSampler_kappa = RandomUnderSampler_kappa + metrics.cohen_kappa_score(Y_test, RandomUnderSampler_predict)
        # SMOTENC
        clf.fit(SMOTENC_X_train,SMOTENC_Y_train)
        SMOTENC_predict = clf.predict(X_test)

        SMOTENC_precision = SMOTENC_precision + metrics.classification_report(Y_test, SMOTENC_predict, output_dict=True)['0']['precision']
        SMOTENC_gmean = SMOTENC_gmean + Gmean(Y_test, SMOTENC_predict)
        SMOTENC_recall = SMOTENC_recall + metrics.classification_report(Y_test, SMOTENC_predict, output_dict=True)['0']['recall']
        SMOTENC_f1 = SMOTENC_f1 + metrics.classification_report(Y_test, SMOTENC_predict, output_dict=True)['0']['f1-score']
        SMOTENC_roc = SMOTENC_roc + metrics.roc_auc_score(Y_test, SMOTENC_predict)
        SMOTENC_mcc = SMOTENC_mcc + metrics.matthews_corrcoef(Y_test, SMOTENC_predict)
        SMOTENC_kappa = SMOTENC_kappa + metrics.cohen_kappa_score(Y_test, SMOTENC_predict)
        # TomekLink
        clf.fit(TomekLink_X_train,TomekLink_Y_train)
        TomekLink_predict = clf.predict(X_test)

        TomekLink_precision = TomekLink_precision + metrics.classification_report(Y_test, TomekLink_predict, output_dict=True)['0']['precision']
        TomekLink_gmean = TomekLink_gmean + Gmean(Y_test, TomekLink_predict)
        TomekLink_recall = TomekLink_recall + metrics.classification_report(Y_test, TomekLink_predict, output_dict=True)['0']['recall']
        TomekLink_f1 = TomekLink_f1 + metrics.classification_report(Y_test, TomekLink_predict, output_dict=True)['0']['f1-score']
        TomekLink_roc = TomekLink_roc + metrics.roc_auc_score(Y_test, TomekLink_predict)
        TomekLink_mcc = TomekLink_mcc + metrics.matthews_corrcoef(Y_test, TomekLink_predict)
        TomekLink_kappa = TomekLink_kappa + metrics.cohen_kappa_score(Y_test, TomekLink_predict)

    print('=============three series================')
    print('precision:',us_precision/10)
    print('Gmean:',us_gmean/10)
    print('us_recall:',us_recall/10)
    print('f1:',us_f1/10)
    print('roc:',us_roc/10)
    print('mcc:',us_mcc/10)
    print('kappa:',us_kappa/10)

    print('=============smote================')
    print('precision:',smote_precision/10)
    print('Gmean:',smote_gmean/10)
    print('us_recall:',smote_recall/10)
    print('f1:',smote_f1/10)
    print('roc:',smote_roc/10)
    print('mcc:',smote_mcc/10)
    print('kappa:',smote_kappa/10)

    print('=============adasyn================')
    print('precision:',adasyn_precision/10)
    print('Gmean:',adasyn_gmean/10)
    print('us_recall:',adasyn_recall/10)
    print('f1:',adasyn_f1/10)
    print('roc:',adasyn_roc/10)
    print('mcc:',adasyn_mcc/10)
    print('kappa:',adasyn_kappa/10)

    print('=============SMOTEN================')
    print('precision:',smoten_precision/10)
    print('Gmean:',smoten_gmean/10)
    print('us_recall:',smoten_recall/10)
    print('f1:',smoten_f1/10)
    print('roc:',smoten_roc/10)
    print('mcc:',smoten_mcc/10)
    print('kappa:',smoten_kappa/10)

    print('=============RandomOverSampler================')
    print('precision:',randomover_precision/10)
    print('Gmean:',randomover_gmean/10)
    print('us_recall:',randomover_recall/10)
    print('f1:',randomover_f1/10)
    print('roc:',randomover_roc/10)
    print('mcc:',randomover_mcc/10)
    print('kappa:',randomover_kappa/10)

    print('=============BorderlineSMOTE================')
    print('precision:',BorderlineSMOTE_precision/10)
    print('Gmean:',BorderlineSMOTE_gmean/10)
    print('us_recall:',BorderlineSMOTE_recall/10)
    print('f1:',BorderlineSMOTE_f1/10)
    print('roc:',BorderlineSMOTE_roc/10)
    print('mcc:',BorderlineSMOTE_mcc/10)
    print('kappa:',BorderlineSMOTE_kappa/10)

    print('=============SVMSMOTE================')
    print('precision:',SVMSMOTE_precision/10)
    print('Gmean:',SVMSMOTE_gmean/10)
    print('us_recall:',SVMSMOTE_recall/10)
    print('f1:',SVMSMOTE_f1/10)
    print('roc:',SVMSMOTE_roc/10)
    print('mcc:',SVMSMOTE_mcc/10)
    print('kappa:',SVMSMOTE_kappa/10)

    print('=============ClusterCentroids================')
    print('precision:',ClusterCentroids_precision/10)
    print('Gmean:',ClusterCentroids_gmean/10)
    print('us_recall:',ClusterCentroids_recall/10)
    print('f1:',ClusterCentroids_f1/10)
    print('roc:',ClusterCentroids_roc/10)
    print('mcc:',ClusterCentroids_mcc/10)
    print('kappa:',ClusterCentroids_kappa/10)

    print('=============EditedNearestNeighbours================')
    print('precision:',EditedNearestNeighbours_precision/10)
    print('Gmean:',EditedNearestNeighbours_gmean/10)
    print('us_recall:',EditedNearestNeighbours_recall/10)
    print('f1:',EditedNearestNeighbours_f1/10)
    print('roc:',EditedNearestNeighbours_roc/10)
    print('mcc:',EditedNearestNeighbours_mcc/10)
    print('kappa:',EditedNearestNeighbours_kappa/10)

    print('=============OneSidedSelection================')
    print('precision:',OneSidedSelection_precision/10)
    print('Gmean:',OneSidedSelection_gmean/10)
    print('us_recall:',OneSidedSelection_recall/10)
    print('f1:',OneSidedSelection_f1/10)
    print('roc:',OneSidedSelection_roc/10)
    print('mcc:',OneSidedSelection_mcc/10)
    print('kappa:',OneSidedSelection_kappa/10)

    print('=============SMOTEENN================')
    print('precision:',SMOTEENN_precision/10)
    print('Gmean:',SMOTEENN_gmean/10)
    print('us_recall:',SMOTEENN_recall/10)
    print('f1:',SMOTEENN_f1/10)
    print('roc:',SMOTEENN_roc/10)
    print('mcc:',SMOTEENN_mcc/10)
    print('kappa:',SMOTEENN_kappa/10)

    print('=============SMOTETomek================')
    print('precision:',SMOTETomek_precision/10)
    print('Gmean:',SMOTETomek_gmean/10)
    print('us_recall:',SMOTETomek_recall/10)
    print('f1:',SMOTETomek_f1/10)
    print('roc:',SMOTETomek_roc/10)
    print('mcc:',SMOTETomek_mcc/10)
    print('kappa:',SMOTETomek_kappa/10)

    print('=============RandomUnderSampler================')
    print('precision:',RandomUnderSampler_precision/10)
    print('Gmean:',RandomUnderSampler_gmean/10)
    print('us_recall:',RandomUnderSampler_recall/10)
    print('f1:',RandomUnderSampler_f1/10)
    print('roc:',RandomUnderSampler_roc/10)
    print('mcc:',RandomUnderSampler_mcc/10)
    print('kappa:',RandomUnderSampler_kappa/10)

    print('=============SMOTENC================')
    print('precision:',SMOTENC_precision/10)
    print('Gmean:',SMOTENC_gmean/10)
    print('us_recall:',SMOTENC_recall/10)
    print('f1:',SMOTENC_f1/10)
    print('roc:',SMOTENC_roc/10)
    print('mcc:',SMOTENC_mcc/10)
    print('kappa:',SMOTENC_kappa/10)

    print('=============TomekLink================')
    print('precision:',TomekLink_precision/10)
    print('Gmean:',TomekLink_gmean/10)
    print('us_recall:',TomekLink_recall/10)
    print('f1:',TomekLink_f1/10)
    print('roc:',TomekLink_roc/10)
    print('mcc:',TomekLink_mcc/10)
    print('kappa:',TomekLink_kappa/10)
