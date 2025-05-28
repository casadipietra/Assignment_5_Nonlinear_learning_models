import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn import metrics
import numpy as np


def split_sample(sample,train_size,permute):
    if permute==True:
        sample=sample.sample(frac=1).reset_index(drop=True)
    cut=train_size*len(sample)
    cut=int(round(cut))
    train_sample=sample.iloc[:cut,:]
    test_sample=sample.iloc[cut:,:]
    return train_sample, test_sample

def k_fold(train_sample,kern,function,c_param,gam,degr,fold,return_training_err=True):
    list_=[]
    list_2=[]
    rang=[int(len(train_sample)/10)*i for i in range(1,(fold+1))]
    for i in rang:
        if i<=(len(train_sample)/10):
            test=train_sample.iloc[:i] 
            train=train_sample.iloc[i:]
        if i>(len(train_sample)/10):
            test=train_sample.iloc[rang[rang.index(i)-1]:i]
            train_sample1=train_sample.iloc[:i-int(len(train_sample)/10)]
            train_sample2=train_sample.iloc[i:]
            train=pd.concat([train_sample1,train_sample2])
        clf=function(C=c_param,kernel=kern,gamma=gam,degree=degr)
        clf.fit(train.iloc[:,:-1],np.array(train.iloc[:,-1]).reshape((-1)))
        y_pred=clf.predict(test.iloc[:,:-1])
        #create mean F1 measure to evaluate performance on validation data
        TP=sum([1 if np.array(test.iloc[:,-1])[i]==y_pred[i]==1 else 0 for i in range(len(y_pred))])

        FP=sum([1 if (np.array(test.iloc[:,-1])[i]!=1) & (y_pred[i]==1) else 0 for i in range(len(y_pred))])
        FN=sum([1 if (np.array(test.iloc[:,-1])[i]==1) & (y_pred[i]!=1) else 0 for i in range(len(y_pred))])
        if TP+FP==0:
             precision=0
        else:
             precision=TP/(TP+FP)
        recall=TP/(TP+FN)
        if precision+recall==0:
                F1=0
        else:
    
                F1=(2*precision*recall)/(precision+recall)
        list_.append(F1)
        #do the same on training data
        if return_training_err==True:
            y_pred=clf.predict(train.iloc[:,:-1])
            TP=sum([1 if np.array(train.iloc[:,-1])[i]==y_pred[i]==1 else 0 for i in range(len(y_pred))])

            FP=sum([1 if (np.array(train.iloc[:,-1])[i]!=1) & (y_pred[i]==1) else 0 for i in range(len(y_pred))])
            FN=sum([1 if (np.array(train.iloc[:,-1])[i]==1) & (y_pred[i]!=1) else 0 for i in range(len(y_pred))])
            if TP+FP==0:
                precision=0
            else:
                precision=TP/(TP+FP)
            recall=TP/(TP+FN)
            if precision+recall==0:
                F1=0
            else:
    
                F1=(2*precision*recall)/(precision+recall)
            list_2.append(F1)
            return np.sum(list_)/len(list_),np.sum(list_2)/len(list_2)        
    return np.sum(list_)/len(list_) , 0

def grid_search_rbf(C_param_list,gam_list,df,fold,return_training_err=True):
    #implements grid search of c param and gamma parameter for rbf
    #returns dataframe containing mean F1 value for each combination of params
    validation_df=pd.DataFrame(columns=C_param_list,index=gam_list)
    training_df=pd.DataFrame(columns=C_param_list,index=gam_list)
    for c_val in C_param_list:
        for gam_val in gam_list:
            f1_v,f1_t=k_fold(df,'rbf',svm.SVC,c_val,gam_val,3,fold,return_training_err)
            validation_df.loc[gam_val,c_val]=float(f1_v)
            training_df.loc[gam_val,c_val]=float(f1_t)
    return validation_df, training_df

def grid_search_linear(C_param_list,df,fold,return_training_err=True):
    validation_df=pd.DataFrame(columns=C_param_list)
    training_df=pd.DataFrame(columns=C_param_list)
    for c_val in C_param_list:
        return_training_err=True
        f1_v,f1_t=k_fold(df,'linear',svm.SVC,c_val,'auto',3,fold,return_training_err)

        validation_df.loc[0,c_val]=float(f1_v)
        training_df.loc[0,c_val]=float(f1_t)
    return validation_df, training_df

def grid_search_polynomial(C_param_list,degree_list,df,fold,return_training_err=True):
    validation_df=pd.DataFrame(columns=C_param_list,index=degree_list)
    training_df=pd.DataFrame(columns=C_param_list,index=degree_list)
    for c_val in C_param_list:
        for degr in degree_list:
            f1_v,f1_t=k_fold(df,'poly',svm.SVC,c_val,'auto',degr,fold,return_training_err)
            
            validation_df.loc[degr,c_val]=float(f1_v)
            training_df.loc[degr,c_val]=float(f1_t)
    return validation_df, training_df

def nested_10_fold_cv(C_param_list_cd, gam_list, kern, train_sample, degree_list, inner_fold):
    fold_size = len(train_sample) // 10
    list_f1 = []

    for i in range(10):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i != 9 else len(train_sample)  # pour éviter de perdre les dernières lignes

        test = train_sample.iloc[start_idx:end_idx]
        train = pd.concat([train_sample.iloc[:start_idx], train_sample.iloc[end_idx:]])

        # Grid search selon le kernel
        if kern == 'rbf':
            degr = 3
            validation_df, _ = grid_search_rbf(C_param_list_cd, gam_list, train, inner_fold, return_training_err=False)
            c_param_cd = validation_df.max()[validation_df.max() == validation_df.max().max()].index[0]
            gam_param_cd = validation_df[c_param_cd][validation_df[c_param_cd] == validation_df[c_param_cd].max()].index[0]

        elif kern == 'poly':
            gam_param_cd = 'auto'
            validation_df, _ = grid_search_polynomial(C_param_list_cd, degree_list, train, inner_fold, return_training_err=False)
            c_param_cd = validation_df.max()[validation_df.max() == validation_df.max().max()].index[0]
            degr = validation_df[c_param_cd][validation_df[c_param_cd] == validation_df[c_param_cd].max()].index[0]

        elif kern == 'linear':
            degr = 3
            gam_param_cd = 'auto'
            validation_df, _ = grid_search_linear(C_param_list_cd, train, inner_fold, return_training_err=False)
            c_param_cd = validation_df.max()[validation_df.max() == validation_df.max().max()].index[0]

        else:
            raise ValueError(f"Unsupported kernel: {kern}")

        # Entraînement
        clf = svm.SVC(C=c_param_cd, kernel=kern, gamma=gam_param_cd, degree=degr)
        clf.fit(train.iloc[:, :-1], train.iloc[:, -1])

        # Prédictions
        y_true = test.iloc[:, -1].to_numpy()
        y_pred = clf.predict(test.iloc[:, :-1])

        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true != 1) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred != 1))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        list_f1.append(f1)

    return np.mean(list_f1)

if __name__ == '__main__':
    split_sample()
    k_fold()
    grid_search_rbf()
    grid_search_linear()
    grid_search_polynomial()
    nested_10_fold_cv()
    
    
    
    
