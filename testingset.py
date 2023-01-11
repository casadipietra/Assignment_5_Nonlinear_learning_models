

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn import metrics
import numpy as np

sns.set_style('darkgrid')

def split_sample(sample,train_size,permute):
    if permute==True:
        sample=sample.sample(frac=1).reset_index(drop=True)
    cut=train_size*len(sample)
    cut=int(round(cut))
    train_sample=sample.iloc[:cut,:]
    test_sample=sample.iloc[cut:,:]
    return train_sample, test_sample

#Import data
#Checkboard dataset
cb_labels_test=pd.read_csv('checkerboardDataset/labelsTest.csv',header=None)
cb_labels_train=pd.read_csv('checkerboardDataset/labelsTrain.csv',header=None)
cb_x_test=pd.read_csv('checkerboardDataset/Xtest.csv',header=None)
cb_x_train=pd.read_csv('checkerboardDataset/Xtrain.csv',header=None)
cb_x_test['labels']=cb_labels_test
cb_x_train['labels']=cb_labels_train
#put all the data together
cb_df=cb_x_train.append(cb_x_test,ignore_index=True)
#mix the data
cb_df,nul=split_sample(cb_df,1,permute=True)



#Linear dataset
ld_labels_test=pd.read_csv('linearDataset/labelsTest.csv',header=None)
ld_labels_train=pd.read_csv('linearDataset/labelsTrain.csv',header=None)
ld_x_test=pd.read_csv('linearDataset/Xtest.csv',header=None)
ld_x_train=pd.read_csv('linearDataset/Xtrain.csv',header=None)

#Ripley dataset
rd_labels_test=pd.read_csv('RipleyDataset/labelsTest.csv',header=None)
rd_labels_train=pd.read_csv('RipleyDataset/labelsTrain.csv',header=None)
rd_x_test=pd.read_csv('RipleyDataset/Xtest.csv',header=None)
rd_x_train=pd.read_csv('RipleyDataset/Xtrain.csv',header=None)

#Part I: Fitting SVM and tuning hyperparameters.
#Task 1 : For each dataset make a conjecture concerning a type of kernel to be used. Comment.
print('Task 1\n')

#Plot scatter plot for checkboard training dataset
X=cb_x_train.loc[cb_labels_train[cb_labels_train==1].dropna().index].iloc[:,0]
Y=cb_x_train.loc[cb_labels_train[cb_labels_train==1].dropna().index].iloc[:,1]
plt.scatter(X,Y,label='1')
X=cb_x_train.loc[cb_labels_train[cb_labels_train==-1].dropna().index].iloc[:,0]
Y=cb_x_train.loc[cb_labels_train[cb_labels_train==-1].dropna().index].iloc[:,1]
plt.scatter(X,Y,label='-1')
plt.legend()
plt.title('Checkerboard training dataset scatter plot')
plt.xlabel('0')
plt.ylabel('1')
plt.show()

print('On the scatter plot of the checkerboard dataset we can see clearly that the data is not linearly separable, we can make the hypothesis that the linear kernel isn\'t going to work.')
print('As such using the Radial Basis Function would be best suited for this dataset')

#Plot scatter plot for linear training dataset
X=ld_x_train.loc[ld_labels_train[ld_labels_train==1].dropna().index].iloc[:,0]
Y=ld_x_train.loc[ld_labels_train[ld_labels_train==1].dropna().index].iloc[:,1]
plt.scatter(X,Y,label='1')
X=ld_x_train.loc[ld_labels_train[ld_labels_train==-1].dropna().index].iloc[:,0]
Y=ld_x_train.loc[ld_labels_train[ld_labels_train==-1].dropna().index].iloc[:,1]
plt.scatter(X,Y,label='-1')
plt.legend()
plt.title('Linear training dataset scatter plot')
plt.xlabel('0')
plt.ylabel('1')
plt.show()

print('\nOn the scatter plot for the linear dataset, we can see that the data is clearly linearly separable, we should use the linear kernel')

#Plot scatter plot for Ripley training dataset
X=rd_x_train.loc[rd_labels_train[rd_labels_train==1].dropna().index].iloc[:,0]
Y=rd_x_train.loc[rd_labels_train[rd_labels_train==1].dropna().index].iloc[:,1]
plt.scatter(X,Y,label='1')
X=rd_x_train.loc[rd_labels_train[rd_labels_train==0].dropna().index].iloc[:,0]
Y=rd_x_train.loc[rd_labels_train[rd_labels_train==0].dropna().index].iloc[:,1]
plt.scatter(X,Y,label='0')
plt.legend()
plt.title('Ripley training dataset scatter plot')
plt.xlabel('0')
plt.ylabel('1')
plt.show()

print('\nFor the Ripley dataset, a polynomial would probably be the best kernel')

#Task 2 For each dataset and a kernel (according to your hypothesis) choose optimal hyperparameters of the model (C and kernel parameters if any) via cross-validation 
print('\nTask 2')

def k_fold_10(train_sample,kern,function,c_param,gam):
    list_=[]
    list_2=[]
    rang=[5*10*i for i in range(1,11)]
    for i in rang:
        if i<=50:
            test=train_sample.iloc[:i] 
            train=train_sample.iloc[i:]
        if i>50:
            test=train_sample.iloc[rang[rang.index(i)-1]:i]
            train_sample1=train_sample.iloc[:i-50]
            train_sample2=train_sample.iloc[i:]
            train=pd.concat([train_sample1,train_sample2])
        clf=function(C=c_param,kernel=kern,gamma=gam)
        clf.fit(train.iloc[:,:-1],np.array(train.iloc[:,-1]).reshape((-1)))
        #check against testing data
        y_pred=clf.predict(test.iloc[:,:-1])
        #create mean F1 measure to evaluate performance on validation data
        TP=sum([1 if np.array(test.iloc[:,-1])[i]==y_pred[i]==1 else 0 for i in range(len(y_pred))])
        
        FP=sum([1 if (np.array(test.iloc[:,-1])[i]!=1) & (y_pred[i]==1) else 0 for i in range(len(y_pred))])
        FN=sum([1 if (np.array(test.iloc[:,-1])[i]==1) & (y_pred[i]!=1) else 0 for i in range(len(y_pred))])
        precision=TP/(TP+FP)
        recall=TP/(TP+FN)
        F1_validation=(2*precision*recall)/(precision+recall)
        list_.append(F1_validation)

    return np.sum(list_)/len(list_)




#Find the optimal hyperparameters for the rbf 
#define range and step for both parameters
C_param_list=np.arange(300,400,5)
gam_list=np.arange(2,12,0.5)
def grid_search_rbf(C_param_list,gam_list,df):
    #implements grid search of c param and gamma parameter for rbf
    #returns dataframe containing mean F1 value for each combination of params
    df_validation=pd.DataFrame(columns=C_param_list,index=gam_list)
  
    for c_val in C_param_list:
        for gam_val in gam_list:
            f1_val=k_fold_10(df,'rbf',svm.SVC,c_val,gam_val)
            df_validation.loc[gam_val,c_val]=float(f1_val)

    return df_validation

#call the grid search function on validation data
grid_df_cb_validation=grid_search_rbf(C_param_list,gam_list,cb_x_train)
print('\nOur maximum F1 average for the RBF kernel on the checkboard is :', max(grid_df_cb_validation.max()))
#retrive the column index for maximum value
c_param=grid_df_cb_validation.max()[grid_df_cb_validation.max()==max(grid_df_cb_validation.max())].index[0]
#retrieve row value for maximum index
gam_param=grid_df_cb_validation.loc[:,c_param][grid_df_cb_validation.loc[:,c_param]==max(grid_df_cb_validation.loc[:,c_param])].index[0]

#plot for validation data
sns.heatmap(np.array(grid_df_cb_validation,dtype=float),xticklabels=gam_list,yticklabels=C_param_list)
plt.show()




