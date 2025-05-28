
from downloadrequirements import install_if_needed

# ✅ Vérifie les requirements au lancement
install_if_needed()

# Ensuite, tu peux importer tes libs normalement
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn import metrics
import numpy as np

# 🎯 Ton vrai code commence ici
print("✅ Environnement prêt. Lancement du code principal...")


from homework import *

sns.set_style('darkgrid')
np.random.seed(10)

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
ld_x_test['labels']=ld_labels_test
ld_x_train['labels']=ld_labels_train
#put all the data together
ld_df=ld_x_train.append(ld_x_test,ignore_index=True)
#mix the data
ld_df,nul=split_sample(ld_df,1,permute=True)

#Ripley dataset
rd_labels_test=pd.read_csv('RipleyDataset/labelsTest.csv',header=None)
rd_labels_train=pd.read_csv('RipleyDataset/labelsTrain.csv',header=None)
#change 0 to -1 in the labels
for i in range(len(rd_labels_test)):
    if rd_labels_test[0][i]==0:
        rd_labels_test[0][i]=-1
for i in range(len(rd_labels_train)):
    if rd_labels_train[0][i]==0:
        rd_labels_train[0][i]=-1
rd_x_test=pd.read_csv('RipleyDataset/Xtest.csv',header=None)
rd_x_train=pd.read_csv('RipleyDataset/Xtrain.csv',header=None)
rd_x_test['labels']=rd_labels_test
rd_x_train['labels']=rd_labels_train
#put all the data together
rd_df=rd_x_train.append(rd_x_test,ignore_index=True)
#mix the data
rd_df,nul=split_sample(rd_df,1,permute=True)

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

print('On the scatter plot of the checkboard dataset we can see clearly that the data is not linearly separable, we can make the hypothesis that the linear kernel isn\'t going to work.')
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
X=rd_x_train.loc[rd_labels_train[rd_labels_train==-1].dropna().index].iloc[:,0]
Y=rd_x_train.loc[rd_labels_train[rd_labels_train==-1].dropna().index].iloc[:,1]
plt.scatter(X,Y,label='-1')
plt.legend()
plt.title('Ripley training dataset scatter plot')
plt.xlabel('0')
plt.ylabel('1')
plt.show()

print('\nFor the Ripley dataset, a polynomial would probably be the best kernel')

#Task 2 For each dataset and a kernel (according to your hypothesis) choose optimal hyperparameters of the model (C and kernel parameters if any) via cross-validation 
print('\nTask 2')




#Find the optimal hyperparameters for the rbf 
#define range and step for both parameters
#C_param_list_cd=np.arange(300,400,5)
C_param_list_cd=np.logspace(1, 5, 10)
gam_list=np.logspace(2, 5, 10)
#gam_list=np.arange(2,12,0.5)


#call the grid search function
validation_df,training_df=grid_search_rbf(C_param_list_cd,gam_list,cb_df,10)
print('\nOur maximum F1 average for the RBF kernel on the checkboard is :', max(validation_df.max()))
#retrive the column index for maximum value
c_param_cd=validation_df.max()[validation_df.max()==max(validation_df.max())].index[0]
#retrieve row value for maximum index
gam_param_cd=validation_df.loc[:,c_param_cd][validation_df.loc[:,c_param_cd]==max(validation_df.loc[:,c_param_cd])].index[0]


#Optimal parameters for the linear kernel
#In this case we only use the C value.

#C_param_list_ld=np.arange(0.1,5,0.1)
C_param_list_ld=np.logspace(1, 5, 10)
     
validation_df_ld,training_df_ld=grid_search_linear(C_param_list_ld,ld_df,10)
print('\nOur maximum F1 average for the linear kernel on the linear dataset is :', max(validation_df_ld.max()))
#Changing the value of C doesn't seem to change the error that much.
c_param_ld=validation_df_ld.max()[validation_df_ld.max()==max(validation_df_ld.max())].index[0]

#Optimal parameters for the polynomial Kernel
#C_param_list_rd=np.arange(15,30,2)
C_param_list_rd=np.logspace(1, 5, 10)
degree_list_rd=np.logspace(1, 5, 10)
#degree_list_rd=np.arange(1,10,1)

validation_df_rd,training_df_rd=grid_search_polynomial(C_param_list_rd,degree_list_rd,rd_df,10)
print('\nOur maximum F1 average for the polynomial kernel on the Ripley dataset is :', max(validation_df_rd.max()))

c_param_rd=validation_df_rd.max()[validation_df_rd.max()==max(validation_df_rd.max())].index[0]
degree_param_rd=validation_df_rd.loc[:,c_param_rd][validation_df_rd.loc[:,c_param_rd]==max(validation_df_rd.loc[:,c_param_rd])].index[0]

print('\nTask 3')
######Checkboard dataset 
#plot heatmap for validation data
sns.heatmap(np.array(validation_df,dtype=float),xticklabels=gam_list,yticklabels=C_param_list_cd)
plt.title('Heatmap of the RBF kernel on the validation data of \n the Checkboard dataset ')
plt.xlabel('Gamma value')
plt.ylabel('C value')
plt.show()

#plot heatmap for training data
sns.heatmap(np.array(training_df,dtype=float),xticklabels=gam_list,yticklabels=C_param_list_cd)
plt.title('Heatmap of the polynomial kernel on the training data of \n the Checkboard dataset ')
plt.xlabel('Gamma value')
plt.ylabel('C value')
plt.show()

#######Linear dataset
#plot heatmap for validation data
sns.heatmap(np.array(validation_df_ld,dtype=float),xticklabels=C_param_list_ld)
plt.title('Heatmap of the linear kernel on the validation data of \n the linear dataset ')
plt.xlabel('C value')
plt.show()

#plot heatmap for training data
sns.heatmap(np.array(training_df_ld,dtype=float),xticklabels=C_param_list_ld)
plt.title('Heatmap of the linear kernel on the training data of \n the linear dataset ')
plt.xlabel('C value')
plt.show()

####Ripley dataset
#plot heatmap for validation data
sns.heatmap(np.array(validation_df_rd,dtype=float),xticklabels=degree_list_rd,yticklabels=C_param_list_rd)
plt.title('Heatmap of the polynomial kernel on the validation data of \n the Ripley dataset ')
plt.xlabel('Degree of polynomial')
plt.ylabel('C value')
plt.show()

#plot heatmap for training data
sns.heatmap(np.array(training_df_rd,dtype=float),xticklabels=degree_list_rd,yticklabels=C_param_list_rd)
plt.title('Heatmap of the polynomial kernel on the training data of \n the Ripley dataset ')
plt.xlabel('Degree of polynomial')
plt.ylabel('C value')
plt.show()

#Part II: Model fitting and selection via nested cross-validation.
print('\nPart 2 Task 1')


#we use the second data set
train_sample=cb_df
#define degree for linear and rbf cases where it is not used, can be any value
degr=3

#define the grid for C
#C_param_list_cd=np.arange(1,100,5)
C_param_list_cd=np.logspace(1, 5, 10)
#grid for gamma
#gam_list=np.arange(2,6,2)
gam_list=np.logspace(1, 5, 10)

#nested cross validation for RBF kernel
nest_cv_rbf=nested_10_fold_cv(C_param_list_cd,gam_list,'rbf',train_sample,degr,5)
print('\nUsing the RBF kernel we get a F1 average of: ',nest_cv_rbf)
#nested cross validation for linear kernel
nest_cv_linear=nested_10_fold_cv(C_param_list_cd,gam_list,'linear',train_sample,degr,5)
print('\nUsing the linear kernel we get:  ',nest_cv_linear)

#define the grid for the degree
#degree_list=np.arange(2,10,1)
degree_list=np.logspace(1, 5, 10)
##nested cross validation for polynomial kernel

nest_cv_polynomial=nested_10_fold_cv(C_param_list_cd,gam_list,'poly',train_sample,degree_list,5)
print('\nUsing the polynomial kernel we get:  ',nest_cv_polynomial)
print('\nWe can see that the polynomial kernel is surpisingly accurate only just less than the RBF')

print('\nWe can say that the rbf kernel is the best suited for this dataset')


