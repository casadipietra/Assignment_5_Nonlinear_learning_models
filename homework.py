import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

import numpy as np
import pandas as pd
from sklearn import svm

def k_fold(train_sample, kern, function, c_param, gam, degr, fold, return_training_err=True):
    """
    Perform a manual K‐fold cross‐validation.

    Parameters:
    - train_sample        : DataFrame containing features + 'labels' column at the end.
    - kern                : kernel type for SVC ('rbf', 'linear', or 'poly').
    - function            : classifier constructor (e.g., svm.SVC).
    - c_param             : value of C to test (float).
    - gam                 : value of gamma to test (float or 'auto').
    - degr                : degree for polynomial kernel (int) or None.
                            (ignored if kern is 'rbf' or 'linear')
    - fold                : number of folds (int).
    - return_training_err : if True, also compute F1 on training splits.


    How it works:
    1. Compute fold_size = len(train_sample) // fold.
    2. Build a list of cut‐points (multiples of fold_size) for each fold.
    3. For each cut‐point i:
       a. Define the test split as rows [prev_cut:i] (or [0:i] for the first fold).
       b. Define the training split as all other rows.
       c. Instantiate SVC with (kernel=kern, C=c_param, gamma=gam, degree=degr or default).
       d. Fit on train split; predict on test split to compute F1.
       e. If return_training_err, also predict on the train split itself and compute F1.
    4. After looping through all folds, average the collected F1 scores.

    Note: The 'labels' must be in the last column of train_sample.

        Returns:
    - (mean_f1_val, mean_f1_train) if return_training_err is True,
      otherwise (mean_f1_val, 0).

    """
     
    list_ = []    # will hold F1 scores on validation folds
    list_2 = []   # will hold F1 scores on training folds
    n = len(train_sample)
    fold_size = n // fold

    # build list of cut points at multiples of fold_size
    rang = [fold_size * i for i in range(1, fold + 1)]

    for i in rang:
        # define test/train split for this fold
        if i <= fold_size:
            test = train_sample.iloc[:i]
            train = train_sample.iloc[i:]
        else:
            prev_cut = rang[rang.index(i) - 1]
            test = train_sample.iloc[prev_cut:i]
            train_part1 = train_sample.iloc[:prev_cut]
            train_part2 = train_sample.iloc[i:]
            train = pd.concat([train_part1, train_part2]).reset_index(drop=True)

        # instantiate the classifier with current hyperparameters
        clf = function(kernel=kern, C=c_param, gamma=gam, degree=(int(degr) if degr is not None else 3))
        # fit on training split
        X_train = train.iloc[:, :-1].to_numpy()
        y_train = train.iloc[:, -1].to_numpy().reshape(-1)
        clf.fit(X_train, y_train)

        # predict on validation split
        X_test = test.iloc[:, :-1].to_numpy()
        y_test = test.iloc[:, -1].to_numpy().reshape(-1)
        y_pred = clf.predict(X_test)

        # compute F1 on this validation split
        TP = np.sum((y_test == 1) & (y_pred == 1))
        FP = np.sum((y_test != 1) & (y_pred == 1))
        FN = np.sum((y_test == 1) & (y_pred != 1))
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        list_.append(F1)

        # if requested, also compute F1 on the same training split
        if return_training_err:
            y_pred_train = clf.predict(X_train)
            TP_t = np.sum((y_train == 1) & (y_pred_train == 1))
            FP_t = np.sum((y_train != 1) & (y_pred_train == 1))
            FN_t = np.sum((y_train == 1) & (y_pred_train != 1))
            precision_t = TP_t / (TP_t + FP_t) if (TP_t + FP_t) > 0 else 0
            recall_t = TP_t / (TP_t + FN_t) if (TP_t + FN_t) > 0 else 0
            F1_t = (2 * precision_t * recall_t) / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0
            list_2.append(F1_t)

    # compute mean F1 over all folds
    mean_f1_val = np.mean(list_)
    if return_training_err:
        mean_f1_train = np.mean(list_2) if len(list_2) > 0 else 0
    else:
        mean_f1_train = 0

    return mean_f1_val, mean_f1_train


def grid_search_rbf(C_param_list,gam_list,df,fold,return_training_err=True):
    """
    Perform grid search over C and gamma for an RBF‐kernel SVM.

    Parameters:
    - C_param_list        : list of C values to test.
    - gam_list            : list of gamma values to test.
    - df                  : DataFrame containing features + 'labels' column.
    - fold                : number of folds for cross‐validation (passed to k_fold).
    - return_training_err : if True, collect F1 on training folds as well.

    Returns:
    - validation_df : DataFrame of shape (len(gam_list) × len(C_param_list)),
                      where index = gam_list, columns = C_param_list,
                      and entries = mean F1 on validation splits.
    - training_df   : same shape, but entries = mean F1 on training splits.
    """
    validation_df=pd.DataFrame(columns=C_param_list,index=gam_list)
    training_df=pd.DataFrame(columns=C_param_list,index=gam_list)
    for c_val in C_param_list:
        for gam_val in gam_list:
            #print(f"🔍 Testing C={c_val:.5f}, gamma={gam_val:.5f}")
            f1_v, f1_t = k_fold(df, 'rbf', svm.SVC, c_val, gam_val, 3, fold, return_training_err)
            #print(f"✅ F1 Validation: {f1_v:.5f}, F1 Training: {f1_t:.5f}")
            validation_df.loc[gam_val, c_val] = float(f1_v)
            training_df.loc[gam_val, c_val] = float(f1_t)

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
    grid_accumulator = None  # to accumulate validation grids

    for i in range(10):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i != 9 else len(train_sample)

        test = train_sample.iloc[start_idx:end_idx]
        train = pd.concat([train_sample.iloc[:start_idx], train_sample.iloc[end_idx:]])

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

        # Accumulate validation grid for averaging later
        if grid_accumulator is None:
            grid_accumulator = validation_df.copy()
        else:
            grid_accumulator += validation_df

        clf = svm.SVC(C=c_param_cd, kernel=kern, gamma=gam_param_cd, degree=degr)
        clf.fit(train.iloc[:, :-1], train.iloc[:, -1])

        y_true = test.iloc[:, -1].to_numpy()
        y_pred = clf.predict(test.iloc[:, :-1])

        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true != 1) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred != 1))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        list_f1.append(f1)

    # Average grid over all folds
    averaged_validation_df = grid_accumulator / 10

    return np.mean(list_f1), averaged_validation_df




from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot_3d_surface(validation_df, title='F1 Validation Score'):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    C_vals = validation_df.columns.astype(float)
    gamma_vals = validation_df.index.astype(float)
    C_mesh, gamma_mesh = np.meshgrid(C_vals, gamma_vals)

    Z = validation_df.values.astype(float)

    ax.plot_surface(np.log10(C_mesh), np.log10(gamma_mesh), Z, cmap='viridis', edgecolor='k')
    ax.set_xlabel('log10(C)')
    ax.set_ylabel('log10(Gamma)')
    ax.set_zlabel('F1 Score')
    ax.set_title(title)
    plt.show()


if __name__ == '__main__':
    split_sample()
    k_fold()
    grid_search_rbf()
    grid_search_linear()
    grid_search_polynomial()
    nested_10_fold_cv()
    
    plot_3d_surface()
    
    
