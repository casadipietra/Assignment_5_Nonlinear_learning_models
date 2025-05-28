import time
start = time.time()

from downloadrequirements import install_if_needed

# ✅ Verify requirements at startup
install_if_needed()

# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn import svm
from sklearn import metrics

# Import custom functions (including nested_10_fold_cv, grid_search_*, split_sample, etc.)
from homework import *

sns.set_style('darkgrid')
np.random.seed(10)

print("✅ Environment ready. Running main code...")

# ==========================================
# Load and prepare datasets
# ==========================================

# --- Checkerboard dataset ---
cb_labels_test = pd.read_csv('checkerboardDataset/labelsTest.csv', header=None)
cb_labels_train = pd.read_csv('checkerboardDataset/labelsTrain.csv', header=None)
cb_x_test = pd.read_csv('checkerboardDataset/Xtest.csv', header=None)
cb_x_train = pd.read_csv('checkerboardDataset/Xtrain.csv', header=None)

cb_x_test['labels'] = cb_labels_test
cb_x_train['labels'] = cb_labels_train

# Combine train and test, then shuffle
cb_df = cb_x_train.append(cb_x_test, ignore_index=True)
cb_df, _ = split_sample(cb_df, 1, permute=True)

# --- Linear dataset ---
ld_labels_test = pd.read_csv('linearDataset/labelsTest.csv', header=None)
ld_labels_train = pd.read_csv('linearDataset/labelsTrain.csv', header=None)
ld_x_test = pd.read_csv('linearDataset/Xtest.csv', header=None)
ld_x_train = pd.read_csv('linearDataset/Xtrain.csv', header=None)

ld_x_test['labels'] = ld_labels_test
ld_x_train['labels'] = ld_labels_train

ld_df = ld_x_train.append(ld_x_test, ignore_index=True)
ld_df, _ = split_sample(ld_df, 0.2, permute=True)

# --- Ripley dataset ---
rd_labels_test = pd.read_csv('RipleyDataset/labelsTest.csv', header=None)
rd_labels_train = pd.read_csv('RipleyDataset/labelsTrain.csv', header=None)

# Convert 0→-1 in labels
rd_labels_test[0] = rd_labels_test[0].replace({0: -1})
rd_labels_train[0] = rd_labels_train[0].replace({0: -1})

rd_x_test = pd.read_csv('RipleyDataset/Xtest.csv', header=None)
rd_x_train = pd.read_csv('RipleyDataset/Xtrain.csv', header=None)

rd_x_test['labels'] = rd_labels_test
rd_x_train['labels'] = rd_labels_train

rd_df = rd_x_train.append(rd_x_test, ignore_index=True)
rd_df, _ = split_sample(rd_df, 0.2, permute=True)

# ==========================================
# Plotting function for nested CV grid
# ==========================================
def plot_3d_nested_grid(validation_df, kernel='rbf', title='Nested CV Validation Scores'):
    """
    Plot a 3D surface for the averaged validation grid returned by nested_10_fold_cv.

    Args:
        validation_df (pd.DataFrame): Grid of validation scores (index = gamma or degree, columns = C values).
        kernel (str): Kernel type, 'rbf' or 'poly'. Determines label for Y-axis.
        title (str): Title for the plot.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    C_vals = validation_df.columns.astype(float)
    param_vals = validation_df.index.astype(float)  # gamma for RBF, degree for poly

    C_mesh, param_mesh = np.meshgrid(C_vals, param_vals)
    Z = validation_df.values.astype(float)

    # Always use log10 scale for C
    log_C = np.log10(C_mesh)

    if kernel == 'rbf':
        log_param = np.log10(param_mesh)
        ax.set_ylabel('log10(Gamma)')
    elif kernel == 'poly':
        log_param = param_mesh
        ax.set_ylabel('Degree')
    else:
        log_param = param_mesh
        ax.set_ylabel('Parameter')

    ax.plot_surface(log_C, log_param, Z, cmap='viridis', edgecolor='k')
    ax.set_xlabel('log10(C)')
    ax.set_zlabel('F1 Score')
    ax.set_title(title)
    plt.show()

# ==========================================
# Part II: Model fitting and selection via nested cross-validation
# ==========================================
print('\nPart 2 Task 1')

# Use the checkerboard dataset for nested CV
train_sample = cb_df

# For linear and RBF kernels, degree is unused (can be any placeholder)
degr = 3

# Grids for hyperparameters
C_param_list_cd = np.logspace(-2, 2.5, num=10)
gam_list = np.logspace(-2, 1, num=10)
degree_list = np.logspace(-2, 2, num=10)

# --- Nested CV for RBF kernel ---
nest_cv_rbf, rbf_grid = nested_10_fold_cv(
    C_param_list_cd, gam_list, 'rbf', train_sample, degr, inner_fold=5
)
print('\nUsing the RBF kernel we get an average F1 of:', nest_cv_rbf)
print('The grid of validation scores is:\n', rbf_grid)
best_gamma, best_C = rbf_grid.astype(float).stack().idxmax()
best_val = rbf_grid.loc[best_gamma, best_C]

print(f"Maximum validation score = {best_val:.4f}  at  C = {best_C}  and  gamma = {best_gamma}")

plot_3d_nested_grid(rbf_grid, kernel='rbf', title='Nested CV RBF Kernel F1 Score')

# ==========================================
# Learning curve with best hyperparameters
# ==========================================

from sklearn.model_selection import learning_curve

# On fixe C_opt et gamma_opt obtenus par nested CV
C_opt, gamma_opt = best_C, best_gamma  

clf = svm.SVC(C=C_opt, kernel='rbf', gamma=gamma_opt)

train_sizes, train_scores, test_scores = learning_curve(
    clf, 
    X = train_sample.iloc[:, :-1], 
    y = train_sample.iloc[:, -1],
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='f1'
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(7,5))
plt.plot(train_sizes, train_scores_mean, 'o-', label="Train F1")
plt.plot(train_sizes, test_scores_mean, 'o-', label="Valid F1")
plt.xlabel("Fraction du jeu d'entraînement")
plt.ylabel("F1-score")
plt.title("Learning Curve (C={}, γ={})".format(C_opt, gamma_opt))
plt.legend()
plt.show()

# ==========================================
# Final model training and ROC/Confusion on Ripley test set
# ==========================================
# Train final classifier on all of train_sample


from sklearn.metrics import roc_curve, auc

clf_final = svm.SVC(C=best_C, kernel='rbf', gamma=best_gamma, probability=True)
clf_final.fit(train_sample.iloc[:, :-1], train_sample.iloc[:, -1])

# Prepare Ripley test set
X_test = rd_df.iloc[:, :-1].to_numpy()
y_test = rd_df.iloc[:, -1].to_numpy()

# Compute scores for ROC
y_score = clf_final.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve (final model on Ripley test set)")
plt.legend(loc="lower right")
plt.show()

print("plot Nested CV for RBF kernel finished — elapsed time: {:.2f} seconds".format(time.time() - start))
"""
# --- Nested CV for Linear kernel ---
nest_cv_linear, linear_grid = nested_10_fold_cv(
    C_param_list_cd, gam_list, 'linear', train_sample, degr, inner_fold=5
)
print('\nUsing the linear kernel we get an average F1 of:', nest_cv_linear)
# For the linear kernel, gamma doesn’t vary; you can still visualize the grid if desired:
# plot_3d_nested_grid(linear_grid, kernel='rbf', title='Nested CV Linear Kernel F1 Score')

# --- Nested CV for Polynomial kernel ---
nest_cv_poly, poly_grid = nested_10_fold_cv(
    C_param_list_cd, gam_list, 'poly', train_sample, degree_list, inner_fold=5
)
print('\nUsing the polynomial kernel we get an average F1 of:', nest_cv_poly)
plot_3d_nested_grid(poly_grid, kernel='poly', title='Nested CV Polynomial Kernel F1 Score')

print('\nWe observe that the polynomial kernel performs almost as well as RBF.')
print('Conclusion: The RBF kernel is best suited for this dataset.')

print("Task 3 finished — elapsed time: {:.2f} seconds".format(time.time() - start))
"""