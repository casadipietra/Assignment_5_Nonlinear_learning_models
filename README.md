
<h1 align="center">
  <a>Assignment 5 – Nonlinear learning models 
    (Support Vector Machines (SVM))</a>
</h1>
<h3 align="center">
  <a>Machine Learning @ Uni-Konstanz 2021</a>
</h3>

## Idea 📓

- Consider binary classification tasks.

- Learn to work with Support Vector Machines (SVMs) with various standard kernels: construct, train, and test SVM classifiers.

- Consider the case of d-dimensional features with d=2.

***

**The goal of all these tasks is to learn how to construct and train (fit) SVMs with the particular examples of kernels, namely the linear, polynomial, and RBF kernels, which are considered to be one of the most effective off-shelf classification tools for high-dimensional datasets with complex structure.** You will have to carry out the binary classification tasks for d-dimensional features (d=2) for several given datasets, which includes model selection, hyperparameter tuning, generalization properties assessment.

***

## Data 📦

- [1: Linearly separable dataset]
- [2: Checkboard Dataset] 
- [3: Ripley Dataset]

All these datasets are taken from from the EPFL advanced machine learning class taught by Prof. Aude Billard: 

https://github.com/epfl-lasa/ML_toolbox

***

For an example on how to load these datasets, please take a look at `examples\example_datasets.m`.

***

## Tasks 📝

See the supporting document `Assignment-5-task.pdf` (Task 1 on Page 9). The following points contain the task:
	
1. **Part I: Fitting SVM and tuning hyperparameters**.

	- *Task 1* : For each dataset make a conjecture concerning a type of kernel to be used. Comment.
  	- *Task 2* : For each dataset and a kernel (according to your hypothesis) choose optimal hyperparameters of the model (C and kernel parameters if any) via cross-validation using the strategy followed in the tutorial and mean F1 measure for evaluating the model performance. Recall that once the best model has been selected via CV it needs to be retrained on the total training set.
  	- *Task 3* : For each dataset plot mean F1 score for the grid of parameters both for the training and validation sets (see heatmaps in Tutorial 5).

2. **Part II: Model fitting and selection via nested cross-validation**.
	
	- *Task 1* : For dataset 2 evaluate performance of SVMs with different kernels via nested cross-validation with F1 score. Make conclusions about the best model.
Hint:  For that see my comments in feedback to Assignment 1 (on model selection), Elements of Statistical Learning (on wrong ways of cross-validation), or for a completely naive exposition Scenario 3 in https://sebastianraschka.com/faq/docs/evaluate-a-model.html.

## Remark ⚠️
** In case you are running into computationally heavy computations, make the grid for grid search less fine. Mind using logarithmic scale as in Tutorial 5.

## Notes ⚠️

**Write your assignment code following the same rules as for the previous assignments and use the assisting code from Tutorial 5**.



<br>

***Work well!***

<br>

### Acknowledgement

This code heavily borrows from the [ML_toolbox](https://github.com/epfl-lasa/ML_toolbox) repository: "_a Machine learning toolbox containing algorithms for dimensionality reduction, clustering, classification and regression along with examples and tutorials which accompany the Master level course [Advanced Machine Learning](http://lasa.epfl.ch/teaching/lectures/ML_MSc_Advanced/index.php)  and [Machine Learning Programming](http://edu.epfl.ch/coursebook/fr/machine-learning-programming-MICRO-401) taught at [EPFL](https://www.epfl.ch/) by [Prof. Aude Billard](http://lasa.epfl.ch/people/member.php?SCIPER=115671)_".

The main authors of the toolbox and accompanying tutorials were the TA's from Spring 2016/2017 semesters:  
[Guillaume de Chambrier](http://chambrierg.com/), [Nadia Figueroa](http://lasa.epfl.ch/people/member.php?SCIPER=238387) and [Denys Lamotte](http://lasa.epfl.ch/people/member.php?SCIPER=231543)

#### 3rd Party Software
This toolbox includes 3rd party software:
- [Matlab Toolbox for Dimensionality Reduction](https://lvdmaaten.github.io/drtoolbox/)
- [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
- [Kernel Methods Toolbox](https://github.com/steven2358/kmbox)
- [SparseBayes Software](http://www.miketipping.com/downloads.htm)

You DO NOT need to install these, they are already pre-packaged in this toolbox.
