---
title: Machine Learning with Tree-Based Models in Python
date: 2021-12-07 11:22:08 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Machine Learning with Tree-Based Models in Python
=====================================================







 This is the memo of the 24th course of ‘Data Scientist with Python’ track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/machine-learning-with-tree-based-models-in-python)**
 .





---



# **1. Classification and Regression Trees(CART)**
-------------------------------------------------




## **1.1 Decision tree for classification**



![Desktop View]({{ site.baseurl }}/assets/datacamp/machine-learning-with-tree-based-models-in-python/capture-1.png?w=1024)

####
**Train your first classification tree**



 In this exercise you’ll work with the
 [Wisconsin Breast Cancer Dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
 from the UCI machine learning repository. You’ll predict whether a tumor is malignant or benign based on two features: the mean radius of the tumor (
 `radius_mean`
 ) and its mean number of concave points (
 `concave points_mean`
 ).





```python

# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth of 6
dt = DecisionTreeClassifier(max_depth=6, random_state=SEED)

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict test set labels
y_pred = dt.predict(X_test)
print(y_pred[0:5])

# [0 0 0 1 0]

```


####
**Evaluate the classification tree**




```python

# Import accuracy_score
from sklearn.metrics import accuracy_score

# Predict test set labels
y_pred = dt.predict(X_test)

# Compute test set accuracy
acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))

# Test set accuracy: 0.89

```



 Using only two features, your tree was able to achieve an accuracy of 89%!



####
**Logistic regression vs classification tree**



 A classification tree divides the feature space into
 **rectangular regions**
 . In contrast, a linear model such as logistic regression produces only a single linear decision boundary dividing the feature space into two decision regions.





```

help(plot_labeled_decision_regions)

Signature: plot_labeled_decision_regions(X, y, models)
Docstring:
Function producing a scatter plot of the instances contained
in the 2D dataset (X,y) along with the decision
regions of two trained classification models contained in the
list 'models'.

Parameters
----------
X: pandas DataFrame corresponding to two numerical features
y: pandas Series corresponding the class labels
models: list containing two trained classifiers
File:      /tmp/tmpzto071yc/<ipython-input-1-9e70bec83095>
Type:      function

```




```python

# Import LogisticRegression from sklearn.linear_model
from sklearn.linear_model import  LogisticRegression

# Instatiate logreg
logreg = LogisticRegression(random_state=1)

# Fit logreg to the training set
logreg.fit(X_train, y_train)

# Define a list called clfs containing the two classifiers logreg and dt
clfs = [logreg, dt]

# Review the decision regions of the two classifiers
plot_labeled_decision_regions(X_test, y_test, clfs)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/machine-learning-with-tree-based-models-in-python/capture1-1.png?w=1024)


 Notice how the decision boundary produced by logistic regression is linear while the boundaries produced by the classification tree divide the feature space into rectangular regions.





---


## **1.2 Classification tree Learning**



**Terms**



* **Decision Tree:**
 data structure consisting of a hierarchy of nodes
* **Node:**
 question or prediction



**Node**



* **Root:**
 no parent node, question giving rise to two children nodes
* **Internal node:**
 one parent node, question giving rise to two children nodes
* **Leaf:**
 one parent node, no children nodes –> prediction



![Desktop View]({{ site.baseurl }}/assets/datacamp/machine-learning-with-tree-based-models-in-python/capture2-1.png?w=1024)


**[Information gain in decision trees](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees)**


[Gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)



####
**Growing a classification tree**


* The existence of a node depends on the state of its predecessors.
* The impurity of a node can be determined using different criteria such as entropy and the gini-index.
* When the information gain resulting from splitting a node is null, the node is declared as a leaf.
* When an internal node is split, the split is performed in such a way so that information gain is maximized.


####
**Using entropy as a criterion**



 In this exercise, you’ll train a classification tree on the Wisconsin Breast Cancer dataset using entropy as an information criterion.





```python

# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# Instantiate dt_entropy, set 'entropy' as the information criterion
dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)

# Fit dt_entropy to the training set
dt_entropy.fit(X_train, y_train)

```


####
**Entropy vs Gini index**



 In this exercise you’ll compare the test set accuracy of
 `dt_entropy`
 to the accuracy of another tree named
 `dt_gini`
 .





```python

# Import accuracy_score from sklearn.metrics
from sklearn.metrics import accuracy_score

# Use dt_entropy to predict test set labels
y_pred= dt_entropy.predict(X_test)

# Evaluate accuracy_entropy
accuracy_entropy = accuracy_score(y_test, y_pred)

# Print accuracy_entropy
print('Accuracy achieved by using entropy: ', accuracy_entropy)

# Print accuracy_gini
print('Accuracy achieved by using the gini index: ', accuracy_gini)

# Accuracy achieved by using entropy:  0.929824561404
# Accuracy achieved by using the gini index:  0.929824561404

```



 Notice how the two models achieve exactly the same accuracy. Most of the time, the gini index and entropy lead to the same results. The gini index is slightly faster to compute and is the default criterion used in the
 `DecisionTreeClassifier`
 model of scikit-learn.





---


## **1.3 Decision tree for regression**


####
**Train your first regression tree**



 In this exercise, you’ll train a regression tree to predict the
 `mpg`
 (miles per gallon) consumption of cars in the
 [auto-mpg dataset](https://www.kaggle.com/uciml/autompg-dataset)
 using all the six available features.





```python

# Import DecisionTreeRegressor from sklearn.tree
from sklearn.tree import DecisionTreeRegressor

# Instantiate dt
dt = DecisionTreeRegressor(max_depth=8,
             min_samples_leaf=0.13,
            random_state=3)

# Fit dt to the training set
dt.fit(X_train, y_train)

```


####
**Evaluate the regression tree**



 In this exercise, you will evaluate the test set performance of
 `dt`
 using the Root Mean Squared Error (RMSE) metric. The RMSE of a model measures, on average, how much the model’s predictions differ from the actual labels.




 The RMSE of a model can be obtained by computing the square root of the model’s Mean Squared Error (MSE).





```python

# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute y_pred
y_pred = dt.predict(X_test)

# Compute mse_dt
mse_dt = MSE(y_test, y_pred)

# Compute rmse_dt
rmse_dt = mse_dt ** 0.5

# Print rmse_dt
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))

# Test set RMSE of dt: 4.37

```


####
**Linear regression vs regression tree**




```python

# Predict test set labels
y_pred_lr = lr.predict(X_test)

# Compute mse_lr
mse_lr = MSE(y_test, y_pred_lr)

# Compute rmse_lr
rmse_lr = mse_lr ** 0.5

# Print rmse_lr
print('Linear Regression test set RMSE: {:.2f}'.format(rmse_lr))

# Print rmse_dt
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_dt))

# Linear Regression test set RMSE: 5.10
# Regression Tree test set RMSE: 4.37

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/machine-learning-with-tree-based-models-in-python/capture3-2.png?w=1024)



---



# **2. The Bias-Variance Tradeoff**
----------------------------------


## **2.1 Generalization Error**



**[Bias–variance tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)**




![Desktop View]({{ site.baseurl }}/assets/datacamp/machine-learning-with-tree-based-models-in-python/capture4-2.png?w=908)


![Desktop View]({{ site.baseurl }}/assets/datacamp/machine-learning-with-tree-based-models-in-python/capture5-2.png?w=809)

####
**Overfitting and underfitting**



![Desktop View]({{ site.baseurl }}/assets/datacamp/machine-learning-with-tree-based-models-in-python/capture6-2.png?w=1024)


 A: complex = overfit = low bias = high variance


 B: simple = underfit = high bias = low variance





---


## **2.2 Diagnose bias and variance problems**


####
**Instantiate the model**



 In the following set of exercises, you’ll diagnose the bias and variance problems of a regression tree.





```python

# Import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split

# Set SEED for reproducibility
SEED = 1

# Split the data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Instantiate a DecisionTreeRegressor dt
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26, random_state=SEED)

```


####
**Evaluate the 10-fold CV error**



 In this exercise, you’ll evaluate the 10-fold CV Root Mean Squared Error (RMSE) achieved by the regression tree
 `dt`
 that you instantiated in the previous exercise.





```python

# Compute the array containing the 10-folds CV MSEs
MSE_CV_scores = - cross_val_score(dt, X_train, y_train, cv=10,
                       scoring='neg_mean_squared_error',
                       n_jobs=-1)

# Compute the 10-folds CV RMSE
RMSE_CV = (MSE_CV_scores.mean())**(0.5)

# Print RMSE_CV
print('CV RMSE: {:.2f}'.format(RMSE_CV))

# CV RMSE: 5.14

```



 A very good practice is to keep the test set untouched until you are confident about your model’s performance.




 CV is a great technique to get an estimate of a model’s performance without affecting the test set.



####
**Evaluate the training error**




```python

# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict the labels of the training set
y_pred_train = dt.predict(X_train)

# Evaluate the training set RMSE of dt
RMSE_train = (MSE(y_train, y_pred_train))**(0.5)

# Print RMSE_train
print('Train RMSE: {:.2f}'.format(RMSE_train))

# Train RMSE: 5.15

```



 Notice how the training error is roughly equal to the 10-folds CV error you obtained in the previous exercice.



####
**High bias or high variance?**



 In this exercise you’ll diagnose whether the regression tree
 `dt`
 you trained in the previous exercise suffers from a bias or a variance problem.




 The training set RMSE (
 `RMSE_train`
 ) and the CV RMSE (
 `RMSE_CV`
 ) achieved by
 `dt`
 are available in your workspace. In addition, we have also loaded a variable called
 `baseline_RMSE`
 which corresponds to the root mean-squared error achieved by the regression-tree trained with the
 `disp`
 feature only (it is the RMSE achieved by the regression tree trained in chapter 1, lesson 3).




 Here
 `baseline_RMSE`
 serves as the baseline RMSE.


 When above baseline, the model is considered to be underfitting.


 When below baseline, the model is considered ‘good enough’.




 Does
 `dt`
 suffer from a high bias or a high variance problem?





```

RMSE_train = 5.15
RMSE_CV = 5.14
baseline_RMSE = 5.1

```



`dt`
 suffers from high bias because
 `RMSE_CV`
 ≈
 `RMSE_train`
 and both scores are greater than
 `baseline_RMSE`
 .




`dt`
 is indeed underfitting the training set as the model is too constrained to capture the nonlinear dependencies between features and labels.





---


## **2.3 Ensemble Learning**


####
**Define the ensemble**



 In the following set of exercises, you’ll work with the
 [Indian Liver Patient Dataset](https://www.kaggle.com/jeevannagaraj/indian-liver-patient-dataset)
 from the UCI Machine learning repository.




 In this exercise, you’ll instantiate three classifiers to predict whether a patient suffers from a liver disease using all the features present in the dataset.





```

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier

# Set seed for reproducibility
SEED=1

# Instantiate lr
lr = LogisticRegression(random_state=SEED)

# Instantiate knn
knn = KNN(n_neighbors=27)

# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)

# Define the list classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]

```


####
**Evaluate individual classifiers**



 In this exercise you’ll evaluate the performance of the models in the list
 `classifiers`
 that we defined in the previous exercise. You’ll do so by fitting each classifier on the training set and evaluating its test set accuracy.





```

from sklearn.metrics import accuracy_score

# Iterate over the pre-defined list of classifiers
for clf_name, clf in classifiers:

    # Fit clf to the training set
    clf.fit(X_train, y_train)

    # Predict y_pred
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))

```




```

Logistic Regression : 0.747
K Nearest Neighbours : 0.724
Classification Tree : 0.730

```


####
**Better performance with a Voting Classifier**



 Finally, you’ll evaluate the performance of a voting classifier that takes the outputs of the models defined in the list
 `classifiers`
 and assigns labels by majority voting.





```

classifiers

[('Logistic Regression',
  LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
            intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
            penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
            verbose=0, warm_start=False)),
 ('K Nearest Neighbours',
  KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
             metric_params=None, n_jobs=1, n_neighbors=27, p=2,
             weights='uniform')),
 ('Classification Tree',
  DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=0.13, min_samples_split=2,
              min_weight_fraction_leaf=0.0, presort=False, random_state=1,
              splitter='best'))]

```




```python

# Import VotingClassifier from sklearn.ensemble
from sklearn.ensemble import VotingClassifier

# Instantiate a VotingClassifier vc
vc = VotingClassifier(estimators=classifiers)

# Fit vc to the training set
vc.fit(X_train, y_train)

# Evaluate the test set predictions
y_pred = vc.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Voting Classifier: {:.3f}'.format(accuracy))

# Voting Classifier: 0.753

```



 Notice how the voting classifier achieves a test set accuracy of 75.3%. This value is greater than that achieved by
 `LogisticRegression`
 .





---



# **3. Bagging and Random Forests**
----------------------------------


## **3.1 Bagging(bootstrap)**



 boostrap = sample with replacement



####
**Define the bagging classifier**



 In the following exercises you’ll work with the
 [Indian Liver Patient](https://www.kaggle.com/uciml/indian-liver-patient-records)
 dataset from the UCI machine learning repository. Your task is to predict whether a patient suffers from a liver disease using 10 features including Albumin, age and gender. You’ll do so using a Bagging Classifier.





```python

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier

# Instantiate dt
dt = DecisionTreeClassifier(random_state=1)

# Instantiate bc
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=1)

```


####
**Evaluate Bagging performance**



 Now that you instantiated the bagging classifier, it’s time to train it and evaluate its test set accuracy.





```python

# Fit bc to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate acc_test
acc_test = accuracy_score(y_test, y_pred)
print('Test set accuracy of bc: {:.2f}'.format(acc_test))

# Test set accuracy of bc: 0.71

```



 A single tree
 `dt`
 would have achieved an accuracy of 63% which is 8% lower than
 `bc`
 ‘s accuracy!





---


## **3.2 Out of Bag(OOB) Evaluation**



![Desktop View]({{ site.baseurl }}/assets/datacamp/machine-learning-with-tree-based-models-in-python/capture-2.png?w=1024)


 What is OOB evaluation? = similar to Cross Validation


 How OOB evaluation works = use OOB samples to evaluate the model


 OOB sample = training data which are NOT selected by boostrap



####
**Prepare the ground**



 In the following exercises, you’ll compare the OOB accuracy to the test set accuracy of a bagging classifier trained on the Indian Liver Patient dataset.





```python

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier

# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf=8, random_state=1)

# Instantiate bc
bc = BaggingClassifier(base_estimator=dt,
            n_estimators=50,
            oob_score=True,
            random_state=1)

```


####
**OOB Score vs Test Set Score**



 Now that you instantiated
 `bc`
 , you will fit it to the training set and evaluate its test set and OOB accuracies.





```python

# Fit bc to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate test set accuracy
acc_test = accuracy_score(y_test, y_pred)

# Evaluate OOB accuracy
acc_oob = bc.oob_score_

# Print acc_test and acc_oob
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))

# Test set accuracy: 0.698, OOB accuracy: 0.704

```




---


## **3.3 Random Forests (RF)**


####
**Train an RF regressor**



 In the following exercises you’ll predict bike rental demand in the Capital Bikeshare program in Washington, D.C using historical weather data from the
 [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand)
 dataset available through Kaggle.




 As a first step, you’ll define a random forests regressor and fit it to the training set.





```

X_train.head(3)
      hr  holiday  workingday  temp   hum  windspeed  instant  mnth  yr  \
1236  12        0           1  0.72  0.45     0.0000    14240     8   1
1349   5        0           0  0.64  0.89     0.1940    14353     8   1
327   15        0           0  0.80  0.55     0.1642    13331     7   1

      Clear to partly cloudy  Light Precipitation  Misty
1236                       1                    0      0
1349                       1                    0      0
327                        1                    0      0


y_train.head(3)
1236    305
1349     16
327     560
Name: cnt, dtype: int64

```




```python

# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# Instantiate rf
rf = RandomForestRegressor(n_estimators=25,
            random_state=2)

# Fit rf to the training set
rf.fit(X_train, y_train)

```


####
**Evaluate the RF regressor**




```python

# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Predict the test set labels
y_pred = rf.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred) ** 0.5

# Print rmse_test
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

# Test set RMSE of rf: 51.97

```



 The test set RMSE achieved by
 `rf`
 is significantly smaller than that achieved by a single CART!



####
**Visualizing features importances**



 In this exercise, you’ll determine which features were the most predictive according to the random forests regressor
 `rf`
 that you trained in a previous exercise.





```python

# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_,
                        index= X_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/machine-learning-with-tree-based-models-in-python/capture2-2.png?w=1024)


 Apparently,
 `hr`
 and
 `workingday`
 are the most important features according to
 `rf`
 . The importances of these two features add up to more than 90%!





---



# **4. Boosting**
----------------


## **4.1 Adaboost(Adaptive Boosting)**



![Desktop View]({{ site.baseurl }}/assets/datacamp/machine-learning-with-tree-based-models-in-python/capture3-3.png?w=1024)


![Desktop View]({{ site.baseurl }}/assets/datacamp/machine-learning-with-tree-based-models-in-python/capture4-3.png?w=1024)

####
**Define the AdaBoost classifier**



 In the following exercises you’ll revisit the
 [Indian Liver Patient](https://www.kaggle.com/uciml/indian-liver-patient-records)
 dataset which was introduced in a previous chapter.




 Your task is to predict whether a patient suffers from a liver disease using 10 features including Albumin, age and gender. However, this time, you’ll be training an AdaBoost ensemble to perform the classification task.




 In addition, given that this dataset is imbalanced, you’ll be using the ROC AUC score as a metric instead of accuracy.




 As a first step, you’ll start by instantiating an AdaBoost classifier.





```

X_train.head(1)
     Age  Total_Bilirubin  Direct_Bilirubin  Alkaline_Phosphotase  \
150   56              1.1               0.5                   180

     Alamine_Aminotransferase  Aspartate_Aminotransferase  Total_Protiens  \
150                        30                          42             6.9

     Albumin  Albumin_and_Globulin_Ratio  Is_male
150      3.8                         1.2        1


y_train.head(3)
150    0
377    0
473    0
Name: Liver_disease, dtype: int64

```




```python

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)

# Instantiate ada
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1)

```


####
**Train the AdaBoost classifier**



 Now that you’ve instantiated the AdaBoost classifier
 `ada`
 , it’s time train it. You will also predict the probabilities of obtaining the positive class in the test set.





```python

# Fit ada to the training set
ada.fit(X_train, y_train)

# Compute the probabilities of obtaining the positive class
y_pred_proba = ada.predict_proba(X_test)[:,1]

```




```

ada.predict_proba(X_test)
array([[ 0.57664817,  0.42335183],
       [ 0.48575393,  0.51424607],
       [ 0.34361394,  0.65638606],
       [ 0.50742464,  0.49257536],

```


####
**Evaluate the AdaBoost classifier**




```python

# Import roc_auc_score
from sklearn.metrics import roc_auc_score

# Evaluate test-set roc_auc_score
ada_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print roc_auc_score
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))

# ROC AUC score: 0.71

```



 Not bad! This untuned AdaBoost classifier achieved a ROC AUC score of 0.71!





---


## **4.2 Gradient Boosting (GB)**



![Desktop View]({{ site.baseurl }}/assets/datacamp/machine-learning-with-tree-based-models-in-python/capture5-3.png?w=1024)


![Desktop View]({{ site.baseurl }}/assets/datacamp/machine-learning-with-tree-based-models-in-python/capture6-3.png?w=1024)


![Desktop View]({{ site.baseurl }}/assets/datacamp/machine-learning-with-tree-based-models-in-python/capture7-1.png?w=1024)

###
**Define the GB regressor**



 You’ll now revisit the
 [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand)
 dataset that was introduced in the previous chapter.




 Recall that your task is to predict the bike rental demand using historical weather data from the Capital Bikeshare program in Washington, D.C.. For this purpose, you’ll be using a gradient boosting regressor.




 As a first step, you’ll start by instantiating a gradient boosting regressor which you will train in the next exercise.





```

 X_train.head()
      hr  holiday  workingday  temp   hum  windspeed  instant  mnth  yr  \
1236  12        0           1  0.72  0.45     0.0000    14240     8   1
1349   5        0           0  0.64  0.89     0.1940    14353     8   1
327   15        0           0  0.80  0.55     0.1642    13331     7   1
104    8        0           1  0.80  0.49     0.1343    13108     7   1
850   10        0           0  0.80  0.59     0.4179    13854     8   1

      Clear to partly cloudy  Light Precipitation  Misty
1236                       1                    0      0
1349                       1                    0      0
327                        1                    0      0
104                        1                    0      0
850                        1                    0      0


y_train.head()
1236    305
1349     16
327     560
104     550
850     364
Name: cnt, dtype: int64

```




```python

# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Instantiate gb
gb = GradientBoostingRegressor(max_depth=4,
            n_estimators=200,
            random_state=2)

```


####
**Train the GB regressor**




```python

# Fit gb to the training set
gb.fit(X_train, y_train)

# Predict test set labels
y_pred = gb.predict(X_test)

```


####
**Evaluate the GB regressor**




```python

# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute MSE
mse_test = MSE(y_test, y_pred)

# Compute RMSE
rmse_test = mse_test ** 0.5

# Print RMSE
print('Test set RMSE of gb: {:.3f}'.format(rmse_test))

# Test set RMSE of gb: 52.065

```




---


## **4.3 Stochastic Gradient Boosting (SGB)**


####
**Regression with SGB**



 As in the exercises from the previous lesson, you’ll be working with the
 [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand)
 dataset. In the following set of exercises, you’ll solve this bike count regression problem using stochastic gradient boosting.





```python

# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Instantiate sgbr
sgbr = GradientBoostingRegressor(max_depth=4,
            subsample=0.9,
            max_features=0.75,
            n_estimators=200,
            random_state=2)

```


####
**Train the SGB regressor**




```python

# Fit sgbr to the training set
sgbr.fit(X_train, y_train)

# Predict test set labels
y_pred = sgbr.predict(X_test)

```


####
**Evaluate the SGB regressor**




```python

# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute test set MSE
mse_test = MSE(y_test, y_pred)

# Compute test set RMSE
rmse_test = mse_test ** 0.5

# Print rmse_test
print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))

# Test set RMSE of sgbr: 49.979

```



 The stochastic gradient boosting regressor achieves a lower test set RMSE than the gradient boosting regressor (which was
 `52.065`
 )!





---



# **5. Model Tuning**
--------------------


## **5.1 Tuning a CART’s Hyperprameters**



![Desktop View]({{ site.baseurl }}/assets/datacamp/machine-learning-with-tree-based-models-in-python/capture10-2.png?w=1024)

####
**show hyperparameters**




```

print(dt.get_params)
<bound method BaseEstimator.get_params of DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=1,
            splitter='best')>

```


####
**Set the tree’s hyperparameter grid**




```python

# Define params_dt
params_dt = {
    'max_depth':[2,3,4],
    'min_samples_leaf':[0.12, 0.14, 0.16, 0.18]}

```


####
**Search for the optimal tree**



 In this exercise, you’ll perform grid search using 5-fold cross validation to find
 `dt`
 ‘s optimal hyperparameters. Note that because grid search is an exhaustive process, it may take a lot time to train the model.





```python

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_dt
grid_dt = GridSearchCV(estimator=dt,
                       param_grid=params_dt,
                       scoring='roc_auc',
                       cv=5,
                       n_jobs=-1)

```


####
**Evaluate the optimal tree**




```python

# Import roc_auc_score from sklearn.metrics
from sklearn.metrics import roc_auc_score

# Extract the best estimator
best_model = grid_dt.best_estimator_

# Predict the test set probabilities of the positive class
y_pred_proba = best_model.predict_proba(X_test)[:,1]

# Compute test_roc_auc
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print test_roc_auc
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))

#  Test set ROC AUC score: 0.610

```




```

best_model
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=0.12, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=1,
            splitter='best')

```



 An untuned classification-tree would achieve a ROC AUC score of
 `0.54`
 !





---


## **5.2 Tuning a RF’s Hyperparameters**


####
**Random forests hyperparameters**




```

rf.get_params
<bound method BaseEstimator.get_params of RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=-1,
           oob_score=False, random_state=2, verbose=0, warm_start=False)>


```


####
**Set the hyperparameter grid of RF**




```python

# Define the dictionary 'params_rf'
params_rf = {
    'n_estimators':[100, 350, 500],
    'max_features':['log2', 'auto', 'sqrt'],
    'min_samples_leaf':[2, 10, 30]}

```


####
**Search for the optimal forest**




```python

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       scoring='neg_mean_squared_error',
                       cv=3,
                       verbose=1,
                       n_jobs=-1)

```


####
**Evaluate the optimal forest**




```python

# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Extract the best estimator
best_model = grid_rf.best_estimator_

# Predict test set labels
y_pred = best_model.predict(X_test)

# Compute rmse_test
rmse_test = MSE(y_test, y_pred) ** 0.5

# Print rmse_test
print('Test RMSE of best model: {:.3f}'.format(rmse_test))

# Test RMSE of best model: 50.569

```



 The End.


 Thank you for reading.



