---
title: Extreme Gradient Boosting with XGBoost
date: 2021-12-07 11:22:08 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Extreme Gradient Boosting with XGBoost
=========================================







 This is the memo of the 5th course (23 courses in all) of ‘Machine Learning Scientist with Python’ skill track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/extreme-gradient-boosting-with-xgboost)**
 .



#####
 PREREQUISITES


* [Supervised Learning with scikit-learn](https://www.datacamp.com/courses/supervised-learning-with-scikit-learn)
* [Machine Learning with the Experts: School Budgets](https://www.datacamp.com/courses/machine-learning-with-the-experts-school-budgets)


###
**Course Description**



 Do you know the basics of supervised learning and want to use state-of-the-art models on real-world datasets? Gradient boosting is currently one of the most popular techniques for efficient modeling of tabular datasets of all sizes. XGboost is a very fast, scalable implementation of gradient boosting, with models using XGBoost regularly winning online data science competitions and being used at scale across different industries. In this course, you’ll learn how to use this powerful library alongside pandas and scikit-learn to build and tune supervised learning models. You’ll work with real-world datasets to solve classification and regression problems.



###
**Table of contents**


1. Classification with XGBoost
2. [Regression with XGBoost](https://datascience103579984.wordpress.com/2019/10/24/extreme-gradient-boosting-with-xgboost-from-datacamp/2/)
3. [Fine-tuning your XGBoost model](https://datascience103579984.wordpress.com/2019/10/24/extreme-gradient-boosting-with-xgboost-from-datacamp/3/)
4. [Using XGBoost in pipelines](https://datascience103579984.wordpress.com/2019/10/24/extreme-gradient-boosting-with-xgboost-from-datacamp/4/)






---



# **1. Classification with XGBoost**
-----------------------------------



 This chapter will introduce you to the fundamental idea behind XGBoost—boosted learners. Once you understand how XGBoost works, you’ll apply it to solve a common classification problem found in industry: predicting whether a customer will stop being a customer at some point in the future.



###
 1.1 Reminder of supervised learning



![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/9.png?w=1024)

### **1.1.1 Which of these is a classification problem?**



 Given below are 4 potential machine learning problems you might encounter in the wild. Pick the one that is a classification problem.



* Given past performance of stocks and various other financial data, predicting the exact price of a given stock (Google) tomorrow.
* Given a large dataset of user behaviors on a website, generating an informative segmentation of the users based on their behaviors.
* **Predicting whether a given user will click on an ad given the ad content and metadata associated with the user.**
* Given a user’s past behavior on a video platform, presenting him/her with a series of recommended videos to watch next.


### **1.1.2 Which of these is a binary classification problem?**



 A classification problem involves predicting the category a given data point belongs to out of a finite set of possible categories. Depending on how many possible categories there are to predict, a classification problem can be either binary or multi-class. Let’s do another quick refresher here. Your job is to pick the
 **binary**
 classification problem out of the following list of supervised learning problems.



* **Predicting whether a given image contains a cat.**
* Predicting the emotional valence of a sentence (Valence can be positive, negative, or neutral).
* Recommending the most tax-efficient strategy for tax filing in an automated accounting system.
* Given a list of symptoms, generating a rank-ordered list of most likely diseases.




---


## **1.2 Introducing XGBoost**



![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/10.png?w=980)

### **1.2.1 XGBoost: Fit/Predict**



 It’s time to create your first XGBoost model! As Sergey showed you in the video, you can use the scikit-learn
 `.fit()`
 /
 `.predict()`
 paradigm that you are already familiar to build your XGBoost models, as the
 `xgboost`
 library has a scikit-learn compatible API!




 Here, you’ll be working with churn data. This dataset contains imaginary data from a ride-sharing app with user behaviors over their first month of app usage in a set of imaginary cities as well as whether they used the service 5 months after sign-up. It has been pre-loaded for you into a DataFrame called
 `churn_data`
 – explore it in the Shell!




 Your goal is to use the first month’s worth of data to predict whether the app’s users will remain users of the service at the 5 month mark. This is a typical setup for a churn prediction problem. To do this, you’ll split the data into training and test sets, fit a small
 `xgboost`
 model on the training set, and evaluate its performance on the test set by computing its accuracy.




`pandas`
 and
 `numpy`
 have been imported as
 `pd`
 and
 `np`
 , and
 `train_test_split`
 has been imported from
 `sklearn.model_selection`
 . Additionally, the arrays for the features and the target have been created as
 `X`
 and
 `y`
 .





```

churn_data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 50000 entries, 0 to 49999
Data columns (total 13 columns):
avg_dist                       50000 non-null float64
avg_rating_by_driver           49799 non-null float64
avg_rating_of_driver           41878 non-null float64
avg_inc_price                  50000 non-null float64
inc_pct                        50000 non-null float64
weekday_pct                    50000 non-null float64
fancy_car_user                 50000 non-null bool
city_Carthag                   50000 non-null int64
city_Harko                     50000 non-null int64
phone_iPhone                   50000 non-null int64
first_month_cat_more_1_trip    50000 non-null int64
first_month_cat_no_trips       50000 non-null int64
month_5_still_here             50000 non-null int64
dtypes: bool(1), float64(6), int64(6)
memory usage: 4.6 MB

```




```

churn_data.head(2)
   avg_dist  avg_rating_by_driver  avg_rating_of_driver  avg_inc_price  inc_pct  ...  city_Harko  phone_iPhone  first_month_cat_more_1_trip  first_month_cat_no_trips  month_5_still_here
0      3.67                   5.0                   4.7            1.1     15.4  ...           1             1                            1                         0                   1
1      8.26                   5.0                   5.0            1.0      0.0  ...           0             0                            0                         1                   0

[2 rows x 13 columns]

```




```python

# Import xgboost
import xgboost as xgb

# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]

# Create the training and test sets
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)

# Fit the classifier to the training set
xg_cl.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))

# accuracy: 0.743300

```



 Your model has an accuracy of around 74%. In Chapter 3, you’ll learn about ways to fine tune your XGBoost models. For now, let’s refresh our memories on how decision trees work.





---


## **1.3 What is a decision tree?**



![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/12.png?w=990)
![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/11.png?w=1024)



### **1.3.1 Decision trees**



 Your task in this exercise is to make a simple decision tree using scikit-learn’s
 `DecisionTreeClassifier`
 on the
 `breast cancer`
 dataset that comes pre-loaded with scikit-learn.




 This dataset contains numeric measurements of various dimensions of individual tumors (such as perimeter and texture) from breast biopsies and a single outcome value (the tumor is either malignant, or benign).




 We’ve preloaded the dataset of samples (measurements) into
 `X`
 and the target values per tumor into
 `y`
 . Now, you have to split the complete dataset into training and testing sets, and then train a
 `DecisionTreeClassifier`
 . You’ll specify a parameter called
 `max_depth`
 . Many other parameters can be modified within this model, and you can check all of them out
 [here](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
 .





```python

# Import the necessary modules
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the classifier: dt_clf_4
dt_clf_4 = DecisionTreeClassifier(max_depth=4)

# Fit the classifier to the training set
dt_clf_4.fit(X_train, y_train)

# Predict the labels of the test set: y_pred_4
y_pred_4 = dt_clf_4.predict(X_test)

# Compute the accuracy of the predictions: accuracy
accuracy = float(np.sum(y_pred_4==y_test))/y_test.shape[0]
print("accuracy:", accuracy)

# accuracy: 0.9649122807017544

```



 It’s now time to learn about what gives XGBoost its state-of-the-art performance: Boosting.





---


## **1.4 What is Boosting?**



![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/13.png?w=990)
![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/14.png?w=1024)




<https://xgboost.readthedocs.io/en/latest/tutorials/model.html>



### **1.4.1 Measuring accuracy**



 You’ll now practice using XGBoost’s learning API through its baked in cross-validation capabilities. As Sergey discussed in the previous video, XGBoost gets its lauded performance and efficiency gains by utilizing its own optimized data structure for datasets called a
 `DMatrix`
 .




 In the previous exercise, the input datasets were converted into
 `DMatrix`
 data on the fly, but when you use the
 `xgboost`
`cv`
 object, you have to first explicitly convert your data into a
 `DMatrix`
 . So, that’s what you will do here before running cross-validation on
 `churn_data`
 .





```python

# Create the DMatrix: churn_dmatrix
churn_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:logistic", "max_depth":3}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=3, num_boost_round=5, metrics="error", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Print the accuracy
print(((1-cv_results["test-error-mean"]).iloc[-1]))

```




```

       train-error-mean  train-error-std  test-error-mean  test-error-std
    0           0.28232         0.002366          0.28378        0.001932
    1           0.26951         0.001855          0.27190        0.001932
    2           0.25605         0.003213          0.25798        0.003963
    3           0.25090         0.001845          0.25434        0.003827
    4           0.24654         0.001981          0.24852        0.000934
    0.75148

```



`cv_results`
 stores the training and test mean and standard deviation of the error per boosting round (tree built) as a DataFrame. From
 `cv_results`
 , the final round
 `'test-error-mean'`
 is extracted and converted into an accuracy, where accuracy is
 `1-error`
 . The final accuracy of around 75% is an improvement from earlier!



### **1.4.2 Measuring AUC**



 Now that you’ve used cross-validation to compute average out-of-sample accuracy (after converting from an error), it’s very easy to compute any other metric you might be interested in. All you have to do is pass it (or a list of metrics) in as an argument to the
 `metrics`
 parameter of
 `xgb.cv()`
 .




 Your job in this exercise is to compute another common metric used in binary classification – the area under the curve (
 `"auc"`
 ). As before,
 `churn_data`
 is available in your workspace, along with the DMatrix
 `churn_dmatrix`
 and parameter dictionary
 `params`
 .





```python

# Perform cross_validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=3, num_boost_round=5, metrics="auc", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Print the AUC
print((cv_results["test-auc-mean"]).iloc[-1])

```




```

       train-auc-mean  train-auc-std  test-auc-mean  test-auc-std
    0        0.768893       0.001544       0.767863      0.002820
    1        0.790864       0.006758       0.789157      0.006846
    2        0.815872       0.003900       0.814476      0.005997
    3        0.822959       0.002018       0.821682      0.003912
    4        0.827528       0.000769       0.826191      0.001937
    0.826191

```



 An AUC of 0.84 is quite strong. As you have seen, XGBoost’s learning API makes it very easy to compute any metric you may be interested in. In Chapter 3, you’ll learn about techniques to fine-tune your XGBoost models to improve their performance even further. For now, it’s time to learn a little about exactly
 **when**
 to use XGBoost.





---


## **1.5 When should I use XGBoost?**



![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/15.png?w=927)
![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/16.png?w=972)



### **1.5.1 Using XGBoost**



 XGBoost is a powerful library that scales very well to many samples and works for a variety of supervised learning problems. That said, as Sergey described in the video, you shouldn’t always pick it as your default machine learning library when starting a new project, since there are some situations in which it is not the best option. In this exercise, your job is to consider the below examples and select the one which would be the best use of XGBoost.



* Visualizing the similarity between stocks by comparing the time series of their historical prices relative to each other.
* Predicting whether a person will develop cancer using genetic data with millions of genes, 23 examples of genomes of people that didn’t develop cancer, 3 genomes of people who wound up getting cancer.
* Clustering documents into topics based on the terms used in them.
* **Predicting the likelihood that a given user will click an ad from a very large clickstream log with millions of users and their web interactions.**



# **2. Regression with XGBoost**
-------------------------------



 After a brief review of supervised regression, you’ll apply XGBoost to the regression task of predicting house prices in Ames, Iowa. You’ll learn about the two kinds of base learners that XGboost can use as its weak learners, and review how to evaluate the quality of your regression models.



## **2.1 Regression review**


### **2.1.1 Which of these is a regression problem?**



 Here are 4 potential machine learning problems you might encounter in the wild. Pick the one that is a clear example of a regression problem.



* Recommending a restaurant to a user given their past history of restaurant visits and reviews for a dining aggregator app.
* Predicting which of several thousand diseases a given person is most likely to have given their symptoms.
* Tagging an email as spam/not spam based on its content and metadata (sender, time sent, etc.).
* **Predicting the expected payout of an auto insurance claim given claim properties (car, accident type, driver prior history, etc.).**




---


## **2.2 Objective (loss) functions and base learners**



![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/17.png?w=1024)

### **2.2.1 Decision trees as base learners**



 It’s now time to build an XGBoost model to predict house prices – not in Boston, Massachusetts, as you saw in the video, but in Ames, Iowa! This dataset of housing prices has been pre-loaded into a DataFrame called
 `df`
 . If you explore it in the Shell, you’ll see that there are a variety of features about the house and its location in the city.




 In this exercise, your goal is to use trees as base learners. By default, XGBoost uses trees as base learners, so you don’t have to specify that you want to use trees here with
 `booster="gbtree"`
 .




`xgboost`
 has been imported as
 `xgb`
 and the arrays for the features and the target are available in
 `X`
 and
 `y`
 , respectively.





```

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 57 columns):
MSSubClass              1460 non-null int64
LotFrontage             1460 non-null float64
LotArea                 1460 non-null int64
OverallQual             1460 non-null int64
OverallCond             1460 non-null int64
YearBuilt               1460 non-null int64
Remodeled               1460 non-null int64
GrLivArea               1460 non-null int64
BsmtFullBath            1460 non-null int64
BsmtHalfBath            1460 non-null int64
FullBath                1460 non-null int64
HalfBath                1460 non-null int64
BedroomAbvGr            1460 non-null int64
Fireplaces              1460 non-null int64
GarageArea              1460 non-null int64
MSZoning_FV             1460 non-null int64
MSZoning_RH             1460 non-null int64
MSZoning_RL             1460 non-null int64
MSZoning_RM             1460 non-null int64
Neighborhood_Blueste    1460 non-null int64
Neighborhood_BrDale     1460 non-null int64
Neighborhood_BrkSide    1460 non-null int64
Neighborhood_ClearCr    1460 non-null int64
Neighborhood_CollgCr    1460 non-null int64
Neighborhood_Crawfor    1460 non-null int64
Neighborhood_Edwards    1460 non-null int64
Neighborhood_Gilbert    1460 non-null int64
Neighborhood_IDOTRR     1460 non-null int64
Neighborhood_MeadowV    1460 non-null int64
Neighborhood_Mitchel    1460 non-null int64
Neighborhood_NAmes      1460 non-null int64
Neighborhood_NPkVill    1460 non-null int64
Neighborhood_NWAmes     1460 non-null int64
Neighborhood_NoRidge    1460 non-null int64
Neighborhood_NridgHt    1460 non-null int64
Neighborhood_OldTown    1460 non-null int64
Neighborhood_SWISU      1460 non-null int64
Neighborhood_Sawyer     1460 non-null int64
Neighborhood_SawyerW    1460 non-null int64
Neighborhood_Somerst    1460 non-null int64
Neighborhood_StoneBr    1460 non-null int64
Neighborhood_Timber     1460 non-null int64
Neighborhood_Veenker    1460 non-null int64
BldgType_2fmCon         1460 non-null int64
BldgType_Duplex         1460 non-null int64
BldgType_Twnhs          1460 non-null int64
BldgType_TwnhsE         1460 non-null int64
HouseStyle_1.5Unf       1460 non-null int64
HouseStyle_1Story       1460 non-null int64
HouseStyle_2.5Fin       1460 non-null int64
HouseStyle_2.5Unf       1460 non-null int64
HouseStyle_2Story       1460 non-null int64
HouseStyle_SFoyer       1460 non-null int64
HouseStyle_SLvl         1460 non-null int64
PavedDrive_P            1460 non-null int64
PavedDrive_Y            1460 non-null int64
SalePrice               1460 non-null int64
dtypes: float64(1), int64(56)
memory usage: 650.2 KB


```




```python

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the XGBRegressor: xg_reg
xg_reg = xgb.XGBRegressor(objective="reg:linear", n_estimators=10, seed=123)

# Fit the regressor to the training set
xg_reg.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_reg.predict(X_test)

# Compute the rmse: rmse
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

# RMSE: 78847.401758

```



 Next, you’ll train an XGBoost model using linear base learners and XGBoost’s learning API. Will it perform better or worse?



### **2.2.2 Linear base learners**



 Now that you’ve used trees as base models in XGBoost, let’s use the other kind of base model that can be used with XGBoost – a linear learner. This model, although not as commonly used in XGBoost, allows you to create a regularized linear regression using XGBoost’s powerful learning API. However, because it’s uncommon, you have to use XGBoost’s own non-scikit-learn compatible functions to build the model, such as
 `xgb.train()`
 .




 In order to do this you must create the parameter dictionary that describes the kind of booster you want to use (similarly to how
 [you created the dictionary in Chapter 1](https://campus.datacamp.com/courses/extreme-gradient-boosting-with-xgboost/10555?ex=9)
 when you used
 `xgb.cv()`
 ). The key-value pair that defines the booster type (base model) you need is
 `"booster":"gblinear"`
 .




 Once you’ve created the model, you can use the
 `.train()`
 and
 `.predict()`
 methods of the model just like you’ve done in the past.




 Here, the data has already been split into training and testing sets, so you can dive right into creating the
 `DMatrix`
 objects required by the XGBoost learning API.





```python

# Convert the training and testing sets into DMatrixes: DM_train, DM_test
DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test =  xgb.DMatrix(data=X_test, label=y_test)

# Create the parameter dictionary: params
params = {"booster":"gblinear", "objective":"reg:linear"}

# Train the model: xg_reg
xg_reg = xgb.train(params = params, dtrain=DM_train, num_boost_round=5)

# Predict the labels of the test set: preds
preds = xg_reg.predict(DM_test)

# Compute and print the RMSE
rmse = np.sqrt(mean_squared_error(y_test,preds))
print("RMSE: %f" % (rmse))

# RMSE: 44159.721661

```



 It looks like linear base learners performed better!



### **2.2.3 Evaluating model quality**



 It’s now time to begin evaluating model quality.




 Here, you will compare the RMSE and MAE of a cross-validated XGBoost model on the Ames housing data. As in previous exercises, all necessary modules have been pre-loaded and the data is available in the DataFrame
 `df`
 .





```python

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":4}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4, num_boost_round=5, metrics='rmse', as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print((cv_results["test-rmse-mean"]).tail(1))

```




```

       train-rmse-mean  train-rmse-std  test-rmse-mean  test-rmse-std
    0    141767.535156      429.449158   142980.433594    1193.789595
    1    102832.542969      322.468977   104891.392578    1223.157953
    2     75872.617188      266.473250    79478.937500    1601.344539
    3     57245.651368      273.626997    62411.924804    2220.148314
    4     44401.295899      316.422824    51348.281250    2963.379118

    4    51348.28125
    Name: test-rmse-mean, dtype: float64

```




```python

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":4}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4, num_boost_round=5, metrics='mae', as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print((cv_results["test-mae-mean"]).tail(1))

```




```

       train-mae-mean  train-mae-std  test-mae-mean  test-mae-std
    0   127343.570313     668.341212  127633.986328   2403.992416
    1    89770.060547     456.948723   90122.496093   2107.910017
    2    63580.789063     263.407042   64278.561524   1887.563581
    3    45633.140625     151.885298   46819.169922   1459.812547
    4    33587.090821      87.001007   35670.651367   1140.608182

    4    35670.651367
    Name: test-mae-mean, dtype: float64

```




---


## **2.3 Regularization and base learners in XGBoost**



![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/1-5.png?w=936)
![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/2-5.png?w=983)
![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/3-5.png?w=1024)



### **2.3.1 Using regularization in XGBoost**



 Having seen an example of l1 regularization in the video, you’ll now vary the l2 regularization penalty – also known as
 `"lambda"`
 – and see its effect on overall model performance on the Ames housing dataset.





```python

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

reg_params = [1, 10, 100]

# Create the initial parameter dictionary for varying l2 strength: params
params = {"objective":"reg:squarederror","max_depth":3}

# Create an empty list for storing rmses as a function of l2 complexity
rmses_l2 = []

# Iterate over reg_params
for reg in reg_params:

    # Update l2 strength
    params["lambda"] = reg

    # Pass this updated param dictionary into cv
    cv_results_rmse = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2, num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)

    # Append best rmse (final round) to rmses_l2
    rmses_l2.append(cv_results_rmse["test-rmse-mean"].tail(1).values[0])

# Look at best rmse per l2 param
print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2", "rmse"]))

```




```

    Best rmse as a function of l2:
        l2          rmse
    0    1  52275.357421
    1   10  57746.064453
    2  100  76624.628907

```



 It looks like as as the value of
 `'lambda'`
 increases, so does the RMSE.



### **2.3.2 Visualizing individual XGBoost trees**



 Now that you’ve used XGBoost to both build and evaluate regression as well as classification models, you should get a handle on how to visually explore your models. Here, you will visualize individual trees from the fully boosted model that XGBoost creates using the entire housing dataset.




 XGBoost has a
 `plot_tree()`
 function that makes this type of visualization easy. Once you train a model using the XGBoost learning API, you can pass it to the
 `plot_tree()`
 function along with the number of trees you want to plot using the
 `num_trees`
 argument.





```python

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":2}

# Train the model: xg_reg
xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)

# Plot the first tree
xgb.plot_tree(xg_reg, num_trees=0)
plt.show()

# Plot the fifth tree
xgb.plot_tree(xg_reg, num_trees=4)
plt.show()

# Plot the last tree sideways
xgb.plot_tree(xg_reg, num_trees=9, rankdir="LR")
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/4-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/5-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/6-4.png?w=1024)




 Have a look at each of the plots. They provide insight into how the model arrived at its final decisions and what splits it made to arrive at those decisions. This allows us to identify which features are the most important in determining house price. In the next exercise, you’ll learn another way of visualizing feature importances.



### **2.3.3 Visualizing feature importances: What features are most important in my dataset**



 Another way to visualize your XGBoost models is to examine the importance of each feature column in the original dataset within the model.




 One simple way of doing this involves counting the number of times each feature is split on across all boosting rounds (trees) in the model, and then visualizing the result as a bar graph, with the features ordered according to how many times they appear. XGBoost has a
 `plot_importance()`
 function that allows you to do exactly this, and you’ll get a chance to use it in this exercise!





```python

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:squarederror", "max_depth":4}

# Train the model: xg_reg
xg_reg = xgb.train(dtrain=housing_dmatrix, params=params, num_boost_round=10)

# Plot the feature importances
xgb.plot_importance(xg_reg)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/7-3.png?w=1024)


 It looks like
 `GrLivArea`
 is the most important feature.




# **3. Fine-tuning your XGBoost model**
--------------------------------------



 This chapter will teach you how to make your XGBoost models as performant as possible. You’ll learn about the variety of parameters that can be adjusted to alter the behavior of XGBoost and how to tune them efficiently so that you can supercharge the performance of your models.



## **3.1 Why tune your model?**


### **3.1.1 When is tuning your model a bad idea?**



 Now that you’ve seen the effect that tuning has on the overall performance of your XGBoost model, let’s turn the question on its head and see if you can figure out when tuning your model might not be the best idea.
 **Given that model tuning can be time-intensive and complicated, which of the following scenarios would NOT call for careful tuning of your model**
 ?



* You have lots of examples from some dataset and very many features at your disposal.
* **You are very short on time before you must push an initial model to production and have little data to train your model on.**
* You have access to a multi-core (64 cores) server with lots of memory (200GB RAM) and no time constraints.
* You must squeeze out every last bit of performance out of your xgboost model.


### **3.1.2 Tuning the number of boosting rounds**



 Let’s start with parameter tuning by seeing how the number of boosting rounds (number of trees you build) impacts the out-of-sample performance of your XGBoost model. You’ll use
 `xgb.cv()`
 inside a
 `for`
 loop and build one model per
 `num_boost_round`
 parameter.




 Here, you’ll continue working with the Ames housing dataset. The features are available in the array
 `X`
 , and the target vector is contained in
 `y`
 .





```python

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree: params
params = {"objective":"reg:squarederror", "max_depth":3}

# Create list of number of boosting rounds
num_rounds = [5, 10, 15]

# Empty list to store final round rmse per XGBoost model
final_rmse_per_round = []

# Iterate over num_rounds and build one model per num_boost_round parameter
for curr_num_rounds in num_rounds:

    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=curr_num_rounds, metrics="rmse", as_pandas=True, seed=123)

    # Append final round RMSE
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses,columns=["num_boosting_rounds","rmse"]))

```




```

       num_boosting_rounds          rmse
    0                    5  50903.300781
    1                   10  34774.194011
    2                   15  32895.097005

```



 As you can see, increasing the number of boosting rounds decreases the RMSE.



### **3.1.3 Automated boosting round selection using early_stopping**



 Now, instead of attempting to cherry pick the best possible number of boosting rounds, you can very easily have XGBoost automatically select the number of boosting rounds for you within
 `xgb.cv()`
 . This is done using a technique called
 **early stopping**
 .




**Early stopping**
 works by testing the XGBoost model after every boosting round against a hold-out dataset and stopping the creation of additional boosting rounds (thereby finishing training of the model early) if the hold-out metric (
 `"rmse"`
 in our case) does not improve for a given number of rounds. Here you will use the
 `early_stopping_rounds`
 parameter in
 `xgb.cv()`
 with a large possible number of boosting rounds (50). Bear in mind that if the holdout metric continuously improves up through when
 `num_boosting_rounds`
 is reached, then early stopping does not occur.




 Here, the
 `DMatrix`
 and parameter dictionary have been created for you. Your task is to use cross-validation with early stopping. Go for it!





```python

# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree: params
params = {"objective":"reg:squarederror", "max_depth":4}

# Perform cross-validation with early stopping: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, num_boost_round=50, early_stopping_rounds=10, metrics='rmse', as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

```




```

        train-rmse-mean  train-rmse-std  test-rmse-mean  test-rmse-std
    0     141871.630208      403.632409   142640.651042     705.571916
    1     103057.031250       73.772931   104907.666667     111.114933
    2      75975.963541      253.734987    79262.059895     563.766991
    3      57420.529948      521.653556    61620.135417    1087.690754
    4      44552.955729      544.169200    50437.561198    1846.448222
...
    45     11356.552734      565.368794    30758.543620    1947.456345
    46     11193.556966      552.298481    30729.972005    1985.699316
    47     11071.315430      604.089695    30732.664062    1966.998275
    48     10950.778646      574.862348    30712.240885    1957.751118
    49     10824.865560      576.666458    30720.854818    1950.511520

```




---


## **3.2 Overview of XGBoost’s hyperparameters**



![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/1.png?w=996)
![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/2.png?w=929)




**Linear based models are rarely used!**



### **3.2.1 Tuning eta**



 It’s time to practice tuning other XGBoost hyperparameters in earnest and observing their effect on model performance! You’ll begin by tuning the
 `"eta"`
 , also known as the learning rate.




 The learning rate in XGBoost is a parameter that can range between
 `0`
 and
 `1`
 , with higher values of
 `"eta"`
 penalizing feature weights more strongly, causing much stronger regularization.





```python

# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree (boosting round)
params = {"objective":"reg:squarederror", "max_depth":3}

# Create list of eta values and empty list to store final round rmse per xgboost model
eta_vals = [0.001, 0.01, 0.1]
best_rmse = []

# Systematically vary the eta
for curr_val in eta_vals:

    params["eta"] = curr_val

    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)

    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(eta_vals, best_rmse)), columns=["eta","best_rmse"]))

'''
         eta      best_rmse
    0  0.001  196653.989583
    1  0.010  188532.578125
    2  0.100  122784.299479
'''

```


### **3.2.2 Tuning max_depth**



 In this exercise, your job is to tune
 `max_depth`
 , which is the parameter that dictates the maximum depth that each tree in a boosting round can grow to. Smaller values will lead to shallower trees, and larger values to deeper trees.





```python

# Create your housing DMatrix
housing_dmatrix = xgb.DMatrix(data=X,label=y)

# Create the parameter dictionary
params = {"objective":"reg:squarederror"}

# Create list of max_depth values
max_depths = [2,5,10,20]
best_rmse = []

# Systematically vary the max_depth
for curr_val in max_depths:

    params["max_depths"] = curr_val

    # Perform cross-validation
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2, early_stopping_rounds=5, num_boost_round=10, metrics='rmse', seed=123, as_pandas=True)


    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(max_depths, best_rmse)),columns=["max_depth","best_rmse"]))

'''
       max_depth     best_rmse
    0          2  35922.521485
    1          5  35922.521485
    2         10  35922.521485
    3         20  35922.521485
'''

```


### **3.2.3 Tuning colsample_bytree**



 Now, it’s time to tune
 `"colsample_bytree"`
 . You’ve already seen this if you’ve ever worked with scikit-learn’s
 `RandomForestClassifier`
 or
 `RandomForestRegressor`
 , where it just was called
 `max_features`
 . In both
 `xgboost`
 and
 `sklearn`
 , this parameter (although named differently) simply specifies the fraction of features to choose from at every split in a given tree. In
 `xgboost`
 ,
 `colsample_bytree`
 must be specified as a float between 0 and 1.





```python

# Create your housing DMatrix
housing_dmatrix = xgb.DMatrix(data=X,label=y)

# Create the parameter dictionary
params={"objective":"reg:squarederror","max_depth":3}

# Create list of hyperparameter values: colsample_bytree_vals
colsample_bytree_vals = [0.1,0.5,0.8,1]
best_rmse = []

# Systematically vary the hyperparameter value
for curr_val in colsample_bytree_vals:

    params['colsample_bytree'] = curr_val

    # Perform cross-validation
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2,
                 num_boost_round=10, early_stopping_rounds=5,
                 metrics="rmse", as_pandas=True, seed=123)

    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(colsample_bytree_vals, best_rmse)), columns=["colsample_bytree","best_rmse"]))

'''
       colsample_bytree     best_rmse
    0               0.1  48193.453125
    1               0.5  36013.542968
    2               0.8  35932.962891
    3               1.0  35836.042969
'''

```



 There are several other individual parameters that you can tune, such as
 `"subsample"`
 , which dictates the fraction of the training data that is used during any given boosting round. Next up: Grid Search and Random Search to tune XGBoost hyperparameters more efficiently!





---


## **3.3 Review of grid search and random search**



![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/3.png?w=994)
![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/4.png?w=973)



### **3.3.1 Grid search with XGBoost**



 Now that you’ve learned how to tune parameters individually with XGBoost, let’s take your parameter tuning to the next level by using scikit-learn’s
 `GridSearch`
 and
 `RandomizedSearch`
 capabilities with internal cross-validation using the
 `GridSearchCV`
 and
 `RandomizedSearchCV`
 functions. You will use these to find the best model exhaustively from a collection of possible parameter values across multiple parameters simultaneously. Let’s get to work, starting with
 `GridSearchCV`
 !





```python

# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'n_estimators': [50],
    'max_depth': [2, 5]
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor()

# Perform grid search: grid_mse
grid_mse = GridSearchCV(param_grid=gbm_param_grid, estimator=gbm, scoring='neg_mean_squared_error', cv=4, verbose=1)


# Fit grid_mse to the data
grid_mse.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))


'''
    Best parameters found:  {'colsample_bytree': 0.7, 'max_depth': 5, 'n_estimators': 50}
    Lowest RMSE found:  29916.562522854438
'''

```


### **3.3.2 Random search with XGBoost**



 Often,
 `GridSearchCV`
 can be really time consuming, so in practice, you may want to use
 `RandomizedSearchCV`
 instead, as you will do in this exercise. The good news is you only have to make a few modifications to your
 `GridSearchCV`
 code to do
 `RandomizedSearchCV`
 . The key difference is you have to specify a
 `param_distributions`
 parameter instead of a
 `param_grid`
 parameter.





```python

# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
    'n_estimators': [25],
    'max_depth': range(2, 12)
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor(n_estimators=10)

# Perform random search: grid_mse
randomized_mse = RandomizedSearchCV(param_distributions=gbm_param_grid, estimator=gbm, scoring='neg_mean_squared_error', n_iter=5, cv=4, verbose=1)


# Fit randomized_mse to the data
randomized_mse.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))


'''
    Fitting 4 folds for each of 5 candidates, totalling 20 fits
    Best parameters found:  {'n_estimators': 25, 'max_depth': 6}
    Lowest RMSE found:  36909.98213965752
'''

```




---


## **3.4 Limits of grid search and random search**



 The search space size can be massive for Grid Search in certain cases, whereas for Random Search the number of hyperparameters has a significant effect on how long it takes to run.




# **4. Using XGBoost in pipelines**
----------------------------------



 Take your XGBoost skills to the next level by incorporating your models into two end-to-end machine learning pipelines. You’ll learn how to tune the most important XGBoost hyperparameters efficiently within a pipeline, and get an introduction to some more advanced preprocessing techniques.



## **4.1 Review of pipelines using sklearn**



![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/1-4.png?w=993)
![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/2-4.png?w=1007)
![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/3-4.png?w=942)



### **4.1.1 Exploratory data analysis**



 Before diving into the nitty gritty of pipelines and preprocessing, let’s do some exploratory analysis of the original, unprocessed
 [Ames housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
 . When you worked with this data in previous chapters, we preprocessed it for you so you could focus on the core XGBoost concepts. In this chapter, you’ll do the preprocessing yourself!




 A smaller version of this original, unprocessed dataset has been pre-loaded into a
 `pandas`
 DataFrame called
 `df`
 . Your task is to explore
 `df`
 in the Shell and pick the option that is
 **incorrect**
 . The larger purpose of this exercise is to understand the kinds of transformations you will need to perform in order to be able to use XGBoost.





```

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 21 columns):
MSSubClass      1460 non-null int64
LotFrontage     1201 non-null float64
LotArea         1460 non-null int64
OverallQual     1460 non-null int64
OverallCond     1460 non-null int64
YearBuilt       1460 non-null int64
Remodeled       1460 non-null int64
GrLivArea       1460 non-null int64
BsmtFullBath    1460 non-null int64
BsmtHalfBath    1460 non-null int64
FullBath        1460 non-null int64
HalfBath        1460 non-null int64
BedroomAbvGr    1460 non-null int64
Fireplaces      1460 non-null int64
GarageArea      1460 non-null int64
MSZoning        1460 non-null object
PavedDrive      1460 non-null object
Neighborhood    1460 non-null object
BldgType        1460 non-null object
HouseStyle      1460 non-null object
SalePrice       1460 non-null int64
dtypes: float64(1), int64(15), object(5)
memory usage: 239.6+ KB

```


### **4.1.2 Encoding categorical columns I: LabelEncoder**



 Now that you’ve seen what will need to be done to get the housing data ready for XGBoost, let’s go through the process step-by-step.




 First, you will need to fill in missing values – as you saw previously, the column
 `LotFrontage`
 has many missing values. Then, you will need to encode any categorical columns in the dataset using one-hot encoding so that they are encoded numerically. You can watch
 [this video](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/preprocessing-and-pipelines?ex=1)
 from
 [Supervised Learning with scikit-learn](https://www.datacamp.com/courses/supervised-learning-with-scikit-learn)
 for a refresher on the idea.




 The data has five categorical columns:
 `MSZoning`
 ,
 `PavedDrive`
 ,
 `Neighborhood`
 ,
 `BldgType`
 , and
 `HouseStyle`
 . Scikit-learn has a
 [LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
 function that converts the values in each categorical column into integers. You’ll practice using this here.





```python

# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Fill missing values with 0
df.LotFrontage = df.LotFrontage.fillna(0)

# Create a boolean mask for categorical columns
categorical_mask = (df.dtypes == object)

# Get list of categorical column names
categorical_columns = df.columns[categorical_mask].tolist()

# Print the head of the categorical columns
print(df[categorical_columns].head())

# Create LabelEncoder object: le
le = LabelEncoder()

# Apply LabelEncoder to categorical columns
df[categorical_columns] = df[categorical_columns].apply(lambda x: le.fit_transform(x))

# Print the head of the LabelEncoded categorical columns
print(df[categorical_columns].head())

```




```

  MSZoning PavedDrive Neighborhood BldgType HouseStyle
0       RL          Y      CollgCr     1Fam     2Story
1       RL          Y      Veenker     1Fam     1Story
2       RL          Y      CollgCr     1Fam     2Story
3       RL          Y      Crawfor     1Fam     2Story
4       RL          Y      NoRidge     1Fam     2Story

   MSZoning  PavedDrive  Neighborhood  BldgType  HouseStyle
0         3           2             5         0           5
1         3           2            24         0           2
2         3           2             5         0           5
3         3           2             6         0           5
4         3           2            15         0           5

```


### **4.1.3 Encoding categorical columns II: OneHotEncoder**



 Okay – so you have your categorical columns encoded numerically. Can you now move onto using pipelines and XGBoost? Not yet! In the categorical columns of this dataset, there is no natural ordering between the entries. As an example: Using
 `LabelEncoder`
 , the
 `CollgCr`
`Neighborhood`
 was encoded as
 `5`
 , while the
 `Veenker`
`Neighborhood`
 was encoded as
 `24`
 , and
 `Crawfor`
 as
 `6`
 . Is
 `Veenker`
 “greater” than
 `Crawfor`
 and
 `CollgCr`
 ? No – and allowing the model to assume this natural ordering may result in poor performance.




 As a result, there is another step needed: You have to apply a one-hot encoding to create binary, or “dummy” variables. You can do this using scikit-learn’s
 [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
 .





```python

# Import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

# Create OneHotEncoder: ohe
ohe = OneHotEncoder(categorical_features=categorical_mask, sparse=False)

# Apply OneHotEncoder to categorical columns - output is no longer a dataframe: df_encoded
df_encoded = ohe.fit_transform(df)

# Print the shape of the original DataFrame
print(df.shape)
# (1460, 21)

# Print the shape of the transformed array
print(df_encoded.shape)
# (1460, 62)

```


### **4.1.4 Encoding categorical columns III: DictVectorizer**



 Alright, one final trick before you dive into pipelines. The two step process you just went through –
 `LabelEncoder`
 followed by
 `OneHotEncoder`
 – can be simplified by using a
 [DictVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html)
 .




 Using a
 `DictVectorizer`
 on a DataFrame that has been converted to a dictionary allows you to get label encoding as well as one-hot encoding in one go.




 Your task is to work through this strategy in this exercise!





```python

# Import DictVectorizer
from sklearn.feature_extraction import DictVectorizer

# Convert df into a dictionary: df_dict
df_dict = df.to_dict(orient='records')

# Create the DictVectorizer object: dv
dv = DictVectorizer(sparse=False)

# Apply dv on df: df_encoded
df_encoded = dv.fit_transform(df_dict)

# Print the resulting first five rows
print(df_encoded[:5,:])

# Print the vocabulary
print(dv.vocabulary_)
'''
{'MSSubClass': 23, 'LotFrontage': 22, 'LotArea': 21, 'OverallQual': 55, 'OverallCond': 54, 'YearBuilt': 61, 'Remodeled': 59, 'GrLivArea': 11, 'BsmtFullBath': 6, 'BsmtHalfBath': 7,

...,

'Neighborhood=BrDale': 31, 'Neighborhood=SWISU': 47, 'MSZoning=RH': 26, 'Neighborhood=Blueste': 30}
'''

```




```

type(df_dict)
list

df_dict
[{'BedroomAbvGr': 3,
  'BldgType': '1Fam',
  'BsmtFullBath': 1,
  'BsmtHalfBath': 0,
  'Fireplaces': 0,
  'FullBath': 2,
  'GarageArea': 548,
  'GrLivArea': 1710,
  'HalfBath': 1,
  'HouseStyle': '2Story',
  'LotArea': 8450,
  'LotFrontage': 65.0,
  'MSSubClass': 60,
  'MSZoning': 'RL',
  'Neighborhood': 'CollgCr',
  'OverallCond': 5,
  'OverallQual': 7,
  'PavedDrive': 'Y',
  'Remodeled': 0,
  'SalePrice': 208500,
  'YearBuilt': 2003},
......
]

```



 Besides simplifying the process into one step,
 `DictVectorizer`
 has useful attributes such as
 `vocabulary_`
 which maps the names of the features to their indices.



### **4.1.5 Preprocessing within a pipeline**



 Now that you’ve seen what steps need to be taken individually to properly process the Ames housing data, let’s use the much cleaner and more succinct
 `DictVectorizer`
 approach and put it alongside an
 `XGBoostRegressor`
 inside of a scikit-learn pipeline.





```python

# Import necessary modules
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor(objective="reg:squarederror"))]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Fit the pipeline
xgb_pipeline.fit(X.to_dict('records'), y)

```




---


## **4.2 Incorporating XGBoost into pipelines**


### **4.2.1 Cross-validating your XGBoost model**



 In this exercise, you’ll go one step further by using the pipeline you’ve created to preprocess
 **and**
 cross-validate your model.





```python

# Import necessary modules
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor(max_depth=2, objective="reg:squarederror"))]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Cross-validate the model
cross_val_scores = cross_val_score(xgb_pipeline, X.to_dict(orient='records'), y, scoring='neg_mean_squared_error')

# Print the 10-fold RMSE
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))
# 10-fold RMSE:  31233.18564354353

```


### **4.2.2 Kidney disease case study I: Categorical Imputer**



 You’ll now continue your exploration of using pipelines with a dataset that requires significantly more wrangling. The
 [chronic kidney disease dataset](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease)
 contains both categorical and numeric features, but contains lots of missing values. The goal here is to predict who has chronic kidney disease given various blood indicators as features.




 As Sergey mentioned in the video, you’ll be introduced to a new library,
 [`sklearn_pandas`](https://github.com/pandas-dev/sklearn-pandas)
 , that allows you to chain many more processing steps inside of a pipeline than are currently supported in scikit-learn. Specifically, you’ll be able to impute missing categorical values directly using the
 `Categorical_Imputer()`
 class in
 `sklearn_pandas`
 , and the
 `DataFrameMapper()`
 class to apply any arbitrary sklearn-compatible transformer on DataFrame columns, where the resulting output can be either a NumPy array or DataFrame.




 We’ve also created a transformer called a
 `Dictifier`
 that encapsulates converting a DataFrame using
 `.to_dict("records")`
 without you having to do it explicitly (and so that it works in a pipeline). Finally, we’ve also provided the list of feature names in
 `kidney_feature_names`
 , the target name in
 `kidney_target_name`
 , the features in
 `X`
 , and the target in
 `y`
 .




 In this exercise, your task is to apply the
 `CategoricalImputer`
 to impute all of the categorical columns in the dataset. You can refer to how the numeric imputation mapper was created as a template. Notice the keyword arguments
 `input_df=True`
 and
 `df_out=True`
 ? This is so that you can work with DataFrames instead of arrays. By default, the transformers are passed a
 `numpy`
 array of the selected columns as input, and as a result, the output of the DataFrame mapper is also an array. Scikit-learn transformers have historically been designed to work with
 `numpy`
 arrays, not
 `pandas`
 DataFrames, even though their basic indexing interfaces are similar.





```python

# Import necessary modules
from sklearn_pandas import DataFrameMapper
from sklearn_pandas import CategoricalImputer

# Check number of nulls in each feature column
nulls_per_column = X.isnull().sum()
print(nulls_per_column)

# Create a boolean mask for categorical columns
categorical_feature_mask = X.dtypes == object

# Get list of categorical column names
categorical_columns = X.columns[categorical_feature_mask].tolist()

# Get list of non-categorical column names
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()

# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper(
                                            [([numeric_feature], Imputer(strategy="median")) for numeric_feature in non_categorical_columns],
                                            input_df=True,
                                            df_out=True
                                           )

# Apply categorical imputer
categorical_imputation_mapper = DataFrameMapper(
                                                [(category_feature, CategoricalImputer()) for category_feature in categorical_columns],
                                                input_df=True,
                                                df_out=True
                                               )

```




```

print(nulls_per_column)
age        9
bp        12
sg        47
al        46
su        49
bgr       44
bu        19
sc        17
sod       87
pot       88
hemo      52
pcv       71
wc       106
rc       131
rbc      152
pc        65
pcc        4
ba         4
htn        2
dm         2
cad        2
appet      1
pe         1
ane        1
dtype: int64

```


### **4.2.3 Kidney disease case study II: Feature Union**



 Having separately imputed numeric as well as categorical columns, your task is now to use scikit-learn’s
 [FeatureUnion](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html)
 to concatenate their results, which are contained in two separate transformer objects –
 `numeric_imputation_mapper`
 , and
 `categorical_imputation_mapper`
 , respectively.




 You may have already encountered
 `FeatureUnion`
 in
 [Machine Learning with the Experts: School Budgets](https://campus.datacamp.com/courses/machine-learning-with-the-experts-school-budgets/improving-your-model?ex=7)
 . Just like with pipelines, you have to pass it a list of
 `(string, transformer)`
 tuples, where the first half of each tuple is the name of the transformer.





```python

# Import FeatureUnion
from sklearn.pipeline import FeatureUnion

# Combine the numeric and categorical transformations
numeric_categorical_union = FeatureUnion([
                                          ("num_mapper", numeric_imputation_mapper),
                                          ("cat_mapper", categorical_imputation_mapper)
                                         ])

```


### **4.2.4 Kidney disease case study III: Full pipeline**



 It’s time to piece together all of the transforms along with an
 `XGBClassifier`
 to build the full pipeline!




 Besides the
 `numeric_categorical_union`
 that you created in the previous exercise, there are two other transforms needed: the
 `Dictifier()`
 transform which we created for you, and the
 `DictVectorizer()`
 .




 After creating the pipeline, your task is to cross-validate it to see how well it performs.





```python

# Create full pipeline
pipeline = Pipeline([
                     ("featureunion", numeric_categorical_union),
                     ("dictifier", Dictifier()),
                     ("vectorizer", DictVectorizer(sort=False)),
                     ("clf", xgb.XGBClassifier(max_depth=3))
                    ])

# Perform cross-validation
cross_val_scores = cross_val_score(pipeline, kidney_data, y, scoring="roc_auc", cv=3)

# Print avg. AUC
print("3-fold AUC: ", np.mean(cross_val_scores))
# 3-fold AUC:  0.998637406769937

```




---


## **4.3 Tuning XGBoost hyperparameters**


### **4.3.1 Bringing it all together**



 Alright, it’s time to bring together everything you’ve learned so far! In this final exercise of the course, you will combine your work from the previous exercises into one end-to-end XGBoost pipeline to really cement your understanding of preprocessing and pipelines in XGBoost.




 Your work from the previous 3 exercises, where you preprocessed the data and set up your pipeline, has been pre-loaded. Your job is to perform a randomized search and identify the best hyperparameters.





```python

# Create the parameter grid
gbm_param_grid = {
    'clf__learning_rate': np.arange(0.05, 1, 0.05),
    'clf__max_depth': np.arange(3, 10, 1),
    'clf__n_estimators': np.arange(50, 200, 50)
}

# Perform RandomizedSearchCV
randomized_roc_auc = RandomizedSearchCV(estimator=pipeline, param_distributions=gbm_param_grid, n_iter=2, scoring='roc_auc', verbose=1)

# Fit the estimator
randomized_roc_auc.fit(X, y)

# Compute metrics
print(randomized_roc_auc.best_score_)
print(randomized_roc_auc.best_estimator_)

```




```

Fitting 3 folds for each of 2 candidates, totalling 6 fits
0.9975202094090647
Pipeline(memory=None,
         steps=[('featureunion',
                 FeatureUnion(n_jobs=None,
                              transformer_list=[('num_mapper',
                                                 DataFrameMapper(default=False,
                                                                 df_out=True,
                                                                 features=[(['age'],
                                                                            Imputer(axis=0,
                                                                                    copy=True,
                                                                                    missing_values='NaN',
                                                                                    strategy='median',
                                                                                    verbose=0)),
                                                                           (['bp'],
                                                                            Imputer(axis=0,
                                                                                    copy=True,
                                                                                    missing_values='NaN',
                                                                                    strategy='median',
                                                                                    verbose=0)),
                                                                           (['sg'],
                                                                            Imputer(axis=0,
                                                                                    copy=...
                 XGBClassifier(base_score=0.5, booster='gbtree',
                               colsample_bylevel=1, colsample_bynode=1,
                               colsample_bytree=1, gamma=0,
                               learning_rate=0.9000000000000001,
                               max_delta_step=0, max_depth=5,
                               min_child_weight=1, missing=None,
                               n_estimators=150, n_jobs=1, nthread=None,
                               objective='binary:logistic', random_state=0,
                               reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                               seed=None, silent=None, subsample=1,
                               verbosity=1))],
         verbose=False)

```




---


## **4.4 Final Thoughts**



![Desktop View]({{ site.baseurl }}/assets/datacamp/extreme-gradient-boosting-with-xgboost/5-3.png?w=1024)


 Thank you for reading and hope you’ve learned a lot.



