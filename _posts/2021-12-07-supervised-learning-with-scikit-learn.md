---
title: Supervised Learning with scikit-learn
date: 2021-12-07 11:22:07 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Supervised Learning with scikit-learn
========================================







 This is the memo of the 21th course of ‘Data Scientist with Python’ track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/supervised-learning-with-scikit-learn)**
 .





---



 1. Classification
-------------------




###
**Machine learning introduction**



**What is machine learning?**
 Giving computers the ability to learn to make decisions from data without being explicitly programmed




**Examples of machine learning:**
 Learning to predict whether an email is spam or not (supervised)


 Clustering Wikipedia entries into different categories (unsupervised)



####
**Types of Machine Learning**


* supervised learning
* unsupervised learning
* reinforcement learning



**Supervised learning:**


 Predictor variables/
 **features**
 and a
 **target variable**




 Aim: Predict the target variable, given the predictor variables


 Classification: Target variable consists of categories


 Regression: Target variable is continuous




**Unsupervised learning:**


 Uncovering hidden patterns from unlabeled data




 Example of unsupervised learning:


 Grouping customers into distinct categories (Clustering)




**Reinforcement learning:**
 Software agents interact with an environment


 Learn how to optimize their behavior


 Given a system of rewards and punishments




 Applications


 Economics


 Genetics


 Game playing (AlphaGo)



####
 Naming conventions


* Features = predictor variables = independent variables
* Target variable = dependent variable = response variable




---


####
 Features of
 **Supervised learning**


* Automate time-consuming or expensive manual tasks (ex. Doctor’s diagnosis)
* Make predictions about the future (ex. Will a customer click on an ad or not)
* Need labeled data (Historical data with labels etc.)


####
**Popular libraries**


* scikit-learning (basic)
* TensorFlow
* keras




---


###
**Exploratory data analysis**


####
**Numerical EDA**



 In this chapter, you’ll be working with a dataset obtained from the
 [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records)
 consisting of votes made by US House of Representatives Congressmen.




 Your goal will be to predict their party affiliation (‘Democrat’ or ‘Republican’) based on how they voted on certain key issues.




 Here, it’s worth noting that we have preprocessed this dataset to deal with missing values. This is so that your focus can be directed towards understanding how to train and evaluate supervised learning models.




 Once you have mastered these fundamentals, you will be introduced to preprocessing techniques in Chapter 4 and have the chance to apply them there yourself – including on this very same dataset!




 Before thinking about what supervised learning models you can apply to this, however, you need to perform Exploratory data analysis (EDA) in order to understand the structure of the data.





```

df.head()
        party  infants  water  budget  physician  salvador  religious  \
0  republican        0      1       0          1         1          1
1  republican        0      1       0          1         1          1
2    democrat        0      1       1          0         1          1
3    democrat        0      1       1          0         1          1
4    democrat        1      1       1          0         1          1

   satellite  aid  missile  immigration  synfuels  education  superfund  \
0          0    0        0            1         0          1          1
1          0    0        0            0         0          1          1
2          0    0        0            0         1          0          1
3          0    0        0            0         1          0          1
4          0    0        0            0         1          0          1

   crime  duty_free_exports  eaa_rsa
0      1                  0        1
1      1                  0        1
2      1                  0        0
3      0                  0        1
4      1                  1        1

```




```

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 435 entries, 0 to 434
Data columns (total 17 columns):
party                435 non-null object
infants              435 non-null int64
water                435 non-null int64
budget               435 non-null int64
physician            435 non-null int64
salvador             435 non-null int64
religious            435 non-null int64
satellite            435 non-null int64
aid                  435 non-null int64
missile              435 non-null int64
immigration          435 non-null int64
synfuels             435 non-null int64
education            435 non-null int64
superfund            435 non-null int64
crime                435 non-null int64
duty_free_exports    435 non-null int64
eaa_rsa              435 non-null int64
dtypes: int64(16), object(1)
memory usage: 57.9+ KB

```


###
**Visual EDA**



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture-21.png)


 Above is a
 `countplot`
 of the
 `'education'`
 bill, generated from the following code:





```

plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

```



 In
 `sns.countplot()`
 , we specify the x-axis data to be
 `'education'`
 , and hue to be
 `'party'`
 . Recall that
 `'party'`
 is also our target variable. So the resulting plot shows the difference in voting behavior between the two parties for the
 `'education'`
 bill, with each party colored differently. We manually specified the color to be
 `'RdBu'`
 , as the Republican party has been traditionally associated with red, and the Democratic party with blue.




 It seems like Democrats voted resoundingly
 *against*
 this bill, compared to Republicans.





```

plt.figure()
sns.countplot(x='missile', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture1-20.png)


 Democrats vote resoundingly in
 *favor*
 of missile, compared to Republicans.





---


###
**The classification challenge**


####
**k-Nearest Neighbors: Fit**



[k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)





```python

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

```


####
**k-Nearest Neighbors: Predict**




```python

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party']
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))

# Prediction: ['democrat']

```



 How sure can you be of its predictions? In other words, how can you measure its performance?





---


###
**Measuring model performance**


####
**The digits recognition dataset: MNIST**



 In the following exercises, you’ll be working with the
 [MNIST](http://yann.lecun.com/exdb/mnist/)
 digits recognition dataset, which has 10 classes, the digits 0 through 9! A reduced version of the MNIST dataset is one of scikit-learn’s included datasets, and that is the one we will use in this exercise.




 Each sample in this scikit-learn dataset is an 8×8 image representing a handwritten digit. Each pixel is represented by an integer in the range 0 to 16, indicating varying levels of black.




 It is a famous dataset in machine learning and computer vision, and frequently used as a benchmark to evaluate the performance of a new model.





```python

# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
#dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])

print(digits.DESCR)
/*
Optical Recognition of Handwritten Digits Data Set
===================================================

Notes
-----
Data Set Characteristics:
    :Number of Instances: 5620
    :Number of Attributes: 64
    :Attribute Information: 8x8 image of integer pixels in the range 0..16.
    :Missing Attribute Values: None
    :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
    :Date: July; 1998
...
*/

# Print the shape of the images and data keys
print(digits.images.shape)
(1797, 8, 8)

print(digits.data.shape)
(1797, 64)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture2-19.png)

####
**Train/Test Split + Fit/Predict/Accuracy**




```python

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
#  Stratify the split according to the labels so that they are distributed in the training and test sets as they are in the original dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))
# 0.983333333333

```



 Incredibly, this out of the box k-NN classifier with 7 neighbors has learned from the training data and predicted the labels of the images in the test set with 98% accuracy, and it did so in less than a second! This is one illustration of how incredibly useful machine learning techniques can be.





---


####
**Overfitting and underfitting**



 In this exercise, you will compute and plot the training and testing accuracy scores for a variety of different neighbor values. By observing how the accuracy scores differ for the training and testing sets with different values of k, you will develop your intuition for overfitting and underfitting.





```python

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture3-18.png)


 It looks like the test accuracy is highest when using 3 and 5 neighbors. Using 8 neighbors or more seems to result in a simple model that underfits the data.





---



# **2. Regression**
------------------


###
**Introduction to regression**


####
**Importing data for supervised learning**



 In this chapter, you will work with
 [Gapminder](https://www.gapminder.org/data/)
 data that we have consolidated into one CSV file available in the workspace as
 `'gapminder.csv'`
 . Specifically, your goal will be to use this data to predict the life expectancy in a given country based on features such as the country’s GDP, fertility rate, and population.





```python

# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Dimensions of y before reshaping: (139,)
# Dimensions of X before reshaping: (139,)


# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

# Dimensions of y after reshaping: (139, 1)
# Dimensions of X after reshaping: (139, 1)

```


####
**Exploring the Gapminder data**




```

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 139 entries, 0 to 138
Data columns (total 9 columns):
population         139 non-null float64
fertility          139 non-null float64
HIV                139 non-null float64
CO2                139 non-null float64
BMI_male           139 non-null float64
GDP                139 non-null float64
BMI_female         139 non-null float64
life               139 non-null float64
child_mortality    139 non-null float64
dtypes: float64(9)
memory usage: 9.9 KB

```




```

sns.heatmap(df.corr(), square=True, cmap='RdYlGn')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture4-17.png)

###
**The basics of linear regression**



 We suppose that y and x have a linear relationship that can be model by


 y = ax + b


 An linear regression is to find a, b that minimize the sum of the squared residual (= Ordinary Least Squares, OLS)




 Why squared residual?


 Residuals may be positive and negative.


 They cancel each other. square residual can solve this problem.




![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture6-11.png)

 green lines are residuals



 When we have n variables of x,


 y = a1*x1 + a2*x2 + … an*xn + b


 we find a1, a2, … an, b that minimize the sum of the squared residual.



####
**Fit & predict for regression**



 In this exercise, you will use the
 `'fertility'`
 feature of the Gapminder dataset. Since the goal is to predict life expectancy, the target variable here is
 `'life'`
 .


 You will also compute and print the R2 score using sckit-learn’s
 `.score()`
 method.





```python

# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2
print(reg.score(X_fertility, y))
0.619244216774

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture7-12.png)

####
**Train/test split for regression**



 In this exercise, you will split the Gapminder dataset into training and testing sets, and then fit and predict a linear regression over
 **all**
 features. In addition to computing the R2 score, you will also compute the Root Mean Squared Error (RMSE), which is another commonly used metric to evaluate regression models.





```python

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
# R^2: 0.838046873142936


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
# Root Mean Squared Error: 3.2476010800377213


```


###
**Cross-validation**



 What is cross validation?


<https://en.wikipedia.org/wiki/Cross-validation_(statistics)>



####
**5-fold cross-validation**



 In this exercise, you will practice 5-fold cross validation on the Gapminder data. By default, scikit-learn’s
 `cross_val_score()`
 function uses R2R2 as the metric of choice for regression.





```python

# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)
# [ 0.81720569  0.82917058  0.90214134  0.80633989  0.94495637]

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
# Average 5-Fold CV Score: 0.8599627722793232

```


####
**K-Fold CV comparison**



 Cross validation is essential but do not forget that the more folds you use, the more computationally expensive cross-validation becomes. In this exercise, you will explore this for yourself. Your job is to perform 3-fold cross-validation and then 10-fold cross-validation on the Gapminder dataset.





```python

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg, X, y, cv=3)
print(np.mean(cvscores_3))
# 0.871871278262

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg, X, y, cv=10)
print(np.mean(cvscores_10))
# 0.843612862013

```




```

%timeit cross_val_score(reg, X, y, cv=3)
100 loops, best of 3: 8.73 ms per loop

%timeit cross_val_score(reg, X, y, cv=10)
10 loops, best of 3: 27.5 ms per loop

```




---


###
**Regularized regression**


####
**[Regularization I: Lasso](https://en.wikipedia.org/wiki/Lasso_(statistics))**



 In this exercise, you will fit a lasso regression to the Gapminder data you have been working with and plot the coefficients. Just as with the Boston data, you will find that the coefficients of some features are shrunk to 0, with only the most important ones remaining.





```

df.columns
Index(['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'life', 'child_mortality'],
      dtype='object')

X: ['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'child_mortality']
y: life

```




```python

# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
#  [-0.         -0.         -0.          0.          0.          0.         -0.
     -0.07087587]


# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture8-9.png)


 According to the lasso algorithm, it seems like
 `'child_mortality'`
 is the most important feature when predicting life expectancy.



####
**Regularization II: Ridge**



 Lasso is great for feature selection, but when building regression models, Ridge regression should be your first choice.




 Recall that lasso performs regularization by adding to the loss function a penalty term of the
 *absolute*
 value of each coefficient multiplied by some alpha. This is also known as L1L1 regularization because the regularization term is the L1L1 norm of the coefficients. This is not the only way to regularize, however.




 If instead you took the sum of the
 *squared*
 values of the coefficients multiplied by some alpha – like in Ridge regression – you would be computing the L2L2norm. In this exercise, you will practice fitting ridge regression models over a range of different alphas, and plot cross-validated R2R2 scores for each, using this function that we have defined for you, which plots the R2R2 score as well as standard error for each alpha:





```

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


```



 Don’t worry about the specifics of the above function works. The motivation behind this exercise is for you to see how the R2R2 score varies with different alphas, and to understand the importance of selecting the right value for alpha. You’ll learn how to tune alpha in the next chapter.





```python

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha

    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)

    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))

    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture9-8.png)


 Notice how the cross-validation scores change with different alphas.





---



# **3. Fine-tuning your model**
------------------------------


###
**confusion matrix**



 What is confusion matrix


<https://en.wikipedia.org/wiki/Confusion_matrix>




![{\displaystyle \mathrm {TPR} ={\frac {\mathrm {TP} }{\mathrm {P} }}={\frac {\mathrm {TP} }{\mathrm {TP} +\mathrm {FN} }}=1-\mathrm {FNR} }](https://wikimedia.org/api/rest_v1/media/math/render/svg/f02ea353bf60bfdd9557d2c98fe18c34cd8db835)

[sensitivity](https://en.wikipedia.org/wiki/Sensitivity_(test))
 ,
 [recall](https://en.wikipedia.org/wiki/Precision_and_recall#Recall)
 ,
 [hit rate](https://en.wikipedia.org/wiki/Hit_rate)
 , or
 [true positive rate](https://en.wikipedia.org/wiki/Sensitivity_(test))
 (TPR)



![{\displaystyle \mathrm {TNR} ={\frac {\mathrm {TN} }{\mathrm {N} }}={\frac {\mathrm {TN} }{\mathrm {TN} +\mathrm {FP} }}=1-\mathrm {FPR} }](https://wikimedia.org/api/rest_v1/media/math/render/svg/8f2c867f0641e498ec8a59de63697a3a45d66b07)

[specificity](https://en.wikipedia.org/wiki/Specificity_(tests))
 ,
 [selectivity](https://en.wikipedia.org/wiki/Specificity_(tests))
 or
 [true negative rate](https://en.wikipedia.org/wiki/Specificity_(tests))
 (TNR)



![{\displaystyle \mathrm {PPV} ={\frac {\mathrm {TP} }{\mathrm {TP} +\mathrm {FP} }}=1-\mathrm {FDR} }](https://wikimedia.org/api/rest_v1/media/math/render/svg/d854b1544fc77735d575ce0d30e34d7f1eacf707)

[precision](https://en.wikipedia.org/wiki/Information_retrieval#Precision)
 or
 [positive predictive value](https://en.wikipedia.org/wiki/Positive_predictive_value)
 (PPV)



![{\displaystyle \mathrm {ACC} ={\frac {\mathrm {TP} +\mathrm {TN} }{\mathrm {P} +\mathrm {N} }}={\frac {\mathrm {TP} +\mathrm {TN} }{\mathrm {TP} +\mathrm {TN} +\mathrm {FP} +\mathrm {FN} }}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/47deb47eb7ac214423d0a6afd05ec0af362fef9b)

[accuracy](https://en.wikipedia.org/wiki/Accuracy)
 (ACC)



![{\displaystyle \mathrm {F} _{1}=2\cdot {\frac {\mathrm {PPV} \cdot \mathrm {TPR} }{\mathrm {PPV} +\mathrm {TPR} }}={\frac {2\mathrm {TP} }{2\mathrm {TP} +\mathrm {FP} +\mathrm {FN} }}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/0e5f071c6418f444fadc9f5f9b0358beed3e094c)

[F1 score](https://en.wikipedia.org/wiki/F1_score)
 is the
 [harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean#Harmonic_mean_of_two_numbers)
 of
 [precision](https://en.wikipedia.org/wiki/Information_retrieval#Precision)
 and
 [sensitivity](https://en.wikipedia.org/wiki/Sensitivity_(test))


####
**illustration for TPR, TNR and PPV**



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/metric_example.png?w=1024)

[source](http://corysimon.github.io/articles/classification-metrics/)


####
**Metrics for classification**



 In this exercise, you will dive more deeply into evaluating the performance of binary classifiers by computing a confusion matrix and generating a classification report.




 Here, you’ll work with the
 [PIMA Indians](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
 dataset obtained from the UCI Machine Learning Repository. The goal is to predict whether or not a given female patient will contract diabetes based on features such as BMI, age, and number of pregnancies.




 Therefore, it is a binary classification problem. A target value of
 `0`
 indicates that the patient does
 *not*
 have diabetes, while a value of
 `1`
 indicates that the patient
 *does*
 have diabetes.





```

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
pregnancies    768 non-null int64
glucose        768 non-null int64
diastolic      768 non-null int64
triceps        768 non-null float64
insulin        768 non-null float64
bmi            768 non-null float64
dpf            768 non-null float64
age            768 non-null int64
diabetes       768 non-null int64
dtypes: float64(4), int64(5)
memory usage: 54.1 KB

```




```python

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

```




```

[[176  30]
 [ 52  50]]

             precision    recall  f1-score   support

          0       0.77      0.85      0.81       206
          1       0.62      0.49      0.55       102

avg / total       0.72      0.73      0.72       308

```




---


###
**Logistic regression and the ROC curve**



 What is logistic regression?


<https://en.wikipedia.org/wiki/Logistic_regression>




 What is ROC?


[Receiver operating characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)




 Further Reading:
 [scikit-learn document](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py)




![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/roc_space-2.png?w=1024)

####
**Building a logistic regression model**




```

X.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 8 columns):
pregnancies    768 non-null int64
glucose        768 non-null int64
diastolic      768 non-null int64
triceps        768 non-null float64
insulin        768 non-null float64
bmi            768 non-null float64
dpf            768 non-null float64
age            768 non-null int64
dtypes: float64(4), int64(4)
memory usage: 48.1 KB


y
0      1
1      0
2      1
      ..

765    0
766    1
767    0
Name: diabetes, dtype: int64

```




```python

# Import the necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


```




```

[[176  30]
 [ 35  67]]

             precision    recall  f1-score   support

          0       0.83      0.85      0.84       206
          1       0.69      0.66      0.67       102

avg / total       0.79      0.79      0.79       308

```


####
**Plotting an ROC curve**

**.predict_proba()**




```python

# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

```




```

logreg.predict_proba(X_test)
# False, True
# 0, 1
# Negative, Positive
array([[ 0.60409835,  0.39590165],
       [ 0.76042394,  0.23957606],
       [ 0.79670177,  0.20329823],
       ...
       [ 0.84686912,  0.15313088],
       [ 0.97617225,  0.02382775],
       [ 0.40380502,  0.59619498]])

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture1-21.png?w=1024)

####
**Precision-recall Curve**



 There are other ways to visually evaluate model performance. One such way is the precision-recall curve, which is generated by plotting the precision and recall for different thresholds.


 Note that here, the class is positive (1) if the individual
 *has*
 diabetes.




![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture2-20.png?w=1024)

* A recall of 1 corresponds to a classifier with a low threshold in which
 *all*
 females who contract diabetes were correctly classified as such, at the expense of many misclassifications of those who did
 *not*
 have diabetes.
* Precision is undefined for a classifier which makes
 *no*
 positive predictions, that is, classifies
 *everyone*
 as
 *not*
 having diabetes.
* When the threshold is very close to 1, precision is also 1, because the classifier is absolutely certain about its predictions.



 recall or sensitivity, TPR = 1 means all true positive are detected. We can predict all to positive to get a recall of 1.




 precision, PPV = 1 means no false positive are detected. We can predict less positive to get a higher precision.



####
**Area under the ROC curve**


####
**AUC(**
 Area Under the Curve
 **) computation**




```python

# diabetes data set
df.head()
   pregnancies  glucose  diastolic   triceps     insulin   bmi    dpf  age  \
0            6      148         72  35.00000  155.548223  33.6  0.627   50
1            1       85         66  29.00000  155.548223  26.6  0.351   31
2            8      183         64  29.15342  155.548223  23.3  0.672   32
3            1       89         66  23.00000   94.000000  28.1  0.167   21
4            0      137         40  35.00000  168.000000  43.1  2.288   33

   diabetes
0         1
1         0
2         1
3         0
4         1

```




```python

# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))


```




```

AUC: 0.8254806777079764

AUC scores computed using 5-fold cross-validation:
[ 0.80148148  0.8062963   0.81481481  0.86245283  0.8554717 ]

```




---


###
**Hyperparameter tuning**


####
**Hyperparameter tuning with GridSearchCV**



 You will now practice this yourself, but by using logistic regression on the diabetes dataset.




 Like the alpha parameter of lasso and ridge regularization that you saw earlier, logistic regression also has a regularization parameter: CC. CC controls the
 *inverse*
 of the regularization strength, and this is what you will tune in this exercise. A large CC can lead to an
 *overfit*
 model, while a small CC can lead to an
 *underfit*
 model.




 The hyperparameter space for CC has been setup for you. Your job is to use GridSearchCV and logistic regression to find the optimal CC in this hyperparameter space.





```python

# diabetes data set

```




```python

# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))


```




```

Tuned Logistic Regression Parameters: {'C': 3.7275937203149381}
Best score is 0.7708333333333334

```


####
**Hyperparameter tuning with RandomizedSearchCV**



 GridSearchCV can be computationally expensive, especially if you are searching over a large hyperparameter space and dealing with multiple hyperparameters. A solution to this is to use
 `RandomizedSearchCV`
 , in which not all hyperparameter values are tried out. Instead, a fixed number of hyperparameter settings is sampled from specified probability distributions. You’ll practice using
 `RandomizedSearchCV`
 in this exercise and see how this works.





```python

# diabetes data set

```




```python

# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


```




```

Tuned Decision Tree Parameters: {'criterion': 'entropy', 'max_depth': 3, 'max_features': 7, 'min_samples_leaf': 1}
Best score is 0.7317708333333334

```



 Note that
 `RandomizedSearchCV`
 will never outperform
 `GridSearchCV`
 . Instead, it is valuable because it saves on computation time.



###
**Hold-out set for final evaluation**


####
**Hold-out set in practice I: Classification**



 In addition to CC, logistic regression has a
 `'penalty'`
 hyperparameter which specifies whether to use
 `'l1'`
 or
 `'l2'`
 regularization. Your job in this exercise is to create a hold-out set, tune the
 `'C'`
 and
 `'penalty'`
 hyperparameters of a logistic regression classifier using
 `GridSearchCV`
 on the training set.





```python

# diabetes data set

```




```python

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))


```




```

Tuned Logistic Regression Parameter: {'C': 0.43939705607607948, 'penalty': 'l1'}
Tuned Logistic Regression Accuracy: 0.7652173913043478

```


####
**Hold-out set in practice II: Regression**



 Remember lasso and ridge regression from the previous chapter? Lasso used the L1 penalty to regularize, while ridge used the L2 penalty. There is another type of regularized regression known as the elastic net. In elastic net regularization, the penalty term is a linear combination of the L1 and L2 penalties:




**a∗L1+b∗L2**




 In scikit-learn, this term is represented by the
 `'l1_ratio'`
 parameter: An
 `'l1_ratio'`
 of
 `1`
 corresponds to an L1L1 penalty, and anything lower is a combination of L1L1 and L2L2.




 In this exercise, you will
 `GridSearchCV`
 to tune the
 `'l1_ratio'`
 of an elastic net model trained on the Gapminder data. As in the previous exercise, use a hold-out set to evaluate your model’s performance.





```

df.head()
   population  fertility  HIV        CO2  BMI_male      GDP  BMI_female  life  \
0  34811059.0       2.73  0.1   3.328945  24.59620  12314.0    129.9049  75.3
1  19842251.0       6.43  2.0   1.474353  22.25083   7103.0    130.1247  58.3
2  40381860.0       2.24  0.5   4.785170  27.50170  14646.0    118.8915  75.5
3   2975029.0       1.40  0.1   1.804106  25.35542   7383.0    132.8108  72.5
4  21370348.0       1.96  0.1  18.016313  27.56373  41312.0    117.3755  81.5

   child_mortality
0             29.5
1            192.0
2             15.4
3             20.0
4              5.2


```




```python

# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))


```




```

Tuned ElasticNet l1 ratio: {'l1_ratio': 0.20689655172413793}
Tuned ElasticNet R squared: 0.8668305372460283
Tuned ElasticNet MSE: 10.05791413339844

```




---



**Preprocessing and pipelines**
--------------------------------


###
**Preprocessing data**


####
**Exploring categorical features**




```

df.head()
   population  fertility  HIV        CO2  BMI_male      GDP  BMI_female  life  \
0  34811059.0       2.73  0.1   3.328945  24.59620  12314.0    129.9049  75.3
1  19842251.0       6.43  2.0   1.474353  22.25083   7103.0    130.1247  58.3
2  40381860.0       2.24  0.5   4.785170  27.50170  14646.0    118.8915  75.5
3   2975029.0       1.40  0.1   1.804106  25.35542   7383.0    132.8108  72.5
4  21370348.0       1.96  0.1  18.016313  27.56373  41312.0    117.3755  81.5

   child_mortality                      Region
0             29.5  Middle East & North Africa
1            192.0          Sub-Saharan Africa
2             15.4                     America
3             20.0       Europe & Central Asia
4              5.2         East Asia & Pacific

```




```python

# Import pandas
import pandas as pd

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture3-19.png?w=1024)

####
**Creating dummy variables**




```python

# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df, drop_first=True)

# Print the new columns of df_region
print(df_region.columns)

```




```

Index(['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'life', 'child_mortality', 'Region_America',
       'Region_East Asia & Pacific', 'Region_Europe & Central Asia',
       'Region_Middle East & North Africa', 'Region_South Asia',
       'Region_Sub-Saharan Africa'],
      dtype='object')

# Region_America has been dropped
Index(['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'life', 'child_mortality', 'Region_East Asia & Pacific',
       'Region_Europe & Central Asia', 'Region_Middle East & North Africa',
       'Region_South Asia', 'Region_Sub-Saharan Africa'],
      dtype='object')


df_region.head()
   population  fertility  HIV        CO2  BMI_male      GDP  BMI_female  life  \
0  34811059.0       2.73  0.1   3.328945  24.59620  12314.0    129.9049  75.3
1  19842251.0       6.43  2.0   1.474353  22.25083   7103.0    130.1247  58.3
2  40381860.0       2.24  0.5   4.785170  27.50170  14646.0    118.8915  75.5

   child_mortality  Region_East Asia & Pacific  Region_Europe & Central Asia  \
0             29.5                           0                             0
1            192.0                           0                             0
2             15.4                           0                             0

   Region_Middle East & North Africa  Region_South Asia  \
0                                  1                  0
1                                  0                  0
2                                  0                  0

   Region_Sub-Saharan Africa
0                          0
1                          1
2                          0

```


####
**Regression with categorical features**




```python

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)

# Print the cross-validated scores
print(ridge_cv)
[ 0.86808336  0.80623545  0.84004203  0.7754344   0.87503712]

```




---


###
**Handling missing data**


####
**Dropping missing data**



 Now, it’s time for you to take care of these yourself!




 The unprocessed dataset has been loaded into a DataFrame
 `df`
 . Explore it in the IPython Shell with the
 `.head()`
 method. You will see that there are certain data points labeled with a
 `'?'`
 . These denote missing values. As you saw in the video, different datasets encode missing values in different ways. Sometimes it may be a
 `'9999'`
 , other times a
 `0`
 – real-world data can be very messy! If you’re lucky, the missing values will already be encoded as
 `NaN`
 . We use
 `NaN`
 because it is an efficient and simplified way of internally representing missing data, and it lets us take advantage of pandas methods such as
 `.dropna()`
 and
 `.fillna()`
 , as well as scikit-learn’s Imputation transformer
 `Imputer()`
 .




 In this exercise, your job is to convert the
 `'?'`
 s to NaNs, and then drop the rows that contain them from the DataFrame.





```

df.head(3)
        party infants water budget physician salvador religious satellite aid  \
0  republican       0     1      0         1        1         1         0   0
1  republican       0     1      0         1        1         1         0   0
2    democrat       ?     1      1         ?        1         1         0   0

  missile immigration synfuels education superfund crime duty_free_exports  \
0       0           1        ?         1         1     1                 0
1       0           0        0         1         1     1                 0
2       0           0        1         0         1     1                 0

  eaa_rsa
0       1
1       ?
2       0

```




```python

# Convert '?' to NaN
df[df == '?'] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))


```




```

party                  0
infants               12
water                 48
budget                11
physician             11
salvador              15
religious             11
satellite             14
aid                   15
missile               22
immigration            7
synfuels              21
education             31
superfund             25
crime                 17
duty_free_exports     28
eaa_rsa              104
dtype: int64
Shape of Original DataFrame: (435, 17)


Shape of DataFrame After Dropping All Rows with Missing Values: (232, 17)

```



 When many values in your dataset are missing, if you drop them, you may end up throwing away valuable information along with the missing data. It’s better instead to develop an imputation strategy. This is where domain knowledge is useful, but in the absence of it, you can impute missing values with the mean or the median of the row or column that the missing value is in.



####
**Imputing missing data in a ML Pipeline I**



 As you’ve come to appreciate, there are many steps to building a model, from creating training and test sets, to fitting a classifier or regressor, to tuning its parameters, to evaluating its performance on new data. Imputation can be seen as the first step of this machine learning process, the entirety of which can be viewed within the context of a pipeline. Scikit-learn provides a pipeline constructor that allows you to piece together these steps into one process and thereby simplify your workflow.




 You’ll now practice setting up a pipeline with two steps: the imputation step, followed by the instantiation of a classifier. You’ve seen three classifiers in this course so far: k-NN, logistic regression, and the decision tree. You will now be introduced to a fourth one – the Support Vector Machine, or
 [SVM](http://scikit-learn.org/stable/modules/svm.html)
 .





```python

# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
# axis=0 for column
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]

```


####
**Imputing missing data in a ML Pipeline II**



 Having setup the steps of the pipeline in the previous exercise, you will now use it on the voting dataset to classify a Congressman’s party affiliation.




 What makes pipelines so incredibly useful is the simple interface that they provide. You can use the
 `.fit()`
 and
 `.predict()`
 methods on pipelines just as you did with your classifiers and regressors!





```python

# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))


```




```

             precision    recall  f1-score   support

   democrat       0.99      0.96      0.98        85
 republican       0.94      0.98      0.96        46

avg / total       0.97      0.97      0.97       131

```


###
**Centering and scaling**


####
**Centering and scaling your data**



 You will now explore scaling for yourself on a new dataset –
 [White Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
 !




 We have used the
 `'quality'`
 feature of the wine to create a binary target variable: If
 `'quality'`
 is less than
 `5`
 , the target variable is
 `1`
 , and otherwise, it is
 `0`
 .




 Notice how some features seem to have different units of measurement.
 `'density'`
 , for instance, takes values between 0.98 and 1.04, while
 `'total sulfur dioxide'`
 ranges from 9 to 440. As a result, it may be worth scaling the features here. Your job in this exercise is to scale the features and compute the mean and standard deviation of the unscaled features compared to the scaled features.





```python

# white wine quality data set
df.head(3)
   fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
0            7.0              0.27         0.36            20.7      0.045
1            6.3              0.30         0.34             1.6      0.049
2            8.1              0.28         0.40             6.9      0.050

   free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
0                 45.0                 170.0   1.0010  3.00       0.45
1                 14.0                 132.0   0.9940  3.30       0.49
2                 30.0                  97.0   0.9951  3.26       0.44

   alcohol  quality
0      8.8        6
1      9.5        6
2     10.1        6

```




```python

# Import scale
from sklearn.preprocessing import scale

# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X)))
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled)))
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))

```




```

Mean of Unscaled Features: 18.432687072460002
Standard Deviation of Unscaled Features: 41.54494764094571

Mean of Scaled Features: 2.7314972981668206e-15
Standard Deviation of Scaled Features: 0.9999999999999999

```


####
**Centering and scaling in a pipeline**



 With regard to whether or not scaling is effective, the proof is in the pudding! See for yourself whether or not scaling the features of the White Wine Quality dataset has any impact on its performance.




 You will use a k-NN classifier as part of a pipeline that includes scaling, and for the purposes of comparison, a k-NN classifier trained on the unscaled data has been provided.





```python

# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))

Accuracy with Scaling: 0.7700680272108843
Accuracy without Scaling: 0.6979591836734694

```



 It looks like scaling has significantly improved model performance!





---


####
**Bringing it all together I: Pipeline for classification**



 It is time now to piece together everything you have learned so far into a pipeline for classification! Your job in this exercise is to build a pipeline that includes scaling and hyperparameter tuning to classify wine quality.




 You’ll return to using the SVM classifier you were briefly introduced to earlier in this chapter. The hyperparameters you will tune are C and gamma. C controls the regularization strength. It is analogous to the C you tuned for logistic regression in Chapter 3, while gamma controls the kernel coefficient:





```python

# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))


```




```

Accuracy: 0.7795918367346939
             precision    recall  f1-score   support

      False       0.83      0.85      0.84       662
       True       0.67      0.63      0.65       318

avg / total       0.78      0.78      0.78       980

Tuned Model Parameters: {'SVM__C': 10, 'SVM__gamma': 0.1}

```


####
**Bringing it all together II: Pipeline for regression**



 Your job is to build a pipeline that imputes the missing data, scales the features, and fits an ElasticNet to the Gapminder data. You will then tune the
 `l1_ratio`
 of your ElasticNet using GridSearchCV.





```python

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))

#    Tuned ElasticNet Alpha: {'elasticnet__l1_ratio': 1.0}
#    Tuned ElasticNet R squared: 0.8862016570888217

```



 The End.


 Thank you for reading.



---



 1. Classification
-------------------




###
**Machine learning introduction**



**What is machine learning?**
 Giving computers the ability to learn to make decisions from data without being explicitly programmed




**Examples of machine learning:**
 Learning to predict whether an email is spam or not (supervised)


 Clustering Wikipedia entries into different categories (unsupervised)



####
**Types of Machine Learning**


* supervised learning
* unsupervised learning
* reinforcement learning



**Supervised learning:**


 Predictor variables/
 **features**
 and a
 **target variable**




 Aim: Predict the target variable, given the predictor variables


 Classification: Target variable consists of categories


 Regression: Target variable is continuous




**Unsupervised learning:**


 Uncovering hidden patterns from unlabeled data




 Example of unsupervised learning:


 Grouping customers into distinct categories (Clustering)




**Reinforcement learning:**
 Software agents interact with an environment


 Learn how to optimize their behavior


 Given a system of rewards and punishments




 Applications


 Economics


 Genetics


 Game playing (AlphaGo)



####
 Naming conventions


* Features = predictor variables = independent variables
* Target variable = dependent variable = response variable




---


####
 Features of
 **Supervised learning**


* Automate time-consuming or expensive manual tasks (ex. Doctor’s diagnosis)
* Make predictions about the future (ex. Will a customer click on an ad or not)
* Need labeled data (Historical data with labels etc.)


####
**Popular libraries**


* scikit-learning (basic)
* TensorFlow
* keras




---


###
**Exploratory data analysis**


####
**Numerical EDA**



 In this chapter, you’ll be working with a dataset obtained from the
 [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records)
 consisting of votes made by US House of Representatives Congressmen.




 Your goal will be to predict their party affiliation (‘Democrat’ or ‘Republican’) based on how they voted on certain key issues.




 Here, it’s worth noting that we have preprocessed this dataset to deal with missing values. This is so that your focus can be directed towards understanding how to train and evaluate supervised learning models.




 Once you have mastered these fundamentals, you will be introduced to preprocessing techniques in Chapter 4 and have the chance to apply them there yourself – including on this very same dataset!




 Before thinking about what supervised learning models you can apply to this, however, you need to perform Exploratory data analysis (EDA) in order to understand the structure of the data.





```

df.head()
        party  infants  water  budget  physician  salvador  religious  \
0  republican        0      1       0          1         1          1
1  republican        0      1       0          1         1          1
2    democrat        0      1       1          0         1          1
3    democrat        0      1       1          0         1          1
4    democrat        1      1       1          0         1          1

   satellite  aid  missile  immigration  synfuels  education  superfund  \
0          0    0        0            1         0          1          1
1          0    0        0            0         0          1          1
2          0    0        0            0         1          0          1
3          0    0        0            0         1          0          1
4          0    0        0            0         1          0          1

   crime  duty_free_exports  eaa_rsa
0      1                  0        1
1      1                  0        1
2      1                  0        0
3      0                  0        1
4      1                  1        1

```




```

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 435 entries, 0 to 434
Data columns (total 17 columns):
party                435 non-null object
infants              435 non-null int64
water                435 non-null int64
budget               435 non-null int64
physician            435 non-null int64
salvador             435 non-null int64
religious            435 non-null int64
satellite            435 non-null int64
aid                  435 non-null int64
missile              435 non-null int64
immigration          435 non-null int64
synfuels             435 non-null int64
education            435 non-null int64
superfund            435 non-null int64
crime                435 non-null int64
duty_free_exports    435 non-null int64
eaa_rsa              435 non-null int64
dtypes: int64(16), object(1)
memory usage: 57.9+ KB

```


###
**Visual EDA**



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture-21.png)


 Above is a
 `countplot`
 of the
 `'education'`
 bill, generated from the following code:





```

plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

```



 In
 `sns.countplot()`
 , we specify the x-axis data to be
 `'education'`
 , and hue to be
 `'party'`
 . Recall that
 `'party'`
 is also our target variable. So the resulting plot shows the difference in voting behavior between the two parties for the
 `'education'`
 bill, with each party colored differently. We manually specified the color to be
 `'RdBu'`
 , as the Republican party has been traditionally associated with red, and the Democratic party with blue.




 It seems like Democrats voted resoundingly
 *against*
 this bill, compared to Republicans.





```

plt.figure()
sns.countplot(x='missile', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture1-20.png)


 Democrats vote resoundingly in
 *favor*
 of missile, compared to Republicans.





---


###
**The classification challenge**


####
**k-Nearest Neighbors: Fit**



[k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)





```python

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

```


####
**k-Nearest Neighbors: Predict**




```python

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party']
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))

# Prediction: ['democrat']

```



 How sure can you be of its predictions? In other words, how can you measure its performance?





---


###
**Measuring model performance**


####
**The digits recognition dataset: MNIST**



 In the following exercises, you’ll be working with the
 [MNIST](http://yann.lecun.com/exdb/mnist/)
 digits recognition dataset, which has 10 classes, the digits 0 through 9! A reduced version of the MNIST dataset is one of scikit-learn’s included datasets, and that is the one we will use in this exercise.




 Each sample in this scikit-learn dataset is an 8×8 image representing a handwritten digit. Each pixel is represented by an integer in the range 0 to 16, indicating varying levels of black.




 It is a famous dataset in machine learning and computer vision, and frequently used as a benchmark to evaluate the performance of a new model.





```python

# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
#dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])

print(digits.DESCR)
/*
Optical Recognition of Handwritten Digits Data Set
===================================================

Notes
-----
Data Set Characteristics:
    :Number of Instances: 5620
    :Number of Attributes: 64
    :Attribute Information: 8x8 image of integer pixels in the range 0..16.
    :Missing Attribute Values: None
    :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
    :Date: July; 1998
...
*/

# Print the shape of the images and data keys
print(digits.images.shape)
(1797, 8, 8)

print(digits.data.shape)
(1797, 64)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture2-19.png)

####
**Train/Test Split + Fit/Predict/Accuracy**




```python

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
#  Stratify the split according to the labels so that they are distributed in the training and test sets as they are in the original dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))
# 0.983333333333

```



 Incredibly, this out of the box k-NN classifier with 7 neighbors has learned from the training data and predicted the labels of the images in the test set with 98% accuracy, and it did so in less than a second! This is one illustration of how incredibly useful machine learning techniques can be.





---


####
**Overfitting and underfitting**



 In this exercise, you will compute and plot the training and testing accuracy scores for a variety of different neighbor values. By observing how the accuracy scores differ for the training and testing sets with different values of k, you will develop your intuition for overfitting and underfitting.





```python

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture3-18.png)


 It looks like the test accuracy is highest when using 3 and 5 neighbors. Using 8 neighbors or more seems to result in a simple model that underfits the data.





---



# **2. Regression**
------------------


###
**Introduction to regression**


####
**Importing data for supervised learning**



 In this chapter, you will work with
 [Gapminder](https://www.gapminder.org/data/)
 data that we have consolidated into one CSV file available in the workspace as
 `'gapminder.csv'`
 . Specifically, your goal will be to use this data to predict the life expectancy in a given country based on features such as the country’s GDP, fertility rate, and population.





```python

# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Dimensions of y before reshaping: (139,)
# Dimensions of X before reshaping: (139,)


# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

# Dimensions of y after reshaping: (139, 1)
# Dimensions of X after reshaping: (139, 1)

```


####
**Exploring the Gapminder data**




```

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 139 entries, 0 to 138
Data columns (total 9 columns):
population         139 non-null float64
fertility          139 non-null float64
HIV                139 non-null float64
CO2                139 non-null float64
BMI_male           139 non-null float64
GDP                139 non-null float64
BMI_female         139 non-null float64
life               139 non-null float64
child_mortality    139 non-null float64
dtypes: float64(9)
memory usage: 9.9 KB

```




```

sns.heatmap(df.corr(), square=True, cmap='RdYlGn')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture4-17.png)

###
**The basics of linear regression**



 We suppose that y and x have a linear relationship that can be model by


 y = ax + b


 An linear regression is to find a, b that minimize the sum of the squared residual (= Ordinary Least Squares, OLS)




 Why squared residual?


 Residuals may be positive and negative.


 They cancel each other. square residual can solve this problem.




![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture6-11.png)

 green lines are residuals



 When we have n variables of x,


 y = a1*x1 + a2*x2 + … an*xn + b


 we find a1, a2, … an, b that minimize the sum of the squared residual.



####
**Fit & predict for regression**



 In this exercise, you will use the
 `'fertility'`
 feature of the Gapminder dataset. Since the goal is to predict life expectancy, the target variable here is
 `'life'`
 .


 You will also compute and print the R2 score using sckit-learn’s
 `.score()`
 method.





```python

# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2
print(reg.score(X_fertility, y))
0.619244216774

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture7-12.png)

####
**Train/test split for regression**



 In this exercise, you will split the Gapminder dataset into training and testing sets, and then fit and predict a linear regression over
 **all**
 features. In addition to computing the R2 score, you will also compute the Root Mean Squared Error (RMSE), which is another commonly used metric to evaluate regression models.





```python

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
# R^2: 0.838046873142936


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
# Root Mean Squared Error: 3.2476010800377213


```


###
**Cross-validation**



 What is cross validation?


<https://en.wikipedia.org/wiki/Cross-validation_(statistics)>



####
**5-fold cross-validation**



 In this exercise, you will practice 5-fold cross validation on the Gapminder data. By default, scikit-learn’s
 `cross_val_score()`
 function uses R2R2 as the metric of choice for regression.





```python

# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)
# [ 0.81720569  0.82917058  0.90214134  0.80633989  0.94495637]

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
# Average 5-Fold CV Score: 0.8599627722793232

```


####
**K-Fold CV comparison**



 Cross validation is essential but do not forget that the more folds you use, the more computationally expensive cross-validation becomes. In this exercise, you will explore this for yourself. Your job is to perform 3-fold cross-validation and then 10-fold cross-validation on the Gapminder dataset.





```python

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg, X, y, cv=3)
print(np.mean(cvscores_3))
# 0.871871278262

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg, X, y, cv=10)
print(np.mean(cvscores_10))
# 0.843612862013

```




```

%timeit cross_val_score(reg, X, y, cv=3)
100 loops, best of 3: 8.73 ms per loop

%timeit cross_val_score(reg, X, y, cv=10)
10 loops, best of 3: 27.5 ms per loop

```




---


###
**Regularized regression**


####
**[Regularization I: Lasso](https://en.wikipedia.org/wiki/Lasso_(statistics))**



 In this exercise, you will fit a lasso regression to the Gapminder data you have been working with and plot the coefficients. Just as with the Boston data, you will find that the coefficients of some features are shrunk to 0, with only the most important ones remaining.





```

df.columns
Index(['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'life', 'child_mortality'],
      dtype='object')

X: ['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'child_mortality']
y: life

```




```python

# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
#  [-0.         -0.         -0.          0.          0.          0.         -0.
     -0.07087587]


# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture8-9.png)


 According to the lasso algorithm, it seems like
 `'child_mortality'`
 is the most important feature when predicting life expectancy.



####
**Regularization II: Ridge**



 Lasso is great for feature selection, but when building regression models, Ridge regression should be your first choice.




 Recall that lasso performs regularization by adding to the loss function a penalty term of the
 *absolute*
 value of each coefficient multiplied by some alpha. This is also known as L1L1 regularization because the regularization term is the L1L1 norm of the coefficients. This is not the only way to regularize, however.




 If instead you took the sum of the
 *squared*
 values of the coefficients multiplied by some alpha – like in Ridge regression – you would be computing the L2L2norm. In this exercise, you will practice fitting ridge regression models over a range of different alphas, and plot cross-validated R2R2 scores for each, using this function that we have defined for you, which plots the R2R2 score as well as standard error for each alpha:





```

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


```



 Don’t worry about the specifics of the above function works. The motivation behind this exercise is for you to see how the R2R2 score varies with different alphas, and to understand the importance of selecting the right value for alpha. You’ll learn how to tune alpha in the next chapter.





```python

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha

    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)

    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))

    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture9-8.png)


 Notice how the cross-validation scores change with different alphas.





---



# **3. Fine-tuning your model**
------------------------------


###
**confusion matrix**



 What is confusion matrix


<https://en.wikipedia.org/wiki/Confusion_matrix>




![{\displaystyle \mathrm {TPR} ={\frac {\mathrm {TP} }{\mathrm {P} }}={\frac {\mathrm {TP} }{\mathrm {TP} +\mathrm {FN} }}=1-\mathrm {FNR} }](https://wikimedia.org/api/rest_v1/media/math/render/svg/f02ea353bf60bfdd9557d2c98fe18c34cd8db835)

[sensitivity](https://en.wikipedia.org/wiki/Sensitivity_(test))
 ,
 [recall](https://en.wikipedia.org/wiki/Precision_and_recall#Recall)
 ,
 [hit rate](https://en.wikipedia.org/wiki/Hit_rate)
 , or
 [true positive rate](https://en.wikipedia.org/wiki/Sensitivity_(test))
 (TPR)



![{\displaystyle \mathrm {TNR} ={\frac {\mathrm {TN} }{\mathrm {N} }}={\frac {\mathrm {TN} }{\mathrm {TN} +\mathrm {FP} }}=1-\mathrm {FPR} }](https://wikimedia.org/api/rest_v1/media/math/render/svg/8f2c867f0641e498ec8a59de63697a3a45d66b07)

[specificity](https://en.wikipedia.org/wiki/Specificity_(tests))
 ,
 [selectivity](https://en.wikipedia.org/wiki/Specificity_(tests))
 or
 [true negative rate](https://en.wikipedia.org/wiki/Specificity_(tests))
 (TNR)



![{\displaystyle \mathrm {PPV} ={\frac {\mathrm {TP} }{\mathrm {TP} +\mathrm {FP} }}=1-\mathrm {FDR} }](https://wikimedia.org/api/rest_v1/media/math/render/svg/d854b1544fc77735d575ce0d30e34d7f1eacf707)

[precision](https://en.wikipedia.org/wiki/Information_retrieval#Precision)
 or
 [positive predictive value](https://en.wikipedia.org/wiki/Positive_predictive_value)
 (PPV)



![{\displaystyle \mathrm {ACC} ={\frac {\mathrm {TP} +\mathrm {TN} }{\mathrm {P} +\mathrm {N} }}={\frac {\mathrm {TP} +\mathrm {TN} }{\mathrm {TP} +\mathrm {TN} +\mathrm {FP} +\mathrm {FN} }}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/47deb47eb7ac214423d0a6afd05ec0af362fef9b)

[accuracy](https://en.wikipedia.org/wiki/Accuracy)
 (ACC)



![{\displaystyle \mathrm {F} _{1}=2\cdot {\frac {\mathrm {PPV} \cdot \mathrm {TPR} }{\mathrm {PPV} +\mathrm {TPR} }}={\frac {2\mathrm {TP} }{2\mathrm {TP} +\mathrm {FP} +\mathrm {FN} }}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/0e5f071c6418f444fadc9f5f9b0358beed3e094c)

[F1 score](https://en.wikipedia.org/wiki/F1_score)
 is the
 [harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean#Harmonic_mean_of_two_numbers)
 of
 [precision](https://en.wikipedia.org/wiki/Information_retrieval#Precision)
 and
 [sensitivity](https://en.wikipedia.org/wiki/Sensitivity_(test))


####
**illustration for TPR, TNR and PPV**



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/metric_example.png?w=1024)

[source](http://corysimon.github.io/articles/classification-metrics/)


####
**Metrics for classification**



 In this exercise, you will dive more deeply into evaluating the performance of binary classifiers by computing a confusion matrix and generating a classification report.




 Here, you’ll work with the
 [PIMA Indians](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
 dataset obtained from the UCI Machine Learning Repository. The goal is to predict whether or not a given female patient will contract diabetes based on features such as BMI, age, and number of pregnancies.




 Therefore, it is a binary classification problem. A target value of
 `0`
 indicates that the patient does
 *not*
 have diabetes, while a value of
 `1`
 indicates that the patient
 *does*
 have diabetes.





```

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
pregnancies    768 non-null int64
glucose        768 non-null int64
diastolic      768 non-null int64
triceps        768 non-null float64
insulin        768 non-null float64
bmi            768 non-null float64
dpf            768 non-null float64
age            768 non-null int64
diabetes       768 non-null int64
dtypes: float64(4), int64(5)
memory usage: 54.1 KB

```




```python

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

```




```

[[176  30]
 [ 52  50]]

             precision    recall  f1-score   support

          0       0.77      0.85      0.81       206
          1       0.62      0.49      0.55       102

avg / total       0.72      0.73      0.72       308

```




---


###
**Logistic regression and the ROC curve**



 What is logistic regression?


<https://en.wikipedia.org/wiki/Logistic_regression>




 What is ROC?


[Receiver operating characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)




 Further Reading:
 [scikit-learn document](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py)




![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/roc_space-2.png?w=1024)

####
**Building a logistic regression model**




```

X.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 8 columns):
pregnancies    768 non-null int64
glucose        768 non-null int64
diastolic      768 non-null int64
triceps        768 non-null float64
insulin        768 non-null float64
bmi            768 non-null float64
dpf            768 non-null float64
age            768 non-null int64
dtypes: float64(4), int64(4)
memory usage: 48.1 KB


y
0      1
1      0
2      1
      ..

765    0
766    1
767    0
Name: diabetes, dtype: int64

```




```python

# Import the necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


```




```

[[176  30]
 [ 35  67]]

             precision    recall  f1-score   support

          0       0.83      0.85      0.84       206
          1       0.69      0.66      0.67       102

avg / total       0.79      0.79      0.79       308

```


####
**Plotting an ROC curve**

**.predict_proba()**




```python

# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

```




```

logreg.predict_proba(X_test)
# False, True
# 0, 1
# Negative, Positive
array([[ 0.60409835,  0.39590165],
       [ 0.76042394,  0.23957606],
       [ 0.79670177,  0.20329823],
       ...
       [ 0.84686912,  0.15313088],
       [ 0.97617225,  0.02382775],
       [ 0.40380502,  0.59619498]])

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture1-21.png?w=1024)

####
**Precision-recall Curve**



 There are other ways to visually evaluate model performance. One such way is the precision-recall curve, which is generated by plotting the precision and recall for different thresholds.


 Note that here, the class is positive (1) if the individual
 *has*
 diabetes.




![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture2-20.png?w=1024)

* A recall of 1 corresponds to a classifier with a low threshold in which
 *all*
 females who contract diabetes were correctly classified as such, at the expense of many misclassifications of those who did
 *not*
 have diabetes.
* Precision is undefined for a classifier which makes
 *no*
 positive predictions, that is, classifies
 *everyone*
 as
 *not*
 having diabetes.
* When the threshold is very close to 1, precision is also 1, because the classifier is absolutely certain about its predictions.



 recall or sensitivity, TPR = 1 means all true positive are detected. We can predict all to positive to get a recall of 1.




 precision, PPV = 1 means no false positive are detected. We can predict less positive to get a higher precision.



####
**Area under the ROC curve**


####
**AUC(**
 Area Under the Curve
 **) computation**




```python

# diabetes data set
df.head()
   pregnancies  glucose  diastolic   triceps     insulin   bmi    dpf  age  \
0            6      148         72  35.00000  155.548223  33.6  0.627   50
1            1       85         66  29.00000  155.548223  26.6  0.351   31
2            8      183         64  29.15342  155.548223  23.3  0.672   32
3            1       89         66  23.00000   94.000000  28.1  0.167   21
4            0      137         40  35.00000  168.000000  43.1  2.288   33

   diabetes
0         1
1         0
2         1
3         0
4         1

```




```python

# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))


```




```

AUC: 0.8254806777079764

AUC scores computed using 5-fold cross-validation:
[ 0.80148148  0.8062963   0.81481481  0.86245283  0.8554717 ]

```




---


###
**Hyperparameter tuning**


####
**Hyperparameter tuning with GridSearchCV**



 You will now practice this yourself, but by using logistic regression on the diabetes dataset.




 Like the alpha parameter of lasso and ridge regularization that you saw earlier, logistic regression also has a regularization parameter: CC. CC controls the
 *inverse*
 of the regularization strength, and this is what you will tune in this exercise. A large CC can lead to an
 *overfit*
 model, while a small CC can lead to an
 *underfit*
 model.




 The hyperparameter space for CC has been setup for you. Your job is to use GridSearchCV and logistic regression to find the optimal CC in this hyperparameter space.





```python

# diabetes data set

```




```python

# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))


```




```

Tuned Logistic Regression Parameters: {'C': 3.7275937203149381}
Best score is 0.7708333333333334

```


####
**Hyperparameter tuning with RandomizedSearchCV**



 GridSearchCV can be computationally expensive, especially if you are searching over a large hyperparameter space and dealing with multiple hyperparameters. A solution to this is to use
 `RandomizedSearchCV`
 , in which not all hyperparameter values are tried out. Instead, a fixed number of hyperparameter settings is sampled from specified probability distributions. You’ll practice using
 `RandomizedSearchCV`
 in this exercise and see how this works.





```python

# diabetes data set

```




```python

# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


```




```

Tuned Decision Tree Parameters: {'criterion': 'entropy', 'max_depth': 3, 'max_features': 7, 'min_samples_leaf': 1}
Best score is 0.7317708333333334

```



 Note that
 `RandomizedSearchCV`
 will never outperform
 `GridSearchCV`
 . Instead, it is valuable because it saves on computation time.



###
**Hold-out set for final evaluation**


####
**Hold-out set in practice I: Classification**



 In addition to CC, logistic regression has a
 `'penalty'`
 hyperparameter which specifies whether to use
 `'l1'`
 or
 `'l2'`
 regularization. Your job in this exercise is to create a hold-out set, tune the
 `'C'`
 and
 `'penalty'`
 hyperparameters of a logistic regression classifier using
 `GridSearchCV`
 on the training set.





```python

# diabetes data set

```




```python

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))


```




```

Tuned Logistic Regression Parameter: {'C': 0.43939705607607948, 'penalty': 'l1'}
Tuned Logistic Regression Accuracy: 0.7652173913043478

```


####
**Hold-out set in practice II: Regression**



 Remember lasso and ridge regression from the previous chapter? Lasso used the L1 penalty to regularize, while ridge used the L2 penalty. There is another type of regularized regression known as the elastic net. In elastic net regularization, the penalty term is a linear combination of the L1 and L2 penalties:




**a∗L1+b∗L2**




 In scikit-learn, this term is represented by the
 `'l1_ratio'`
 parameter: An
 `'l1_ratio'`
 of
 `1`
 corresponds to an L1L1 penalty, and anything lower is a combination of L1L1 and L2L2.




 In this exercise, you will
 `GridSearchCV`
 to tune the
 `'l1_ratio'`
 of an elastic net model trained on the Gapminder data. As in the previous exercise, use a hold-out set to evaluate your model’s performance.





```

df.head()
   population  fertility  HIV        CO2  BMI_male      GDP  BMI_female  life  \
0  34811059.0       2.73  0.1   3.328945  24.59620  12314.0    129.9049  75.3
1  19842251.0       6.43  2.0   1.474353  22.25083   7103.0    130.1247  58.3
2  40381860.0       2.24  0.5   4.785170  27.50170  14646.0    118.8915  75.5
3   2975029.0       1.40  0.1   1.804106  25.35542   7383.0    132.8108  72.5
4  21370348.0       1.96  0.1  18.016313  27.56373  41312.0    117.3755  81.5

   child_mortality
0             29.5
1            192.0
2             15.4
3             20.0
4              5.2


```




```python

# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))


```




```

Tuned ElasticNet l1 ratio: {'l1_ratio': 0.20689655172413793}
Tuned ElasticNet R squared: 0.8668305372460283
Tuned ElasticNet MSE: 10.05791413339844

```




---



**Preprocessing and pipelines**
--------------------------------


###
**Preprocessing data**


####
**Exploring categorical features**




```

df.head()
   population  fertility  HIV        CO2  BMI_male      GDP  BMI_female  life  \
0  34811059.0       2.73  0.1   3.328945  24.59620  12314.0    129.9049  75.3
1  19842251.0       6.43  2.0   1.474353  22.25083   7103.0    130.1247  58.3
2  40381860.0       2.24  0.5   4.785170  27.50170  14646.0    118.8915  75.5
3   2975029.0       1.40  0.1   1.804106  25.35542   7383.0    132.8108  72.5
4  21370348.0       1.96  0.1  18.016313  27.56373  41312.0    117.3755  81.5

   child_mortality                      Region
0             29.5  Middle East & North Africa
1            192.0          Sub-Saharan Africa
2             15.4                     America
3             20.0       Europe & Central Asia
4              5.2         East Asia & Pacific

```




```python

# Import pandas
import pandas as pd

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture3-19.png?w=1024)

####
**Creating dummy variables**




```python

# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df, drop_first=True)

# Print the new columns of df_region
print(df_region.columns)

```




```

Index(['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'life', 'child_mortality', 'Region_America',
       'Region_East Asia & Pacific', 'Region_Europe & Central Asia',
       'Region_Middle East & North Africa', 'Region_South Asia',
       'Region_Sub-Saharan Africa'],
      dtype='object')

# Region_America has been dropped
Index(['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'life', 'child_mortality', 'Region_East Asia & Pacific',
       'Region_Europe & Central Asia', 'Region_Middle East & North Africa',
       'Region_South Asia', 'Region_Sub-Saharan Africa'],
      dtype='object')


df_region.head()
   population  fertility  HIV        CO2  BMI_male      GDP  BMI_female  life  \
0  34811059.0       2.73  0.1   3.328945  24.59620  12314.0    129.9049  75.3
1  19842251.0       6.43  2.0   1.474353  22.25083   7103.0    130.1247  58.3
2  40381860.0       2.24  0.5   4.785170  27.50170  14646.0    118.8915  75.5

   child_mortality  Region_East Asia & Pacific  Region_Europe & Central Asia  \
0             29.5                           0                             0
1            192.0                           0                             0
2             15.4                           0                             0

   Region_Middle East & North Africa  Region_South Asia  \
0                                  1                  0
1                                  0                  0
2                                  0                  0

   Region_Sub-Saharan Africa
0                          0
1                          1
2                          0

```


####
**Regression with categorical features**




```python

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)

# Print the cross-validated scores
print(ridge_cv)
[ 0.86808336  0.80623545  0.84004203  0.7754344   0.87503712]

```




---


###
**Handling missing data**


####
**Dropping missing data**



 Now, it’s time for you to take care of these yourself!




 The unprocessed dataset has been loaded into a DataFrame
 `df`
 . Explore it in the IPython Shell with the
 `.head()`
 method. You will see that there are certain data points labeled with a
 `'?'`
 . These denote missing values. As you saw in the video, different datasets encode missing values in different ways. Sometimes it may be a
 `'9999'`
 , other times a
 `0`
 – real-world data can be very messy! If you’re lucky, the missing values will already be encoded as
 `NaN`
 . We use
 `NaN`
 because it is an efficient and simplified way of internally representing missing data, and it lets us take advantage of pandas methods such as
 `.dropna()`
 and
 `.fillna()`
 , as well as scikit-learn’s Imputation transformer
 `Imputer()`
 .




 In this exercise, your job is to convert the
 `'?'`
 s to NaNs, and then drop the rows that contain them from the DataFrame.





```

df.head(3)
        party infants water budget physician salvador religious satellite aid  \
0  republican       0     1      0         1        1         1         0   0
1  republican       0     1      0         1        1         1         0   0
2    democrat       ?     1      1         ?        1         1         0   0

  missile immigration synfuels education superfund crime duty_free_exports  \
0       0           1        ?         1         1     1                 0
1       0           0        0         1         1     1                 0
2       0           0        1         0         1     1                 0

  eaa_rsa
0       1
1       ?
2       0

```




```python

# Convert '?' to NaN
df[df == '?'] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))


```




```

party                  0
infants               12
water                 48
budget                11
physician             11
salvador              15
religious             11
satellite             14
aid                   15
missile               22
immigration            7
synfuels              21
education             31
superfund             25
crime                 17
duty_free_exports     28
eaa_rsa              104
dtype: int64
Shape of Original DataFrame: (435, 17)


Shape of DataFrame After Dropping All Rows with Missing Values: (232, 17)

```



 When many values in your dataset are missing, if you drop them, you may end up throwing away valuable information along with the missing data. It’s better instead to develop an imputation strategy. This is where domain knowledge is useful, but in the absence of it, you can impute missing values with the mean or the median of the row or column that the missing value is in.



####
**Imputing missing data in a ML Pipeline I**



 As you’ve come to appreciate, there are many steps to building a model, from creating training and test sets, to fitting a classifier or regressor, to tuning its parameters, to evaluating its performance on new data. Imputation can be seen as the first step of this machine learning process, the entirety of which can be viewed within the context of a pipeline. Scikit-learn provides a pipeline constructor that allows you to piece together these steps into one process and thereby simplify your workflow.




 You’ll now practice setting up a pipeline with two steps: the imputation step, followed by the instantiation of a classifier. You’ve seen three classifiers in this course so far: k-NN, logistic regression, and the decision tree. You will now be introduced to a fourth one – the Support Vector Machine, or
 [SVM](http://scikit-learn.org/stable/modules/svm.html)
 .





```python

# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
# axis=0 for column
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]

```


####
**Imputing missing data in a ML Pipeline II**



 Having setup the steps of the pipeline in the previous exercise, you will now use it on the voting dataset to classify a Congressman’s party affiliation.




 What makes pipelines so incredibly useful is the simple interface that they provide. You can use the
 `.fit()`
 and
 `.predict()`
 methods on pipelines just as you did with your classifiers and regressors!





```python

# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))


```




```

             precision    recall  f1-score   support

   democrat       0.99      0.96      0.98        85
 republican       0.94      0.98      0.96        46

avg / total       0.97      0.97      0.97       131

```


###
**Centering and scaling**


####
**Centering and scaling your data**



 You will now explore scaling for yourself on a new dataset –
 [White Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
 !




 We have used the
 `'quality'`
 feature of the wine to create a binary target variable: If
 `'quality'`
 is less than
 `5`
 , the target variable is
 `1`
 , and otherwise, it is
 `0`
 .




 Notice how some features seem to have different units of measurement.
 `'density'`
 , for instance, takes values between 0.98 and 1.04, while
 `'total sulfur dioxide'`
 ranges from 9 to 440. As a result, it may be worth scaling the features here. Your job in this exercise is to scale the features and compute the mean and standard deviation of the unscaled features compared to the scaled features.





```python

# white wine quality data set
df.head(3)
   fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
0            7.0              0.27         0.36            20.7      0.045
1            6.3              0.30         0.34             1.6      0.049
2            8.1              0.28         0.40             6.9      0.050

   free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
0                 45.0                 170.0   1.0010  3.00       0.45
1                 14.0                 132.0   0.9940  3.30       0.49
2                 30.0                  97.0   0.9951  3.26       0.44

   alcohol  quality
0      8.8        6
1      9.5        6
2     10.1        6

```




```python

# Import scale
from sklearn.preprocessing import scale

# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X)))
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled)))
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))

```




```

Mean of Unscaled Features: 18.432687072460002
Standard Deviation of Unscaled Features: 41.54494764094571

Mean of Scaled Features: 2.7314972981668206e-15
Standard Deviation of Scaled Features: 0.9999999999999999

```


####
**Centering and scaling in a pipeline**



 With regard to whether or not scaling is effective, the proof is in the pudding! See for yourself whether or not scaling the features of the White Wine Quality dataset has any impact on its performance.




 You will use a k-NN classifier as part of a pipeline that includes scaling, and for the purposes of comparison, a k-NN classifier trained on the unscaled data has been provided.





```python

# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))

Accuracy with Scaling: 0.7700680272108843
Accuracy without Scaling: 0.6979591836734694

```



 It looks like scaling has significantly improved model performance!





---


####
**Bringing it all together I: Pipeline for classification**



 It is time now to piece together everything you have learned so far into a pipeline for classification! Your job in this exercise is to build a pipeline that includes scaling and hyperparameter tuning to classify wine quality.




 You’ll return to using the SVM classifier you were briefly introduced to earlier in this chapter. The hyperparameters you will tune are C and gamma. C controls the regularization strength. It is analogous to the C you tuned for logistic regression in Chapter 3, while gamma controls the kernel coefficient:





```python

# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))


```




```

Accuracy: 0.7795918367346939
             precision    recall  f1-score   support

      False       0.83      0.85      0.84       662
       True       0.67      0.63      0.65       318

avg / total       0.78      0.78      0.78       980

Tuned Model Parameters: {'SVM__C': 10, 'SVM__gamma': 0.1}

```


####
**Bringing it all together II: Pipeline for regression**



 Your job is to build a pipeline that imputes the missing data, scales the features, and fits an ElasticNet to the Gapminder data. You will then tune the
 `l1_ratio`
 of your ElasticNet using GridSearchCV.





```python

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))

#    Tuned ElasticNet Alpha: {'elasticnet__l1_ratio': 1.0}
#    Tuned ElasticNet R squared: 0.8862016570888217

```



 The End.


 Thank you for reading.



---



 1. Classification
-------------------




###
**Machine learning introduction**



**What is machine learning?**
 Giving computers the ability to learn to make decisions from data without being explicitly programmed




**Examples of machine learning:**
 Learning to predict whether an email is spam or not (supervised)


 Clustering Wikipedia entries into different categories (unsupervised)



####
**Types of Machine Learning**


* supervised learning
* unsupervised learning
* reinforcement learning



**Supervised learning:**


 Predictor variables/
 **features**
 and a
 **target variable**




 Aim: Predict the target variable, given the predictor variables


 Classification: Target variable consists of categories


 Regression: Target variable is continuous




**Unsupervised learning:**


 Uncovering hidden patterns from unlabeled data




 Example of unsupervised learning:


 Grouping customers into distinct categories (Clustering)




**Reinforcement learning:**
 Software agents interact with an environment


 Learn how to optimize their behavior


 Given a system of rewards and punishments




 Applications


 Economics


 Genetics


 Game playing (AlphaGo)



####
 Naming conventions


* Features = predictor variables = independent variables
* Target variable = dependent variable = response variable




---


####
 Features of
 **Supervised learning**


* Automate time-consuming or expensive manual tasks (ex. Doctor’s diagnosis)
* Make predictions about the future (ex. Will a customer click on an ad or not)
* Need labeled data (Historical data with labels etc.)


####
**Popular libraries**


* scikit-learning (basic)
* TensorFlow
* keras




---


###
**Exploratory data analysis**


####
**Numerical EDA**



 In this chapter, you’ll be working with a dataset obtained from the
 [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records)
 consisting of votes made by US House of Representatives Congressmen.




 Your goal will be to predict their party affiliation (‘Democrat’ or ‘Republican’) based on how they voted on certain key issues.




 Here, it’s worth noting that we have preprocessed this dataset to deal with missing values. This is so that your focus can be directed towards understanding how to train and evaluate supervised learning models.




 Once you have mastered these fundamentals, you will be introduced to preprocessing techniques in Chapter 4 and have the chance to apply them there yourself – including on this very same dataset!




 Before thinking about what supervised learning models you can apply to this, however, you need to perform Exploratory data analysis (EDA) in order to understand the structure of the data.





```

df.head()
        party  infants  water  budget  physician  salvador  religious  \
0  republican        0      1       0          1         1          1
1  republican        0      1       0          1         1          1
2    democrat        0      1       1          0         1          1
3    democrat        0      1       1          0         1          1
4    democrat        1      1       1          0         1          1

   satellite  aid  missile  immigration  synfuels  education  superfund  \
0          0    0        0            1         0          1          1
1          0    0        0            0         0          1          1
2          0    0        0            0         1          0          1
3          0    0        0            0         1          0          1
4          0    0        0            0         1          0          1

   crime  duty_free_exports  eaa_rsa
0      1                  0        1
1      1                  0        1
2      1                  0        0
3      0                  0        1
4      1                  1        1

```




```

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 435 entries, 0 to 434
Data columns (total 17 columns):
party                435 non-null object
infants              435 non-null int64
water                435 non-null int64
budget               435 non-null int64
physician            435 non-null int64
salvador             435 non-null int64
religious            435 non-null int64
satellite            435 non-null int64
aid                  435 non-null int64
missile              435 non-null int64
immigration          435 non-null int64
synfuels             435 non-null int64
education            435 non-null int64
superfund            435 non-null int64
crime                435 non-null int64
duty_free_exports    435 non-null int64
eaa_rsa              435 non-null int64
dtypes: int64(16), object(1)
memory usage: 57.9+ KB

```


###
**Visual EDA**



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture-21.png)


 Above is a
 `countplot`
 of the
 `'education'`
 bill, generated from the following code:





```

plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

```



 In
 `sns.countplot()`
 , we specify the x-axis data to be
 `'education'`
 , and hue to be
 `'party'`
 . Recall that
 `'party'`
 is also our target variable. So the resulting plot shows the difference in voting behavior between the two parties for the
 `'education'`
 bill, with each party colored differently. We manually specified the color to be
 `'RdBu'`
 , as the Republican party has been traditionally associated with red, and the Democratic party with blue.




 It seems like Democrats voted resoundingly
 *against*
 this bill, compared to Republicans.





```

plt.figure()
sns.countplot(x='missile', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture1-20.png)


 Democrats vote resoundingly in
 *favor*
 of missile, compared to Republicans.





---


###
**The classification challenge**


####
**k-Nearest Neighbors: Fit**



[k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)





```python

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

```


####
**k-Nearest Neighbors: Predict**




```python

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party']
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))

# Prediction: ['democrat']

```



 How sure can you be of its predictions? In other words, how can you measure its performance?





---


###
**Measuring model performance**


####
**The digits recognition dataset: MNIST**



 In the following exercises, you’ll be working with the
 [MNIST](http://yann.lecun.com/exdb/mnist/)
 digits recognition dataset, which has 10 classes, the digits 0 through 9! A reduced version of the MNIST dataset is one of scikit-learn’s included datasets, and that is the one we will use in this exercise.




 Each sample in this scikit-learn dataset is an 8×8 image representing a handwritten digit. Each pixel is represented by an integer in the range 0 to 16, indicating varying levels of black.




 It is a famous dataset in machine learning and computer vision, and frequently used as a benchmark to evaluate the performance of a new model.





```python

# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
#dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])

print(digits.DESCR)
/*
Optical Recognition of Handwritten Digits Data Set
===================================================

Notes
-----
Data Set Characteristics:
    :Number of Instances: 5620
    :Number of Attributes: 64
    :Attribute Information: 8x8 image of integer pixels in the range 0..16.
    :Missing Attribute Values: None
    :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
    :Date: July; 1998
...
*/

# Print the shape of the images and data keys
print(digits.images.shape)
(1797, 8, 8)

print(digits.data.shape)
(1797, 64)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture2-19.png)

####
**Train/Test Split + Fit/Predict/Accuracy**




```python

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
#  Stratify the split according to the labels so that they are distributed in the training and test sets as they are in the original dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))
# 0.983333333333

```



 Incredibly, this out of the box k-NN classifier with 7 neighbors has learned from the training data and predicted the labels of the images in the test set with 98% accuracy, and it did so in less than a second! This is one illustration of how incredibly useful machine learning techniques can be.





---


####
**Overfitting and underfitting**



 In this exercise, you will compute and plot the training and testing accuracy scores for a variety of different neighbor values. By observing how the accuracy scores differ for the training and testing sets with different values of k, you will develop your intuition for overfitting and underfitting.





```python

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture3-18.png)


 It looks like the test accuracy is highest when using 3 and 5 neighbors. Using 8 neighbors or more seems to result in a simple model that underfits the data.





---



# **2. Regression**
------------------


###
**Introduction to regression**


####
**Importing data for supervised learning**



 In this chapter, you will work with
 [Gapminder](https://www.gapminder.org/data/)
 data that we have consolidated into one CSV file available in the workspace as
 `'gapminder.csv'`
 . Specifically, your goal will be to use this data to predict the life expectancy in a given country based on features such as the country’s GDP, fertility rate, and population.





```python

# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Dimensions of y before reshaping: (139,)
# Dimensions of X before reshaping: (139,)


# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

# Dimensions of y after reshaping: (139, 1)
# Dimensions of X after reshaping: (139, 1)

```


####
**Exploring the Gapminder data**




```

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 139 entries, 0 to 138
Data columns (total 9 columns):
population         139 non-null float64
fertility          139 non-null float64
HIV                139 non-null float64
CO2                139 non-null float64
BMI_male           139 non-null float64
GDP                139 non-null float64
BMI_female         139 non-null float64
life               139 non-null float64
child_mortality    139 non-null float64
dtypes: float64(9)
memory usage: 9.9 KB

```




```

sns.heatmap(df.corr(), square=True, cmap='RdYlGn')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture4-17.png)

###
**The basics of linear regression**



 We suppose that y and x have a linear relationship that can be model by


 y = ax + b


 An linear regression is to find a, b that minimize the sum of the squared residual (= Ordinary Least Squares, OLS)




 Why squared residual?


 Residuals may be positive and negative.


 They cancel each other. square residual can solve this problem.




![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture6-11.png)

 green lines are residuals



 When we have n variables of x,


 y = a1*x1 + a2*x2 + … an*xn + b


 we find a1, a2, … an, b that minimize the sum of the squared residual.



####
**Fit & predict for regression**



 In this exercise, you will use the
 `'fertility'`
 feature of the Gapminder dataset. Since the goal is to predict life expectancy, the target variable here is
 `'life'`
 .


 You will also compute and print the R2 score using sckit-learn’s
 `.score()`
 method.





```python

# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2
print(reg.score(X_fertility, y))
0.619244216774

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture7-12.png)

####
**Train/test split for regression**



 In this exercise, you will split the Gapminder dataset into training and testing sets, and then fit and predict a linear regression over
 **all**
 features. In addition to computing the R2 score, you will also compute the Root Mean Squared Error (RMSE), which is another commonly used metric to evaluate regression models.





```python

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
# R^2: 0.838046873142936


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
# Root Mean Squared Error: 3.2476010800377213


```


###
**Cross-validation**



 What is cross validation?


<https://en.wikipedia.org/wiki/Cross-validation_(statistics)>



####
**5-fold cross-validation**



 In this exercise, you will practice 5-fold cross validation on the Gapminder data. By default, scikit-learn’s
 `cross_val_score()`
 function uses R2R2 as the metric of choice for regression.





```python

# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)
# [ 0.81720569  0.82917058  0.90214134  0.80633989  0.94495637]

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
# Average 5-Fold CV Score: 0.8599627722793232

```


####
**K-Fold CV comparison**



 Cross validation is essential but do not forget that the more folds you use, the more computationally expensive cross-validation becomes. In this exercise, you will explore this for yourself. Your job is to perform 3-fold cross-validation and then 10-fold cross-validation on the Gapminder dataset.





```python

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg, X, y, cv=3)
print(np.mean(cvscores_3))
# 0.871871278262

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg, X, y, cv=10)
print(np.mean(cvscores_10))
# 0.843612862013

```




```

%timeit cross_val_score(reg, X, y, cv=3)
100 loops, best of 3: 8.73 ms per loop

%timeit cross_val_score(reg, X, y, cv=10)
10 loops, best of 3: 27.5 ms per loop

```




---


###
**Regularized regression**


####
**[Regularization I: Lasso](https://en.wikipedia.org/wiki/Lasso_(statistics))**



 In this exercise, you will fit a lasso regression to the Gapminder data you have been working with and plot the coefficients. Just as with the Boston data, you will find that the coefficients of some features are shrunk to 0, with only the most important ones remaining.





```

df.columns
Index(['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'life', 'child_mortality'],
      dtype='object')

X: ['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'child_mortality']
y: life

```




```python

# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
#  [-0.         -0.         -0.          0.          0.          0.         -0.
     -0.07087587]


# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture8-9.png)


 According to the lasso algorithm, it seems like
 `'child_mortality'`
 is the most important feature when predicting life expectancy.



####
**Regularization II: Ridge**



 Lasso is great for feature selection, but when building regression models, Ridge regression should be your first choice.




 Recall that lasso performs regularization by adding to the loss function a penalty term of the
 *absolute*
 value of each coefficient multiplied by some alpha. This is also known as L1L1 regularization because the regularization term is the L1L1 norm of the coefficients. This is not the only way to regularize, however.




 If instead you took the sum of the
 *squared*
 values of the coefficients multiplied by some alpha – like in Ridge regression – you would be computing the L2L2norm. In this exercise, you will practice fitting ridge regression models over a range of different alphas, and plot cross-validated R2R2 scores for each, using this function that we have defined for you, which plots the R2R2 score as well as standard error for each alpha:





```

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


```



 Don’t worry about the specifics of the above function works. The motivation behind this exercise is for you to see how the R2R2 score varies with different alphas, and to understand the importance of selecting the right value for alpha. You’ll learn how to tune alpha in the next chapter.





```python

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha

    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)

    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))

    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture9-8.png)


 Notice how the cross-validation scores change with different alphas.





---



# **3. Fine-tuning your model**
------------------------------


###
**confusion matrix**



 What is confusion matrix


<https://en.wikipedia.org/wiki/Confusion_matrix>




![{\displaystyle \mathrm {TPR} ={\frac {\mathrm {TP} }{\mathrm {P} }}={\frac {\mathrm {TP} }{\mathrm {TP} +\mathrm {FN} }}=1-\mathrm {FNR} }](https://wikimedia.org/api/rest_v1/media/math/render/svg/f02ea353bf60bfdd9557d2c98fe18c34cd8db835)

[sensitivity](https://en.wikipedia.org/wiki/Sensitivity_(test))
 ,
 [recall](https://en.wikipedia.org/wiki/Precision_and_recall#Recall)
 ,
 [hit rate](https://en.wikipedia.org/wiki/Hit_rate)
 , or
 [true positive rate](https://en.wikipedia.org/wiki/Sensitivity_(test))
 (TPR)



![{\displaystyle \mathrm {TNR} ={\frac {\mathrm {TN} }{\mathrm {N} }}={\frac {\mathrm {TN} }{\mathrm {TN} +\mathrm {FP} }}=1-\mathrm {FPR} }](https://wikimedia.org/api/rest_v1/media/math/render/svg/8f2c867f0641e498ec8a59de63697a3a45d66b07)

[specificity](https://en.wikipedia.org/wiki/Specificity_(tests))
 ,
 [selectivity](https://en.wikipedia.org/wiki/Specificity_(tests))
 or
 [true negative rate](https://en.wikipedia.org/wiki/Specificity_(tests))
 (TNR)



![{\displaystyle \mathrm {PPV} ={\frac {\mathrm {TP} }{\mathrm {TP} +\mathrm {FP} }}=1-\mathrm {FDR} }](https://wikimedia.org/api/rest_v1/media/math/render/svg/d854b1544fc77735d575ce0d30e34d7f1eacf707)

[precision](https://en.wikipedia.org/wiki/Information_retrieval#Precision)
 or
 [positive predictive value](https://en.wikipedia.org/wiki/Positive_predictive_value)
 (PPV)



![{\displaystyle \mathrm {ACC} ={\frac {\mathrm {TP} +\mathrm {TN} }{\mathrm {P} +\mathrm {N} }}={\frac {\mathrm {TP} +\mathrm {TN} }{\mathrm {TP} +\mathrm {TN} +\mathrm {FP} +\mathrm {FN} }}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/47deb47eb7ac214423d0a6afd05ec0af362fef9b)

[accuracy](https://en.wikipedia.org/wiki/Accuracy)
 (ACC)



![{\displaystyle \mathrm {F} _{1}=2\cdot {\frac {\mathrm {PPV} \cdot \mathrm {TPR} }{\mathrm {PPV} +\mathrm {TPR} }}={\frac {2\mathrm {TP} }{2\mathrm {TP} +\mathrm {FP} +\mathrm {FN} }}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/0e5f071c6418f444fadc9f5f9b0358beed3e094c)

[F1 score](https://en.wikipedia.org/wiki/F1_score)
 is the
 [harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean#Harmonic_mean_of_two_numbers)
 of
 [precision](https://en.wikipedia.org/wiki/Information_retrieval#Precision)
 and
 [sensitivity](https://en.wikipedia.org/wiki/Sensitivity_(test))


####
**illustration for TPR, TNR and PPV**



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/metric_example.png?w=1024)

[source](http://corysimon.github.io/articles/classification-metrics/)


####
**Metrics for classification**



 In this exercise, you will dive more deeply into evaluating the performance of binary classifiers by computing a confusion matrix and generating a classification report.




 Here, you’ll work with the
 [PIMA Indians](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
 dataset obtained from the UCI Machine Learning Repository. The goal is to predict whether or not a given female patient will contract diabetes based on features such as BMI, age, and number of pregnancies.




 Therefore, it is a binary classification problem. A target value of
 `0`
 indicates that the patient does
 *not*
 have diabetes, while a value of
 `1`
 indicates that the patient
 *does*
 have diabetes.





```

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
pregnancies    768 non-null int64
glucose        768 non-null int64
diastolic      768 non-null int64
triceps        768 non-null float64
insulin        768 non-null float64
bmi            768 non-null float64
dpf            768 non-null float64
age            768 non-null int64
diabetes       768 non-null int64
dtypes: float64(4), int64(5)
memory usage: 54.1 KB

```




```python

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

```




```

[[176  30]
 [ 52  50]]

             precision    recall  f1-score   support

          0       0.77      0.85      0.81       206
          1       0.62      0.49      0.55       102

avg / total       0.72      0.73      0.72       308

```




---


###
**Logistic regression and the ROC curve**



 What is logistic regression?


<https://en.wikipedia.org/wiki/Logistic_regression>




 What is ROC?


[Receiver operating characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)




 Further Reading:
 [scikit-learn document](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py)




![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/roc_space-2.png?w=1024)

####
**Building a logistic regression model**




```

X.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 8 columns):
pregnancies    768 non-null int64
glucose        768 non-null int64
diastolic      768 non-null int64
triceps        768 non-null float64
insulin        768 non-null float64
bmi            768 non-null float64
dpf            768 non-null float64
age            768 non-null int64
dtypes: float64(4), int64(4)
memory usage: 48.1 KB


y
0      1
1      0
2      1
      ..

765    0
766    1
767    0
Name: diabetes, dtype: int64

```




```python

# Import the necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


```




```

[[176  30]
 [ 35  67]]

             precision    recall  f1-score   support

          0       0.83      0.85      0.84       206
          1       0.69      0.66      0.67       102

avg / total       0.79      0.79      0.79       308

```


####
**Plotting an ROC curve**

**.predict_proba()**




```python

# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

```




```

logreg.predict_proba(X_test)
# False, True
# 0, 1
# Negative, Positive
array([[ 0.60409835,  0.39590165],
       [ 0.76042394,  0.23957606],
       [ 0.79670177,  0.20329823],
       ...
       [ 0.84686912,  0.15313088],
       [ 0.97617225,  0.02382775],
       [ 0.40380502,  0.59619498]])

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture1-21.png?w=1024)

####
**Precision-recall Curve**



 There are other ways to visually evaluate model performance. One such way is the precision-recall curve, which is generated by plotting the precision and recall for different thresholds.


 Note that here, the class is positive (1) if the individual
 *has*
 diabetes.




![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture2-20.png?w=1024)

* A recall of 1 corresponds to a classifier with a low threshold in which
 *all*
 females who contract diabetes were correctly classified as such, at the expense of many misclassifications of those who did
 *not*
 have diabetes.
* Precision is undefined for a classifier which makes
 *no*
 positive predictions, that is, classifies
 *everyone*
 as
 *not*
 having diabetes.
* When the threshold is very close to 1, precision is also 1, because the classifier is absolutely certain about its predictions.



 recall or sensitivity, TPR = 1 means all true positive are detected. We can predict all to positive to get a recall of 1.




 precision, PPV = 1 means no false positive are detected. We can predict less positive to get a higher precision.



####
**Area under the ROC curve**


####
**AUC(**
 Area Under the Curve
 **) computation**




```python

# diabetes data set
df.head()
   pregnancies  glucose  diastolic   triceps     insulin   bmi    dpf  age  \
0            6      148         72  35.00000  155.548223  33.6  0.627   50
1            1       85         66  29.00000  155.548223  26.6  0.351   31
2            8      183         64  29.15342  155.548223  23.3  0.672   32
3            1       89         66  23.00000   94.000000  28.1  0.167   21
4            0      137         40  35.00000  168.000000  43.1  2.288   33

   diabetes
0         1
1         0
2         1
3         0
4         1

```




```python

# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))


```




```

AUC: 0.8254806777079764

AUC scores computed using 5-fold cross-validation:
[ 0.80148148  0.8062963   0.81481481  0.86245283  0.8554717 ]

```




---


###
**Hyperparameter tuning**


####
**Hyperparameter tuning with GridSearchCV**



 You will now practice this yourself, but by using logistic regression on the diabetes dataset.




 Like the alpha parameter of lasso and ridge regularization that you saw earlier, logistic regression also has a regularization parameter: CC. CC controls the
 *inverse*
 of the regularization strength, and this is what you will tune in this exercise. A large CC can lead to an
 *overfit*
 model, while a small CC can lead to an
 *underfit*
 model.




 The hyperparameter space for CC has been setup for you. Your job is to use GridSearchCV and logistic regression to find the optimal CC in this hyperparameter space.





```python

# diabetes data set

```




```python

# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))


```




```

Tuned Logistic Regression Parameters: {'C': 3.7275937203149381}
Best score is 0.7708333333333334

```


####
**Hyperparameter tuning with RandomizedSearchCV**



 GridSearchCV can be computationally expensive, especially if you are searching over a large hyperparameter space and dealing with multiple hyperparameters. A solution to this is to use
 `RandomizedSearchCV`
 , in which not all hyperparameter values are tried out. Instead, a fixed number of hyperparameter settings is sampled from specified probability distributions. You’ll practice using
 `RandomizedSearchCV`
 in this exercise and see how this works.





```python

# diabetes data set

```




```python

# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


```




```

Tuned Decision Tree Parameters: {'criterion': 'entropy', 'max_depth': 3, 'max_features': 7, 'min_samples_leaf': 1}
Best score is 0.7317708333333334

```



 Note that
 `RandomizedSearchCV`
 will never outperform
 `GridSearchCV`
 . Instead, it is valuable because it saves on computation time.



###
**Hold-out set for final evaluation**


####
**Hold-out set in practice I: Classification**



 In addition to CC, logistic regression has a
 `'penalty'`
 hyperparameter which specifies whether to use
 `'l1'`
 or
 `'l2'`
 regularization. Your job in this exercise is to create a hold-out set, tune the
 `'C'`
 and
 `'penalty'`
 hyperparameters of a logistic regression classifier using
 `GridSearchCV`
 on the training set.





```python

# diabetes data set

```




```python

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))


```




```

Tuned Logistic Regression Parameter: {'C': 0.43939705607607948, 'penalty': 'l1'}
Tuned Logistic Regression Accuracy: 0.7652173913043478

```


####
**Hold-out set in practice II: Regression**



 Remember lasso and ridge regression from the previous chapter? Lasso used the L1 penalty to regularize, while ridge used the L2 penalty. There is another type of regularized regression known as the elastic net. In elastic net regularization, the penalty term is a linear combination of the L1 and L2 penalties:




**a∗L1+b∗L2**




 In scikit-learn, this term is represented by the
 `'l1_ratio'`
 parameter: An
 `'l1_ratio'`
 of
 `1`
 corresponds to an L1L1 penalty, and anything lower is a combination of L1L1 and L2L2.




 In this exercise, you will
 `GridSearchCV`
 to tune the
 `'l1_ratio'`
 of an elastic net model trained on the Gapminder data. As in the previous exercise, use a hold-out set to evaluate your model’s performance.





```

df.head()
   population  fertility  HIV        CO2  BMI_male      GDP  BMI_female  life  \
0  34811059.0       2.73  0.1   3.328945  24.59620  12314.0    129.9049  75.3
1  19842251.0       6.43  2.0   1.474353  22.25083   7103.0    130.1247  58.3
2  40381860.0       2.24  0.5   4.785170  27.50170  14646.0    118.8915  75.5
3   2975029.0       1.40  0.1   1.804106  25.35542   7383.0    132.8108  72.5
4  21370348.0       1.96  0.1  18.016313  27.56373  41312.0    117.3755  81.5

   child_mortality
0             29.5
1            192.0
2             15.4
3             20.0
4              5.2


```




```python

# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))


```




```

Tuned ElasticNet l1 ratio: {'l1_ratio': 0.20689655172413793}
Tuned ElasticNet R squared: 0.8668305372460283
Tuned ElasticNet MSE: 10.05791413339844

```




---



**Preprocessing and pipelines**
--------------------------------


###
**Preprocessing data**


####
**Exploring categorical features**




```

df.head()
   population  fertility  HIV        CO2  BMI_male      GDP  BMI_female  life  \
0  34811059.0       2.73  0.1   3.328945  24.59620  12314.0    129.9049  75.3
1  19842251.0       6.43  2.0   1.474353  22.25083   7103.0    130.1247  58.3
2  40381860.0       2.24  0.5   4.785170  27.50170  14646.0    118.8915  75.5
3   2975029.0       1.40  0.1   1.804106  25.35542   7383.0    132.8108  72.5
4  21370348.0       1.96  0.1  18.016313  27.56373  41312.0    117.3755  81.5

   child_mortality                      Region
0             29.5  Middle East & North Africa
1            192.0          Sub-Saharan Africa
2             15.4                     America
3             20.0       Europe & Central Asia
4              5.2         East Asia & Pacific

```




```python

# Import pandas
import pandas as pd

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture3-19.png?w=1024)

####
**Creating dummy variables**




```python

# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df, drop_first=True)

# Print the new columns of df_region
print(df_region.columns)

```




```

Index(['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'life', 'child_mortality', 'Region_America',
       'Region_East Asia & Pacific', 'Region_Europe & Central Asia',
       'Region_Middle East & North Africa', 'Region_South Asia',
       'Region_Sub-Saharan Africa'],
      dtype='object')

# Region_America has been dropped
Index(['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'life', 'child_mortality', 'Region_East Asia & Pacific',
       'Region_Europe & Central Asia', 'Region_Middle East & North Africa',
       'Region_South Asia', 'Region_Sub-Saharan Africa'],
      dtype='object')


df_region.head()
   population  fertility  HIV        CO2  BMI_male      GDP  BMI_female  life  \
0  34811059.0       2.73  0.1   3.328945  24.59620  12314.0    129.9049  75.3
1  19842251.0       6.43  2.0   1.474353  22.25083   7103.0    130.1247  58.3
2  40381860.0       2.24  0.5   4.785170  27.50170  14646.0    118.8915  75.5

   child_mortality  Region_East Asia & Pacific  Region_Europe & Central Asia  \
0             29.5                           0                             0
1            192.0                           0                             0
2             15.4                           0                             0

   Region_Middle East & North Africa  Region_South Asia  \
0                                  1                  0
1                                  0                  0
2                                  0                  0

   Region_Sub-Saharan Africa
0                          0
1                          1
2                          0

```


####
**Regression with categorical features**




```python

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)

# Print the cross-validated scores
print(ridge_cv)
[ 0.86808336  0.80623545  0.84004203  0.7754344   0.87503712]

```




---


###
**Handling missing data**


####
**Dropping missing data**



 Now, it’s time for you to take care of these yourself!




 The unprocessed dataset has been loaded into a DataFrame
 `df`
 . Explore it in the IPython Shell with the
 `.head()`
 method. You will see that there are certain data points labeled with a
 `'?'`
 . These denote missing values. As you saw in the video, different datasets encode missing values in different ways. Sometimes it may be a
 `'9999'`
 , other times a
 `0`
 – real-world data can be very messy! If you’re lucky, the missing values will already be encoded as
 `NaN`
 . We use
 `NaN`
 because it is an efficient and simplified way of internally representing missing data, and it lets us take advantage of pandas methods such as
 `.dropna()`
 and
 `.fillna()`
 , as well as scikit-learn’s Imputation transformer
 `Imputer()`
 .




 In this exercise, your job is to convert the
 `'?'`
 s to NaNs, and then drop the rows that contain them from the DataFrame.





```

df.head(3)
        party infants water budget physician salvador religious satellite aid  \
0  republican       0     1      0         1        1         1         0   0
1  republican       0     1      0         1        1         1         0   0
2    democrat       ?     1      1         ?        1         1         0   0

  missile immigration synfuels education superfund crime duty_free_exports  \
0       0           1        ?         1         1     1                 0
1       0           0        0         1         1     1                 0
2       0           0        1         0         1     1                 0

  eaa_rsa
0       1
1       ?
2       0

```




```python

# Convert '?' to NaN
df[df == '?'] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))


```




```

party                  0
infants               12
water                 48
budget                11
physician             11
salvador              15
religious             11
satellite             14
aid                   15
missile               22
immigration            7
synfuels              21
education             31
superfund             25
crime                 17
duty_free_exports     28
eaa_rsa              104
dtype: int64
Shape of Original DataFrame: (435, 17)


Shape of DataFrame After Dropping All Rows with Missing Values: (232, 17)

```



 When many values in your dataset are missing, if you drop them, you may end up throwing away valuable information along with the missing data. It’s better instead to develop an imputation strategy. This is where domain knowledge is useful, but in the absence of it, you can impute missing values with the mean or the median of the row or column that the missing value is in.



####
**Imputing missing data in a ML Pipeline I**



 As you’ve come to appreciate, there are many steps to building a model, from creating training and test sets, to fitting a classifier or regressor, to tuning its parameters, to evaluating its performance on new data. Imputation can be seen as the first step of this machine learning process, the entirety of which can be viewed within the context of a pipeline. Scikit-learn provides a pipeline constructor that allows you to piece together these steps into one process and thereby simplify your workflow.




 You’ll now practice setting up a pipeline with two steps: the imputation step, followed by the instantiation of a classifier. You’ve seen three classifiers in this course so far: k-NN, logistic regression, and the decision tree. You will now be introduced to a fourth one – the Support Vector Machine, or
 [SVM](http://scikit-learn.org/stable/modules/svm.html)
 .





```python

# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
# axis=0 for column
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]

```


####
**Imputing missing data in a ML Pipeline II**



 Having setup the steps of the pipeline in the previous exercise, you will now use it on the voting dataset to classify a Congressman’s party affiliation.




 What makes pipelines so incredibly useful is the simple interface that they provide. You can use the
 `.fit()`
 and
 `.predict()`
 methods on pipelines just as you did with your classifiers and regressors!





```python

# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))


```




```

             precision    recall  f1-score   support

   democrat       0.99      0.96      0.98        85
 republican       0.94      0.98      0.96        46

avg / total       0.97      0.97      0.97       131

```


###
**Centering and scaling**


####
**Centering and scaling your data**



 You will now explore scaling for yourself on a new dataset –
 [White Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
 !




 We have used the
 `'quality'`
 feature of the wine to create a binary target variable: If
 `'quality'`
 is less than
 `5`
 , the target variable is
 `1`
 , and otherwise, it is
 `0`
 .




 Notice how some features seem to have different units of measurement.
 `'density'`
 , for instance, takes values between 0.98 and 1.04, while
 `'total sulfur dioxide'`
 ranges from 9 to 440. As a result, it may be worth scaling the features here. Your job in this exercise is to scale the features and compute the mean and standard deviation of the unscaled features compared to the scaled features.





```python

# white wine quality data set
df.head(3)
   fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
0            7.0              0.27         0.36            20.7      0.045
1            6.3              0.30         0.34             1.6      0.049
2            8.1              0.28         0.40             6.9      0.050

   free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
0                 45.0                 170.0   1.0010  3.00       0.45
1                 14.0                 132.0   0.9940  3.30       0.49
2                 30.0                  97.0   0.9951  3.26       0.44

   alcohol  quality
0      8.8        6
1      9.5        6
2     10.1        6

```




```python

# Import scale
from sklearn.preprocessing import scale

# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X)))
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled)))
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))

```




```

Mean of Unscaled Features: 18.432687072460002
Standard Deviation of Unscaled Features: 41.54494764094571

Mean of Scaled Features: 2.7314972981668206e-15
Standard Deviation of Scaled Features: 0.9999999999999999

```


####
**Centering and scaling in a pipeline**



 With regard to whether or not scaling is effective, the proof is in the pudding! See for yourself whether or not scaling the features of the White Wine Quality dataset has any impact on its performance.




 You will use a k-NN classifier as part of a pipeline that includes scaling, and for the purposes of comparison, a k-NN classifier trained on the unscaled data has been provided.





```python

# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))

Accuracy with Scaling: 0.7700680272108843
Accuracy without Scaling: 0.6979591836734694

```



 It looks like scaling has significantly improved model performance!





---


####
**Bringing it all together I: Pipeline for classification**



 It is time now to piece together everything you have learned so far into a pipeline for classification! Your job in this exercise is to build a pipeline that includes scaling and hyperparameter tuning to classify wine quality.




 You’ll return to using the SVM classifier you were briefly introduced to earlier in this chapter. The hyperparameters you will tune are C and gamma. C controls the regularization strength. It is analogous to the C you tuned for logistic regression in Chapter 3, while gamma controls the kernel coefficient:





```python

# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))


```




```

Accuracy: 0.7795918367346939
             precision    recall  f1-score   support

      False       0.83      0.85      0.84       662
       True       0.67      0.63      0.65       318

avg / total       0.78      0.78      0.78       980

Tuned Model Parameters: {'SVM__C': 10, 'SVM__gamma': 0.1}

```


####
**Bringing it all together II: Pipeline for regression**



 Your job is to build a pipeline that imputes the missing data, scales the features, and fits an ElasticNet to the Gapminder data. You will then tune the
 `l1_ratio`
 of your ElasticNet using GridSearchCV.





```python

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))

#    Tuned ElasticNet Alpha: {'elasticnet__l1_ratio': 1.0}
#    Tuned ElasticNet R squared: 0.8862016570888217

```



 The End.


 Thank you for reading.



---



 1. Classification
-------------------




###
**Machine learning introduction**



**What is machine learning?**
 Giving computers the ability to learn to make decisions from data without being explicitly programmed




**Examples of machine learning:**
 Learning to predict whether an email is spam or not (supervised)


 Clustering Wikipedia entries into different categories (unsupervised)



####
**Types of Machine Learning**


* supervised learning
* unsupervised learning
* reinforcement learning



**Supervised learning:**


 Predictor variables/
 **features**
 and a
 **target variable**




 Aim: Predict the target variable, given the predictor variables


 Classification: Target variable consists of categories


 Regression: Target variable is continuous




**Unsupervised learning:**


 Uncovering hidden patterns from unlabeled data




 Example of unsupervised learning:


 Grouping customers into distinct categories (Clustering)




**Reinforcement learning:**
 Software agents interact with an environment


 Learn how to optimize their behavior


 Given a system of rewards and punishments




 Applications


 Economics


 Genetics


 Game playing (AlphaGo)



####
 Naming conventions


* Features = predictor variables = independent variables
* Target variable = dependent variable = response variable




---


####
 Features of
 **Supervised learning**


* Automate time-consuming or expensive manual tasks (ex. Doctor’s diagnosis)
* Make predictions about the future (ex. Will a customer click on an ad or not)
* Need labeled data (Historical data with labels etc.)


####
**Popular libraries**


* scikit-learning (basic)
* TensorFlow
* keras




---


###
**Exploratory data analysis**


####
**Numerical EDA**



 In this chapter, you’ll be working with a dataset obtained from the
 [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records)
 consisting of votes made by US House of Representatives Congressmen.




 Your goal will be to predict their party affiliation (‘Democrat’ or ‘Republican’) based on how they voted on certain key issues.




 Here, it’s worth noting that we have preprocessed this dataset to deal with missing values. This is so that your focus can be directed towards understanding how to train and evaluate supervised learning models.




 Once you have mastered these fundamentals, you will be introduced to preprocessing techniques in Chapter 4 and have the chance to apply them there yourself – including on this very same dataset!




 Before thinking about what supervised learning models you can apply to this, however, you need to perform Exploratory data analysis (EDA) in order to understand the structure of the data.





```

df.head()
        party  infants  water  budget  physician  salvador  religious  \
0  republican        0      1       0          1         1          1
1  republican        0      1       0          1         1          1
2    democrat        0      1       1          0         1          1
3    democrat        0      1       1          0         1          1
4    democrat        1      1       1          0         1          1

   satellite  aid  missile  immigration  synfuels  education  superfund  \
0          0    0        0            1         0          1          1
1          0    0        0            0         0          1          1
2          0    0        0            0         1          0          1
3          0    0        0            0         1          0          1
4          0    0        0            0         1          0          1

   crime  duty_free_exports  eaa_rsa
0      1                  0        1
1      1                  0        1
2      1                  0        0
3      0                  0        1
4      1                  1        1

```




```

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 435 entries, 0 to 434
Data columns (total 17 columns):
party                435 non-null object
infants              435 non-null int64
water                435 non-null int64
budget               435 non-null int64
physician            435 non-null int64
salvador             435 non-null int64
religious            435 non-null int64
satellite            435 non-null int64
aid                  435 non-null int64
missile              435 non-null int64
immigration          435 non-null int64
synfuels             435 non-null int64
education            435 non-null int64
superfund            435 non-null int64
crime                435 non-null int64
duty_free_exports    435 non-null int64
eaa_rsa              435 non-null int64
dtypes: int64(16), object(1)
memory usage: 57.9+ KB

```


###
**Visual EDA**



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture-21.png)


 Above is a
 `countplot`
 of the
 `'education'`
 bill, generated from the following code:





```

plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

```



 In
 `sns.countplot()`
 , we specify the x-axis data to be
 `'education'`
 , and hue to be
 `'party'`
 . Recall that
 `'party'`
 is also our target variable. So the resulting plot shows the difference in voting behavior between the two parties for the
 `'education'`
 bill, with each party colored differently. We manually specified the color to be
 `'RdBu'`
 , as the Republican party has been traditionally associated with red, and the Democratic party with blue.




 It seems like Democrats voted resoundingly
 *against*
 this bill, compared to Republicans.





```

plt.figure()
sns.countplot(x='missile', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture1-20.png)


 Democrats vote resoundingly in
 *favor*
 of missile, compared to Republicans.





---


###
**The classification challenge**


####
**k-Nearest Neighbors: Fit**



[k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)





```python

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

```


####
**k-Nearest Neighbors: Predict**




```python

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party']
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))

# Prediction: ['democrat']

```



 How sure can you be of its predictions? In other words, how can you measure its performance?





---


###
**Measuring model performance**


####
**The digits recognition dataset: MNIST**



 In the following exercises, you’ll be working with the
 [MNIST](http://yann.lecun.com/exdb/mnist/)
 digits recognition dataset, which has 10 classes, the digits 0 through 9! A reduced version of the MNIST dataset is one of scikit-learn’s included datasets, and that is the one we will use in this exercise.




 Each sample in this scikit-learn dataset is an 8×8 image representing a handwritten digit. Each pixel is represented by an integer in the range 0 to 16, indicating varying levels of black.




 It is a famous dataset in machine learning and computer vision, and frequently used as a benchmark to evaluate the performance of a new model.





```python

# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
#dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])

print(digits.DESCR)
/*
Optical Recognition of Handwritten Digits Data Set
===================================================

Notes
-----
Data Set Characteristics:
    :Number of Instances: 5620
    :Number of Attributes: 64
    :Attribute Information: 8x8 image of integer pixels in the range 0..16.
    :Missing Attribute Values: None
    :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
    :Date: July; 1998
...
*/

# Print the shape of the images and data keys
print(digits.images.shape)
(1797, 8, 8)

print(digits.data.shape)
(1797, 64)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture2-19.png)

####
**Train/Test Split + Fit/Predict/Accuracy**




```python

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
#  Stratify the split according to the labels so that they are distributed in the training and test sets as they are in the original dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))
# 0.983333333333

```



 Incredibly, this out of the box k-NN classifier with 7 neighbors has learned from the training data and predicted the labels of the images in the test set with 98% accuracy, and it did so in less than a second! This is one illustration of how incredibly useful machine learning techniques can be.





---


####
**Overfitting and underfitting**



 In this exercise, you will compute and plot the training and testing accuracy scores for a variety of different neighbor values. By observing how the accuracy scores differ for the training and testing sets with different values of k, you will develop your intuition for overfitting and underfitting.





```python

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture3-18.png)


 It looks like the test accuracy is highest when using 3 and 5 neighbors. Using 8 neighbors or more seems to result in a simple model that underfits the data.





---



# **2. Regression**
------------------


###
**Introduction to regression**


####
**Importing data for supervised learning**



 In this chapter, you will work with
 [Gapminder](https://www.gapminder.org/data/)
 data that we have consolidated into one CSV file available in the workspace as
 `'gapminder.csv'`
 . Specifically, your goal will be to use this data to predict the life expectancy in a given country based on features such as the country’s GDP, fertility rate, and population.





```python

# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Dimensions of y before reshaping: (139,)
# Dimensions of X before reshaping: (139,)


# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

# Dimensions of y after reshaping: (139, 1)
# Dimensions of X after reshaping: (139, 1)

```


####
**Exploring the Gapminder data**




```

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 139 entries, 0 to 138
Data columns (total 9 columns):
population         139 non-null float64
fertility          139 non-null float64
HIV                139 non-null float64
CO2                139 non-null float64
BMI_male           139 non-null float64
GDP                139 non-null float64
BMI_female         139 non-null float64
life               139 non-null float64
child_mortality    139 non-null float64
dtypes: float64(9)
memory usage: 9.9 KB

```




```

sns.heatmap(df.corr(), square=True, cmap='RdYlGn')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture4-17.png)

###
**The basics of linear regression**



 We suppose that y and x have a linear relationship that can be model by


 y = ax + b


 An linear regression is to find a, b that minimize the sum of the squared residual (= Ordinary Least Squares, OLS)




 Why squared residual?


 Residuals may be positive and negative.


 They cancel each other. square residual can solve this problem.




![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture6-11.png)

 green lines are residuals



 When we have n variables of x,


 y = a1*x1 + a2*x2 + … an*xn + b


 we find a1, a2, … an, b that minimize the sum of the squared residual.



####
**Fit & predict for regression**



 In this exercise, you will use the
 `'fertility'`
 feature of the Gapminder dataset. Since the goal is to predict life expectancy, the target variable here is
 `'life'`
 .


 You will also compute and print the R2 score using sckit-learn’s
 `.score()`
 method.





```python

# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2
print(reg.score(X_fertility, y))
0.619244216774

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture7-12.png)

####
**Train/test split for regression**



 In this exercise, you will split the Gapminder dataset into training and testing sets, and then fit and predict a linear regression over
 **all**
 features. In addition to computing the R2 score, you will also compute the Root Mean Squared Error (RMSE), which is another commonly used metric to evaluate regression models.





```python

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
# R^2: 0.838046873142936


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
# Root Mean Squared Error: 3.2476010800377213


```


###
**Cross-validation**



 What is cross validation?


<https://en.wikipedia.org/wiki/Cross-validation_(statistics)>



####
**5-fold cross-validation**



 In this exercise, you will practice 5-fold cross validation on the Gapminder data. By default, scikit-learn’s
 `cross_val_score()`
 function uses R2R2 as the metric of choice for regression.





```python

# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)
# [ 0.81720569  0.82917058  0.90214134  0.80633989  0.94495637]

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
# Average 5-Fold CV Score: 0.8599627722793232

```


####
**K-Fold CV comparison**



 Cross validation is essential but do not forget that the more folds you use, the more computationally expensive cross-validation becomes. In this exercise, you will explore this for yourself. Your job is to perform 3-fold cross-validation and then 10-fold cross-validation on the Gapminder dataset.





```python

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg, X, y, cv=3)
print(np.mean(cvscores_3))
# 0.871871278262

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg, X, y, cv=10)
print(np.mean(cvscores_10))
# 0.843612862013

```




```

%timeit cross_val_score(reg, X, y, cv=3)
100 loops, best of 3: 8.73 ms per loop

%timeit cross_val_score(reg, X, y, cv=10)
10 loops, best of 3: 27.5 ms per loop

```




---


###
**Regularized regression**


####
**[Regularization I: Lasso](https://en.wikipedia.org/wiki/Lasso_(statistics))**



 In this exercise, you will fit a lasso regression to the Gapminder data you have been working with and plot the coefficients. Just as with the Boston data, you will find that the coefficients of some features are shrunk to 0, with only the most important ones remaining.





```

df.columns
Index(['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'life', 'child_mortality'],
      dtype='object')

X: ['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'child_mortality']
y: life

```




```python

# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
#  [-0.         -0.         -0.          0.          0.          0.         -0.
     -0.07087587]


# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture8-9.png)


 According to the lasso algorithm, it seems like
 `'child_mortality'`
 is the most important feature when predicting life expectancy.



####
**Regularization II: Ridge**



 Lasso is great for feature selection, but when building regression models, Ridge regression should be your first choice.




 Recall that lasso performs regularization by adding to the loss function a penalty term of the
 *absolute*
 value of each coefficient multiplied by some alpha. This is also known as L1L1 regularization because the regularization term is the L1L1 norm of the coefficients. This is not the only way to regularize, however.




 If instead you took the sum of the
 *squared*
 values of the coefficients multiplied by some alpha – like in Ridge regression – you would be computing the L2L2norm. In this exercise, you will practice fitting ridge regression models over a range of different alphas, and plot cross-validated R2R2 scores for each, using this function that we have defined for you, which plots the R2R2 score as well as standard error for each alpha:





```

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


```



 Don’t worry about the specifics of the above function works. The motivation behind this exercise is for you to see how the R2R2 score varies with different alphas, and to understand the importance of selecting the right value for alpha. You’ll learn how to tune alpha in the next chapter.





```python

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha

    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)

    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))

    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture9-8.png)


 Notice how the cross-validation scores change with different alphas.





---



# **3. Fine-tuning your model**
------------------------------


###
**confusion matrix**



 What is confusion matrix


<https://en.wikipedia.org/wiki/Confusion_matrix>




![{\displaystyle \mathrm {TPR} ={\frac {\mathrm {TP} }{\mathrm {P} }}={\frac {\mathrm {TP} }{\mathrm {TP} +\mathrm {FN} }}=1-\mathrm {FNR} }](https://wikimedia.org/api/rest_v1/media/math/render/svg/f02ea353bf60bfdd9557d2c98fe18c34cd8db835)

[sensitivity](https://en.wikipedia.org/wiki/Sensitivity_(test))
 ,
 [recall](https://en.wikipedia.org/wiki/Precision_and_recall#Recall)
 ,
 [hit rate](https://en.wikipedia.org/wiki/Hit_rate)
 , or
 [true positive rate](https://en.wikipedia.org/wiki/Sensitivity_(test))
 (TPR)



![{\displaystyle \mathrm {TNR} ={\frac {\mathrm {TN} }{\mathrm {N} }}={\frac {\mathrm {TN} }{\mathrm {TN} +\mathrm {FP} }}=1-\mathrm {FPR} }](https://wikimedia.org/api/rest_v1/media/math/render/svg/8f2c867f0641e498ec8a59de63697a3a45d66b07)

[specificity](https://en.wikipedia.org/wiki/Specificity_(tests))
 ,
 [selectivity](https://en.wikipedia.org/wiki/Specificity_(tests))
 or
 [true negative rate](https://en.wikipedia.org/wiki/Specificity_(tests))
 (TNR)



![{\displaystyle \mathrm {PPV} ={\frac {\mathrm {TP} }{\mathrm {TP} +\mathrm {FP} }}=1-\mathrm {FDR} }](https://wikimedia.org/api/rest_v1/media/math/render/svg/d854b1544fc77735d575ce0d30e34d7f1eacf707)

[precision](https://en.wikipedia.org/wiki/Information_retrieval#Precision)
 or
 [positive predictive value](https://en.wikipedia.org/wiki/Positive_predictive_value)
 (PPV)



![{\displaystyle \mathrm {ACC} ={\frac {\mathrm {TP} +\mathrm {TN} }{\mathrm {P} +\mathrm {N} }}={\frac {\mathrm {TP} +\mathrm {TN} }{\mathrm {TP} +\mathrm {TN} +\mathrm {FP} +\mathrm {FN} }}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/47deb47eb7ac214423d0a6afd05ec0af362fef9b)

[accuracy](https://en.wikipedia.org/wiki/Accuracy)
 (ACC)



![{\displaystyle \mathrm {F} _{1}=2\cdot {\frac {\mathrm {PPV} \cdot \mathrm {TPR} }{\mathrm {PPV} +\mathrm {TPR} }}={\frac {2\mathrm {TP} }{2\mathrm {TP} +\mathrm {FP} +\mathrm {FN} }}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/0e5f071c6418f444fadc9f5f9b0358beed3e094c)

[F1 score](https://en.wikipedia.org/wiki/F1_score)
 is the
 [harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean#Harmonic_mean_of_two_numbers)
 of
 [precision](https://en.wikipedia.org/wiki/Information_retrieval#Precision)
 and
 [sensitivity](https://en.wikipedia.org/wiki/Sensitivity_(test))


####
**illustration for TPR, TNR and PPV**



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/metric_example.png?w=1024)

[source](http://corysimon.github.io/articles/classification-metrics/)


####
**Metrics for classification**



 In this exercise, you will dive more deeply into evaluating the performance of binary classifiers by computing a confusion matrix and generating a classification report.




 Here, you’ll work with the
 [PIMA Indians](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
 dataset obtained from the UCI Machine Learning Repository. The goal is to predict whether or not a given female patient will contract diabetes based on features such as BMI, age, and number of pregnancies.




 Therefore, it is a binary classification problem. A target value of
 `0`
 indicates that the patient does
 *not*
 have diabetes, while a value of
 `1`
 indicates that the patient
 *does*
 have diabetes.





```

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
pregnancies    768 non-null int64
glucose        768 non-null int64
diastolic      768 non-null int64
triceps        768 non-null float64
insulin        768 non-null float64
bmi            768 non-null float64
dpf            768 non-null float64
age            768 non-null int64
diabetes       768 non-null int64
dtypes: float64(4), int64(5)
memory usage: 54.1 KB

```




```python

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

```




```

[[176  30]
 [ 52  50]]

             precision    recall  f1-score   support

          0       0.77      0.85      0.81       206
          1       0.62      0.49      0.55       102

avg / total       0.72      0.73      0.72       308

```




---


###
**Logistic regression and the ROC curve**



 What is logistic regression?


<https://en.wikipedia.org/wiki/Logistic_regression>




 What is ROC?


[Receiver operating characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)




 Further Reading:
 [scikit-learn document](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py)




![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/roc_space-2.png?w=1024)

####
**Building a logistic regression model**




```

X.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 8 columns):
pregnancies    768 non-null int64
glucose        768 non-null int64
diastolic      768 non-null int64
triceps        768 non-null float64
insulin        768 non-null float64
bmi            768 non-null float64
dpf            768 non-null float64
age            768 non-null int64
dtypes: float64(4), int64(4)
memory usage: 48.1 KB


y
0      1
1      0
2      1
      ..

765    0
766    1
767    0
Name: diabetes, dtype: int64

```




```python

# Import the necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


```




```

[[176  30]
 [ 35  67]]

             precision    recall  f1-score   support

          0       0.83      0.85      0.84       206
          1       0.69      0.66      0.67       102

avg / total       0.79      0.79      0.79       308

```


####
**Plotting an ROC curve**

**.predict_proba()**




```python

# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

```




```

logreg.predict_proba(X_test)
# False, True
# 0, 1
# Negative, Positive
array([[ 0.60409835,  0.39590165],
       [ 0.76042394,  0.23957606],
       [ 0.79670177,  0.20329823],
       ...
       [ 0.84686912,  0.15313088],
       [ 0.97617225,  0.02382775],
       [ 0.40380502,  0.59619498]])

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture1-21.png?w=1024)

####
**Precision-recall Curve**



 There are other ways to visually evaluate model performance. One such way is the precision-recall curve, which is generated by plotting the precision and recall for different thresholds.


 Note that here, the class is positive (1) if the individual
 *has*
 diabetes.




![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture2-20.png?w=1024)

* A recall of 1 corresponds to a classifier with a low threshold in which
 *all*
 females who contract diabetes were correctly classified as such, at the expense of many misclassifications of those who did
 *not*
 have diabetes.
* Precision is undefined for a classifier which makes
 *no*
 positive predictions, that is, classifies
 *everyone*
 as
 *not*
 having diabetes.
* When the threshold is very close to 1, precision is also 1, because the classifier is absolutely certain about its predictions.



 recall or sensitivity, TPR = 1 means all true positive are detected. We can predict all to positive to get a recall of 1.




 precision, PPV = 1 means no false positive are detected. We can predict less positive to get a higher precision.



####
**Area under the ROC curve**


####
**AUC(**
 Area Under the Curve
 **) computation**




```python

# diabetes data set
df.head()
   pregnancies  glucose  diastolic   triceps     insulin   bmi    dpf  age  \
0            6      148         72  35.00000  155.548223  33.6  0.627   50
1            1       85         66  29.00000  155.548223  26.6  0.351   31
2            8      183         64  29.15342  155.548223  23.3  0.672   32
3            1       89         66  23.00000   94.000000  28.1  0.167   21
4            0      137         40  35.00000  168.000000  43.1  2.288   33

   diabetes
0         1
1         0
2         1
3         0
4         1

```




```python

# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))


```




```

AUC: 0.8254806777079764

AUC scores computed using 5-fold cross-validation:
[ 0.80148148  0.8062963   0.81481481  0.86245283  0.8554717 ]

```




---


###
**Hyperparameter tuning**


####
**Hyperparameter tuning with GridSearchCV**



 You will now practice this yourself, but by using logistic regression on the diabetes dataset.




 Like the alpha parameter of lasso and ridge regularization that you saw earlier, logistic regression also has a regularization parameter: CC. CC controls the
 *inverse*
 of the regularization strength, and this is what you will tune in this exercise. A large CC can lead to an
 *overfit*
 model, while a small CC can lead to an
 *underfit*
 model.




 The hyperparameter space for CC has been setup for you. Your job is to use GridSearchCV and logistic regression to find the optimal CC in this hyperparameter space.





```python

# diabetes data set

```




```python

# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))


```




```

Tuned Logistic Regression Parameters: {'C': 3.7275937203149381}
Best score is 0.7708333333333334

```


####
**Hyperparameter tuning with RandomizedSearchCV**



 GridSearchCV can be computationally expensive, especially if you are searching over a large hyperparameter space and dealing with multiple hyperparameters. A solution to this is to use
 `RandomizedSearchCV`
 , in which not all hyperparameter values are tried out. Instead, a fixed number of hyperparameter settings is sampled from specified probability distributions. You’ll practice using
 `RandomizedSearchCV`
 in this exercise and see how this works.





```python

# diabetes data set

```




```python

# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


```




```

Tuned Decision Tree Parameters: {'criterion': 'entropy', 'max_depth': 3, 'max_features': 7, 'min_samples_leaf': 1}
Best score is 0.7317708333333334

```



 Note that
 `RandomizedSearchCV`
 will never outperform
 `GridSearchCV`
 . Instead, it is valuable because it saves on computation time.



###
**Hold-out set for final evaluation**


####
**Hold-out set in practice I: Classification**



 In addition to CC, logistic regression has a
 `'penalty'`
 hyperparameter which specifies whether to use
 `'l1'`
 or
 `'l2'`
 regularization. Your job in this exercise is to create a hold-out set, tune the
 `'C'`
 and
 `'penalty'`
 hyperparameters of a logistic regression classifier using
 `GridSearchCV`
 on the training set.





```python

# diabetes data set

```




```python

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))


```




```

Tuned Logistic Regression Parameter: {'C': 0.43939705607607948, 'penalty': 'l1'}
Tuned Logistic Regression Accuracy: 0.7652173913043478

```


####
**Hold-out set in practice II: Regression**



 Remember lasso and ridge regression from the previous chapter? Lasso used the L1 penalty to regularize, while ridge used the L2 penalty. There is another type of regularized regression known as the elastic net. In elastic net regularization, the penalty term is a linear combination of the L1 and L2 penalties:




**a∗L1+b∗L2**




 In scikit-learn, this term is represented by the
 `'l1_ratio'`
 parameter: An
 `'l1_ratio'`
 of
 `1`
 corresponds to an L1L1 penalty, and anything lower is a combination of L1L1 and L2L2.




 In this exercise, you will
 `GridSearchCV`
 to tune the
 `'l1_ratio'`
 of an elastic net model trained on the Gapminder data. As in the previous exercise, use a hold-out set to evaluate your model’s performance.





```

df.head()
   population  fertility  HIV        CO2  BMI_male      GDP  BMI_female  life  \
0  34811059.0       2.73  0.1   3.328945  24.59620  12314.0    129.9049  75.3
1  19842251.0       6.43  2.0   1.474353  22.25083   7103.0    130.1247  58.3
2  40381860.0       2.24  0.5   4.785170  27.50170  14646.0    118.8915  75.5
3   2975029.0       1.40  0.1   1.804106  25.35542   7383.0    132.8108  72.5
4  21370348.0       1.96  0.1  18.016313  27.56373  41312.0    117.3755  81.5

   child_mortality
0             29.5
1            192.0
2             15.4
3             20.0
4              5.2


```




```python

# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))


```




```

Tuned ElasticNet l1 ratio: {'l1_ratio': 0.20689655172413793}
Tuned ElasticNet R squared: 0.8668305372460283
Tuned ElasticNet MSE: 10.05791413339844

```




---



**Preprocessing and pipelines**
--------------------------------


###
**Preprocessing data**


####
**Exploring categorical features**




```

df.head()
   population  fertility  HIV        CO2  BMI_male      GDP  BMI_female  life  \
0  34811059.0       2.73  0.1   3.328945  24.59620  12314.0    129.9049  75.3
1  19842251.0       6.43  2.0   1.474353  22.25083   7103.0    130.1247  58.3
2  40381860.0       2.24  0.5   4.785170  27.50170  14646.0    118.8915  75.5
3   2975029.0       1.40  0.1   1.804106  25.35542   7383.0    132.8108  72.5
4  21370348.0       1.96  0.1  18.016313  27.56373  41312.0    117.3755  81.5

   child_mortality                      Region
0             29.5  Middle East & North Africa
1            192.0          Sub-Saharan Africa
2             15.4                     America
3             20.0       Europe & Central Asia
4              5.2         East Asia & Pacific

```




```python

# Import pandas
import pandas as pd

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture3-19.png?w=1024)

####
**Creating dummy variables**




```python

# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df, drop_first=True)

# Print the new columns of df_region
print(df_region.columns)

```




```

Index(['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'life', 'child_mortality', 'Region_America',
       'Region_East Asia & Pacific', 'Region_Europe & Central Asia',
       'Region_Middle East & North Africa', 'Region_South Asia',
       'Region_Sub-Saharan Africa'],
      dtype='object')

# Region_America has been dropped
Index(['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'life', 'child_mortality', 'Region_East Asia & Pacific',
       'Region_Europe & Central Asia', 'Region_Middle East & North Africa',
       'Region_South Asia', 'Region_Sub-Saharan Africa'],
      dtype='object')


df_region.head()
   population  fertility  HIV        CO2  BMI_male      GDP  BMI_female  life  \
0  34811059.0       2.73  0.1   3.328945  24.59620  12314.0    129.9049  75.3
1  19842251.0       6.43  2.0   1.474353  22.25083   7103.0    130.1247  58.3
2  40381860.0       2.24  0.5   4.785170  27.50170  14646.0    118.8915  75.5

   child_mortality  Region_East Asia & Pacific  Region_Europe & Central Asia  \
0             29.5                           0                             0
1            192.0                           0                             0
2             15.4                           0                             0

   Region_Middle East & North Africa  Region_South Asia  \
0                                  1                  0
1                                  0                  0
2                                  0                  0

   Region_Sub-Saharan Africa
0                          0
1                          1
2                          0

```


####
**Regression with categorical features**




```python

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)

# Print the cross-validated scores
print(ridge_cv)
[ 0.86808336  0.80623545  0.84004203  0.7754344   0.87503712]

```




---


###
**Handling missing data**


####
**Dropping missing data**



 Now, it’s time for you to take care of these yourself!




 The unprocessed dataset has been loaded into a DataFrame
 `df`
 . Explore it in the IPython Shell with the
 `.head()`
 method. You will see that there are certain data points labeled with a
 `'?'`
 . These denote missing values. As you saw in the video, different datasets encode missing values in different ways. Sometimes it may be a
 `'9999'`
 , other times a
 `0`
 – real-world data can be very messy! If you’re lucky, the missing values will already be encoded as
 `NaN`
 . We use
 `NaN`
 because it is an efficient and simplified way of internally representing missing data, and it lets us take advantage of pandas methods such as
 `.dropna()`
 and
 `.fillna()`
 , as well as scikit-learn’s Imputation transformer
 `Imputer()`
 .




 In this exercise, your job is to convert the
 `'?'`
 s to NaNs, and then drop the rows that contain them from the DataFrame.





```

df.head(3)
        party infants water budget physician salvador religious satellite aid  \
0  republican       0     1      0         1        1         1         0   0
1  republican       0     1      0         1        1         1         0   0
2    democrat       ?     1      1         ?        1         1         0   0

  missile immigration synfuels education superfund crime duty_free_exports  \
0       0           1        ?         1         1     1                 0
1       0           0        0         1         1     1                 0
2       0           0        1         0         1     1                 0

  eaa_rsa
0       1
1       ?
2       0

```




```python

# Convert '?' to NaN
df[df == '?'] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))


```




```

party                  0
infants               12
water                 48
budget                11
physician             11
salvador              15
religious             11
satellite             14
aid                   15
missile               22
immigration            7
synfuels              21
education             31
superfund             25
crime                 17
duty_free_exports     28
eaa_rsa              104
dtype: int64
Shape of Original DataFrame: (435, 17)


Shape of DataFrame After Dropping All Rows with Missing Values: (232, 17)

```



 When many values in your dataset are missing, if you drop them, you may end up throwing away valuable information along with the missing data. It’s better instead to develop an imputation strategy. This is where domain knowledge is useful, but in the absence of it, you can impute missing values with the mean or the median of the row or column that the missing value is in.



####
**Imputing missing data in a ML Pipeline I**



 As you’ve come to appreciate, there are many steps to building a model, from creating training and test sets, to fitting a classifier or regressor, to tuning its parameters, to evaluating its performance on new data. Imputation can be seen as the first step of this machine learning process, the entirety of which can be viewed within the context of a pipeline. Scikit-learn provides a pipeline constructor that allows you to piece together these steps into one process and thereby simplify your workflow.




 You’ll now practice setting up a pipeline with two steps: the imputation step, followed by the instantiation of a classifier. You’ve seen three classifiers in this course so far: k-NN, logistic regression, and the decision tree. You will now be introduced to a fourth one – the Support Vector Machine, or
 [SVM](http://scikit-learn.org/stable/modules/svm.html)
 .





```python

# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
# axis=0 for column
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]

```


####
**Imputing missing data in a ML Pipeline II**



 Having setup the steps of the pipeline in the previous exercise, you will now use it on the voting dataset to classify a Congressman’s party affiliation.




 What makes pipelines so incredibly useful is the simple interface that they provide. You can use the
 `.fit()`
 and
 `.predict()`
 methods on pipelines just as you did with your classifiers and regressors!





```python

# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))


```




```

             precision    recall  f1-score   support

   democrat       0.99      0.96      0.98        85
 republican       0.94      0.98      0.96        46

avg / total       0.97      0.97      0.97       131

```


###
**Centering and scaling**


####
**Centering and scaling your data**



 You will now explore scaling for yourself on a new dataset –
 [White Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
 !




 We have used the
 `'quality'`
 feature of the wine to create a binary target variable: If
 `'quality'`
 is less than
 `5`
 , the target variable is
 `1`
 , and otherwise, it is
 `0`
 .




 Notice how some features seem to have different units of measurement.
 `'density'`
 , for instance, takes values between 0.98 and 1.04, while
 `'total sulfur dioxide'`
 ranges from 9 to 440. As a result, it may be worth scaling the features here. Your job in this exercise is to scale the features and compute the mean and standard deviation of the unscaled features compared to the scaled features.





```python

# white wine quality data set
df.head(3)
   fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
0            7.0              0.27         0.36            20.7      0.045
1            6.3              0.30         0.34             1.6      0.049
2            8.1              0.28         0.40             6.9      0.050

   free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
0                 45.0                 170.0   1.0010  3.00       0.45
1                 14.0                 132.0   0.9940  3.30       0.49
2                 30.0                  97.0   0.9951  3.26       0.44

   alcohol  quality
0      8.8        6
1      9.5        6
2     10.1        6

```




```python

# Import scale
from sklearn.preprocessing import scale

# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X)))
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled)))
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))

```




```

Mean of Unscaled Features: 18.432687072460002
Standard Deviation of Unscaled Features: 41.54494764094571

Mean of Scaled Features: 2.7314972981668206e-15
Standard Deviation of Scaled Features: 0.9999999999999999

```


####
**Centering and scaling in a pipeline**



 With regard to whether or not scaling is effective, the proof is in the pudding! See for yourself whether or not scaling the features of the White Wine Quality dataset has any impact on its performance.




 You will use a k-NN classifier as part of a pipeline that includes scaling, and for the purposes of comparison, a k-NN classifier trained on the unscaled data has been provided.





```python

# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))

Accuracy with Scaling: 0.7700680272108843
Accuracy without Scaling: 0.6979591836734694

```



 It looks like scaling has significantly improved model performance!





---


####
**Bringing it all together I: Pipeline for classification**



 It is time now to piece together everything you have learned so far into a pipeline for classification! Your job in this exercise is to build a pipeline that includes scaling and hyperparameter tuning to classify wine quality.




 You’ll return to using the SVM classifier you were briefly introduced to earlier in this chapter. The hyperparameters you will tune are C and gamma. C controls the regularization strength. It is analogous to the C you tuned for logistic regression in Chapter 3, while gamma controls the kernel coefficient:





```python

# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))


```




```

Accuracy: 0.7795918367346939
             precision    recall  f1-score   support

      False       0.83      0.85      0.84       662
       True       0.67      0.63      0.65       318

avg / total       0.78      0.78      0.78       980

Tuned Model Parameters: {'SVM__C': 10, 'SVM__gamma': 0.1}

```


####
**Bringing it all together II: Pipeline for regression**



 Your job is to build a pipeline that imputes the missing data, scales the features, and fits an ElasticNet to the Gapminder data. You will then tune the
 `l1_ratio`
 of your ElasticNet using GridSearchCV.





```python

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))

#    Tuned ElasticNet Alpha: {'elasticnet__l1_ratio': 1.0}
#    Tuned ElasticNet R squared: 0.8862016570888217

```



 The End.


 Thank you for reading.



---



 1. Classification
-------------------




###
**Machine learning introduction**



**What is machine learning?**
 Giving computers the ability to learn to make decisions from data without being explicitly programmed




**Examples of machine learning:**
 Learning to predict whether an email is spam or not (supervised)


 Clustering Wikipedia entries into different categories (unsupervised)



####
**Types of Machine Learning**


* supervised learning
* unsupervised learning
* reinforcement learning



**Supervised learning:**


 Predictor variables/
 **features**
 and a
 **target variable**




 Aim: Predict the target variable, given the predictor variables


 Classification: Target variable consists of categories


 Regression: Target variable is continuous




**Unsupervised learning:**


 Uncovering hidden patterns from unlabeled data




 Example of unsupervised learning:


 Grouping customers into distinct categories (Clustering)




**Reinforcement learning:**
 Software agents interact with an environment


 Learn how to optimize their behavior


 Given a system of rewards and punishments




 Applications


 Economics


 Genetics


 Game playing (AlphaGo)



####
 Naming conventions


* Features = predictor variables = independent variables
* Target variable = dependent variable = response variable




---


####
 Features of
 **Supervised learning**


* Automate time-consuming or expensive manual tasks (ex. Doctor’s diagnosis)
* Make predictions about the future (ex. Will a customer click on an ad or not)
* Need labeled data (Historical data with labels etc.)


####
**Popular libraries**


* scikit-learning (basic)
* TensorFlow
* keras




---


###
**Exploratory data analysis**


####
**Numerical EDA**



 In this chapter, you’ll be working with a dataset obtained from the
 [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records)
 consisting of votes made by US House of Representatives Congressmen.




 Your goal will be to predict their party affiliation (‘Democrat’ or ‘Republican’) based on how they voted on certain key issues.




 Here, it’s worth noting that we have preprocessed this dataset to deal with missing values. This is so that your focus can be directed towards understanding how to train and evaluate supervised learning models.




 Once you have mastered these fundamentals, you will be introduced to preprocessing techniques in Chapter 4 and have the chance to apply them there yourself – including on this very same dataset!




 Before thinking about what supervised learning models you can apply to this, however, you need to perform Exploratory data analysis (EDA) in order to understand the structure of the data.





```

df.head()
        party  infants  water  budget  physician  salvador  religious  \
0  republican        0      1       0          1         1          1
1  republican        0      1       0          1         1          1
2    democrat        0      1       1          0         1          1
3    democrat        0      1       1          0         1          1
4    democrat        1      1       1          0         1          1

   satellite  aid  missile  immigration  synfuels  education  superfund  \
0          0    0        0            1         0          1          1
1          0    0        0            0         0          1          1
2          0    0        0            0         1          0          1
3          0    0        0            0         1          0          1
4          0    0        0            0         1          0          1

   crime  duty_free_exports  eaa_rsa
0      1                  0        1
1      1                  0        1
2      1                  0        0
3      0                  0        1
4      1                  1        1

```




```

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 435 entries, 0 to 434
Data columns (total 17 columns):
party                435 non-null object
infants              435 non-null int64
water                435 non-null int64
budget               435 non-null int64
physician            435 non-null int64
salvador             435 non-null int64
religious            435 non-null int64
satellite            435 non-null int64
aid                  435 non-null int64
missile              435 non-null int64
immigration          435 non-null int64
synfuels             435 non-null int64
education            435 non-null int64
superfund            435 non-null int64
crime                435 non-null int64
duty_free_exports    435 non-null int64
eaa_rsa              435 non-null int64
dtypes: int64(16), object(1)
memory usage: 57.9+ KB

```


###
**Visual EDA**



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture-21.png)


 Above is a
 `countplot`
 of the
 `'education'`
 bill, generated from the following code:





```

plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

```



 In
 `sns.countplot()`
 , we specify the x-axis data to be
 `'education'`
 , and hue to be
 `'party'`
 . Recall that
 `'party'`
 is also our target variable. So the resulting plot shows the difference in voting behavior between the two parties for the
 `'education'`
 bill, with each party colored differently. We manually specified the color to be
 `'RdBu'`
 , as the Republican party has been traditionally associated with red, and the Democratic party with blue.




 It seems like Democrats voted resoundingly
 *against*
 this bill, compared to Republicans.





```

plt.figure()
sns.countplot(x='missile', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture1-20.png)


 Democrats vote resoundingly in
 *favor*
 of missile, compared to Republicans.





---


###
**The classification challenge**


####
**k-Nearest Neighbors: Fit**



[k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)





```python

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

```


####
**k-Nearest Neighbors: Predict**




```python

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party']
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))

# Prediction: ['democrat']

```



 How sure can you be of its predictions? In other words, how can you measure its performance?





---


###
**Measuring model performance**


####
**The digits recognition dataset: MNIST**



 In the following exercises, you’ll be working with the
 [MNIST](http://yann.lecun.com/exdb/mnist/)
 digits recognition dataset, which has 10 classes, the digits 0 through 9! A reduced version of the MNIST dataset is one of scikit-learn’s included datasets, and that is the one we will use in this exercise.




 Each sample in this scikit-learn dataset is an 8×8 image representing a handwritten digit. Each pixel is represented by an integer in the range 0 to 16, indicating varying levels of black.




 It is a famous dataset in machine learning and computer vision, and frequently used as a benchmark to evaluate the performance of a new model.





```python

# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
#dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])

print(digits.DESCR)
/*
Optical Recognition of Handwritten Digits Data Set
===================================================

Notes
-----
Data Set Characteristics:
    :Number of Instances: 5620
    :Number of Attributes: 64
    :Attribute Information: 8x8 image of integer pixels in the range 0..16.
    :Missing Attribute Values: None
    :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
    :Date: July; 1998
...
*/

# Print the shape of the images and data keys
print(digits.images.shape)
(1797, 8, 8)

print(digits.data.shape)
(1797, 64)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture2-19.png)

####
**Train/Test Split + Fit/Predict/Accuracy**




```python

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
#  Stratify the split according to the labels so that they are distributed in the training and test sets as they are in the original dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))
# 0.983333333333

```



 Incredibly, this out of the box k-NN classifier with 7 neighbors has learned from the training data and predicted the labels of the images in the test set with 98% accuracy, and it did so in less than a second! This is one illustration of how incredibly useful machine learning techniques can be.





---


####
**Overfitting and underfitting**



 In this exercise, you will compute and plot the training and testing accuracy scores for a variety of different neighbor values. By observing how the accuracy scores differ for the training and testing sets with different values of k, you will develop your intuition for overfitting and underfitting.





```python

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture3-18.png)


 It looks like the test accuracy is highest when using 3 and 5 neighbors. Using 8 neighbors or more seems to result in a simple model that underfits the data.





---



# **2. Regression**
------------------


###
**Introduction to regression**


####
**Importing data for supervised learning**



 In this chapter, you will work with
 [Gapminder](https://www.gapminder.org/data/)
 data that we have consolidated into one CSV file available in the workspace as
 `'gapminder.csv'`
 . Specifically, your goal will be to use this data to predict the life expectancy in a given country based on features such as the country’s GDP, fertility rate, and population.





```python

# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Dimensions of y before reshaping: (139,)
# Dimensions of X before reshaping: (139,)


# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

# Dimensions of y after reshaping: (139, 1)
# Dimensions of X after reshaping: (139, 1)

```


####
**Exploring the Gapminder data**




```

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 139 entries, 0 to 138
Data columns (total 9 columns):
population         139 non-null float64
fertility          139 non-null float64
HIV                139 non-null float64
CO2                139 non-null float64
BMI_male           139 non-null float64
GDP                139 non-null float64
BMI_female         139 non-null float64
life               139 non-null float64
child_mortality    139 non-null float64
dtypes: float64(9)
memory usage: 9.9 KB

```




```

sns.heatmap(df.corr(), square=True, cmap='RdYlGn')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture4-17.png)

###
**The basics of linear regression**



 We suppose that y and x have a linear relationship that can be model by


 y = ax + b


 An linear regression is to find a, b that minimize the sum of the squared residual (= Ordinary Least Squares, OLS)




 Why squared residual?


 Residuals may be positive and negative.


 They cancel each other. square residual can solve this problem.




![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture6-11.png)

 green lines are residuals



 When we have n variables of x,


 y = a1*x1 + a2*x2 + … an*xn + b


 we find a1, a2, … an, b that minimize the sum of the squared residual.



####
**Fit & predict for regression**



 In this exercise, you will use the
 `'fertility'`
 feature of the Gapminder dataset. Since the goal is to predict life expectancy, the target variable here is
 `'life'`
 .


 You will also compute and print the R2 score using sckit-learn’s
 `.score()`
 method.





```python

# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2
print(reg.score(X_fertility, y))
0.619244216774

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture7-12.png)

####
**Train/test split for regression**



 In this exercise, you will split the Gapminder dataset into training and testing sets, and then fit and predict a linear regression over
 **all**
 features. In addition to computing the R2 score, you will also compute the Root Mean Squared Error (RMSE), which is another commonly used metric to evaluate regression models.





```python

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
# R^2: 0.838046873142936


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
# Root Mean Squared Error: 3.2476010800377213


```


###
**Cross-validation**



 What is cross validation?


<https://en.wikipedia.org/wiki/Cross-validation_(statistics)>



####
**5-fold cross-validation**



 In this exercise, you will practice 5-fold cross validation on the Gapminder data. By default, scikit-learn’s
 `cross_val_score()`
 function uses R2R2 as the metric of choice for regression.





```python

# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)
# [ 0.81720569  0.82917058  0.90214134  0.80633989  0.94495637]

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
# Average 5-Fold CV Score: 0.8599627722793232

```


####
**K-Fold CV comparison**



 Cross validation is essential but do not forget that the more folds you use, the more computationally expensive cross-validation becomes. In this exercise, you will explore this for yourself. Your job is to perform 3-fold cross-validation and then 10-fold cross-validation on the Gapminder dataset.





```python

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg, X, y, cv=3)
print(np.mean(cvscores_3))
# 0.871871278262

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg, X, y, cv=10)
print(np.mean(cvscores_10))
# 0.843612862013

```




```

%timeit cross_val_score(reg, X, y, cv=3)
100 loops, best of 3: 8.73 ms per loop

%timeit cross_val_score(reg, X, y, cv=10)
10 loops, best of 3: 27.5 ms per loop

```




---


###
**Regularized regression**


####
**[Regularization I: Lasso](https://en.wikipedia.org/wiki/Lasso_(statistics))**



 In this exercise, you will fit a lasso regression to the Gapminder data you have been working with and plot the coefficients. Just as with the Boston data, you will find that the coefficients of some features are shrunk to 0, with only the most important ones remaining.





```

df.columns
Index(['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'life', 'child_mortality'],
      dtype='object')

X: ['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'child_mortality']
y: life

```




```python

# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
#  [-0.         -0.         -0.          0.          0.          0.         -0.
     -0.07087587]


# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture8-9.png)


 According to the lasso algorithm, it seems like
 `'child_mortality'`
 is the most important feature when predicting life expectancy.



####
**Regularization II: Ridge**



 Lasso is great for feature selection, but when building regression models, Ridge regression should be your first choice.




 Recall that lasso performs regularization by adding to the loss function a penalty term of the
 *absolute*
 value of each coefficient multiplied by some alpha. This is also known as L1L1 regularization because the regularization term is the L1L1 norm of the coefficients. This is not the only way to regularize, however.




 If instead you took the sum of the
 *squared*
 values of the coefficients multiplied by some alpha – like in Ridge regression – you would be computing the L2L2norm. In this exercise, you will practice fitting ridge regression models over a range of different alphas, and plot cross-validated R2R2 scores for each, using this function that we have defined for you, which plots the R2R2 score as well as standard error for each alpha:





```

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


```



 Don’t worry about the specifics of the above function works. The motivation behind this exercise is for you to see how the R2R2 score varies with different alphas, and to understand the importance of selecting the right value for alpha. You’ll learn how to tune alpha in the next chapter.





```python

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha

    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)

    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))

    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture9-8.png)


 Notice how the cross-validation scores change with different alphas.





---



# **3. Fine-tuning your model**
------------------------------


###
**confusion matrix**



 What is confusion matrix


<https://en.wikipedia.org/wiki/Confusion_matrix>




![{\displaystyle \mathrm {TPR} ={\frac {\mathrm {TP} }{\mathrm {P} }}={\frac {\mathrm {TP} }{\mathrm {TP} +\mathrm {FN} }}=1-\mathrm {FNR} }](https://wikimedia.org/api/rest_v1/media/math/render/svg/f02ea353bf60bfdd9557d2c98fe18c34cd8db835)

[sensitivity](https://en.wikipedia.org/wiki/Sensitivity_(test))
 ,
 [recall](https://en.wikipedia.org/wiki/Precision_and_recall#Recall)
 ,
 [hit rate](https://en.wikipedia.org/wiki/Hit_rate)
 , or
 [true positive rate](https://en.wikipedia.org/wiki/Sensitivity_(test))
 (TPR)



![{\displaystyle \mathrm {TNR} ={\frac {\mathrm {TN} }{\mathrm {N} }}={\frac {\mathrm {TN} }{\mathrm {TN} +\mathrm {FP} }}=1-\mathrm {FPR} }](https://wikimedia.org/api/rest_v1/media/math/render/svg/8f2c867f0641e498ec8a59de63697a3a45d66b07)

[specificity](https://en.wikipedia.org/wiki/Specificity_(tests))
 ,
 [selectivity](https://en.wikipedia.org/wiki/Specificity_(tests))
 or
 [true negative rate](https://en.wikipedia.org/wiki/Specificity_(tests))
 (TNR)



![{\displaystyle \mathrm {PPV} ={\frac {\mathrm {TP} }{\mathrm {TP} +\mathrm {FP} }}=1-\mathrm {FDR} }](https://wikimedia.org/api/rest_v1/media/math/render/svg/d854b1544fc77735d575ce0d30e34d7f1eacf707)

[precision](https://en.wikipedia.org/wiki/Information_retrieval#Precision)
 or
 [positive predictive value](https://en.wikipedia.org/wiki/Positive_predictive_value)
 (PPV)



![{\displaystyle \mathrm {ACC} ={\frac {\mathrm {TP} +\mathrm {TN} }{\mathrm {P} +\mathrm {N} }}={\frac {\mathrm {TP} +\mathrm {TN} }{\mathrm {TP} +\mathrm {TN} +\mathrm {FP} +\mathrm {FN} }}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/47deb47eb7ac214423d0a6afd05ec0af362fef9b)

[accuracy](https://en.wikipedia.org/wiki/Accuracy)
 (ACC)



![{\displaystyle \mathrm {F} _{1}=2\cdot {\frac {\mathrm {PPV} \cdot \mathrm {TPR} }{\mathrm {PPV} +\mathrm {TPR} }}={\frac {2\mathrm {TP} }{2\mathrm {TP} +\mathrm {FP} +\mathrm {FN} }}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/0e5f071c6418f444fadc9f5f9b0358beed3e094c)

[F1 score](https://en.wikipedia.org/wiki/F1_score)
 is the
 [harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean#Harmonic_mean_of_two_numbers)
 of
 [precision](https://en.wikipedia.org/wiki/Information_retrieval#Precision)
 and
 [sensitivity](https://en.wikipedia.org/wiki/Sensitivity_(test))


####
**illustration for TPR, TNR and PPV**



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/metric_example.png?w=1024)

[source](http://corysimon.github.io/articles/classification-metrics/)


####
**Metrics for classification**



 In this exercise, you will dive more deeply into evaluating the performance of binary classifiers by computing a confusion matrix and generating a classification report.




 Here, you’ll work with the
 [PIMA Indians](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
 dataset obtained from the UCI Machine Learning Repository. The goal is to predict whether or not a given female patient will contract diabetes based on features such as BMI, age, and number of pregnancies.




 Therefore, it is a binary classification problem. A target value of
 `0`
 indicates that the patient does
 *not*
 have diabetes, while a value of
 `1`
 indicates that the patient
 *does*
 have diabetes.





```

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
pregnancies    768 non-null int64
glucose        768 non-null int64
diastolic      768 non-null int64
triceps        768 non-null float64
insulin        768 non-null float64
bmi            768 non-null float64
dpf            768 non-null float64
age            768 non-null int64
diabetes       768 non-null int64
dtypes: float64(4), int64(5)
memory usage: 54.1 KB

```




```python

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

```




```

[[176  30]
 [ 52  50]]

             precision    recall  f1-score   support

          0       0.77      0.85      0.81       206
          1       0.62      0.49      0.55       102

avg / total       0.72      0.73      0.72       308

```




---


###
**Logistic regression and the ROC curve**



 What is logistic regression?


<https://en.wikipedia.org/wiki/Logistic_regression>




 What is ROC?


[Receiver operating characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)




 Further Reading:
 [scikit-learn document](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py)




![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/roc_space-2.png?w=1024)

####
**Building a logistic regression model**




```

X.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 8 columns):
pregnancies    768 non-null int64
glucose        768 non-null int64
diastolic      768 non-null int64
triceps        768 non-null float64
insulin        768 non-null float64
bmi            768 non-null float64
dpf            768 non-null float64
age            768 non-null int64
dtypes: float64(4), int64(4)
memory usage: 48.1 KB


y
0      1
1      0
2      1
      ..

765    0
766    1
767    0
Name: diabetes, dtype: int64

```




```python

# Import the necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


```




```

[[176  30]
 [ 35  67]]

             precision    recall  f1-score   support

          0       0.83      0.85      0.84       206
          1       0.69      0.66      0.67       102

avg / total       0.79      0.79      0.79       308

```


####
**Plotting an ROC curve**

**.predict_proba()**




```python

# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

```




```

logreg.predict_proba(X_test)
# False, True
# 0, 1
# Negative, Positive
array([[ 0.60409835,  0.39590165],
       [ 0.76042394,  0.23957606],
       [ 0.79670177,  0.20329823],
       ...
       [ 0.84686912,  0.15313088],
       [ 0.97617225,  0.02382775],
       [ 0.40380502,  0.59619498]])

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture1-21.png?w=1024)

####
**Precision-recall Curve**



 There are other ways to visually evaluate model performance. One such way is the precision-recall curve, which is generated by plotting the precision and recall for different thresholds.


 Note that here, the class is positive (1) if the individual
 *has*
 diabetes.




![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture2-20.png?w=1024)

* A recall of 1 corresponds to a classifier with a low threshold in which
 *all*
 females who contract diabetes were correctly classified as such, at the expense of many misclassifications of those who did
 *not*
 have diabetes.
* Precision is undefined for a classifier which makes
 *no*
 positive predictions, that is, classifies
 *everyone*
 as
 *not*
 having diabetes.
* When the threshold is very close to 1, precision is also 1, because the classifier is absolutely certain about its predictions.



 recall or sensitivity, TPR = 1 means all true positive are detected. We can predict all to positive to get a recall of 1.




 precision, PPV = 1 means no false positive are detected. We can predict less positive to get a higher precision.



####
**Area under the ROC curve**


####
**AUC(**
 Area Under the Curve
 **) computation**




```python

# diabetes data set
df.head()
   pregnancies  glucose  diastolic   triceps     insulin   bmi    dpf  age  \
0            6      148         72  35.00000  155.548223  33.6  0.627   50
1            1       85         66  29.00000  155.548223  26.6  0.351   31
2            8      183         64  29.15342  155.548223  23.3  0.672   32
3            1       89         66  23.00000   94.000000  28.1  0.167   21
4            0      137         40  35.00000  168.000000  43.1  2.288   33

   diabetes
0         1
1         0
2         1
3         0
4         1

```




```python

# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))


```




```

AUC: 0.8254806777079764

AUC scores computed using 5-fold cross-validation:
[ 0.80148148  0.8062963   0.81481481  0.86245283  0.8554717 ]

```




---


###
**Hyperparameter tuning**


####
**Hyperparameter tuning with GridSearchCV**



 You will now practice this yourself, but by using logistic regression on the diabetes dataset.




 Like the alpha parameter of lasso and ridge regularization that you saw earlier, logistic regression also has a regularization parameter: CC. CC controls the
 *inverse*
 of the regularization strength, and this is what you will tune in this exercise. A large CC can lead to an
 *overfit*
 model, while a small CC can lead to an
 *underfit*
 model.




 The hyperparameter space for CC has been setup for you. Your job is to use GridSearchCV and logistic regression to find the optimal CC in this hyperparameter space.





```python

# diabetes data set

```




```python

# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))


```




```

Tuned Logistic Regression Parameters: {'C': 3.7275937203149381}
Best score is 0.7708333333333334

```


####
**Hyperparameter tuning with RandomizedSearchCV**



 GridSearchCV can be computationally expensive, especially if you are searching over a large hyperparameter space and dealing with multiple hyperparameters. A solution to this is to use
 `RandomizedSearchCV`
 , in which not all hyperparameter values are tried out. Instead, a fixed number of hyperparameter settings is sampled from specified probability distributions. You’ll practice using
 `RandomizedSearchCV`
 in this exercise and see how this works.





```python

# diabetes data set

```




```python

# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


```




```

Tuned Decision Tree Parameters: {'criterion': 'entropy', 'max_depth': 3, 'max_features': 7, 'min_samples_leaf': 1}
Best score is 0.7317708333333334

```



 Note that
 `RandomizedSearchCV`
 will never outperform
 `GridSearchCV`
 . Instead, it is valuable because it saves on computation time.



###
**Hold-out set for final evaluation**


####
**Hold-out set in practice I: Classification**



 In addition to CC, logistic regression has a
 `'penalty'`
 hyperparameter which specifies whether to use
 `'l1'`
 or
 `'l2'`
 regularization. Your job in this exercise is to create a hold-out set, tune the
 `'C'`
 and
 `'penalty'`
 hyperparameters of a logistic regression classifier using
 `GridSearchCV`
 on the training set.





```python

# diabetes data set

```




```python

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))


```




```

Tuned Logistic Regression Parameter: {'C': 0.43939705607607948, 'penalty': 'l1'}
Tuned Logistic Regression Accuracy: 0.7652173913043478

```


####
**Hold-out set in practice II: Regression**



 Remember lasso and ridge regression from the previous chapter? Lasso used the L1 penalty to regularize, while ridge used the L2 penalty. There is another type of regularized regression known as the elastic net. In elastic net regularization, the penalty term is a linear combination of the L1 and L2 penalties:




**a∗L1+b∗L2**




 In scikit-learn, this term is represented by the
 `'l1_ratio'`
 parameter: An
 `'l1_ratio'`
 of
 `1`
 corresponds to an L1L1 penalty, and anything lower is a combination of L1L1 and L2L2.




 In this exercise, you will
 `GridSearchCV`
 to tune the
 `'l1_ratio'`
 of an elastic net model trained on the Gapminder data. As in the previous exercise, use a hold-out set to evaluate your model’s performance.





```

df.head()
   population  fertility  HIV        CO2  BMI_male      GDP  BMI_female  life  \
0  34811059.0       2.73  0.1   3.328945  24.59620  12314.0    129.9049  75.3
1  19842251.0       6.43  2.0   1.474353  22.25083   7103.0    130.1247  58.3
2  40381860.0       2.24  0.5   4.785170  27.50170  14646.0    118.8915  75.5
3   2975029.0       1.40  0.1   1.804106  25.35542   7383.0    132.8108  72.5
4  21370348.0       1.96  0.1  18.016313  27.56373  41312.0    117.3755  81.5

   child_mortality
0             29.5
1            192.0
2             15.4
3             20.0
4              5.2


```




```python

# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))


```




```

Tuned ElasticNet l1 ratio: {'l1_ratio': 0.20689655172413793}
Tuned ElasticNet R squared: 0.8668305372460283
Tuned ElasticNet MSE: 10.05791413339844

```




---



**Preprocessing and pipelines**
--------------------------------


###
**Preprocessing data**


####
**Exploring categorical features**




```

df.head()
   population  fertility  HIV        CO2  BMI_male      GDP  BMI_female  life  \
0  34811059.0       2.73  0.1   3.328945  24.59620  12314.0    129.9049  75.3
1  19842251.0       6.43  2.0   1.474353  22.25083   7103.0    130.1247  58.3
2  40381860.0       2.24  0.5   4.785170  27.50170  14646.0    118.8915  75.5
3   2975029.0       1.40  0.1   1.804106  25.35542   7383.0    132.8108  72.5
4  21370348.0       1.96  0.1  18.016313  27.56373  41312.0    117.3755  81.5

   child_mortality                      Region
0             29.5  Middle East & North Africa
1            192.0          Sub-Saharan Africa
2             15.4                     America
3             20.0       Europe & Central Asia
4              5.2         East Asia & Pacific

```




```python

# Import pandas
import pandas as pd

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/supervised-learning-with-scikit-learn/capture3-19.png?w=1024)

####
**Creating dummy variables**




```python

# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df, drop_first=True)

# Print the new columns of df_region
print(df_region.columns)

```




```

Index(['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'life', 'child_mortality', 'Region_America',
       'Region_East Asia & Pacific', 'Region_Europe & Central Asia',
       'Region_Middle East & North Africa', 'Region_South Asia',
       'Region_Sub-Saharan Africa'],
      dtype='object')

# Region_America has been dropped
Index(['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'life', 'child_mortality', 'Region_East Asia & Pacific',
       'Region_Europe & Central Asia', 'Region_Middle East & North Africa',
       'Region_South Asia', 'Region_Sub-Saharan Africa'],
      dtype='object')


df_region.head()
   population  fertility  HIV        CO2  BMI_male      GDP  BMI_female  life  \
0  34811059.0       2.73  0.1   3.328945  24.59620  12314.0    129.9049  75.3
1  19842251.0       6.43  2.0   1.474353  22.25083   7103.0    130.1247  58.3
2  40381860.0       2.24  0.5   4.785170  27.50170  14646.0    118.8915  75.5

   child_mortality  Region_East Asia & Pacific  Region_Europe & Central Asia  \
0             29.5                           0                             0
1            192.0                           0                             0
2             15.4                           0                             0

   Region_Middle East & North Africa  Region_South Asia  \
0                                  1                  0
1                                  0                  0
2                                  0                  0

   Region_Sub-Saharan Africa
0                          0
1                          1
2                          0

```


####
**Regression with categorical features**




```python

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)

# Print the cross-validated scores
print(ridge_cv)
[ 0.86808336  0.80623545  0.84004203  0.7754344   0.87503712]

```




---


###
**Handling missing data**


####
**Dropping missing data**



 Now, it’s time for you to take care of these yourself!




 The unprocessed dataset has been loaded into a DataFrame
 `df`
 . Explore it in the IPython Shell with the
 `.head()`
 method. You will see that there are certain data points labeled with a
 `'?'`
 . These denote missing values. As you saw in the video, different datasets encode missing values in different ways. Sometimes it may be a
 `'9999'`
 , other times a
 `0`
 – real-world data can be very messy! If you’re lucky, the missing values will already be encoded as
 `NaN`
 . We use
 `NaN`
 because it is an efficient and simplified way of internally representing missing data, and it lets us take advantage of pandas methods such as
 `.dropna()`
 and
 `.fillna()`
 , as well as scikit-learn’s Imputation transformer
 `Imputer()`
 .




 In this exercise, your job is to convert the
 `'?'`
 s to NaNs, and then drop the rows that contain them from the DataFrame.





```

df.head(3)
        party infants water budget physician salvador religious satellite aid  \
0  republican       0     1      0         1        1         1         0   0
1  republican       0     1      0         1        1         1         0   0
2    democrat       ?     1      1         ?        1         1         0   0

  missile immigration synfuels education superfund crime duty_free_exports  \
0       0           1        ?         1         1     1                 0
1       0           0        0         1         1     1                 0
2       0           0        1         0         1     1                 0

  eaa_rsa
0       1
1       ?
2       0

```




```python

# Convert '?' to NaN
df[df == '?'] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))


```




```

party                  0
infants               12
water                 48
budget                11
physician             11
salvador              15
religious             11
satellite             14
aid                   15
missile               22
immigration            7
synfuels              21
education             31
superfund             25
crime                 17
duty_free_exports     28
eaa_rsa              104
dtype: int64
Shape of Original DataFrame: (435, 17)


Shape of DataFrame After Dropping All Rows with Missing Values: (232, 17)

```



 When many values in your dataset are missing, if you drop them, you may end up throwing away valuable information along with the missing data. It’s better instead to develop an imputation strategy. This is where domain knowledge is useful, but in the absence of it, you can impute missing values with the mean or the median of the row or column that the missing value is in.



####
**Imputing missing data in a ML Pipeline I**



 As you’ve come to appreciate, there are many steps to building a model, from creating training and test sets, to fitting a classifier or regressor, to tuning its parameters, to evaluating its performance on new data. Imputation can be seen as the first step of this machine learning process, the entirety of which can be viewed within the context of a pipeline. Scikit-learn provides a pipeline constructor that allows you to piece together these steps into one process and thereby simplify your workflow.




 You’ll now practice setting up a pipeline with two steps: the imputation step, followed by the instantiation of a classifier. You’ve seen three classifiers in this course so far: k-NN, logistic regression, and the decision tree. You will now be introduced to a fourth one – the Support Vector Machine, or
 [SVM](http://scikit-learn.org/stable/modules/svm.html)
 .





```python

# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
# axis=0 for column
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]

```


####
**Imputing missing data in a ML Pipeline II**



 Having setup the steps of the pipeline in the previous exercise, you will now use it on the voting dataset to classify a Congressman’s party affiliation.




 What makes pipelines so incredibly useful is the simple interface that they provide. You can use the
 `.fit()`
 and
 `.predict()`
 methods on pipelines just as you did with your classifiers and regressors!





```python

# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))


```




```

             precision    recall  f1-score   support

   democrat       0.99      0.96      0.98        85
 republican       0.94      0.98      0.96        46

avg / total       0.97      0.97      0.97       131

```


###
**Centering and scaling**


####
**Centering and scaling your data**



 You will now explore scaling for yourself on a new dataset –
 [White Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
 !




 We have used the
 `'quality'`
 feature of the wine to create a binary target variable: If
 `'quality'`
 is less than
 `5`
 , the target variable is
 `1`
 , and otherwise, it is
 `0`
 .




 Notice how some features seem to have different units of measurement.
 `'density'`
 , for instance, takes values between 0.98 and 1.04, while
 `'total sulfur dioxide'`
 ranges from 9 to 440. As a result, it may be worth scaling the features here. Your job in this exercise is to scale the features and compute the mean and standard deviation of the unscaled features compared to the scaled features.





```python

# white wine quality data set
df.head(3)
   fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
0            7.0              0.27         0.36            20.7      0.045
1            6.3              0.30         0.34             1.6      0.049
2            8.1              0.28         0.40             6.9      0.050

   free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
0                 45.0                 170.0   1.0010  3.00       0.45
1                 14.0                 132.0   0.9940  3.30       0.49
2                 30.0                  97.0   0.9951  3.26       0.44

   alcohol  quality
0      8.8        6
1      9.5        6
2     10.1        6

```




```python

# Import scale
from sklearn.preprocessing import scale

# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X)))
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled)))
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))

```




```

Mean of Unscaled Features: 18.432687072460002
Standard Deviation of Unscaled Features: 41.54494764094571

Mean of Scaled Features: 2.7314972981668206e-15
Standard Deviation of Scaled Features: 0.9999999999999999

```


####
**Centering and scaling in a pipeline**



 With regard to whether or not scaling is effective, the proof is in the pudding! See for yourself whether or not scaling the features of the White Wine Quality dataset has any impact on its performance.




 You will use a k-NN classifier as part of a pipeline that includes scaling, and for the purposes of comparison, a k-NN classifier trained on the unscaled data has been provided.





```python

# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))

Accuracy with Scaling: 0.7700680272108843
Accuracy without Scaling: 0.6979591836734694

```



 It looks like scaling has significantly improved model performance!





---


####
**Bringing it all together I: Pipeline for classification**



 It is time now to piece together everything you have learned so far into a pipeline for classification! Your job in this exercise is to build a pipeline that includes scaling and hyperparameter tuning to classify wine quality.




 You’ll return to using the SVM classifier you were briefly introduced to earlier in this chapter. The hyperparameters you will tune are C and gamma. C controls the regularization strength. It is analogous to the C you tuned for logistic regression in Chapter 3, while gamma controls the kernel coefficient:





```python

# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))


```




```

Accuracy: 0.7795918367346939
             precision    recall  f1-score   support

      False       0.83      0.85      0.84       662
       True       0.67      0.63      0.65       318

avg / total       0.78      0.78      0.78       980

Tuned Model Parameters: {'SVM__C': 10, 'SVM__gamma': 0.1}

```


####
**Bringing it all together II: Pipeline for regression**



 Your job is to build a pipeline that imputes the missing data, scales the features, and fits an ElasticNet to the Gapminder data. You will then tune the
 `l1_ratio`
 of your ElasticNet using GridSearchCV.





```python

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))

#    Tuned ElasticNet Alpha: {'elasticnet__l1_ratio': 1.0}
#    Tuned ElasticNet R squared: 0.8862016570888217

```



 The End.


 Thank you for reading.



