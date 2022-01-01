---
title: Model Validation in Python
date: 2021-12-07 11:22:11 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Model Validation in Python
============================







 This is the memo of the 11th course (23 courses in all) of ‘Machine Learning Scientist with Python’ skill track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/model-validation-in-python)**
 .



###
**Course Description**



 Machine learning models are easier to implement now more than ever before. Without proper validation, the results of running new data through a model might not be as accurate as expected. Model validation allows analysts to confidently answer the question, how good is your model? We will answer this question for classification models using the complete set of tic-tac-toe endgame scenarios, and for regression models using fivethirtyeight’s ultimate Halloween candy power ranking dataset. In this course, we will cover the basics of model validation, discuss various validation techniques, and begin to develop tools for creating validated and high performing models.



###
**Table of contents**


* [Basic Modeling in scikit-learn](https://datascience103579984.wordpress.com/2020/01/06/model-validation-in-python-from-datacamp/)
* [Validation Basics](https://datascience103579984.wordpress.com/2020/01/06/model-validation-in-python-from-datacamp/2/)
* [Cross Validation](https://datascience103579984.wordpress.com/2020/01/06/model-validation-in-python-from-datacamp/3/)
* [Selecting the best model with Hyperparameter tuning](https://datascience103579984.wordpress.com/2020/01/06/model-validation-in-python-from-datacamp/4/)





# **1. Basic Modeling in scikit-learn**
--------------------------------------


## **1.1 Introduction to model validation**



![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/1-2.png?w=943)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/2-2.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/3-2.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/4-2.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/5-2.png?w=1024)



### **1.1.1 Seen vs. unseen data**



 Model’s tend to have higher accuracy on observations they have seen before. In the candy dataset, predicting the popularity of Skittles will likely have higher accuracy than predicting the popularity of Andes Mints; Skittles is in the dataset, and Andes Mints is not.




 You’ve built a model based on 50 candies using the dataset
 `X_train`
 and need to report how accurate the model is at predicting the popularity of the 50 candies the model was built on, and the 35 candies (
 `X_test`
 ) it has never seen. You will use the mean absolute error,
 `mae()`
 , as the accuracy metric.





```python

# The model is fit using X_train and y_train
model.fit(X_train, y_train)

# Create vectors of predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Train/Test Errors
train_error = mae(y_true=y_train, y_pred=train_predictions)
test_error = mae(y_true=y_test, y_pred=test_predictions)

# Print the accuracy for seen and unseen data
print("Model error on seen data: {0:.2f}.".format(train_error))
print("Model error on unseen data: {0:.2f}.".format(test_error))

# Model error on seen data: 3.28.
# Model error on unseen data: 11.07.

```



 When models perform differently on training and testing data, you should look to model validation to ensure you have the best performing model. In the next lesson, you will start building models to validate.





---


## **1.2 Regression models**



![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/6-2.png?w=907)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/7-2.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/8-2.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/9-2.png?w=1014)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/10-2.png?w=960)



### **1.2.1 Set parameters and fit a model**



 Predictive tasks fall into one of two categories: regression or classification. In the candy dataset, the outcome is a
 *continuous*
 variable describing how often the candy was chosen over another candy in a series of 1-on-1 match-ups. To predict this value (the win-percentage), you will use a
 **regression**
 model.




 In this exercise, you will specify a few parameters using a random forest regression model
 `rfr`
 .





```python

# Set the number of trees
rfr.n_estimators = 100

# Add a maximum depth
rfr.max_depth = 6

# Set the random state
rfr.random_state = 1111

# Fit the model
rfr.fit(X_train, y_train)

```



 You have updated parameters
 *after*
 the model was initialized. This approach is helpful when you need to update parameters. Before making predictions, let’s see which candy characteristics were most important to the model.



### **1.2.2 Feature importances**



 Although some candy attributes, such as chocolate, may be extremely popular, it doesn’t mean they will be
 *important*
 to model prediction. After a random forest model has been fit, you can review the model’s attribute,
 `.feature_importances_`
 , to see which variables had the biggest impact. You can check how important each variable was in the model by looping over the feature importance array using
 `enumerate()`
 .




 If you are unfamiliar with Python’s
 `enumerate()`
 function, it can loop over a list while also creating an automatic counter.





```python

# Fit the model using X and y
rfr.fit(X_train, y_train)

# Print how important each column is to the model
for i, item in enumerate(rfr.feature_importances_):
      # Use i and item to print out the feature importance of each column
    print("{0:s}: {1:.2f}".format(X_train.columns[i], item))

```




```

chocolate: 0.44
fruity: 0.03
caramel: 0.02
peanutyalmondy: 0.05
nougat: 0.01
crispedricewafer: 0.03
hard: 0.01
bar: 0.02
pluribus: 0.02
sugarpercent: 0.17
pricepercent: 0.19

```



 No surprise here – chocolate
 *is*
 the most important variable. .feature_importances_ is a great way to see which variables were important to your random forest model.





---


## **1.3 Classification models**



![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/11-2.png?w=653)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/12-2.png?w=896)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/13-2.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/14-2.png?w=942)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/15-2.png?w=692)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/16-2.png?w=795)



### **1.3.1 Classification predictions**



 In model validation, it is often important to know more about the predictions than just the final classification. When predicting who will win a game, most people are also interested in
 *how likely*
 it is a team will win.






|
 Probability
  |
 Prediction
  |
 Meaning
  |
| --- | --- | --- |
|
 0 < .50
  |
 0
  |
 Team Loses
  |
|
 .50 +
  |
 1
  |
 Team Wins
  |




 In this exercise, you look at the methods,
 `.predict()`
 and
 `.predict_proba()`
 using the
 `tic_tac_toe`
 dataset. The first method will give a prediction of whether Player One will win the game, and the second method will provide the probability of Player One winning. Use
 `rfc`
 as the random forest classification model.





```python

# Fit the rfc model.
rfc.fit(X_train, y_train)

# Create arrays of predictions
classification_predictions = rfc.predict(X_test)
probability_predictions = rfc.predict_proba(X_test)

# Print out count of binary predictions
print(pd.Series(classification_predictions).value_counts())

# Print the first value from probability_predictions
print('The first predicted probabilities are: {}'.format(probability_predictions[0]))

```




```

1    563
0    204
dtype: int64
The first predicted probabilities are: [0.26524423 0.73475577]

```



 You can see there were 563 observations where Player One was predicted to win the Tic-Tac-Toe game. Also, note that the
 `predicted_probabilities`
 array contains lists with only two values because you only have two possible responses (win or lose). Remember these two methods, as you will use them a lot throughout this course.



### **1.3.2 Reusing model parameters**



 Replicating model performance is vital in model validation. Replication is also important when sharing models with co-workers, reusing models on new data or asking questions on a website such as
 [Stack Overflow](https://stackoverflow.com/)
 . You might use such a site to ask other coders about model errors, output, or performance. The best way to do this is to replicate your work by reusing model parameters.




 In this exercise, you use various methods to recall which parameters were used in a model.





```

rfc = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1111)

# Print the classification model
print(rfc)

# Print the classification model's random state parameter
print('The random state is: {}'.format(rfc.random_state))

# Print all parameters
print('Printing the parameters dictionary: {}'.format(rfc.get_params()))

```




```

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=6, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
            oob_score=False, random_state=1111, verbose=0,
            warm_start=False)

The random state is: 1111

Printing the parameters dictionary: {'bootstrap': True, 'class_weight': None, 'criterion': 'gini', 'max_depth': 6, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 50, 'n_jobs': None, 'oob_score': False, 'random_state': 1111, 'verbose': 0, 'warm_start': False}

```



 Recalling which parameters were used will be helpful going forward. Model validation and performance rely heavily on which parameters were used, and there is no way to replicate a model without keeping track of the parameters used!



### **1.3.3 Random forest classifier**



 This exercise reviews the four modeling steps discussed throughout this chapter using a random forest classification model. You will:



1. Create a random forest classification model.
2. Fit the model using the
 `tic_tac_toe`
 dataset.
3. Make predictions on whether Player One will win (1) or lose (0) the current game.
4. Finally, you will evaluate the overall accuracy of the model.



 Let’s get started!





```

from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier
rfc = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1111)

# Fit rfc using X_train and y_train
rfc.fit(X_train, y_train)

# Create predictions on X_test
predictions = rfc.predict(X_test)
print(predictions[0:5])
# [1 1 1 1 1]


# Print model accuracy using score() and the testing data
print(rfc.score(X_test, y_test))
# 0.817470664928292

```



 That’s all the steps! Notice the first five predictions were all 1, indicating that Player One is predicted to win all five of those games. You also see the model accuracy was only 82%.




 Let’s move on to Chapter 2 and increase our model validation toolbox by learning about splitting datasets, standard accuracy metrics, and the bias-variance tradeoff.




# **2. Validation Basics**
-------------------------


## **2.1 Creating train, test, and validation datasets**



![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/1-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/2-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/3-3.png?w=804)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/4-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/5-3.png?w=942)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/6-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/7-3.png?w=1024)



### **2.1.1 Create one holdout set**



 Your boss has asked you to create a simple random forest model on the
 `tic_tac_toe`
 dataset. She doesn’t want you to spend much time selecting parameters; rather she wants to know how well the model will perform on future data. For future Tic-Tac-Toe games, it would be nice to know if your model can predict which player will win.




 The dataset
 `tic_tac_toe`
 has been loaded for your use.




 Note that in Python,
 `=\`
 indicates the code was too long for one line and has been split across two lines.





```python

# Create dummy variables using pandas
X = pd.get_dummies(tic_tac_toe.iloc[:,0:9])
y = tic_tac_toe.iloc[:, 9]

# Create training and testing datasets. Use 10% for the test set
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.1, random_state=1111)

```



 Good! Remember, without the holdout set, you cannot truly validate a model. Let’s move on to creating two holdout sets.



### **2.1.2 Create two holdout sets**



 You recently created a simple random forest model to predict Tic-Tac-Toe game wins for your boss, and at her request, you did not do any parameter tuning. Unfortunately, the overall model accuracy was too low for her standards. This time around, she has asked you to focus on model performance.




 Before you start testing different models and parameter sets, you will need to split the data into training, validation, and testing datasets. Remember that after splitting the data into training and testing datasets, the validation dataset is created by splitting the training dataset.




 The datasets
 `X`
 and
 `y`
 have been loaded for your use.





```python

# Create temporary training and final testing datasets
X_temp, X_test, y_temp, y_test  =\
    train_test_split(X, y, test_size=0.2, random_state=1111)

# Create the final training and validation datasets
X_train, X_val, y_train, y_val =\
    train_test_split(X_temp, y_temp, test_size=0.25, random_state=1111)

```



 Great! You now have training, validation, and testing datasets, but do you know
 *when*
 you need both validation and testing datasets? Keep going! The next exercise will help make sure you understand when to use validation datasets.



### **2.1.3 Why use holdout sets**



 It is important to understand when you would use three datasets (training, validation, and testing) instead of two (training and testing). There is no point in creating an additional dataset split if you are not going to use it.




 When should you consider using training, validation,
 *and*
 testing datasets?




**When testing parameters, tuning hyper-parameters, or anytime you are frequently evaluating model performance.**




 Correct! Anytime we are evaluating model performance repeatedly we need to create training, validation, and testing datasets.





---


## **2.2 Accuracy metrics: regression models**



![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/8-3.png?w=932)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/9-3.png?w=1018)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/10-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/11-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/12-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/13-3.png?w=1024)



### **2.2.1 Mean absolute error**



 Communicating modeling results can be difficult. However, most clients understand that on average, a predictive model was off by some number. This makes explaining the mean absolute error easy. For example, when predicting the number of wins for a basketball team, if you predict 42, and they end up with 40, you can easily explain that the error was two wins.




 In this exercise, you are interviewing for a new position and are provided with two arrays.
 `y_test`
 , the true number of wins for all 30 NBA teams in 2017 and
 `predictions`
 , which contains a prediction for each team. To test your understanding, you are asked to both manually calculate the MAE and use
 `sklearn`
 .





```

from sklearn.metrics import mean_absolute_error

# Manually calculate the MAE
n = len(predictions)
mae_one = sum(abs(y_test - predictions)) / n
print('With a manual calculation, the error is {}'.format(mae_one))
# With a manual calculation, the error is 5.9


# Use scikit-learn to calculate the MAE
mae_two = mean_absolute_error(y_test, predictions)
print('Using scikit-lean, the error is {}'.format(mae_two))
# Using scikit-lean, the error is 5.9

```



 Well done! These predictions were about six wins off on average. This isn’t too bad considering NBA teams play 82 games a year. Let’s see how these errors would look if you used the mean squared error instead.



### **2.2.2 Mean squared error**



 Let’s focus on the 2017 NBA predictions again. Every year, there are at least a couple of NBA teams that win
 *way*
 more games than expected. If you use the MAE, this accuracy metric does not reflect the bad predictions as much as if you use the MSE. Squaring the large errors from bad predictions will make the accuracy look worse.




 In this example, NBA executives want to better predict team wins. You will use the mean squared error to calculate the prediction error. The actual wins are loaded as
 `y_test`
 and the predictions as
 `predictions`
 .





```

from sklearn.metrics import mean_squared_error

n = len(predictions)
# Finish the manual calculation of the MSE
mse_one = sum((y_test - predictions)**2) / n
print('With a manual calculation, the error is {}'.format(mse_one))
# With a manual calculation, the error is 49.1


# Use the scikit-learn function to calculate MSE
mse_two = mean_squared_error(y_test, predictions)
print('Using scikit-lean, the error is {}'.format(mse_two))
# Using scikit-lean, the error is 49.1

```



 Good job! If you run any additional models, you will try to beat an MSE of 49.1, which is the average squared error of using your model. Although the MSE is not as interpretable as the MAE, it will help us select a model that has fewer ‘large’ errors.



### **2.2.3 Performance on data subsets**



 In professional basketball, there are two conferences, the East and the West. Coaches and fans often only care about how teams in their own conference will do this year.




 You have been working on an NBA prediction model and would like to determine if the predictions were better for the East or West conference. You added a third array to your data called
 `labels`
 , which contains an “E” for the East teams, and a “W” for the West.
 `y_test`
 and
 `predictions`
 have again been loaded for your use.





```python

# Find the East conference teams
east_teams = labels == "E"

# Create arrays for the true and predicted values
true_east = y_test[east_teams]
preds_east = predictions[east_teams]

# Print the accuracy metrics
print('The MAE for East teams is {}'.format(
    mae(true_east, preds_east)))
# The MAE for East teams is 6.733333333333333


# Print the West accuracy
print('The MAE for West conference is {}'.format(west_error))
# The MAE for West conference is 5.01

```



 Great! It looks like the Western conference predictions were about two games better on average. Over the past few seasons, the Western teams have generally won the same number of games as the experts have predicted. Teams in the East are just not as predictable as those in the West.





---


## **2.3 Classification metrics**



![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/14-3.png?w=643)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/15-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/16-3.png?w=933)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/17-2.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/18-1.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/19-1.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/20-1.png?w=1024)



### **2.3.1 Confusion matrices**



 Confusion matrices are a great way to start exploring your model’s accuracy. They provide the values needed to calculate a wide range of metrics, including sensitivity, specificity, and the F1-score.




 You have built a classification model to predict if a person has a broken arm based on an X-ray image. On the testing set, you have the following confusion matrix:






|  |
 Prediction: 0
  |
 Prediction: 1
  |
| --- | --- | --- |
|
 Actual: 0
  |
 324 (TN)
  |
 15 (FP)
  |
|
 Actual: 1
  |
 123 (FN)
  |
 491 (TP)
  |





```python

# Calculate and print the accuracy
accuracy = (324 + 491) / (953)
print("The overall accuracy is {0: 0.2f}".format(accuracy))

# Calculate and print the precision
precision = (491) / (15 + 491)
print("The precision is {0: 0.2f}".format(precision))

# Calculate and print the recall
recall = (491) / (123 + 491)
print("The recall is {0: 0.2f}".format(recall))

```




```

The overall accuracy is  0.86
The precision is  0.97
The recall is  0.80

```



 Well done! In this case, a true positive is a picture of an actual broken arm that was also predicted to be broken. Doctors are okay with a few additional false positives (predicted broken, not actually broken), as long as you don’t miss anyone who needs immediate medical attention.



### **2.3.2 Confusion matrices, again**



 Creating a confusion matrix in Python is simple. The biggest challenge will be making sure you understand the orientation of the matrix. This exercise makes sure you understand the
 `sklearn`
 implementation of confusion matrices. Here, you have created a random forest model using the
 `tic_tac_toe`
 dataset
 `rfc`
 to predict outcomes of 0 (loss) or 1 (a win) for Player One.




*Note:*
 If you read about confusion matrices on another website or for another programming language, the values might be reversed.





```

from sklearn.metrics import confusion_matrix

# Create predictions
test_predictions = rfc.predict(X_test)

# Create and print the confusion matrix
cm = confusion_matrix(y_test, test_predictions)
print(cm)

# Print the true positives (actual 1s that were predicted 1s)
print("The number of true positives is: {}".format(cm[1, 1]))

```




```

[[177 123]
 [ 92 471]]

The number of true positives is: 471

```



 Good job! Row 1, column 1 represents the number of actual 1s that were predicted 1s (the true positives). Always make sure you understand the orientation of the confusion matrix before you start using it!



### **2.3.3 Precision vs. recall**



 The accuracy metrics you use to evaluate your model should
 *always*
 be based on the specific application. For this example, let’s assume you are a really sore loser when it comes to playing Tic-Tac-Toe, but only when you are certain that you are going to win.




 Choose the most appropriate accuracy metric, either precision or recall, to complete this example. But remember,
 *if you think you are going to win, you better win!*




 Use
 `rfc`
 , which is a random forest classification model built on the
 `tic_tac_toe`
 dataset.





```

from sklearn.metrics import precision_score

test_predictions = rfc.predict(X_test)

# Create precision or recall score based on the metric you imported
score = precision_score(y_test, test_predictions)

# Print the final result
print("The precision value is {0:.2f}".format(score))
# The precision value is 0.79

```



 Great job! Precision is the correct metric here. Sore-losers can’t stand losing when they are certain they will win! For that reason, our model needs to be as precise as possible. With a precision of only 79%, you may need to try some other modeling techniques to improve this score.





---


## **2.4 The bias-variance tradeoff**



![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/21.png?w=930)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/22.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/23.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/24.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/25.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/26.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/27.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/28.png?w=1024)



### **2.4.1 Error due to under/over-fitting**



 The candy dataset is prime for overfitting. With only 85 observations, if you use 20% for the testing dataset, you are losing a lot of vital data that could be used for modeling. Imagine the scenario where most of the chocolate candies ended up in the training data and very few in the holdout sample. Our model might
 *only*
 see that chocolate is a vital factor, but fail to find that other attributes are also important. In this exercise, you’ll explore how using too many features (columns) in a random forest model can lead to overfitting.




 A
 *feature*
 represents which columns of the data are used in a decision tree. The parameter
 `max_features`
 limits the number of features available.





```python

# Update the rfr model
rfr = RandomForestRegressor(n_estimators=25,
                            random_state=1111,
                            max_features=2)
rfr.fit(X_train, y_train)

# Print the training and testing accuracies
print('The training error is {0:.2f}'.format(
  mae(y_train, rfr.predict(X_train))))
print('The testing error is {0:.2f}'.format(
  mae(y_test, rfr.predict(X_test))))

```



 The training error is 3.88


 The testing error is 9.15





```python

# Update the rfr model
rfr = RandomForestRegressor(n_estimators=25,
                            random_state=1111,
                            max_features=11)
rfr.fit(X_train, y_train)

# Print the training and testing accuracies
print('The training error is {0:.2f}'.format(
  mae(y_train, rfr.predict(X_train))))
print('The testing error is {0:.2f}'.format(
  mae(y_test, rfr.predict(X_test))))

```



 The training error is 3.57


 The testing error is 10.05





```python

# Update the rfr model
rfr = RandomForestRegressor(n_estimators=25,
                            random_state=1111,
                            max_features=4)
rfr.fit(X_train, y_train)

# Print the training and testing accuracies
print('The training error is {0:.2f}'.format(
  mae(y_train, rfr.predict(X_train))))
print('The testing error is {0:.2f}'.format(
  mae(y_test, rfr.predict(X_test))))

```



 The training error is 3.60


 The testing error is 8.79




 Great job! The chart below shows the performance at various max feature values. Sometimes, setting parameter values can make a huge difference in model performance.








![](https://assets.datacamp.com/production/repositories/3981/datasets/7e30218261b88cc6e57da1e07b73c5803450ccf6/Screen%20Shot%202019-01-13%20at%205.40.29%20PM.png)

### **2.4.2 Am I underfitting?**



 You are creating a random forest model to predict if you will win a future game of Tic-Tac-Toe. Using the
 `tic_tac_toe`
 dataset, you have created training and testing datasets,
 `X_train`
 ,
 `X_test`
 ,
 `y_train`
 , and
 `y_test`
 .




 You have decided to create a bunch of random forest models with varying amounts of trees (1, 2, 3, 4, 5, 10, 20, and 50). The more trees you use, the longer your random forest model will take to run. However, if you don’t use enough trees, you risk underfitting. You have created a for loop to test your model at the different number of trees.





```

from sklearn.metrics import accuracy_score

test_scores, train_scores = [], []
for i in [1, 2, 3, 4, 5, 10, 20, 50]:
    rfc = RandomForestClassifier(n_estimators=i, random_state=1111)
    rfc.fit(X_train, y_train)
    # Create predictions for the X_train and X_test datasets.
    train_predictions = rfc.predict(X_train)
    test_predictions = rfc.predict(X_test)
    # Append the accuracy score for the test and train predictions.
    train_scores.append(round(accuracy_score(y_train, train_predictions), 2))
    test_scores.append(round(accuracy_score(y_test, test_predictions), 2))
# Print the train and test scores.
print("The training scores were: {}".format(train_scores))
print("The testing scores were: {}".format(test_scores))

```




```

The training scores were: [0.94, 0.93, 0.98, 0.97, 0.99, 1.0, 1.0, 1.0]
The testing scores were: [0.83, 0.79, 0.89, 0.91, 0.91, 0.93, 0.97, 0.98]

```



 Excellent! Notice that with only one tree, both the train and test scores are low. As you add more trees, both errors improve. Even at 50 trees, this still might not be enough. Every time you use more trees, you achieve higher accuracy. At some point though, more trees increase training time, but do not decrease testing error.




# **3. Cross Validation**
------------------------


## **3.1 The problems with holdout sets**



![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/1-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/2-4.png?w=880)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/3-4.png?w=735)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/4-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/5-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/6-4.png?w=1024)



### **3.1.2 Two samples**



 After building several classification models based on the
 `tic_tac_toe`
 dataset, you realize that some models do not generalize as well as others. You have created training and testing splits just as you have been taught, so you are curious why your validation process is not working.




 After trying a different training, test split, you noticed differing accuracies for your machine learning model. Before getting too frustrated with the varying results, you have decided to see what else could be going on.





```python

# Create two different samples of 200 observations
sample1 = tic_tac_toe.sample(200, random_state=1111)
sample2 = tic_tac_toe.sample(200, random_state=1171)

# Print the number of common observations
print(len([index for index in sample1.index if index in sample2.index]))
# 40

# Print the number of observations in the Class column for both samples
print(sample1['Class'].value_counts())
print(sample2['Class'].value_counts())

```




```

positive    134
negative     66
Name: Class, dtype: int64
positive    123
negative     77
Name: Class, dtype: int64

```



 Well done! Notice that there are a varying number of positive observations for both sample test sets. Sometimes creating a single test holdout sample is not enough to achieve the high levels of model validation you want. You need to use something more robust.



### **3.1.3 Potential problems**


* Using different data splitting methods may lead to varying data in the final holdout samples.
* If you have limited data, your holdout accuracy may be misleading.



 If our models are not generalizing well or if we have limited data, we should be careful using a single training/validation split. You should use the next lesson’s topic: cross-validation.





---


## **3.2 Cross-validation**



![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/7-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/8-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/9-4.png?w=873)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/10-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/11-4.png?w=933)



### **3.2.1 scikit-learn’s KFold()**



 You just finished running a colleagues code that creates a random forest model and calculates an out-of-sample accuracy. You noticed that your colleague’s code did not have a random state, and the errors you found were completely different than the errors your colleague reported.




 To get a better estimate for how accurate this random forest model will be on new data, you have decided to generate some indices to use for KFold cross-validation.





```

from sklearn.model_selection import KFold

# Use KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1111)

# Create splits
splits = kf.split(X)

# Print the number of indices
for train_index, val_index in splits:
    print("Number of training indices: %s" % len(train_index))
    print("Number of validation indices: %s" % len(val_index))

```




```

Number of training indices: 68
Number of validation indices: 17
Number of training indices: 68
Number of validation indices: 17
Number of training indices: 68
Number of validation indices: 17
Number of training indices: 68
Number of validation indices: 17
Number of training indices: 68
Number of validation indices: 17

```



 Good job! This dataset has 85 rows. You have created five splits – each containing 68 training and 17 validation indices. You can use these indices to complete 5-fold cross-validation.



### **3.2.2 Using KFold indices**



 You have already created
 `splits`
 , which contains indices for the candy-data dataset to complete 5-fold cross-validation. To get a better estimate for how well a colleague’s random forest model will perform on a new data, you want to run this model on the five different training and validation indices you just created.




 In this exercise, you will use these indices to check the accuracy of this model using the five different splits. A for loop has been provided to assist with this process.





```

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rfc = RandomForestRegressor(n_estimators=25, random_state=1111)

# Access the training and validation indices of splits
for train_index, val_index in splits:
    # Setup the training and validation data
    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index]
    # Fit the random forest model
    rfc.fit(X_train, y_train)
    # Make predictions, and print the accuracy
    predictions = rfc.predict(X_val)
    print("Split accuracy: " + str(mean_squared_error(y_val, predictions)))

```




```

Split accuracy: 178.75586448813047
Split accuracy: 98.29560208158634
Split accuracy: 86.2673010849621
Split accuracy: 217.4185114496197
Split accuracy: 140.5437661156536

```



 Nice work!
 `KFold()`
 is a great method for accessing individual indices when completing cross-validation. One drawback is needing a for loop to work through the indices though. In the next lesson, you will look at an automated method for cross-validation using
 `sklearn`
 .





---


## **3.3 sklearn’s cross_val_score()**



![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/12-4.png?w=901)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/13-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/14-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/15-4.png?w=765)



### **3.3.1 scikit-learn’s methods**



 You have decided to build a regression model to predict the number of new employees your company will successfully hire next month. You open up a new Python script to get started, but you quickly realize that
 `sklearn`
 has
 *a lot*
 of different modules. Let’s make sure you understand the names of the modules, the methods, and which module contains which method.




 Follow the instructions below to load in all of the necessary methods for completing cross-validation using
 `sklearn`
 . You will use modules:



* `metrics`
* `model_selection`
* `ensemble`




```python

# Instruction 1: Load the cross-validation method
from sklearn.model_selection import cross_val_score

# Instruction 2: Load the random forest regression model
from sklearn.ensemble import RandomForestRegressor

# Instruction 3: Load the mean squared error method
# Instruction 4: Load the function for creating a scorer
from sklearn.metrics import mean_squared_error, make_scorer

```



 Well done! It is easy to see how all of the methods can get mixed up, but it is important to know the names of the methods you need. You can always review the
 [scikit-learn documentation](https://scikit-learn.org/stable/documentation.html)
 should you need any help.



### **3.3.2 Implement cross_val_score()**



 Your company has created several new candies to sell, but they are not sure if they should release all five of them. To predict the popularity of these new candies, you have been asked to build a regression model using the candy dataset. Remember that the response value is a head-to-head win-percentage against other candies.




 Before you begin trying different regression models, you have decided to run cross-validation on a simple random forest model to get a baseline error to compare with any future results.





```

rfc = RandomForestRegressor(n_estimators=25, random_state=1111)
mse = make_scorer(mean_squared_error)

# Set up cross_val_score
cv = cross_val_score(estimator=rfc,
                     X=X_train,
                     y=y_train,
                     cv=10,
                     scoring=mse)

# Print the mean error
print(cv.mean())
# 155.55845080026586

```



 Nice! You now have a baseline score to build on. If you decide to build additional models or try new techniques, you should try to get an error lower than 155.56. Lower errors indicate that your popularity predictions are improving.





---


## **3.4 Leave-one-out-cross-validation (LOOCV)**



![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/16-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/17-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/18-2.png?w=1024)



### **3.4.1 When to use LOOCV**



 Which of the following are reasons you might
 **NOT**
 run LOOCV on the provided
 `X`
 dataset? The
 `X`
 data has been loaded for you to explore as you see fit.



* **A**
 : The
 `X`
 dataset has 122,624 data points, which might be computationally expensive and slow.
* **B**
 : You cannot run LOOCV on classification problems.
* **C**
 : You want to test different values for 15 different parameters



**A&C**




 Well done! This many observations will definitely slow things down and could be computationally expensive. If you don’t have time to wait while your computer runs through 1,000 models, you might want to use 5 or 10-fold cross-validation.



### **3.4.2 Leave-one-out-cross-validation**



 Let’s assume your favorite candy is not in the candy dataset, and that you are interested in the popularity of this candy. Using 5-fold cross-validation will train on only 80% of the data at a time. The candy dataset
 *only*
 has 85 rows though, and leaving out 20% of the data could hinder our model. However, using leave-one-out-cross-validation allows us to make the most out of our limited dataset and will give you the best estimate for your favorite candy’s popularity!




 In this exercise, you will use
 `cross_val_score()`
 to perform LOOCV.





```

from sklearn.metrics import mean_absolute_error, make_scorer

# Create scorer
mae_scorer = make_scorer(mean_absolute_error)

rfr = RandomForestRegressor(n_estimators=15, random_state=1111)

# Implement LOOCV
scores = cross_val_score(rfr, X=X, y=y, cv=X.shape[0], scoring=mae_scorer)

# Print the mean and standard deviation
print("The mean of the errors is: %s." % np.mean(scores))
print("The standard deviation of the errors is: %s." % np.std(scores))

# The mean of the errors is: 9.464989603398694.
# The standard deviation of the errors is: 7.265762094853885.

```



 Very good! You have come along way with model validation techniques. The final chapter will wrap up model validation by discussing how to select the best model and give an introduction to parameter tuning.




# **4. Selecting the best model with Hyperparameter tuning**
-----------------------------------------------------------


## **4.1 Introduction to hyperparameter tuning**



![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/1-5.png?w=635)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/2-5.png?w=871)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/3-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/4-5.png?w=759)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/5-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/6-5.png?w=940)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/7-5.png?w=890)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/8-5.png?w=825)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/9-5.png?w=551)



### **4.1.1 Creating Hyperparameters**



 For a school assignment, your professor has asked your class to create a random forest model to predict the average test score for the final exam.




 After developing an initial random forest model, you are unsatisfied with the overall accuracy. You realize that there are too many hyperparameters to choose from, and each one has
 *a lot*
 of possible values. You have decided to make a list of possible ranges for the hyperparameters you might use in your next model.




 Your professor has provided de-identified data for the last ten quizzes to act as the training data. There are 30 students in your class.





```python

# Review the parameters of rfr
print(rfr.get_params())

# Maximum Depth
max_depth = [4, 8, 12]

# Minimum samples for a split
min_samples_split = [2, 5, 10]

# Max features
max_features = [4, 6, 8, 10]

```




```

{'bootstrap': True, 'criterion': 'mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 'warn', 'n_jobs': None, 'oob_score': False, 'random_state': 1111, 'verbose': 0, 'warm_start': False}

```



 Good job! Hyperparameter tuning requires selecting parameters to tune, as well the possible values these parameters can be set to.



### **4.1.2 Running a model using ranges**



 You have just finished creating a list of hyperparameters and ranges to use when tuning a predictive model for an assignment. You have used
 `max_depth`
 ,
 `min_samples_split`
 , and
 `max_features`
 as your range variable names.





```

from sklearn.ensemble import RandomForestRegressor

# Fill in rfr using your variables
rfr = RandomForestRegressor(
    n_estimators=100,
    max_depth=random.choice(max_depth),
    min_samples_split=random.choice(min_samples_split),
    max_features=random.choice(max_features))

# Print out the parameters
print(rfr.get_params())

```




```

{'bootstrap': True, 'criterion': 'mse', 'max_depth': 12, 'max_features': 10, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}

```



 Good job! Notice that
 `min_samples_split`
 was randomly set to 2. Since you specified a random state,
 `min_samples_split`
 will always be set to 2 if you only run this model one time.





---


## **4.2 RandomizedSearchCV**



![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/10-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/11-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/12-5.png?w=482)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/13-5.png?w=953)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/14-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/15-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/16-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/17-4.png?w=1024)



### **4.2.1 Preparing for RandomizedSearch**



 Last semester your professor challenged your class to build a predictive model to predict final exam test scores. You tried running a few different models by randomly selecting hyperparameters. However, running each model required you to code it individually.




 After learning about
 `RandomizedSearchCV()`
 , you’re revisiting your professors challenge to build the best model. In this exercise, you will prepare the three necessary inputs for completing a random search.





```

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error

# Finish the dictionary by adding the max_depth parameter
param_dist = {"max_depth": [2, 4, 6, 8],
              "max_features": [2, 4, 6, 8, 10],
              "min_samples_split": [2, 4, 8, 16]}

# Create a random forest regression model
rfr = RandomForestRegressor(n_estimators=10, random_state=1111)

# Create a scorer to use (use the mean squared error)
scorer = make_scorer(mean_squared_error)

```



 Well done! To use
 `RandomizedSearchCV()`
 , you need a distribution dictionary, an estimator, and a scorer—once you’ve got these, you can run a random search to find the best parameters for your model.



### **4.2.2 Implementing RandomizedSearchCV**



 You are hoping that using a random search algorithm will help you improve predictions for a class assignment. You professor has challenged your class to predict the overall final exam average score.




 In preparation for completing a random search, you have created:



* `param_dist`
 : the hyperparameter distributions
* `rfr`
 : a random forest regression model
* `scorer`
 : a scoring method to use




```python

# Import the method for random search
from sklearn.model_selection import RandomizedSearchCV

# Build a random search using param_dist, rfr, and scorer
random_search =\
    RandomizedSearchCV(
        estimator=rfr,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring=scorer)

```



 Nice! Although it takes a lot of steps, hyperparameter tuning with random search is well worth it and can improve the accuracy of your models. Plus, you are already using cross-validation to validate your best model.





---


## **4.3 Selecting your final model**



![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/18-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/19-2.png?w=945)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/20-2.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/21-1.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/22-1.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/23-1.png?w=797)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/24-1.png?w=763)



### **4.3.1 Best classification accuracy**



 You are in a competition at work to build the best model for predicting the winner of a Tic-Tac-Toe game. You already ran a random search and saved the results of the most accurate model to
 `rs`
 .




 Which parameter set produces the best classification accuracy?





```

rs.best_estimator_
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=12, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=4,
            min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=None,
            oob_score=False, random_state=1111, verbose=0,
            warm_start=False)

```



 Perfect! These parameters do produce the best testing accuracy. Good job! Remember, to reuse this model you can use
 `rs.best_estimator_`
 .



### **4.3.2 Selecting the best precision model**



 Your boss has offered to pay for you to see three sports games this year. Of the 41 home games your favorite team plays, you want to ensure you go to three home games that they will
 *definitely*
 win. You build a model to decide which games your team will win.




 To do this, you will build a random search algorithm and focus on model precision (to ensure your team wins). You also want to keep track of your best model and best parameters, so that you can use them again next year (if the model does well, of course). You have already decided on using the random forest classification model
 `rfc`
 and generated a parameter distribution
 `param_dist`
 .





```

from sklearn.metrics import precision_score, make_scorer

# Create a precision scorer
precision = make_scorer(precision_score)
# Finalize the random search
rs = RandomizedSearchCV(
  estimator=rfc, param_distributions=param_dist,
  scoring = precision,
  cv=5, n_iter=10, random_state=1111)
rs.fit(X, y)

# print the mean test scores:
print('The accuracy for each run was: {}.'.format(rs.cv_results_['mean_test_score']))
# print the best model score:
print('The best accuracy for a single model was: {}'.format(rs.best_score_))

```




```

The accuracy for each run was: [0.86446668 0.75302055 0.67570816 0.88459939 0.88381178 0.86917588
 0.68014695 0.81721906 0.87895856 0.92917474].
The best accuracy for a single model was: 0.9291747446879924

```



 Wow – Your model’s precision was 93%! The best model accurately predicts a winning game 93% of the time. If you look at the mean test scores, you can tell some of the other parameter sets did really poorly. Also, since you used cross-validation, you can be confident in your predictions. Well done!





---


## **4.4 Course completed!**



![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/25-1.png?w=786)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/26-1.png?w=346)
![Desktop View]({{ site.baseurl }}/assets/datacamp/model-validation-in-python/27-1.png?w=1024)





---



 Thank you for reading and hope you’ve learned a lot.



