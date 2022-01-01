---
title: Machine Learning with the Experts School Budgets from Datacamp
date: 2021-12-07 11:22:07 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Machine Learning with the Experts: School Budgets from Datacamp
=====================================================================







 This is the memo of the 22th course of ‘Data Scientist with Python’ track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/machine-learning-with-the-experts-school-budgets)**
 .





---



# **1. Exploring the raw data**
------------------------------





 You’re going to be working with school district budget data. This data can be classified in many ways according to certain labels, e.g.
 `Function: Career & Academic Counseling`
 , or
 `Position_Type: Librarian`
 .




 Your goal is to develop a model that predicts the probability for each possible label by relying on some correctly labeled examples.




 This is a supervised Learning, because the model will be trained using labeled examples.



####
**What is the goal of the algorithm?**



 Your goal is to correctly label budget line items by training a supervised model to predict the probability of each possible label, taking most probable label as the correct label.




 It’s a classification problem, because predicted probabilities will be used to select a label class.




 Specifically, we have ourselves a multi-class-multi-label classification problem (quite a mouthful!), because there are 9 broad categories that each take on many possible sub-label instances.



###
**Exploring the data**


####
**Loading the data**




```

df = pd.read_csv('TrainingData.csv', index_col=0)

df.head(3)
                   Function          Use          Sharing   Reporting  \
198                NO_LABEL     NO_LABEL         NO_LABEL    NO_LABEL
209  Student Transportation     NO_LABEL  Shared Services  Non-School
750    Teacher Compensation  Instruction  School Reported      School

    Student_Type Position_Type               Object_Type     Pre_K  \
198     NO_LABEL      NO_LABEL                  NO_LABEL  NO_LABEL
209     NO_LABEL      NO_LABEL    Other Non-Compensation  NO_LABEL
750  Unspecified       Teacher  Base Salary/Compensation  Non PreK

      Operating_Status               Object_Description         ...          \
198      Non-Operating                   Supplemental *         ...
209  PreK-12 Operating  REPAIR AND MAINTENANCE SERVICES         ...
750  PreK-12 Operating     Personal Services - Teachers         ...

                  Sub_Object_Description Location_Description  FTE  \
198  Non-Certificated Salaries And Wages                  NaN  NaN
209                                  NaN      ADMIN. SERVICES  NaN
750                                  NaN                  NaN  1.0

                     Function_Description Facility_or_Department  \
198  Care and Upkeep of Building Services                    NaN
209             STUDENT TRANSPORT SERVICE                    NaN
750                                   NaN                    NaN

    Position_Extra     Total    Program_Description  \
198            NaN  -8291.86                    NaN
209            NaN    618.29   PUPIL TRANSPORTATION
750        TEACHER  49768.82  Instruction - Regular

                                      Fund_Description              Text_1
198  Title I - Disadvantaged Children/Targeted Assi...  TITLE I CARRYOVER
209                                       General Fund                 NaN
750                             General Purpose School                 NaN

[3 rows x 25 columns]


```




```

df.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1560 entries, 198 to 101861
Data columns (total 25 columns):
Function                  1560 non-null object
Use                       1560 non-null object
Sharing                   1560 non-null object
Reporting                 1560 non-null object
Student_Type              1560 non-null object
Position_Type             1560 non-null object
Object_Type               1560 non-null object
Pre_K                     1560 non-null object
Operating_Status          1560 non-null object
Object_Description        1461 non-null object
Text_2                    382 non-null object
SubFund_Description       1183 non-null object
Job_Title_Description     1131 non-null object
Text_3                    677 non-null object
Text_4                    193 non-null object
Sub_Object_Description    364 non-null object
Location_Description      874 non-null object
FTE                       449 non-null float64
Function_Description      1340 non-null object
Facility_or_Department    252 non-null object
Position_Extra            1026 non-null object
Total                     1542 non-null float64
Program_Description       1192 non-null object
Fund_Description          819 non-null object
Text_1                    1132 non-null object
dtypes: float64(2), object(23)
memory usage: 316.9+ KB

```


####
**Summarizing the data**



 There are two numeric columns, called
 `FTE`
 and
 `Total`
 .



* `FTE`
 : Stands for “full-time equivalent”. If the budget item is associated to an employee, this number tells us the percentage of full-time that the employee works. A value of 1 means the associated employee works for the school full-time. A value close to 0 means the item is associated to a part-time or contracted employee.
* `Total`
 : Stands for the total cost of the expenditure. This number tells us how much the budget item cost.



 After printing summary statistics for the numeric data, your job is to plot a histogram of the non-null
 `FTE`
 column to see the distribution of part-time and full-time employees in the dataset.





```python

# Print the summary statistics
print(df.describe())

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Create the histogram
plt.hist(df.FTE.dropna())

# Add title and labels
plt.title('Distribution of %full-time \n employee works')
plt.xlabel('% of full-time')
plt.ylabel('num employees')

# Display the histogram
plt.show()

```




```

              FTE         Total
count  449.000000  1.542000e+03
mean     0.493532  1.446867e+04
std      0.452844  7.916752e+04
min     -0.002369 -1.044084e+06
25%           NaN           NaN
50%           NaN           NaN
75%           NaN           NaN
max      1.047222  1.367500e+06

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/machine-learning-with-the-experts:-school-budgets-from datacamp/capture-23.png?w=1024)


 The high variance in expenditures makes sense (some purchases are cheap some are expensive). Also, it looks like the FTE column is bimodal. That is, there are some part-time and some full-time employees.



###
**Covert object data to category**


####
**Count the data type and value**




```

df.dtypes.value_counts()
object     23
float64     2
dtype: int64

```


####
**Encode the labels as categorical variables**



 There are 9 columns of labels in the dataset. Each of these columns is a category that has
 [many possible values it can take](https://www.drivendata.org/competitions/4/box-plots-for-education/page/15/#labels_list)
 . The 9 labels have been loaded into a list called
 `LABELS`
 .




 You will notice that every label is encoded as an object datatype. Because
 `category`
 datatypes are
 [much more efficient](http://matthewrocklin.com/blog/work/2015/06/18/Categoricals)
 your task is to convert the labels to category types using the
 `.astype()`
 method.





```

LABELS
['Function',
 'Use',
 'Sharing',
 'Reporting',
 'Student_Type',
 'Position_Type',
 'Object_Type',
 'Pre_K',
 'Operating_Status']


df[LABELS].dtypes
Function            object
Use                 object
Sharing             object
Reporting           object
Student_Type        object
Position_Type       object
Object_Type         object
Pre_K               object
Operating_Status    object
dtype: object

```




```python

# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
df[LABELS] = df[LABELS].apply(categorize_label, axis=0)

# Print the converted dtypes
print(df[LABELS].dtypes)


```




```

Function            category
Use                 category
Sharing             category
Reporting           category
Student_Type        category
Position_Type       category
Object_Type         category
Pre_K               category
Operating_Status    category
dtype: object

```




```

'''
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html
apply
axis : {0 or ‘index’, 1 or ‘columns’}, default 0
Axis along which the function is applied:

0 or ‘index’: apply function to each column.
1 or ‘columns’: apply function to each row.
'''

```


####
**Counting unique labels

 .apply(pd.Series.nunique)**




```python

# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Calculate number of unique values for each label: num_unique_labels
num_unique_labels = df[LABELS].apply(pd.Series.nunique)

# Plot number of unique values for each label
num_unique_labels.plot(kind='bar')

# Label the axes
plt.xlabel('Labels')
plt.ylabel('Number of unique values')

# Display the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/machine-learning-with-the-experts:-school-budgets-from datacamp/capture1-22.png?w=1024)



---


###
**How do we measure success?**



 We can use a loss function.


[What is Log Loss?](https://www.kaggle.com/dansbecker/what-is-log-loss)




![Desktop View]({{ site.baseurl }}/assets/datacamp/machine-learning-with-the-experts:-school-budgets-from datacamp/capture2-21.png?w=1024)

####
**Penalizing highly confident wrong answers**



 Log loss provides a steep penalty for predictions that are both wrong and confident, i.e., a high probability is assigned to the incorrect class.




 Suppose you have the following 3 examples:



* A:y=1,p=0.85
* B:y=0,p=0.99
* C:y=0,p=0.51



 Select the ordering of the examples which corresponds to the lowest to highest log loss scores.
 `y`
 is an indicator of whether the example was classified correctly. You shouldn’t need to crunch any numbers!




 Of the two incorrect predictions,
 `B`
 will have a higher log loss because it is confident
 *and*
 wrong.




 Lowest: A, Middle: C, Highest: B


**The lower loss score is ,the better the prediction is.**



####
**Computing log loss with NumPy**



 To see how the log loss metric handles the trade-off between accuracy and confidence, we will use some sample data generated with NumPy and compute the log loss using the provided function
 `compute_log_loss()`
 .




 Your job is to compute the log loss for each sample set provided using the
 `compute_log_loss(predicted_values, actual_values)`
 . It takes the predicted values as the first argument and the actual values as the second argument.





```

actual_labels
# array([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.])

correct_confident
# array([0.95, 0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05, 0.05])

correct_not_confident
# array([0.65, 0.65, 0.65, 0.65, 0.65, 0.35, 0.35, 0.35, 0.35, 0.35])

wrong_not_confident
# array([0.35, 0.35, 0.35, 0.35, 0.35, 0.65, 0.65, 0.65, 0.65, 0.65])

wrong_confident
# array([0.05, 0.05, 0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95])

```




```python

# Compute and print log loss for 1st case
correct_confident_loss = compute_log_loss(correct_confident, actual_labels)
print("Log loss, correct and confident: {}".format(correct_confident_loss))

# Compute log loss for 2nd case
correct_not_confident_loss = compute_log_loss(correct_not_confident, actual_labels)
print("Log loss, correct and not confident: {}".format(correct_not_confident_loss))

# Compute and print log loss for 3rd case
wrong_not_confident_loss = compute_log_loss(wrong_not_confident, actual_labels)
print("Log loss, wrong and not confident: {}".format(wrong_not_confident_loss))

# Compute and print log loss for 4th case
wrong_confident_loss = compute_log_loss(wrong_confident, actual_labels)
print("Log loss, wrong and confident: {}".format(wrong_confident_loss))

# Compute and print log loss for actual labels
actual_labels_loss = compute_log_loss(actual_labels, actual_labels)
print("Log loss, actual labels: {}".format(actual_labels_loss))


```




```

Log loss, correct and confident: 0.05129329438755058
Log loss, correct and not confident: 0.4307829160924542
Log loss, wrong and not confident: 1.049822124498678
Log loss, wrong and confident: 2.9957322735539904
Log loss, actual labels: 9.99200722162646e-15

```



 Log loss penalizes highly confident wrong answers much more than any other type. This will be a good metric to use on your models.





---



# **2. Creating a simple first model**
-------------------------------------


###
**It’s time to build a model**


####
**Setting up a train-test split in scikit-learn**



 It’s finally time to start training models!




 The first step is to split the data into a training set and a test set. Some labels don’t occur very often, but we want to make sure that they appear in both the training and the test sets. We provide a function that will make sure at least
 `min_count`
 examples of each label appear in each split:
 `multilabel_train_test_split`
 .




 Feel free to check out the full code for
 `multilabel_train_test_split`
[here](https://github.com/drivendataorg/box-plots-sklearn/blob/master/src/data/multilabel.py)
 .




 You’ll start with a simple model that uses
 **just the numeric columns**
 of your DataFrame when calling
 `multilabel_train_test_split`
 . The data has been read into a DataFrame
 `df`
 and a list consisting of just the numeric columns is available as
 `NUMERIC_COLUMNS`
 .





```python

# X
df[NUMERIC_COLUMNS].head(3)
     FTE     Total
198  NaN  -8291.86
209  NaN    618.29
750  1.0  49768.82

df[NUMERIC_COLUMNS].shape
(1560, 2)


# y
label_dummies.columns
Out[16]:
Index(['Function_Aides Compensation', 'Function_Career & Academic Counseling',
       'Function_Communications', 'Function_Curriculum Development',
       'Function_Data Processing & Information Services',
       ...
       'Operating_Status_Non-Operating',
       'Operating_Status_Operating, Not PreK-12',
       'Operating_Status_PreK-12 Operating'],
      dtype='object', length=104)

label_dummies.shape
(1560, 104)

```




```python

# Create the new DataFrame: numeric_data_only
numeric_data_only = df[NUMERIC_COLUMNS].fillna(-1000)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(df[LABELS])

# Create training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(numeric_data_only,
                                                               label_dummies,
                                                               size=0.2,
                                                               seed=123)

# Print the info
print("X_train info:")
print(X_train.info())
print("\nX_test info:")
print(X_test.info())
print("\ny_train info:")
print(y_train.info())
print("\ny_test info:")
print(y_test.info())

```




```

X_train info:
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1040 entries, 198 to 101861
Data columns (total 2 columns):
FTE      1040 non-null float64
Total    1040 non-null float64
dtypes: float64(2)
memory usage: 24.4 KB
None

X_test info:
<class 'pandas.core.frame.DataFrame'>
Int64Index: 520 entries, 209 to 448628
Data columns (total 2 columns):
FTE      520 non-null float64
Total    520 non-null float64
dtypes: float64(2)
memory usage: 12.2 KB
None

y_train info:
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1040 entries, 198 to 101861
Columns: 104 entries, Function_Aides Compensation to Operating_Status_PreK-12 Operating
dtypes: float64(104)
memory usage: 853.1 KB
None

y_test info:
<class 'pandas.core.frame.DataFrame'>
Int64Index: 520 entries, 209 to 448628
Columns: 104 entries, Function_Aides Compensation to Operating_Status_PreK-12 Operating
dtypes: float64(104)
memory usage: 426.6 KB
None

```


####
**Training a model**




```python

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Create the DataFrame: numeric_data_only
numeric_data_only = df[NUMERIC_COLUMNS].fillna(-1000)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(df[LABELS])

# Create training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(numeric_data_only,
                                                               label_dummies,
                                                               size=0.2,
                                                               seed=123)

# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression())

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Print the accuracy
print("Accuracy: {}".format(clf.score(X_test, y_test)))

```




```

Accuracy: 0.0

```



 The good news is that your workflow didn’t cause any errors. The bad news is that your model scored the
 **lowest possible accuracy: 0.0**
 !


 But hey, you just threw away ALL of the text data in the budget. Later, you won’t. Before you add the text data, let’s see how the model does when scored by log loss.



###
**Making predictions**


####
**Use your model to predict values on holdout data**



 You’re ready to make some predictions! Remember, the train-test-split you’ve carried out so far is for model development. The original competition provides an additional test set, for which you’ll never actually
 *see*
 the correct labels. This is called the “holdout data.”




 Remember that the original goal is to predict the
 **probability of each label**
 . In this exercise you’ll do just that by using the
 `.predict_proba()`
 method on your trained model.





```python

# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression())

# Fit it to the training data
clf.fit(X_train, y_train)

# Load the holdout data: holdout
holdout = pd.read_csv('HoldoutData.csv', index_col=0)
holdout = holdout[NUMERIC_COLUMNS].fillna(-1000)

# Generate predictions: predictions
predictions = clf.predict_proba(holdout)

```


####
**Writing out your results to a csv for submission**



 At last, you’re ready to submit some predictions for scoring. In this exercise, you’ll write your predictions to a
 `.csv`
 using the
 `.to_csv()`
 method on a pandas DataFrame. Then you’ll evaluate your performance according to the LogLoss metric!




 You’ll need to make sure your submission obeys the
 [correct format](https://www.drivendata.org/competitions/4/page/15/#sub_values)
 .




**Interpreting LogLoss & Beating the Benchmark:**




 When interpreting your log loss score, keep in mind that the score will change based on the number of samples tested. To get a sense of how this
 *very basic*
 model performs, compare your score to the
 **DrivenData benchmark model performance: 2.0455**
 , which merely submitted uniform probabilities for each class.




 Remember, the lower the log loss the better. Is your model’s log loss lower than 2.0455?





```python

# Generate predictions: predictions
predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS].fillna(-1000))

# Format predictions in DataFrame: prediction_df
prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS]).columns,
                             index=holdout.index,
                             data=predictions)


# Save prediction_df to csv
prediction_df.to_csv('predictions.csv')

# Submit the predictions for scoring: score
score = score_submission(pred_path='predictions.csv')

# Print score
print('Your model, trained with numeric data only, yields logloss score: {}'.format(score))

```




```

predictions
array([[0.108789  , 0.04790553, 0.02505001, ..., 0.13005936, 0.03531605,
        0.87679861],
       ...,
       [0.10169308, 0.04809784, 0.02423019, ..., 0.12078767, 0.03686144,
        0.88204561]])

predictions.shape
(2000, 104)


Your model, trained with numeric data only, yields logloss score: 1.9067227623381413

```



 Even though your basic model scored 0.0 accuracy, it nevertheless performs better than the benchmark score of 2.0455. You’ve now got the basics down and have made a first pass at this complicated supervised learning problem. It’s time to step up your game and incorporate the text data.





---


###
**A very brief introduction to NLP(Natural Language Processing)**


####
**Tokenizing text**



[Tokenization](http://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html)
 is the process of chopping up a character sequence into pieces called
 *tokens*
 .




 How do we determine what constitutes a token? Often, tokens are separated by whitespace. But we can specify other delimiters as well. For example, if we decided to tokenize on punctuation, then any punctuation mark would be treated like a whitespace. How we tokenize text in our DataFrame can affect the statistics we use in our model.




 A particular cell in our budget DataFrame may have the string content
 `Title I - Disadvantaged Children/Targeted Assistance`
 . The number of n-grams generated by this text data is sensitive to whether or not we tokenize on punctuation.




 How many tokens (1-grams) are in the string




 Title I – Disadvantaged Children/Targeted Assistance




 if we tokenize on punctuation?




 6 tokens(1-grams)



* Title
* I
* Disadvantaged
* Children
* Targeted
* Assistance


####
**n_grams token**


* one_grams = [‘petro’, ‘vend’, ‘fuel’, ‘and’, ‘fluids’]
* two _grams = [‘petro vend’, ‘vend fuel’, ‘fuel and’, ‘and fluids’]
* three _grams = [‘petro vend fuel’, ‘vend fuel and’, ‘ fuel and fluids’]


###
**Representing text numerically**


####
**Creating a bag-of-words in scikit-learn**



 In this exercise, you’ll study the effects of tokenizing in different ways by comparing the bag-of-words representations resulting from different token patterns.




 You will focus on one feature only, the
 `Position_Extra`
 column, which describes any additional information not captured by the
 `Position_Type`
 label.




 Your task is to turn the raw text in this column into a bag-of-words representation by creating tokens that contain
 *only*
 alphanumeric characters.




 For comparison purposes, the first 15 tokens of
 `vec_basic`
 , which splits
 `df.Position_Extra`
 into tokens when it encounters only
 *whitespace*
 characters, have been printed along with the length of the representation.





```python

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Fill missing values in df.Position_Extra
df.Position_Extra.fillna('', inplace=True)

# Instantiate the CountVectorizer: vec_alphanumeric
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Fit to the data
vec_alphanumeric.fit(df.Position_Extra)

# Print the number of tokens and first 15 tokens
msg = "There are {} tokens in Position_Extra if we split on non-alpha numeric"
print(msg.format(len(vec_alphanumeric.get_feature_names())))
print(vec_alphanumeric.get_feature_names()[:15])


```




```

There are 123 tokens in Position_Extra if we split on non-alpha numeric
['1st', '2nd', '3rd', 'a', 'ab', 'additional', 'adm', 'administrative', 'and', 'any', 'art', 'assessment', 'assistant', 'asst', 'athletic']

```



 Treating only alpha-numeric characters as tokens gives you a smaller number of more meaningful tokens. You’ve got bag-of-words in the bag!



####
**Combining text columns for tokenization**



 In order to get a bag-of-words representation for all of the text data in our DataFrame, you must first convert the text data in each row of the DataFrame into a single string.




 In the previous exercise, this wasn’t necessary because you only looked at one column of data, so each row was already just a single string.
 `CountVectorizer`
 expects each row to just be a single string, so in order to use all of the text columns, you’ll need a method to turn a list of strings into a single string.




 In this exercise, you’ll complete the function definition
 `combine_text_columns()`
 . When completed, this function will convert all training text data in your DataFrame to a single string per row that can be passed to the vectorizer object and made into a bag-of-words using the
 `.fit_transform()`
 method.




 Note that the function uses
 `NUMERIC_COLUMNS`
 and
 `LABELS`
 to determine which columns to drop. These lists have been loaded into the workspace.





```python

# Define combine_text_columns()
def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """ converts all text in each row of data_frame to single vector """

    # Drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1)

    # Replace nans with blanks
    text_data.fillna('', inplace=True)

    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)

```


####
**What’s in a token?**



 Now you will use
 `combine_text_columns`
 to convert all training text data in your DataFrame to a single vector that can be passed to the vectorizer object and made into a bag-of-words using the
 `.fit_transform()`
 method.




 You’ll compare the effect of tokenizing using any non-whitespace characters as a token and using only alphanumeric characters as a token.





```python

# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create the basic token pattern
TOKENS_BASIC = '\\S+(?=\\s+)'

# Create the alphanumeric token pattern
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate basic CountVectorizer: vec_basic
vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC)

# Instantiate alphanumeric CountVectorizer: vec_alphanumeric
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Create the text vector
text_vector = combine_text_columns(df)

# Fit and transform vec_basic
vec_basic.fit_transform(text_vector)

# Print number of tokens of vec_basic
print("There are {} tokens in the dataset".format(len(vec_basic.get_feature_names())))

# Fit and transform vec_alphanumeric
vec_alphanumeric.fit_transform(text_vector)

# Print number of tokens of vec_alphanumeric
print("There are {} alpha-numeric tokens in the dataset".format(len(vec_alphanumeric.get_feature_names())))

```




```

There are 1405 tokens in the dataset
There are 1117 alpha-numeric tokens in the dataset

```




```

text_vector
198       Supplemental *  Operation and Maintenance of P...
209       REPAIR AND MAINTENANCE SERVICES  PUPIL TRANSPO...
                                ...
305347    Extra Duty Pay/Overtime For Support Personnel ...
101861    SALARIES OF REGULAR EMPLOYEES  FEDERAL GDPG FU...
dtype: object

```



 Notice that tokenizing on alpha-numeric tokens reduced the number of tokens. We’ll keep this in mind when building a better model with the Pipeline object next.





---



# **3. Improving your model**
----------------------------


###
**Pipelines, feature & text preprocessing**



**What is pipeline?**




 You can look up at document
 [here](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline)
 . And
 [here](https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html)
 is an example.


 Simply speaking, pipeline just put several machine learning steps(like scaling data, parameter tuning) together.




**Why should you use pipeline?**



* Your code will be much easier to read
* You can easily modify the parameters


####
**Instantiate pipeline**



 For the next few exercises, you’ll reacquaint yourself with pipelines and train a classifier on some synthetic (sample) data of multiple datatypes before using the same techniques on the main dataset.




 In this exercise, your job is to instantiate a pipeline that trains using the
 `numeric`
 column of the sample data.





```

     numeric     text  with_missing label
0 -10.856306               4.433240     b
1   9.973454      foo      4.310229     b
2   2.829785  foo bar      2.469828     a
3 -15.062947               2.852981     b
4  -5.786003  foo bar      1.826475     a


pd.get_dummies(sample_df['label'])
       a    b
0    0.0  1.0
1    0.0  1.0
2    1.0  0.0
3    0.0  1.0

```




```python

# Import Pipeline
from sklearn.pipeline import Pipeline

# Import other necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Split and select numeric data only, no nans
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric']],
                                                    pd.get_dummies(sample_df['label']),
                                                    random_state=22)

# Instantiate Pipeline object: pl
pl = Pipeline([
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit the pipeline to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - numeric, no nans: ", accuracy)

# Accuracy on sample data - numeric, no nans:  0.62

```



 Now it’s time to incorporate numeric data with missing values by adding a preprocessing step!



####
**Preprocessing numeric features**



 In this exercise you’ll improve your pipeline a bit by using the
 `Imputer()`
 imputation transformer from scikit-learn to fill in missing values in your sample data.




 By default, the
 [imputer transformer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html)
 replaces NaNs with the mean value of the column. That’s a good enough imputation strategy for the sample data, so you won’t need to pass anything extra to the imputer.




 After importing the transformer, you will edit the steps list used in the previous exercise by inserting a
 `(name, transform)`
 tuple. Recall that steps are processed sequentially, so make sure the new tuple encoding your
 *preprocessing*
 step is put in the right place.





```python

# Import the Imputer object
from sklearn.preprocessing import Imputer

# Create training and test sets using only numeric data
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing']],
                                                    pd.get_dummies(sample_df['label']),
                                                    random_state=456)

# Insantiate Pipeline object: pl
pl = Pipeline([
        ('imp', Imputer()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit the pipeline to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all numeric, incl nans: ", accuracy)

# Accuracy on sample data - all numeric, incl nans:  0.636

```



 Now you know how to use preprocessing in pipelines with numeric data, and it looks like the accuracy has improved because of it! Text data preprocessing is next!





---


###
**Text features and feature unions**


####
**Preprocessing text features**




```

sample_df['text']
0
1          foo
2      foo bar
3
4      foo bar
5
6      foo bar
7          foo

```




```python

# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Split out only the text data
X_train, X_test, y_train, y_test = train_test_split(sample_df['text'],
                                                    pd.get_dummies(sample_df['label']),
                                                    random_state=456)

# Instantiate Pipeline object: pl
pl = Pipeline([
        ('vec', CountVectorizer()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - just text data: ", accuracy)

# Accuracy on sample data - just text data:  0.808

```


####
**Multiple types of processing: FunctionTransformer**



 The next two exercises will introduce new topics you’ll need to make your pipeline truly excel.




 Any step in the pipeline
 *must*
 be an object that implements the
 `fit`
 and
 `transform`
 methods. The
 [`FunctionTransformer`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
 creates an object with these methods out of any Python function that you pass to it. We’ll use it to help select subsets of data in a way that plays nicely with pipelines.




 You are working with numeric data that needs imputation, and text data that needs to be converted into a bag-of-words. You’ll create functions that separate the text from the numeric variables and see how the
 `.fit()`
 and
 `.transform()`
 methods work.





```python

# Import FunctionTransformer
from sklearn.preprocessing import FunctionTransformer

# Obtain the text data: get_text_data
get_text_data = FunctionTransformer(lambda x: x['text'], validate=False)

# Obtain the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[['numeric', 'with_missing']], validate=False)

# Fit and transform the text data: just_text_data
just_text_data = get_text_data.fit_transform(sample_df)

# Fit and transform the numeric data: just_numeric_data
just_numeric_data = get_numeric_data.fit_transform(sample_df)

# Print head to check results
print('Text Data')
print(just_text_data.head())
print('\nNumeric Data')
print(just_numeric_data.head())

```




```

Text Data
0
1        foo
2    foo bar
3
4    foo bar
Name: text, dtype: object

Numeric Data
     numeric  with_missing
0 -10.856306      4.433240
1   9.973454      4.310229
2   2.829785      2.469828
3 -15.062947      2.852981
4  -5.786003      1.826475

```


####
**Multiple types of processing: FeatureUnion**



 Now that you can separate text and numeric data in your pipeline, you’re ready to perform separate steps on each by nesting pipelines and using
 `FeatureUnion()`
 .




 These tools will allow you to streamline all preprocessing steps for your model, even when multiple datatypes are involved. Here, for example, you don’t want to impute our text data, and you don’t want to create a bag-of-words with our numeric data. Instead, you want to deal with these separately and then join the results together using
 `FeatureUnion()`
 .




 In the end, you’ll still have only two high-level steps in your pipeline: preprocessing and model instantiation. The difference is that the first preprocessing step actually consists of a pipeline for numeric data and a pipeline for text data. The results of those pipelines are joined using
 `FeatureUnion()`
 .





```python

# Import FeatureUnion
from sklearn.pipeline import FeatureUnion

# Split using ALL data in sample_df
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing', 'text']],
                                                    pd.get_dummies(sample_df['label']),
                                                    random_state=22)

# Create a FeatureUnion with nested pipeline: process_and_join_features
process_and_join_features = FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )

# Instantiate nested pipeline: pl
pl = Pipeline([
        ('union', process_and_join_features),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])


# Fit pl to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all data: ", accuracy)

# Accuracy on sample data - all data:  0.928

```




---


###
**Choosing a classification model**


####
**Using FunctionTransformer on the main dataset**



 In this exercise you’re going to use
 `FunctionTransformer`
 on the primary budget data, before instantiating a multiple-datatype pipeline in the next exercise.





```python

# Import FunctionTransformer
from sklearn.preprocessing import FunctionTransformer

# Get the dummy encoding of the labels
dummy_labels = pd.get_dummies(df[LABELS])

# Get the columns that are features in the original df
NON_LABELS = [c for c in df.columns if c not in LABELS]

# Split into training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(df[NON_LABELS],
                                                               dummy_labels,
                                                               0.2,
                                                               seed=123)

# Preprocess the text data: get_text_data
get_text_data = FunctionTransformer(combine_text_columns, validate=False)

# Preprocess the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)


```


####
**Add a model to the pipeline**



 You’re about to take everything you’ve learned so far and implement it in a
 `Pipeline`
 that works with the real,
 [DrivenData](https://www.drivendata.org/)
 budget line item data you’ve been exploring.




**Surprise!**
 The structure of the pipeline is exactly the same as earlier in this chapter:



* the
 **preprocessing step**
 uses
 `FeatureUnion`
 to join the results of nested pipelines that each rely on
 `FunctionTransformer`
 to select multiple datatypes
* the
 **model step**
 stores the model object



 You can then call familiar methods like
 `.fit()`
 and
 `.score()`
 on the
 `Pipeline`
 object
 `pl`
 .





```python

# Complete the pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)

# Accuracy on budget dataset:  0.20384615384615384

```


####
**Try a different class of model**



 Now you’re cruising. One of the great strengths of pipelines is how easy they make the process of testing different models.




 Until now, you’ve been using the model step
 `('clf', OneVsRestClassifier(LogisticRegression()))`
 in your pipeline.




 But what if you want to try a different model? Do you need to build an entirely new pipeline? New nests? New FeatureUnions? Nope! You just have a simple one-line change, as you’ll see in this exercise.




 In particular, you’ll swap out the logistic-regression model and replace it with a
 [random forest](https://en.wikipedia.org/wiki/Random_forest)
 classifier, which uses the statistics of an ensemble of decision trees to generate predictions.





```python

# Import random forest classifer
from sklearn.ensemble import RandomForestClassifier

# Edit model step in pipeline
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', RandomForestClassifier())
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)

# Accuracy on budget dataset:  0.2826923076923077

```



 An accuracy improvement- amazing! All your work building the pipeline is paying off. It’s now very simple to test different models!



####
**Can you adjust the model or parameters to improve accuracy?**



 You just saw a substantial improvement in accuracy by swapping out the model. Pipelines are amazing!




 Can you make it better? Try changing the parameter
 `n_estimators`
 of
 `RandomForestClassifier()`
 , whose default value is
 `10`
 , to
 `15`
 .





```python

# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Add model step to pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', RandomForestClassifier(n_estimators=15))
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)

# Accuracy on budget dataset:  0.3211538461538462

```



 Wow, you’re becoming a master! It’s time to get serious and work with the log loss metric. You’ll learn expert techniques in the next chapter to take the model to the next level.





---



# **4. Learning from the experts**
---------------------------------


###
**Learning from the expert: processing**


####
**Deciding what’s a word**



 Before you build up to the winning pipeline, it will be useful to look a little deeper into how the text features will be processed.




 In this exercise, you will use
 `CountVectorizer`
 on the training data
 `X_train`
 (preloaded into the workspace) to see the effect of tokenization on punctuation.




 Remember, since
 `CountVectorizer`
 expects a vector, you’ll need to use the preloaded function,
 `combine_text_columns`
 before fitting to the training data.





```python

# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create the text vector
text_vector = combine_text_columns(X_train)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate the CountVectorizer: text_features
text_features = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Fit text_features to the text vector
text_features.fit(text_vector)

# Print the first 10 tokens
print(text_features.get_feature_names()[:10])

# ['00a', '12', '1st', '2nd', '3rd', '5th', '70', '70h', '8', 'a']

```




```

text_features.get_feature_names()
['00a',
 '12',
 '1st',
 '2nd',
 '3rd',
 '5th',
 '70',
 '70h',
 '8',
 'a',
 'aaps',
 'ab',
 'acad',
 'academ',
 'academic',
 'accelerated',
 'access',
...

```




---


####
**N-gram range in scikit-learn**



 In this exercise you’ll insert a
 `CountVectorizer`
 instance into your pipeline for the main dataset, and compute multiple n-gram features to be used in the model.




 In order to look for ngram relationships at multiple scales, you will use the
 `ngram_range`
 parameter as Peter discussed in the video.




**Special functions:**
 You’ll notice a couple of new steps provided in the pipeline in this and many of the remaining exercises. Specifically, the
 `dim_red`
 step following the
 `vectorizer`
 step , and the
 `scale`
 step preceeding the
 `clf`
 (classification) step.




 These have been added in order to account for the fact that you’re using a reduced-size sample of the full dataset in this course. To make sure the models perform as the expert competition winner intended, we have to apply a
 [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction)
 technique, which is what the
 `dim_red`
 step does, and we have to
 [scale the features](https://en.wikipedia.org/wiki/Feature_scaling)
 to lie between -1 and 1, which is what the
 `scale`
 step does.




 The
 `dim_red`
 step uses a scikit-learn function called
 `SelectKBest()`
 , applying something called the
 [chi-squared test](https://en.wikipedia.org/wiki/Chi-squared_test)
 to select the K “best” features. The
 `scale`
 step uses a scikit-learn function called
 `MaxAbsScaler()`
 in order to squash the relevant features into the interval -1 to 1.




 You won’t need to do anything extra with these functions here, just complete the vectorizing pipeline steps below. However, notice how easy it was to add more processing steps to our pipeline!





```python

# Import pipeline
from sklearn.pipeline import Pipeline

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Import other preprocessing modules
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import chi2, SelectKBest

# Select 300 best features
chi_k = 300

# Import functional utilities
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.pipeline import FeatureUnion

# Perform preprocessing
get_text_data = FunctionTransformer(combine_text_columns, validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1,2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

```



 Log loss score: 1.2681. Great work! You’ll now add some additional tricks to make the pipeline even better.





---


###
**Learning from the expert: a stats trick**


####
**Implement interaction modeling in scikit-learn**



 It’s time to add interaction features to your model. The
 `PolynomialFeatures`
 object in scikit-learn does just that, but here you’re going to use a custom interaction object,
 `SparseInteractions`
 . Interaction terms are a statistical tool that lets your model express what happens if two features appear together in the same row.




`SparseInteractions`
 does the same thing as
 `PolynomialFeatures`
 , but it uses sparse matrices to do so. You can get the code for
 `SparseInteractions`
 at
 [this GitHub Gist](https://github.com/drivendataorg/box-plots-sklearn/blob/master/src/features/SparseInteractions.py)
 .




`PolynomialFeatures`
 and
 `SparseInteractions`
 both take the argument
 `degree`
 , which tells them what polynomial degree of interactions to compute.




 You’re going to consider interaction terms of
 `degree=2`
 in your pipeline. You will insert these steps
 *after*
 the preprocessing steps you’ve built out so far, but
 *before*
 the classifier steps.




 Pipelines with interaction terms take a while to train (since you’re making n features into n-squared features!), so as long as you set it up right, we’ll do the heavy lifting and tell you what your score is!





```python

# Instantiate pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1, 2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

```



 Log loss score: 1.2256. Nice improvement from 1.2681! The student is becoming the master!



###
**Learning from the expert: a computational trick and the winning model**


####
**Why is hashing a useful trick?**



 A
 [hash](https://en.wikipedia.org/wiki/Feature_hashing#Feature_vectorization_using_the_hashing_trick)
 function takes an input, in your case a token, and outputs a hash value. For example, the input may be a string and the hash value may be an integer.




 We’ve loaded a familiar python datatype, a dictionary called
 `hash_dict`
 , that makes this mapping concept a bit more explicit. In fact,
 [python dictionaries ARE hash tables](http://stackoverflow.com/questions/114830/is-a-python-dictionary-an-example-of-a-hash-table)
 !




 Print
 `hash_dict`
 in the IPython Shell to get a sense of how strings can be mapped to integers.




 By explicitly stating how many possible outputs the hashing function may have, we limit the size of the objects that need to be processed. With these limits known, computation can be made more efficient and we can get results faster, even on large datasets.




**Some problems are memory-bound and not easily parallelizable, and hashing enforces a fixed length computation instead of using a mutable datatype (like a dictionary).**





```

hash_dict
{'and': 780, 'fluids': 354, 'fuel': 895, 'petro': 354, 'vend': 785}

```


####
**Implementing the hashing trick in scikit-learn**



 In this exercise you will check out the scikit-learn implementation of
 `HashingVectorizer`
 before adding it to your pipeline later.




 As you saw in the video,
 `HashingVectorizer`
 acts just like
 `CountVectorizer`
 in that it can accept
 `token_pattern`
 and
 `ngram_range`
 parameters. The important difference is that it creates hash values from the text, so that we get all the computational advantages of hashing!





```python

# Import HashingVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

# Get text data: text_data
text_data = combine_text_columns(X_train)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate the HashingVectorizer: hashing_vec
hashing_vec = HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Fit and transform the Hashing Vectorizer
hashed_text = hashing_vec.fit_transform(text_data)

# Create DataFrame and print the head
hashed_df = pd.DataFrame(hashed_text.data)
print(hashed_df.head())

```



 As you can see, some text is hashed to the same value, but this doesn’t neccessarily hurt performance.



####
**Build the winning model**



 You have arrived! This is where all of your hard work pays off. It’s time to build the model that won DrivenData’s competition.




 You’ve constructed a robust, powerful pipeline capable of processing training
 *and*
 testing data. Now that you understand the data and know all of the tools you need, you can essentially solve the whole problem in a relatively small number of lines of code. Wow!




 All you need to do is add the
 `HashingVectorizer`
 step to the pipeline to replace the
 `CountVectorizer`
 step.




 The parameters
 `non_negative=True`
 ,
 `norm=None`
 , and
 `binary=False`
 make the
 `HashingVectorizer`
 perform similarly to the default settings on the
 `CountVectorizer`
 so you can just replace one with the other.





```python

# Import the hashing vectorizer
from sklearn.feature_extraction.text import HashingVectorizer

# Instantiate the winning model pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                     non_negative=True, norm=None, binary=False,
                                                     ngram_range=(1,2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

```



 Log loss: 1.2258. Looks like the performance is about the same, but this is expected since the HashingVectorizer should work the same as the CountVectorizer. Try this pipeline out on the whole dataset on your local machine to see its full power!



####
**What tactics got the winner the best score?**



 The winner used skillful NLP, efficient computation, and simple but powerful stats tricks to master the budget data.




 Often times simpler is better, and understanding the problem in depth leads to simpler solutions!




[Full solution code url](https://github.com/datacamp/course-resources-ml-with-experts-budgets/blob/master/notebooks/1.0-full-model.ipynb)




 The End.


 Thank you for reading.



