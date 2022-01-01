---
title: Feature Engineering for Machine Learning in Python
date: 2021-07-06 18:36:55 +0100
categories: [Blogging, Demo]
tags: [typography]
math: true
mermaid: true
---
# Feature Engineering for Machine Learning in Python

This is the memo of the 10th course (23 courses in all) of ‘Machine Learning Scientist with Python’ skill track.\
**You can find the original course** [**HERE**](https://www.datacamp.com/courses/feature-engineering-for-machine-learning-in-python).

#### **Course Description**

Every day you read about the amazing breakthroughs in how the newest applications of machine learning are changing the world. Often this reporting glosses over the fact that a huge amount of data munging and feature engineering must be done before any of these fancy models can be used. In this course, you will learn how to do just that. You will work with Stack Overflow Developers survey, and historic US presidential inauguration addresses, to understand how best to preprocess and engineer features from categorical, continuous, and unstructured data. This course will give you hands-on experience on how to prepare any data for your own machine learning models.

#### **Table of contents**

1. [Creating Features](https://datascience103579984.wordpress.com/2020/01/03/feature-engineering-for-machine-learning-in-python-from-datacamp/)
2. [Dealing with Messy Data](https://datascience103579984.wordpress.com/2020/01/03/feature-engineering-for-machine-learning-in-python-from-datacamp/2/)
3. [Conforming to Statistical Assumptions](https://datascience103579984.wordpress.com/2020/01/03/feature-engineering-for-machine-learning-in-python-from-datacamp/3/)
4. [Dealing with Text Data](https://datascience103579984.wordpress.com/2020/01/03/feature-engineering-for-machine-learning-in-python-from-datacamp/4/)

### **1. Creating Features**


#### **1.1 Why generate features?**

* ![](https://datascience103579984.files.wordpress.com/2019/12/1-19.png?w=971)
* ![](https://datascience103579984.files.wordpress.com/2019/12/2-20.png?w=671)
* ![](https://datascience103579984.files.wordpress.com/2019/12/3-20.png?w=922)

**1.1.1 Getting to know your data**

Pandas is one the most popular packages used to work with tabular data in Python. It is generally imported using the alias `pd` and can be used to load a CSV (or other delimited files) using `read_csv()`.

You will be working with a modified subset of the [Stackoverflow survey response data](https://insights.stackoverflow.com/survey/2018/#overview) in the first three chapters of this course. This data set records the details, and preferences of thousands of users of the StackOverflow website.

```python
# Import pandas
import pandas as pd

# Import so_survey_csv into so_survey_df
so_survey_df = pd.read_csv(so_survey_csv)

# Print the first five rows of the DataFrame
print(so_survey_df.head())

# Print the data type of each column
print(so_survey_df.dtypes)
```

```python
      SurveyDate                                    FormalEducation  ConvertedSalary Hobby       Country  ...     VersionControl Age  Years Experience  Gender   RawSalary
0  2/28/18 20:20           Bachelor's degree (BA. BS. B.Eng.. etc.)              NaN   Yes  South Africa  ...                Git  21                13    Male         NaN
1  6/28/18 13:26           Bachelor's degree (BA. BS. B.Eng.. etc.)          70841.0   Yes       Sweeden  ...     Git;Subversion  38                 9    Male   70,841.00
2    6/6/18 3:37           Bachelor's degree (BA. BS. B.Eng.. etc.)              NaN    No       Sweeden  ...                Git  45                11     NaN         NaN
3    5/9/18 1:06  Some college/university study without earning ...          21426.0   Yes       Sweeden  ...  Zip file back-ups  46                12    Male   21,426.00
4  4/12/18 22:41           Bachelor's degree (BA. BS. B.Eng.. etc.)          41671.0   Yes            UK  ...                Git  39                 7    Male  £41,671.00

[5 rows x 11 columns]
SurveyDate                     object
FormalEducation                object
ConvertedSalary               float64
Hobby                          object
Country                        object
StackOverflowJobsRecommend    float64
VersionControl                 object
Age                             int64
Years Experience                int64
Gender                         object
RawSalary                      object
dtype: object
```

**1.1.2 Selecting specific data types**

Often a data set will contain columns with several different data types (like the one you are working with). The majority of machine learning models require you to have a consistent data type across features. Similarly, most feature engineering techniques are applicable to only one type of data at a time. For these reasons among others, you will often want to be able to access just the columns of certain types when working with a DataFrame.

```python
# Create subset of only the numeric columns
so_numeric_df = so_survey_df.select_dtypes(include=['int', 'float'])

# Print the column names contained in so_survey_df_num
print(so_numeric_df.columns)
# Index(['ConvertedSalary', 'StackOverflowJobsRecommend', 'Age', 'Years Experience'], dtype='object')
```

#### **1.2 Dealing with categorical features**

* ![](https://datascience103579984.files.wordpress.com/2019/12/4-20.png?w=987)
* ![](https://datascience103579984.files.wordpress.com/2019/12/5-20.png?w=770)
* ![](https://datascience103579984.files.wordpress.com/2019/12/6-20.png?w=766)
* ![](https://datascience103579984.files.wordpress.com/2019/12/7-20.png?w=959)
* ![](https://datascience103579984.files.wordpress.com/2019/12/8-19.png?w=725)
* ![](https://datascience103579984.files.wordpress.com/2019/12/9-19.png?w=682)
* ![](https://datascience103579984.files.wordpress.com/2019/12/10-19.png?w=902)

**1.2.1 One-hot encoding and dummy variables**

To use categorical variables in a machine learning model, you first need to represent them in a quantitative way. The two most common approaches are to one-hot encode the variables using or to use dummy variables. In this exercise, you will create both types of encoding, and compare the created column sets. We will continue using the same DataFrame from previous lesson loaded as `so_survey_df` and focusing on its `Country` column.

```python
# Convert the Country column to a one hot encoded Data Frame
one_hot_encoded = pd.get_dummies(so_survey_df, columns=['Country'], prefix='OH')

# Print the columns names
print(one_hot_encoded.columns)
```

```python
Index(['SurveyDate', 'FormalEducation', 'ConvertedSalary', 'Hobby', 'StackOverflowJobsRecommend', 'VersionControl', 'Age', 'Years Experience', 'Gender', 'RawSalary', 'OH_France', 'OH_India',
       'OH_Ireland', 'OH_Russia', 'OH_South Africa', 'OH_Spain', 'OH_Sweeden', 'OH_UK', 'OH_USA', 'OH_Ukraine'],
      dtype='object')
```

```python
# Create dummy variables for the Country column
dummy = pd.get_dummies(so_survey_df, columns=['Country'], drop_first=True, prefix='DM')

# Print the columns names
print(dummy.columns)
```

```python
Index(['SurveyDate', 'FormalEducation', 'ConvertedSalary', 'Hobby', 'StackOverflowJobsRecommend', 'VersionControl', 'Age', 'Years Experience', 'Gender', 'RawSalary', 'DM_India', 'DM_Ireland',
       'DM_Russia', 'DM_South Africa', 'DM_Spain', 'DM_Sweeden', 'DM_UK', 'DM_USA', 'DM_Ukraine'],
      dtype='object')
```

Did you notice that the column for France was missing when you created dummy variables? Now you can choose to use one-hot encoding or dummy variables where appropriate.

**1.2.2 Dealing with uncommon categories**

Some features can have many different categories but a very uneven distribution of their occurrences. Take for example Data Science’s favorite languages to code in, some common choices are Python, R, and Julia, but there can be individuals with bespoke choices, like FORTRAN, C etc. In these cases, you may not want to create a feature for each value, but only the more common occurrences.

```python
countries.value_counts()
South Africa    166
USA             164
Spain           134
Sweeden         119
France          115
Russia           97
India            95
UK               95
Ukraine           9
Ireland           5
Name: Country, dtype: int64
```

```python
# Create a series out of the Country column
countries = so_survey_df['Country']

# Get the counts of each category
country_counts = countries.value_counts()

# Create a mask for only categories that occur less than 10 times
mask = countries.isin(country_counts[country_counts < 10].index)

# Label all other categories as Other
countries[mask] = 'Other'

# Print the updated category counts
print(countries.value_counts())
```

```python
South Africa    166
USA             164
Spain           134
Sweeden         119
France          115
Russia           97
India            95
UK               95
Other            14
Name: Country, dtype: int64
```

Good work, now you can work with large data sets while grouping low frequency categories.

#### **1.3 Numeric variables**

* ![](https://datascience103579984.files.wordpress.com/2019/12/11-18.png?w=784)
* ![](https://datascience103579984.files.wordpress.com/2019/12/12-19.png?w=844)
* ![](https://datascience103579984.files.wordpress.com/2019/12/13-17.png?w=875)
* ![](https://datascience103579984.files.wordpress.com/2019/12/14-15.png?w=778)
* ![](https://datascience103579984.files.wordpress.com/2019/12/15-14.png?w=876)

**1.3.1 Binarizing columns**

While numeric values can often be used without any feature engineering, there will be cases when some form of manipulation can be useful. For example on some occasions, you might not care about the magnitude of a value but only care about its direction, or if it exists at all. In these situations, you will want to binarize a column. In the `so_survey_df` data, you have a large number of survey respondents that are working voluntarily (without pay). You will create a new column titled `Paid_Job` indicating whether each person is paid (their salary is greater than zero).

```python
# Create the Paid_Job column filled with zeros
so_survey_df['Paid_Job'] = 0

# Replace all the Paid_Job values where ConvertedSalary is > 0
so_survey_df.loc[so_survey_df['ConvertedSalary']>0, 'Paid_Job'] = 1

# Print the first five rows of the columns
print(so_survey_df[['Paid_Job', 'ConvertedSalary']].head())
```

```python
   Paid_Job  ConvertedSalary
0         0              0.0
1         1          70841.0
2         0              0.0
3         1          21426.0
4         1          41671.0
```

Good work, binarizing columns can also be useful for your target variables.

**1.3.2 Binning values**

For many continuous values you will care less about the exact value of a numeric column, but instead care about the bucket it falls into. This can be useful when plotting values, or simplifying your machine learning models. It is mostly used on continuous variables where accuracy is not the biggest concern e.g. age, height, wages.

Bins are created using `pd.cut(df['column_name'], bins)` where `bins` can be an integer specifying the number of evenly spaced bins, or a list of bin boundaries.

```python
# Bin the continuous variable ConvertedSalary into 5 bins
so_survey_df['equal_binned'] = pd.cut(so_survey_df['ConvertedSalary'], bins=5)

# Print the first 5 rows of the equal_binned column
print(so_survey_df[['equal_binned', 'ConvertedSalary']].head())
```

```python
          equal_binned  ConvertedSalary
0  (-2000.0, 400000.0]              0.0
1  (-2000.0, 400000.0]          70841.0
2  (-2000.0, 400000.0]              0.0
3  (-2000.0, 400000.0]          21426.0
4  (-2000.0, 400000.0]          41671.0
```

```python
# Import numpy
import numpy as np

# Specify the boundaries of the bins
bins = [-np.inf, 10000, 50000, 100000, 150000, np.inf]

# Bin labels
labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']

# Bin the continuous variable ConvertedSalary using these boundaries
so_survey_df['boundary_binned'] = pd.cut(so_survey_df['ConvertedSalary'],
                                         bins=bins, labels=labels)

# Print the first 5 rows of the boundary_binned column
print(so_survey_df[['boundary_binned', 'ConvertedSalary']].head())
```

```python
  boundary_binned  ConvertedSalary
0        Very low              0.0
1          Medium          70841.0
2        Very low              0.0
3             Low          21426.0
4             Low          41671.0
```

Now you can bin columns with equal spacing and predefined boundaries.

### **2. Dealing with Messy Data**


#### **2.1 Why do missing values exist?**

* ![](https://datascience103579984.files.wordpress.com/2019/12/16-12.png?w=789)
* ![](https://datascience103579984.files.wordpress.com/2019/12/17-10.png?w=872)
* ![](https://datascience103579984.files.wordpress.com/2019/12/18-11.png?w=800)
* ![](https://datascience103579984.files.wordpress.com/2019/12/19-11.png?w=746)
* ![](https://datascience103579984.files.wordpress.com/2019/12/20-11.png?w=953)
* ![](https://datascience103579984.files.wordpress.com/2019/12/21-11.png?w=788)

**2.1.1 How sparse is my data?**

Most data sets contain missing values, often represented as NaN (Not a Number). If you are working with Pandas you can easily check how many missing values exist in each column.

Let’s find out how many of the developers taking the survey chose to enter their age (found in the `Age` column of `so_survey_df`) and their gender (`Gender` column of `so_survey_df`).

```python
# Subset the DataFrame
sub_df = so_survey_df[['Age','Gender']]

# Print the number of non-missing values
print(sub_df.notnull().sum())

Age       999
Gender    693
dtype: int64
```

**2.1.2 Finding the missing values**

While having a summary of how much of your data is missing can be useful, often you will need to find the exact locations of these missing values. Using the same subset of the StackOverflow data from the last exercise (`sub_df`), you will show how a value can be flagged as missing.

```python
# Print the locations of the missing values
print(sub_df.head(10).isnull())

     Age  Gender
0  False   False
1  False   False
2  False    True
3  False   False
4  False   False
5  False   False
6  False   False
7  False   False
8  False   False
9  False    True

# Print the locations of the non-missing values
print(sub_df.head(10).notnull())

    Age  Gender
0  True    True
1  True    True
2  True   False
3  True    True
4  True    True
5  True    True
6  True    True
7  True    True
8  True    True
9  True   False
```

#### **2.2 Dealing with missing values (I)**

* ![](https://datascience103579984.files.wordpress.com/2019/12/20-12.png?w=953)
* ![](https://datascience103579984.files.wordpress.com/2019/12/21-12.png?w=788)
* ![](https://datascience103579984.files.wordpress.com/2019/12/22-10.png?w=992)
* ![](https://datascience103579984.files.wordpress.com/2019/12/23-10.png?w=862)
* ![](https://datascience103579984.files.wordpress.com/2019/12/24-10.png?w=939)
* ![](https://datascience103579984.files.wordpress.com/2019/12/25-9.png?w=593)
* ![](https://datascience103579984.files.wordpress.com/2019/12/26-9.png?w=823)
* ![](https://datascience103579984.files.wordpress.com/2019/12/27-9.png?w=910)

**2.2.1 Listwise deletion**

The simplest way to deal with missing values in your dataset when they are occurring entirely at random is to remove those rows, also called ‘listwise deletion’.

Depending on the use case, you will sometimes want to remove all missing values in your data while other times you may want to only remove a particular column if too many values are missing in that column.

```python
# Print the number of rows and columns
print(so_survey_df.shape)
# (999, 11)

# Create a new DataFrame dropping all incomplete rows
no_missing_values_rows = so_survey_df.dropna()

# Print the shape of the new DataFrame
print(no_missing_values_rows.shape)
# (264, 11)

# Create a new DataFrame dropping all columns with incomplete rows
no_missing_values_cols = so_survey_df.dropna(how='any', axis=1)

# Print the shape of the new DataFrame
print(no_missing_values_cols.shape)
# (999, 7)

# Drop all rows where Gender is missing
no_gender = so_survey_df.dropna(subset=['Gender'])

# Print the shape of the new DataFrame
print(no_gender.shape)
# (693, 11)
```

Correct, as you can see dropping all rows that contain any missing values may greatly reduce the size of your dataset. So you need to think carefully and consider several trade-offs when deleting missing values.

**2.2.2 Replacing missing values with constants**

While removing missing data entirely maybe a correct approach in many situations, this may result in a lot of information being omitted from your models.

You may find categorical columns where the missing value is a valid piece of information in itself, such as someone refusing to answer a question in a survey. In these cases, you can fill all missing values with a new category entirely, for example ‘No response given’.

```python
# Print the count of occurrences
print(so_survey_df['Gender'].value_counts())
```

```python
Male                                                                         632
Female                                                                        53
Transgender                                                                    2
Female;Male                                                                    2
Female;Transgender                                                             1
Male;Non-binary. genderqueer. or gender non-conforming                         1
Female;Male;Transgender;Non-binary. genderqueer. or gender non-conforming      1
Non-binary. genderqueer. or gender non-conforming                              1
Name: Gender, dtype: int64
```

```python
# Replace missing values
so_survey_df['Gender'].fillna('Not Given', inplace=True)

# Print the count of each value
print(so_survey_df['Gender'].value_counts())
```

```python
Male                                                                         632
Not Given                                                                    306
Female                                                                        53
Transgender                                                                    2
Female;Male                                                                    2
Female;Transgender                                                             1
Male;Non-binary. genderqueer. or gender non-conforming                         1
Female;Male;Transgender;Non-binary. genderqueer. or gender non-conforming      1
Non-binary. genderqueer. or gender non-conforming                              1
Name: Gender, dtype: int64
```

#### **2.3 Dealing with missing values (II)**

* ![](https://datascience103579984.files.wordpress.com/2019/12/1-20.png?w=790)
* ![](https://datascience103579984.files.wordpress.com/2019/12/2-21.png?w=946)
* ![](https://datascience103579984.files.wordpress.com/2019/12/3-21.png?w=874)
* ![](https://datascience103579984.files.wordpress.com/2019/12/4-21.png?w=1024)
* ![](https://datascience103579984.files.wordpress.com/2019/12/5-21.png?w=1007)
* ![](https://datascience103579984.files.wordpress.com/2019/12/6-21.png?w=999)

**2.3.1 Filling continuous missing values**

In the last lesson, you dealt with different methods of removing data missing values and filling in missing values with a fixed string. These approaches are valid in many cases, particularly when dealing with categorical columns but have limited use when working with continuous values. In these cases, it may be most valid to fill the missing values in the column with a value calculated from the entries present in the column.

```python
 # Print the first five rows of StackOverflowJobsRecommend column
 print(so_survey_df['StackOverflowJobsRecommend'].head(5))

# Fill missing values with the mean
so_survey_df['StackOverflowJobsRecommend'].fillna(so_survey_df['StackOverflowJobsRecommend'].mean(), inplace=True)

# Round the StackOverflowJobsRecommend values
so_survey_df['StackOverflowJobsRecommend'] = round(so_survey_df['StackOverflowJobsRecommend'])

# Print the top 5 rows
print(so_survey_df['StackOverflowJobsRecommend'].head())

0    7.0
1    7.0
2    8.0
3    7.0
4    8.0
Name: StackOverflowJobsRecommend, dtype: float64
```

Nicely done, remember you should only round your values if you are certain it is applicable.

**2.3.2 Imputing values in predictive models**

When working with predictive models you will often have a separate train and test DataFrames. In these cases you want to ensure no information from your test set leaks into your train set. When filling missing values in data to be used in these situations how should approach the two data sets?

**Apply the measures of central tendency (mean/median etc.) calculated on the train set to both the train and test sets.**

Values calculated on the train test should be applied to both DataFrames.

#### **2.4 Dealing with other data issues**

* ![](https://datascience103579984.files.wordpress.com/2019/12/7-21.png?w=606)
* ![](https://datascience103579984.files.wordpress.com/2019/12/8-20.png?w=1008)
* ![](https://datascience103579984.files.wordpress.com/2019/12/9-20.png?w=907)
* ![](https://datascience103579984.files.wordpress.com/2019/12/10-20.png?w=917)

**2.4.1 Dealing with stray characters (I)**

In this exercise, you will work with the `RawSalary` column of `so_survey_df` which contains the wages of the respondents along with the currency symbols and commas, such as _$42,000_. When importing data from Microsoft Excel, more often that not you will come across data in this form.

```python
so_survey_df['RawSalary']
0               NaN
1         70,841.00
2               NaN
3         21,426.00
4        £41,671.00
```

```python
# Remove the commas in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace(',', '')

# Remove the dollar signs in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('$','')
```

**2.4.2 Dealing with stray characters (II)**

In the last exercise, you could tell quickly based off of the `df.head()` call which characters were causing an issue. In many cases this will not be so apparent. There will often be values deep within a column that are preventing you from casting a column as a numeric type so that it can be used in a model or further feature engineering.

One approach to finding these values is to force the column to the data type desired using `pd.to_numeric()`, coercing any values causing issues to NaN, Then filtering the DataFrame by just the rows containing the NaN values.

Try to cast the `RawSalary` column as a float and it will fail as an additional character can now be found in it. Find the character and remove it so the column can be cast as a float.

```python
# Attempt to convert the column to numeric values
numeric_vals = pd.to_numeric(so_survey_df['RawSalary'], errors='coerce')

# Find the indexes of missing values
idx = numeric_vals.isna()

# Print the relevant rows
print(so_survey_df['RawSalary'][idx])

0             NaN
2             NaN
4       £41671.00
6             NaN
8             NaN
...
```

```python
# Replace the offending characters
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('£','')

# Convert the column to float
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].astype('float')

# Print the column
print(so_survey_df['RawSalary'])
```

Remember that even after removing all the relevant characters, you still need to change the type of the column to numeric if you want to plot these continuous values.

**2.4.3 Method chaining**

When applying multiple operations on the same column (like in the previous exercises), you made the changes in several steps, assigning the results back in each step. However, when applying multiple successive operations on the same column, you can “chain” these operations together for clarity and ease of management. This can be achieved by calling multiple methods sequentially:

```python
# Method chaining
df['column'] = df['column'].method1().method2().method3()

# Same as
df['column'] = df['column'].method1()
df['column'] = df['column'].method2()
df['column'] = df['column'].method3()
```

In this exercise you will repeat the steps you performed in the last two exercises, but do so using method chaining.

```python
# Use method chaining
so_survey_df['RawSalary'] = so_survey_df['RawSalary']\
                              .str.replace(',','')\
                              .str.replace('$','')\
                              .str.replace('£','')\
                              .astype('float')

# Print the RawSalary column
print(so_survey_df['RawSalary'])
```

Custom functions can be also used when method chaining using the .apply() method.

### **3. Conforming to Statistical Assumptions**


#### **3.1 Data distributions**

* ![](https://datascience103579984.files.wordpress.com/2019/12/1-21.png?w=992)
* ![](https://datascience103579984.files.wordpress.com/2019/12/2-22.png?w=763)
* ![](https://datascience103579984.files.wordpress.com/2019/12/3-22.png?w=952)
* ![](https://datascience103579984.files.wordpress.com/2019/12/4-22.png?w=765)
* ![](https://datascience103579984.files.wordpress.com/2019/12/5-22.png?w=750)
* ![](https://datascience103579984.files.wordpress.com/2019/12/6-22.png?w=1024)

**3.1.1 What does your data look like? (I)**

Up until now you have focused on creating new features and dealing with issues in your data. Feature engineering can also be used to make the most out of the data that you already have and use it more effectively when creating machine learning models.\
Many algorithms may assume that your data is normally distributed, or at least that all your columns are on the same scale. This will often not be the case, e.g. one feature may be measured in thousands of dollars while another would be number of years. In this exercise, you will create plots to examine the distributions of some numeric columns in the `so_survey_df` DataFrame, stored in `so_numeric_df`.

```python
# Create a histogram
so_numeric_df.hist()
plt.show()
```

![](https://datascience103579984.files.wordpress.com/2019/12/7-22.png?w=1024)

```python
# Create a boxplot of two columns
so_numeric_df[['Age', 'Years Experience']].boxplot()
plt.show()
```

![](https://datascience103579984.files.wordpress.com/2019/12/8-21.png?w=1024)

```python
# Create a boxplot of ConvertedSalary
so_numeric_df[['ConvertedSalary']].boxplot()
plt.show()
```

![](https://datascience103579984.files.wordpress.com/2019/12/9-21.png?w=1024)

**3.1.2 What does your data look like? (II)**

In the previous exercise you looked at the distribution of individual columns. While this is a good start, a more detailed view of how different features interact with each other may be useful as this can impact your decision on what to transform and how.

```python
# Import packages
from matplotlib import pyplot as plt
import seaborn as sns

# Plot pairwise relationships
sns.pairplot(so_numeric_df)

# Show plot
plt.show()
```

![](https://datascience103579984.files.wordpress.com/2019/12/10-21.png?w=1024)

```python
# Print summary statistics
print(so_numeric_df.describe())
```

```python
       ConvertedSalary         Age  Years Experience
count     9.990000e+02  999.000000        999.000000
mean      6.161746e+04   36.003003          9.961962
std       1.760924e+05   13.255127          4.878129
min       0.000000e+00   18.000000          0.000000
25%       0.000000e+00   25.000000          7.000000
50%       2.712000e+04   35.000000         10.000000
75%       7.000000e+04   45.000000         13.000000
max       2.000000e+06   83.000000         27.000000
```

Good work, understanding these summary statistics of a column can be very valuable when deciding what transformations are necessary.

**3.1.3 When don’t you have to transform your data?**

While making sure that all of your data is on the same scale is advisable for most analyses, for which of the following machine learning models is normalizing data not always necessary?

**Decision Trees**

As decision trees split along a singular point, they do not require all the columns to be on the same scale.

#### **3.2 Scaling and transformations**

* ![](https://datascience103579984.files.wordpress.com/2019/12/11-19.png?w=923)
* ![](https://datascience103579984.files.wordpress.com/2019/12/12-20.png?w=868)
* ![](https://datascience103579984.files.wordpress.com/2019/12/13-18.png?w=865)
* ![](https://datascience103579984.files.wordpress.com/2019/12/14-16.png?w=937)
* ![](https://datascience103579984.files.wordpress.com/2019/12/15-15.png?w=877)
* ![](https://datascience103579984.files.wordpress.com/2019/12/16-13.png?w=860)
* ![](https://datascience103579984.files.wordpress.com/2019/12/17-11.png?w=873)
* ![](https://datascience103579984.files.wordpress.com/2019/12/18-12.png?w=902)

**3.2.1 Normalization**

As discussed in the video, in normalization you linearly scale the entire column between 0 and 1, with 0 corresponding with the lowest value in the column, and 1 with the largest.\
When using scikit-learn (the most commonly used machine learning library in Python) you can use a `MinMaxScaler` to apply normalization. _(It is called this as it scales your values between a minimum and maximum value.)_

```python
# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Instantiate MinMaxScaler
MM_scaler = MinMaxScaler()

# Fit MM_scaler to the data
MM_scaler.fit(so_numeric_df[['Age']])

# Transform the data using the fitted scaler
so_numeric_df['Age_MM'] = MM_scaler.transform(so_numeric_df[['Age']])

# Compare the origional and transformed column
print(so_numeric_df[['Age_MM', 'Age']].head())
```

```python
     Age_MM  Age
0  0.046154   21
1  0.307692   38
2  0.415385   45
3  0.430769   46
4  0.323077   39
```

Did you notice that all values have been scaled between 0 and 1?

**3.2.2 Standardization**

While normalization can be useful for scaling a column between two data points, it is hard to compare two scaled columns if even one of them is overly affected by outliers. One commonly used solution to this is called standardization, where instead of having a strict upper and lower bound, you center the data around its mean, and calculate the number of standard deviations away from mean each data point is.

```python
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Instantiate StandardScaler
SS_scaler = StandardScaler()

# Fit SS_scaler to the data
SS_scaler.fit(so_numeric_df[['Age']])

# Transform the data using the fitted scaler
so_numeric_df['Age_SS'] = SS_scaler.transform(so_numeric_df[['Age']])

# Compare the origional and transformed column
print(so_numeric_df[['Age_SS', 'Age']].head())
```

```python
     Age_SS  Age
0 -1.132431   21
1  0.150734   38
2  0.679096   45
3  0.754576   46
4  0.226214   39
```

you can see that the values have been scaled linearly, but not between set values.

**3.2.3 Log transformation**

In the previous exercises you scaled the data linearly, which will not affect the data’s shape. This works great if your data is normally distributed (or closely normally distributed), an assumption that a lot of machine learning models make. Sometimes you will work with data that closely conforms to normality, e.g the height or weight of a population. On the other hand, many variables in the real world do not follow this pattern e.g, wages or age of a population. In this exercise you will use a log transform on the `ConvertedSalary` column in the `so_numeric_df` DataFrame as it has a large amount of its data centered around the lower values, but contains very high values also. These distributions are said to have a long right tail.

```python
# Import PowerTransformer
from sklearn.preprocessing import PowerTransformer

# Instantiate PowerTransformer
pow_trans = PowerTransformer()

# Train the transform on the data
pow_trans.fit(so_numeric_df[['ConvertedSalary']])

# Apply the power transform to the data
so_numeric_df['ConvertedSalary_LG'] = pow_trans.transform(so_numeric_df[['ConvertedSalary']])

# Plot the data before and after the transformation
so_numeric_df[['ConvertedSalary', 'ConvertedSalary_LG']].hist()
plt.show()
```

```python
so_numeric_df.head()
   ConvertedSalary  Age  Years Experience  ConvertedSalary_LG
0              NaN   21                13                 NaN
1          70841.0   38                 9            0.312939
2              NaN   45                11                 NaN
3          21426.0   46                12           -0.652182
4          41671.0   39                 7           -0.135589
```

![](https://datascience103579984.files.wordpress.com/2019/12/19-12.png?w=1024)

Did you notice the change in the shape of the distribution? `ConvertedSalary_LG` column looks much more normal than the original `ConvertedSalary` column.

**3.2.4 When can you use normalization?**

When could you use normalization (`MinMaxScaler`) when working with a dataset?

**When you know the the data has a strict upper and lower bound.**

Normalization scales all points linearly between the upper and lower bound.

#### **3.3 Removing outliers**

* ![](https://datascience103579984.files.wordpress.com/2019/12/20-13.png?w=889)
* ![](https://datascience103579984.files.wordpress.com/2019/12/21-13.png?w=839)
* ![](https://datascience103579984.files.wordpress.com/2019/12/22-11.png?w=722)
* ![](https://datascience103579984.files.wordpress.com/2019/12/23-11.png?w=1024)
* ![](https://datascience103579984.files.wordpress.com/2019/12/24-11.png?w=1024)

**3.3.1 Percentage based outlier removal**

One way to ensure a small portion of data is not having an overly adverse effect is by removing a certain percentage of the largest and/or smallest values in the column. This can be achieved by finding the relevant quantile and trimming the data using it with a mask. This approach is particularly useful if you are concerned that the highest values in your dataset should be avoided. When using this approach, you must remember that even if there are no outliers, this will still remove the same top N percentage from the dataset.

```python
# Find the 95th quantile
quantile = so_numeric_df['ConvertedSalary'].quantile(0.95)

# Trim the outliers
trimmed_df = so_numeric_df[so_numeric_df['ConvertedSalary'] < quantile]

# The original histogram
so_numeric_df[['ConvertedSalary']].hist()
plt.show()
plt.clf()

# The trimmed histogram
trimmed_df[['ConvertedSalary']].hist()
plt.show()
```

* ![](https://datascience103579984.files.wordpress.com/2019/12/25-10.png?w=1024)
* ![](https://datascience103579984.files.wordpress.com/2019/12/26-10.png?w=1024)

In the next exercise, you will work with a more statistically sound approach in removing outliers.

**3.3.2 Statistical outlier removal**

While removing the top N% of your data is useful for ensuring that very spurious points are removed, it does have the disadvantage of always removing the same proportion of points, even if the data is correct. A commonly used alternative approach is to remove data that sits further than three standard deviations from the mean. You can implement this by first calculating the mean and standard deviation of the relevant column to find upper and lower bounds, and applying these bounds as a mask to the DataFrame. This method ensures that only data that is genuinely different from the rest is removed, and will remove fewer points if the data is close together.

```python
# Find the mean and standard dev
std = so_numeric_df['ConvertedSalary'].std()
mean = so_numeric_df['ConvertedSalary'].mean()

# Calculate the cutoff
cut_off = std * 3
lower, upper = mean - cut_off, mean + cut_off

# Trim the outliers
trimmed_df = so_numeric_df[(so_numeric_df['ConvertedSalary'] < upper)
& (so_numeric_df['ConvertedSalary'] > lower)]

# The trimmed box plot
trimmed_df[['ConvertedSalary']].boxplot()
plt.show()
```

* ![](https://datascience103579984.files.wordpress.com/2019/12/27-10.png?w=1024)
* ![](https://datascience103579984.files.wordpress.com/2019/12/28-9.png?w=1024)

Did you notice the scale change on the y-axis?

#### **3.4 Scaling and transforming new data**

* ![](https://datascience103579984.files.wordpress.com/2019/12/29-8.png?w=875)
* ![](https://datascience103579984.files.wordpress.com/2019/12/30-6.png?w=1011)
* ![](https://datascience103579984.files.wordpress.com/2019/12/31-4.png?w=867)

**3.4.1 Train and testing transformations (I)**

So far you have created scalers based on a column, and then applied the scaler to the same data that it was trained on. When creating machine learning models you will generally build your models on historic data (train set) and apply your model to new unseen data (test set). In these cases you will need to ensure that the same scaling is being applied to both the training and test data.\
_To do this in practice you train the scaler on the train set, and keep the trained scaler to apply it to the test set. You should never retrain a scaler on the test set._

For this exercise and the next, we split the `so_numeric_df` DataFrame into train (`so_train_numeric`) and test (`so_test_numeric`) sets.

```python
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Apply a standard scaler to the data
SS_scaler = StandardScaler()

# Fit the standard scaler to the data
SS_scaler.fit(so_train_numeric[['Age']])

# Transform the test data using the fitted scaler
so_test_numeric['Age_ss'] = SS_scaler.transform(so_test_numeric[['Age']])
print(so_test_numeric[['Age', 'Age_ss']].head())
```

```python
     Age    Age_ss
700   35 -0.069265
701   18 -1.343218
702   47  0.829997
703   57  1.579381
704   41  0.380366
```

**Data leakage** is one of the most common mistakes data scientists tend to make, and I hope that you won’t!

**3.4.2 Train and testing transformations (II)**

Similar to applying the same scaler to both your training and test sets, if you have removed outliers from the train set, you probably want to do the same on the test set as well. Once again you should ensure that you use the _thresholds calculated only from the train set_ to remove outliers from the test set.

Similar to the last exercise, we split the `so_numeric_df` DataFrame into train (`so_train_numeric`) and test (`so_test_numeric`) sets.

```python
train_std = so_train_numeric['ConvertedSalary'].std()
train_mean = so_train_numeric['ConvertedSalary'].mean()

cut_off = train_std * 3
train_lower, train_upper = train_mean - cut_off, train_mean + cut_off

# Trim the test DataFrame
trimmed_df = so_test_numeric[(so_test_numeric['ConvertedSalary'] < train_upper) \
                             & (so_test_numeric['ConvertedSalary'] > train_lower)]
```

Very well done. In the next chapter, you will deal with unstructured (text) data.

### **4. Dealing with Text Data**


#### **4.1 Encoding text**

* ![](https://datascience103579984.files.wordpress.com/2020/01/1.png?w=999)
* ![](https://datascience103579984.files.wordpress.com/2020/01/2.png?w=999)
* ![](https://datascience103579984.files.wordpress.com/2020/01/3.png?w=942)
* ![](https://datascience103579984.files.wordpress.com/2020/01/4.png?w=995)
* ![](https://datascience103579984.files.wordpress.com/2020/01/5.png?w=995)
* ![](https://datascience103579984.files.wordpress.com/2020/01/6.png?w=1001)
* ![](https://datascience103579984.files.wordpress.com/2020/01/7.png?w=922)
* ![](https://datascience103579984.files.wordpress.com/2020/01/8.png?w=990)
* ![](https://datascience103579984.files.wordpress.com/2020/01/9.png?w=1001)
* ![](https://datascience103579984.files.wordpress.com/2020/01/10.png?w=986)

**4.1.1 Cleaning up your text**

Unstructured text data cannot be directly used in most analyses. Multiple steps need to be taken to go from a long free form string to a set of numeric columns in the right format that can be ingested by a machine learning model. The first step of this process is to standardize the data and eliminate any characters that could cause problems later on in your analytic pipeline.

In this chapter you will be working with a new dataset containing the inaugural speeches of the presidents of the United States loaded as `speech_df`, with the speeches stored in the `text` column.

```python
# Print the first 5 rows of the text column
print(speech_df.text.head())
```

```python
0    Fellow-Citizens of the Senate and of the House...
1    Fellow Citizens:  I AM again called upon by th...
2    WHEN it was first perceived, in early times, t...
3    Friends and Fellow-Citizens:  CALLED upon to u...
4    PROCEEDING, fellow-citizens, to that qualifica...
Name: text, dtype: object
```

```python
# Replace all non letter characters with a whitespace
speech_df['text_clean'] = speech_df['text'].str.replace('[^a-zA-Z]', ' ')

# Change to lower case
speech_df['text_clean'] = speech_df['text_clean'].str.lower()

# Print the first 5 rows of the text_clean column
print(speech_df['text_clean'].head())
```

```python
0    fellow citizens of the senate and of the house...
1    fellow citizens   i am again called upon by th...
2    when it was first perceived  in early times  t...
3    friends and fellow citizens   called upon to u...
4    proceeding  fellow citizens  to that qualifica...
Name: text_clean, dtype: object
```

Great, now your text strings have been standardized and cleaned up. You can now use this new column (`text_clean`) to extract information about the speeches.

**4.1.2 High level text features**

Once the text has been cleaned and standardized you can begin creating features from the data. The most fundamental information you can calculate about free form text is its size, such as its length and number of words. In this exercise (and the rest of this chapter), you will focus on the cleaned/transformed text column (`text_clean`) you created in the last exercise.

```python
# Find the length of each text
speech_df['char_cnt'] = speech_df['text_clean'].str.len()

# Count the number of words in each text
speech_df['word_cnt'] = speech_df['text_clean'].str.split().str.len()

# Find the average length of word
speech_df['avg_word_length'] = speech_df['char_cnt'] / speech_df['word_cnt']

# Print the first 5 rows of these columns
print(speech_df[['text_clean', 'char_cnt', 'word_cnt', 'avg_word_length']])
```

```python
                                           text_clean  char_cnt  word_cnt  avg_word_length
0   fellow citizens of the senate and of the house...      8616      1432         6.016760
1   fellow citizens   i am again called upon by th...       787       135         5.829630
2   when it was first perceived  in early times  t...     13871      2323         5.971158
```

These features may appear basic but can be quite useful in ML models.

#### **4.2 Word counts**

* ![](https://datascience103579984.files.wordpress.com/2020/01/11.png?w=1012)
* ![](https://datascience103579984.files.wordpress.com/2020/01/12.png?w=993)
* ![](https://datascience103579984.files.wordpress.com/2020/01/13.png?w=986)
* ![](https://datascience103579984.files.wordpress.com/2020/01/14.png?w=594)
* ![](https://datascience103579984.files.wordpress.com/2020/01/15.png?w=995)
* ![](https://datascience103579984.files.wordpress.com/2020/01/16.png?w=685)
* ![](https://datascience103579984.files.wordpress.com/2020/01/17.png?w=1023)
* ![](https://datascience103579984.files.wordpress.com/2020/01/18.png?w=992)
* ![](https://datascience103579984.files.wordpress.com/2020/01/19.png?w=1024)
* ![](https://datascience103579984.files.wordpress.com/2020/01/20.png?w=791)

**4.2.1 Counting words (I)**

Once high level information has been recorded you can begin creating features based on the actual content of each text. One way to do this is to approach it in a similar way to how you worked with categorical variables in the earlier lessons.

* For each unique word in the dataset a column is created.
* For each entry, the number of times this word occurs is counted and the count value is entered into the respective column.

These “count” columns can then be used to train machine learning models.

```python
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Instantiate CountVectorizer
cv = CountVectorizer()

# Fit the vectorizer
cv.fit(speech_df['text_clean'])

# Print feature names
print(cv.get_feature_names())
```

```python
['abandon', 'abandoned', 'abandonment', 'abate', 'abdicated', 'abeyance', 'abhorring', 'abide', 'abiding', 'abilities', 'ability', 'abject', 'able', ...]
```

**4.2.2 Counting words (II)**

Once the vectorizer has been fit to the data, it can be used to transform the text to an array representing the word counts. This array will have a row per block of text and a column for each of the features generated by the vectorizer that you observed in the last exercise.

The vectorizer to you fit in the last exercise (`cv`) is available in your workspace.

```python
# Apply the vectorizer
cv_transformed = cv.transform(speech_df['text_clean'])

# Print the full array
cv_array = cv_transformed.toarray()
print(cv_array)

print(cv_array.shape)
# (58, 9043)
```

```python
[[0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 1 0 ... 0 0 0]
 ...
 [0 1 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]]
 ```

The speeches have 9043 unique words, which is a lot! In the next exercise, you will see how to create a limited set of features.

**4.2.3 Limiting your features**

As you have seen, using the `CountVectorizer` with its default settings creates a feature for every single word in your corpus. This can create far too many features, often including ones that will provide very little analytical value.

For this purpose `CountVectorizer` has parameters that you can set to reduce the number of features:

* `min_df` : Use only words that occur in more than this percentage of documents. This can be used to remove outlier words that will not generalize across texts.
* `max_df` : Use only words that occur in less than this percentage of documents. This is useful to eliminate very common words that occur in every corpus without adding value such as “and” or “the”.

```python
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Specify arguements to limit the number of features generated
cv = CountVectorizer(min_df=0.2, max_df=0.8)

# Fit, transform, and convert into array
cv_transformed = cv.fit_transform(speech_df['text_clean'])
cv_array = cv_transformed.toarray()

# Print the array shape
print(cv_array.shape)
# (58, 818)
```

**4.2.4 Text to DataFrame**

Now that you have generated these count based features in an array you will need to reformat them so that they can be combined with the rest of the dataset. This can be achieved by converting the array into a pandas DataFrame, with the feature names you found earlier as the column names, and then concatenate it with the original DataFrame.

The numpy array (`cv_array`) and the vectorizer (`cv`) you fit in the last exercise are available in your workspace.

```python
# Create a DataFrame with these features
cv_df = pd.DataFrame(cv_array,
                     columns=cv.get_feature_names()).add_prefix('Counts_')

# Add the new columns to the original DataFrame
speech_df_new = pd.concat([speech_df, cv_df], axis=1, sort=False)
print(speech_df_new.head())
```

```python
                Name         Inaugural Address                      Date                                               text                                         text_clean  ...  Counts_years  \
0  George Washington   First Inaugural Address  Thursday, April 30, 1789  Fellow-Citizens of the Senate and of the House...  fellow citizens of the senate and of the house...  ...             1
1  George Washington  Second Inaugural Address     Monday, March 4, 1793  Fellow Citizens:  I AM again called upon by th...  fellow citizens   i am again called upon by th...  ...             0
2         John Adams         Inaugural Address   Saturday, March 4, 1797  WHEN it was first perceived, in early times, t...  when it was first perceived  in early times  t...  ...             3
3   Thomas Jefferson   First Inaugural Address  Wednesday, March 4, 1801  Friends and Fellow-Citizens:  CALLED upon to u...  friends and fellow citizens   called upon to u...  ...             0
4   Thomas Jefferson  Second Inaugural Address     Monday, March 4, 1805  PROCEEDING, fellow-citizens, to that qualifica...  proceeding  fellow citizens  to that qualifica...  ...             2

   Counts_yet  Counts_you  Counts_young  Counts_your
0           0           5             0            9
1           0           0             0            1
2           0           0             0            1
3           2           7             0            7
4           2           4             0            4

[5 rows x 826 columns]
```

With the new features combined with the orginial DataFrame they can be now used for ML models or analysis.

#### **4.3 Term frequency-inverse document frequency**

* ![](https://datascience103579984.files.wordpress.com/2020/01/1-1.png?w=689)
* ![](https://datascience103579984.files.wordpress.com/2020/01/2-1.png?w=980)
* ![](https://datascience103579984.files.wordpress.com/2020/01/3-1.png?w=991)
* ![](https://datascience103579984.files.wordpress.com/2020/01/4-1.png?w=996)
* ![](https://datascience103579984.files.wordpress.com/2020/01/5-1.png?w=996)
* ![](https://datascience103579984.files.wordpress.com/2020/01/6-1.png?w=987)
* ![](https://datascience103579984.files.wordpress.com/2020/01/7-1.png?w=842)
* ![](https://datascience103579984.files.wordpress.com/2020/01/8-1.png?w=1024)

**4.3.1 Tf-idf**

While counts of occurrences of words can be useful to build models, words that occur many times may skew the results undesirably. To limit these common words from overpowering your model a form of normalization can be used. In this lesson you will be using Term frequency-inverse document frequency (Tf-idf) as was discussed in the video. Tf-idf has the effect of reducing the value of common words, while increasing the weight of words that do not occur in many documents.

```python
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Instantiate TfidfVectorizer
tv = TfidfVectorizer(max_features=100, stop_words='english')

# Fit the vectroizer and transform the data
tv_transformed = tv.fit_transform(speech_df['text_clean'])

# Create a DataFrame with these features
tv_df = pd.DataFrame(tv_transformed.toarray(),
                     columns=tv.get_feature_names()).add_prefix('TFIDF_')
print(tv_df.head())
```

```python
   TFIDF_action  TFIDF_administration  TFIDF_america  TFIDF_american  TFIDF_americans  ...  TFIDF_war  TFIDF_way  TFIDF_work  TFIDF_world  TFIDF_years
0      0.000000              0.133415       0.000000        0.105388              0.0  ...   0.000000   0.060755    0.000000     0.045929     0.052694
1      0.000000              0.261016       0.266097        0.000000              0.0  ...   0.000000   0.000000    0.000000     0.000000     0.000000
2      0.000000              0.092436       0.157058        0.073018              0.0  ...   0.024339   0.000000    0.000000     0.063643     0.073018
3      0.000000              0.092693       0.000000        0.000000              0.0  ...   0.036610   0.000000    0.039277     0.095729     0.000000
4      0.041334              0.039761       0.000000        0.031408              0.0  ...   0.094225   0.000000    0.000000     0.054752     0.062817

[5 rows x 100 columns]
```

Did you notice that counting the word occurences and calculating the Tf-idf weights are very similar? This is one of the reasons scikit-learn is very popular, a consistent API.

**4.3.2 Inspecting Tf-idf values**

After creating Tf-idf features you will often want to understand what are the most highest scored words for each corpus. This can be achieved by isolating the row you want to examine and then sorting the the scores from high to low.

The DataFrame from the last exercise (`tv_df`) is available in your workspace.

```python
# Isolate the row to be examined
sample_row = tv_df.iloc[0]

# Print the top 5 words of the sorted output
print(sample_row.sort_values(ascending=False).head())

TFIDF_government    0.367430
TFIDF_public        0.333237
TFIDF_present       0.315182
TFIDF_duty          0.238637
TFIDF_citizens      0.229644
Name: 0, dtype: float64
```

When creating vectors from text, any transformations that you perform before training a machine learning model, you also need to apply on the new unseen (test) data. To achieve this follow the same approach from the last chapter: _fit the vectorizer only on the training data, and apply it to the test data._

For this exercise the `speech_df` DataFrame has been split in two:

* `train_speech_df`: The training set consisting of the first 45 speeches.
* `test_speech_df`: The test set consisting of the remaining speeches.

```python
# Instantiate TfidfVectorizer
tv = TfidfVectorizer(max_features=100, stop_words='english')

# Fit the vectroizer and transform the data
tv_transformed = tv.fit_transform(train_speech_df['text_clean'])

# Transform test data
test_tv_transformed = tv.transform(test_speech_df['text_clean'])

# Create new features for the test set
test_tv_df = pd.DataFrame(test_tv_transformed.toarray(),
                          columns=tv.get_feature_names()).add_prefix('TFIDF_')
print(test_tv_df.head())
```

```python
   TFIDF_action  TFIDF_administration  TFIDF_america  TFIDF_american  TFIDF_authority  ...  TFIDF_war  TFIDF_way  TFIDF_work  TFIDF_world  TFIDF_years
0      0.000000              0.029540       0.233954        0.082703         0.000000  ...   0.079050   0.033313    0.000000     0.299983     0.134749
1      0.000000              0.000000       0.547457        0.036862         0.000000  ...   0.052851   0.066817    0.078999     0.277701     0.126126
2      0.000000              0.000000       0.126987        0.134669         0.000000  ...   0.042907   0.054245    0.096203     0.225452     0.043884
3      0.037094              0.067428       0.267012        0.031463         0.039990  ...   0.030073   0.038020    0.235998     0.237026     0.061516
4      0.000000              0.000000       0.221561        0.156644         0.028442  ...   0.021389   0.081124    0.119894     0.299701     0.153133

[5 rows x 100 columns]
```

**4.4 N-grams**

* ![](https://datascience103579984.files.wordpress.com/2020/01/9-1.png?w=722)
* ![](https://datascience103579984.files.wordpress.com/2020/01/10-1.png?w=1003)
* ![](https://datascience103579984.files.wordpress.com/2020/01/11-1.png?w=1000)
* ![](https://datascience103579984.files.wordpress.com/2020/01/12-1.png?w=897)

**4.4.1 Using longer n-grams**

So far you have created features based on individual words in each of the texts. This can be quite powerful when used in a machine learning model but you may be concerned that by looking at words individually a lot of the context is being ignored. To deal with this when creating models you can use n-grams which are sequence of n words grouped together. For example:

* bigrams: Sequences of two consecutive words
* trigrams: Sequences of three consecutive words

These can be automatically created in your dataset by specifying the `ngram_range` argument as a tuple `(n1, n2)` where all n-grams in the `n1` to `n2` range are included.

```python
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Instantiate a trigram vectorizer
cv_trigram_vec = CountVectorizer(max_features=100,
                                 stop_words='english',
                                 ngram_range=(3,3))

# Fit and apply trigram vectorizer
cv_trigram = cv_trigram_vec.fit_transform(speech_df['text_clean'])

# Print the trigram features
print(cv_trigram_vec.get_feature_names())
```

```python
['ability preserve protect', 'agriculture commerce manufactures', 'america ideal freedom', 'amity mutual concession', 'anchor peace home', 'ask bow heads', ...]
```

**4.4.2 Finding the most common words**

Its always advisable once you have created your features to inspect them to ensure that they are as you would expect. This will allow you to catch errors early, and perhaps influence what further feature engineering you will need to do.

The vectorizer (`cv`) you fit in the last exercise and the sparse array consisting of word counts (`cv_trigram`) is available in your workspace.

```python
# Create a DataFrame of the features
cv_tri_df = pd.DataFrame(cv_trigram.toarray(),
                 columns=cv_trigram_vec.get_feature_names()).add_prefix('Counts_')

# Print the top 5 words in the sorted output
print(cv_tri_df.sum().sort_values(ascending=False).head())
```

```python
Counts_constitution united states    20
Counts_people united states          13
Counts_preserve protect defend       10
Counts_mr chief justice              10
Counts_president united states        8
dtype: int64
```

**4.5 Wrap-up**

* ![](https://datascience103579984.files.wordpress.com/2020/01/13-1.png?w=761)
* ![](https://datascience103579984.files.wordpress.com/2020/01/14-1.png?w=804)
* ![](https://datascience103579984.files.wordpress.com/2020/01/15-1.png?w=778)
* ![](https://datascience103579984.files.wordpress.com/2020/01/16-1.png?w=946)
* ![](https://datascience103579984.files.wordpress.com/2020/01/17-1.png?w=425)

Thank you for reading and hope you’ve learned a lot.
