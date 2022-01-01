---
title: Preprocessing for Machine Learning in Python
date: 2021-12-07 11:22:11 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Preprocessing for Machine Learning in Python
================================================







 This is the memo of the 8th course (23 courses in all) of ‘Machine Learning Scientist with Python’ skill track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/preprocessing-for-machine-learning-in-python)**
 .



###
**Course Description**



 This course covers the basics of how and when to perform data preprocessing. This essential step in any machine learning project is when you get your data ready for modeling. Between importing and cleaning your data and fitting your machine learning model is when preprocessing comes into play. You’ll learn how to standardize your data so that it’s in the right form for your model, create new features to best leverage the information in your dataset, and select the best features to improve your model fit. Finally, you’ll have some practice preprocessing by getting a dataset on UFO sightings ready for modeling.



###
**Table of contents**


1. Introduction to Data Preprocessing
2. [Standardizing Data](https://datascience103579984.wordpress.com/2019/12/23/preprocessing-for-machine-learning-in-python-from-datacamp/2/)
3. [Feature Engineering](https://datascience103579984.wordpress.com/2019/12/23/preprocessing-for-machine-learning-in-python-from-datacamp/3/)
4. [Selecting features for modeling](https://datascience103579984.wordpress.com/2019/12/23/preprocessing-for-machine-learning-in-python-from-datacamp/4/)
5. [Putting it all together](https://datascience103579984.wordpress.com/2019/12/23/preprocessing-for-machine-learning-in-python-from-datacamp/5/)





# **1. Introduction to Data Preprocessing**
------------------------------------------


## **1.1 What is data preprocessing?**



![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/1-11.png?w=835)

### **1.1.1 Missing data – columns**



 We have a dataset comprised of volunteer information from New York City. The dataset has a number of features, but we want to get rid of features that have at least 3 missing values.




 How many features are in the original dataset, and how many features are in the set after columns with at least 3 missing values are removed?





```

volunteer.shape
# (665, 35)

volunteer.dropna(axis=1,thresh=3).shape
# (665, 24)

```


### **1.1.2 Missing data – rows**



 Taking a look at the
 `volunteer`
 dataset again, we want to drop rows where the
 `category_desc`
 column values are missing. We’re going to do this using boolean indexing, by checking to see if we have any null values, and then filtering the dataset so that we only have rows with those values.





```python

# Check how many values are missing in the category_desc column
print(volunteer['category_desc'].isnull().sum())
# 48

# Subset the volunteer dataset
volunteer_subset = volunteer[volunteer['category_desc'].notnull()]

# Print out the shape of the subset
print(volunteer_subset.shape)
# (617, 35)

```




---


## **1.2 Working with data types**



![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/2-12.png?w=975)

### **1.2.1 Exploring data types**



 Taking another look at the dataset comprised of volunteer information from New York City, we want to know what types we’ll be working with as we start to do more preprocessing.




 Which data types are present in the
 `volunteer`
 dataset?





```

set(volunteer.dtypes.values)
{dtype('int64'), dtype('float64'), dtype('O')}

```


### **1.2.2 Converting a column type**



 If you take a look at the
 `volunteer`
 dataset types, you’ll see that the column
 `hits`
 is type
 `object`
 . But, if you actually look at the column, you’ll see that it consists of integers. Let’s convert that column to type
 `int`
 .





```

volunteer["hits"].dtype
# dtype('O')

# Convert the hits column to type int
volunteer["hits"] = volunteer["hits"].astype('int')

volunteer["hits"].dtype
# dtype('int64')

```




---


## **1.3 Class distribution**



![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/3-12.png?w=996)
![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/4-12.png?w=980)



### **1.3.1 Class imbalance**



 In the
 `volunteer`
 dataset, we’re thinking about trying to predict the
 `category_desc`
 variable using the other features in the dataset. First, though, we need to know what the class distribution (and imbalance) is for that label.




 Which descriptions occur less than 50 times in the
 `volunteer`
 dataset?





```

volunteer.category_desc.value_counts()
Strengthening Communities    307
Helping Neighbors in Need    119
Education                     92
Health                        52
Environment                   32
Emergency Preparedness        15
Name: category_desc, dtype: int64

```


### **1.3.2 Stratified sampling**



 We know that the distribution of variables in the
 `category_desc`
 column in the
 `volunteer`
 dataset is uneven. If we wanted to train a model to try to predict
 `category_desc`
 , we would want to train the model on a sample of data that is representative of the entire dataset. Stratified sampling is a way to achieve this.





```python

# Create a data with all columns except category_desc
volunteer_X = volunteer.drop('category_desc', axis=1)

# Create a category_desc labels dataset
volunteer_y = volunteer[['category_desc']]

# Use stratified sampling to split up the dataset according to the volunteer_y dataset
X_train, X_test, y_train, y_test = train_test_split(volunteer_X, volunteer_y, stratify=volunteer_y)

# Print out the category_desc counts on the training y labels
print(y_train['category_desc'].value_counts())

```




```

Strengthening Communities    230
Helping Neighbors in Need     89
Education                     69
Health                        39
Environment                   24
Emergency Preparedness        11
Name: category_desc, dtype: int64

```


# **2. Standardizing Data**
--------------------------


## **2.1 Standardizing Data**



![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/5-12.png?w=813)
![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/6-12.png?w=863)



### **2.1.1 When to standardize**


* A column you want to use for modeling has extremely high variance.
* You have a dataset with several continuous columns on different scales and you’d like to use a linear model to train the data.
* The models you’re working with use some sort of distance metric in a linear space, like the Euclidean metric.


### **2.1.2 Modeling without normalizing**



 Let’s take a look at what might happen to your model’s accuracy if you try to model data without doing some sort of standardization first. Here we have a subset of the
 `wine`
 dataset. One of the columns,
 `Proline`
 , has an extremely high variance compared to the other columns. This is an example of where a technique like log normalization would come in handy, which you’ll learn about in the next section.





```python

# Split the dataset and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Fit the k-nearest neighbors model to the training data
knn.fit(X_train,y_train)

# Score the model on the test data
print(knn.score(X_test,y_test))
# 0.5333333333333333

```




---


## **2.2 Log normalization**



![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/7-12.png?w=789)
![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/8-11.png?w=996)



### **2.2.1 Checking the variance**



 Check the variance of the columns in the
 `wine`
 dataset. Which column is a candidate for normalization?





```

wine.var()
Type                                0.600679
Alcohol                             0.659062
Malic acid                          1.248015
Ash                                 0.075265
Alcalinity of ash                  11.152686
Magnesium                         203.989335
Total phenols                       0.391690
Flavanoids                          0.997719
Nonflavanoid phenols                0.015489
Proanthocyanins                     0.327595
Color intensity                     5.374449
Hue                                 0.052245
OD280/OD315 of diluted wines        0.504086
Proline                         99166.717355
dtype: float64

```



**Proline 99166.717355**



### **2.2.2 Log normalization in Python**



 Now that we know that the
 `Proline`
 column in our wine dataset has a large amount of variance, let’s log normalize it.





```python

# Print out the variance of the Proline column
print(wine.Proline.var())
# 99166.71735542436

# Apply the log normalization function to the Proline column
wine['Proline_log'] = np.log(wine.Proline)

# Check the variance of the Proline column again
print(wine.Proline_log.var())
# 0.17231366191842012

```




---


## **2.3 Scaling data for feature comparison**



![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/9-11.png?w=852)
![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/10-11.png?w=559)
![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/11-10.png?w=750)



### **2.3.1 Scaling data – investigating columns**



 We want to use the
 `Ash`
 ,
 `Alcalinity of ash`
 , and
 `Magnesium`
 columns in the wine dataset to train a linear model, but it’s possible that these columns are all measured in different ways, which would bias a linear model.





```

wine[['Ash','Alcalinity of ash','Magnesium']].describe()
              Ash  Alcalinity of ash   Magnesium
count  178.000000         178.000000  178.000000
mean     2.366517          19.494944   99.741573
std      0.274344           3.339564   14.282484
min      1.360000          10.600000   70.000000
25%      2.210000          17.200000   88.000000
50%      2.360000          19.500000   98.000000
75%      2.557500          21.500000  107.000000
max      3.230000          30.000000  162.000000

```


### **2.3.2 Scaling data – standardizing columns**



 Since we know that the
 `Ash`
 ,
 `Alcalinity of ash`
 , and
 `Magnesium`
 columns in the wine dataset are all on different scales, let’s standardize them in a way that allows for use in a linear model.





```python

# Import StandardScaler from scikit-learn
from sklearn.preprocessing import StandardScaler

# Create the scaler
ss = StandardScaler()

# Take a subset of the DataFrame you want to scale
wine_subset = wine[['Ash','Alcalinity of ash','Magnesium']]

# Apply the scaler to the DataFrame subset
wine_subset_scaled = ss.fit_transform(wine_subset)

```




```

wine_subset_scaled[:5]
array([[ 0.23205254, -1.16959318,  1.91390522],
       [-0.82799632, -2.49084714,  0.01814502],
       [ 1.10933436, -0.2687382 ,  0.08835836],
       [ 0.4879264 , -0.80925118,  0.93091845],
       [ 1.84040254,  0.45194578,  1.28198515]])

```




---


## **2.4 Standardized data and modeling**


### **2.4.1 KNN on non-scaled data**



 Let’s first take a look at the accuracy of a K-nearest neighbors model on the
 `wine`
 dataset without standardizing the data.





```python

# Split the dataset and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y)

# Fit the k-nearest neighbors model to the training data
knn.fit(X_train, y_train)

# Score the model on the test data
print(knn.score(X_test, y_test))
# 0.6444444444444445

```


### **2.4.2 KNN on scaled data**



 The accuracy score on the unscaled
 `wine`
 dataset was decent, but we can likely do better if we scale the dataset. The process is mostly the same as the previous exercise, with the added step of scaling the data.





```python

# Create the scaling method.
ss = StandardScaler()

# Apply the scaling method to the dataset used for modeling.
X_scaled = ss.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# Fit the k-nearest neighbors model to the training data.
knn.fit(X_train,y_train)

# Score the model on the test data.
print(knn.score(X_test,y_test))
# 0.9555555555555556

```


# **3. Feature Engineering**
---------------------------


## **3.1 Feature engineering**



![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/12-11.png?w=862)
![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/13-10.png?w=895)
![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/14-8.png?w=880)



### **3.1.1 Examples for creating new features**


* timestamps
* newspaper headlines



 Timestamps can be broken into days or months, and headlines can be used for natural language processing.



### **3.1.2 Identifying areas for feature engineering**




```

volunteer[['title','created_date','category_desc']].head(1)
                                               title     created_date category_desc
0  Volunteers Needed For Rise Up & Stay Put! Home...  January 13 2011           Strengthening Communities

```



 All of these columns will require some feature engineering before modeling.





---


## **3.2 Encoding categorical variables**



![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/18-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/15-8.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/16-6.png?w=756)
![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/17-4.png?w=669)



### **3.2.1 Encoding categorical variables – binary**



 Take a look at the
 `hiking`
 dataset. There are several columns here that need encoding, one of which is the
 `Accessible`
 column, which needs to be encoded in order to be modeled.
 `Accessible`
 is a binary feature, so it has two values – either
 `Y`
 or
 `N`
 – so it needs to be encoded into 1s and 0s. Use scikit-learn’s
 `LabelEncoder`
 method to do that transformation.





```python

# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
hiking['Accessible_enc'] = enc.fit_transform(hiking.Accessible)

# Compare the two columns
print(hiking[['Accessible', 'Accessible_enc']].head())

```




```

  Accessible  Accessible_enc
0          Y               1
1          N               0
2          N               0
3          N               0
4          N               0

```


### **3.2.2 Encoding categorical variables – one-hot**



 One of the columns in the
 `volunteer`
 dataset,
 `category_desc`
 , gives category descriptions for the volunteer opportunities listed. Because it is a categorical variable with more than two categories, we need to use one-hot encoding to transform this column numerically.





```python

# Transform the category_desc column
category_enc = pd.get_dummies(volunteer["category_desc"])

# Take a look at the encoded columns
print(category_enc.head())

```




```

   Education  Emergency Preparedness            ...              Helping Neighbors in Need  Strengthening Communities
0          0                       0            ...                                      0                          0
1          0                       0            ...                                      0                          1
2          0                       0            ...                                      0                          1
3          0                       0            ...                                      0                          1
4          0                       0            ...                                      0                          0

[5 rows x 6 columns]

```




---


## **3.3 Engineering numerical features**



![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/19-5.png?w=856)
![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/20-5.png?w=1009)



### **3.3.1 Engineering numerical features – taking an average**



 A good use case for taking an aggregate statistic to create a new feature is to take the mean of columns. Here, you have a DataFrame of running times named
 `running_times_5k`
 . For each
 `name`
 in the dataset, take the mean of their 5 run times.





```python

# Create a list of the columns to average
run_columns = ['run1', 'run2', 'run3', 'run4', 'run5']

# Use apply to create a mean column
# axis=1 = row wise
running_times_5k["mean"] = running_times_5k.apply(lambda row: row[run_columns].mean(), axis=1)

# Take a look at the results
print(running_times_5k)

```




```

      name  run1  run2  run3  run4  run5   mean
0      Sue  20.1  18.5  19.6  20.3  18.3  19.36
1     Mark  16.5  17.1  16.9  17.6  17.3  17.08
2     Sean  23.5  25.1  25.2  24.6  23.9  24.46
3     Erin  21.7  21.1  20.9  22.1  22.2  21.60
4    Jenny  25.8  27.1  26.1  26.7  26.9  26.52
5  Russell  30.9  29.6  31.4  30.4  29.9  30.44

```


### **3.3.2 Engineering numerical features – datetime**



 There are several columns in the
 `volunteer`
 dataset comprised of datetimes. Let’s take a look at the
 `start_date_date`
 column and extract just the month to use as a feature for modeling.





```python

# First, convert string column to date column
volunteer["start_date_converted"] = pd.to_datetime(volunteer["start_date_date"])

# Extract just the month from the converted column
volunteer["start_date_month"] = volunteer["start_date_converted"].apply(lambda row: row.month)

# Take a look at the converted and new month columns
print(volunteer[['start_date_converted', 'start_date_month']].head())

```




```

  start_date_converted  start_date_month
0           2011-07-30                 7
1           2011-02-01                 2
2           2011-01-29                 1
3           2011-02-14                 2
4           2011-02-05                 2

```




---


## **3.4 Text classification**



![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/21-5.png?w=662)
![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/24-5.png?w=540)
![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/22-5.png?w=1005)
![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/23-5.png?w=718)



### **3.4.1 Engineering features from strings – extraction**



 The
 `Length`
 column in the
 `hiking`
 dataset is a column of strings, but contained in the column is the mileage for the hike. We’re going to extract this mileage using regular expressions, and then use a lambda in Pandas to apply the extraction to the DataFrame.





```python

# Write a pattern to extract numbers and decimals
def return_mileage(length):
    pattern = re.compile(r"\d+\.\d+")

    # Search the text for matches
    mile = re.match(pattern, length)

    # If a value is returned, use group(0) to return the found value
    if mile is not None:
        return float(mile.group(0))

# Apply the function to the Length column and take a look at both columns
hiking["Length_num"] = hiking['Length'].apply(lambda row: return_mileage(row))
print(hiking[["Length", "Length_num"]].head())

```




```

       Length  Length_num
0   0.8 miles        0.80
1    1.0 mile        1.00
2  0.75 miles        0.75
3   0.5 miles        0.50
4   0.5 miles        0.50

```


### **3.4.2 Engineering features from strings – tf/idf**



 Let’s transform the
 `volunteer`
 dataset’s
 `title`
 column into a text vector, to use in a prediction task in the next exercise.





```python

# Take the title text
title_text = volunteer['title']

# Create the vectorizer method
tfidf_vec = TfidfVectorizer()

# Transform the text into tf-idf vectors
text_tfidf = tfidf_vec.fit_transform(title_text)

```




```

text_tfidf
<665x1136 sparse matrix of type '<class 'numpy.float64'>'
	with 3397 stored elements in Compressed Sparse Row format>

```


### **3.4.3 Text classification using tf/idf vectors**



 Now that we’ve encoded the
 `volunteer`
 dataset’s
 `title`
 column into tf/idf vectors, let’s use those vectors to try to predict the
 `category_desc`
 column.





```

text_tfidf.toarray()
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]])

text_tfidf.toarray().shape
# (617, 1089)

volunteer["category_desc"].head()
1    Strengthening Communities
2    Strengthening Communities
3    Strengthening Communities
4                  Environment
5                  Environment
Name: category_desc, dtype: object

```




```python

# Split the dataset according to the class distribution of category_desc
y = volunteer["category_desc"]
X_train, X_test, y_train, y_test = train_test_split(text_tfidf.toarray(), y, stratify=y)

# Fit the model to the training data
nb.fit(X_train, y_train)

# Print out the model's accuracy
print(nb.score(X_test, y_test))
# 0.567741935483871

```



 Notice that the model doesn’t score very well. We’ll work on selecting the best features for modeling in the next chapter.



# **4. Selecting features for modeling**
---------------------------------------


## **4.1 Feature selection**



![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/1-12.png?w=770)

### **4.1.1 Identifying areas for feature selection**



 Take an exploratory look at the post-feature engineering
 `hiking`
 dataset. Which of the following columns is a good candidate for feature selection?





```

hiking.columns
Index(['Accessible', 'Difficulty', 'Length', 'Limited_Access', 'Location',
       'Name', 'Other_Details', 'Park_Name', 'Prop_ID', 'lat', 'lon',
       'Length_extract', 'accessible_enc', '', 'Easy', 'Easy ',
       'Easy/Moderate', 'Moderate', 'Moderate/Difficult', 'Various'],
      dtype='object')

```


* Length
* Difficulty
* Accessible



 All three of these columns are good candidates for feature selection.





---


## **4.2 Removing redundant features**



![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/2-13.png?w=596)
![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/3-13.png?w=912)



### **4.2.1 Selecting relevant features**



 Now let’s identify the redundant columns in the
 `volunteer`
 dataset and perform feature selection on the dataset to return a DataFrame of the relevant features.




 For example, if you explore the
 `volunteer`
 dataset in the console, you’ll see three features which are related to location:
 `locality`
 ,
 `region`
 , and
 `postalcode`
 . They contain repeated information, so it would make sense to keep only one of the features.




 There are also features that have gone through the feature engineering process: columns like
 `Education`
 and
 `Emergency Preparedness`
 are a product of encoding the categorical variable
 `category_desc`
 , so
 `category_desc`
 itself is redundant now.




 Take a moment to examine the features of
 `volunteer`
 in the console, and try to identify the redundant features.





```python

# Create a list of redundant column names to drop
to_drop = ["locality", "region", "category_desc", "created_date", "vol_requests"]

# Drop those columns from the dataset
volunteer_subset = volunteer.drop(to_drop, axis=1)

# Print out the head of the new dataset
print(volunteer_subset.head())

```




```

volunteer_subset.columns
Index(['title', 'hits', 'postalcode', 'vol_requests_lognorm', 'created_month',
       'Education', 'Emergency Preparedness', 'Environment', 'Health',
       'Helping Neighbors in Need', 'Strengthening Communities'],
      dtype='object')

```


### **4.2.2 Checking for correlated features**



 Let’s take a look at the
 `wine`
 dataset again, which is made up of continuous, numerical features. Run Pearson’s correlation coefficient on the dataset to determine which columns are good candidates for eliminating. Then, remove those columns from the DataFrame.





```python

# Print out the column correlations of the wine dataset
print(wine.corr())

# Take a minute to find the column where the correlation value is greater than 0.75 at least twice
to_drop = "Flavanoids"

# Drop that column from the DataFrame
wine = wine.drop(to_drop, axis=1)

```




```

print(wine.corr())
                              Flavanoids  Total phenols  Malic acid  OD280/OD315 of diluted wines       Hue
Flavanoids                      1.000000       0.864564   -0.411007                      0.787194  0.543479
Total phenols                   0.864564       1.000000   -0.335167                      0.699949  0.433681
Malic acid                     -0.411007      -0.335167    1.000000                     -0.368710 -0.561296
OD280/OD315 of diluted wines    0.787194       0.699949   -0.368710                      1.000000  0.565468
Hue                             0.543479       0.433681   -0.561296                      0.565468  1.000000

```




---


## **4.3 Selecting features using text vectors**



![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/4-13.png?w=985)
![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/5-13.png?w=997)
![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/6-13.png?w=945)



### **4.3.1 Exploring text vectors, part 1**



 Let’s expand on the text vector exploration method we just learned about, using the
 `volunteer`
 dataset’s
 `title`
 tf/idf vectors. In this first part of text vector exploration, we’re going to add to that function we learned about in the slides. We’ll return a list of numbers with the function. In the next exercise, we’ll write another function to collect the top words across all documents, extract them, and then use that list to filter down our
 `text_tfidf`
 vector.





```

vocab
{1048: 'web',
 278: 'designer',
 1017: 'urban',
...}

tfidf_vec
TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 1), norm='l2', preprocessor=None, smooth_idf=True,
        stop_words=None, strip_accents=None, sublinear_tf=False,
        token_pattern='(?u)\\b\\w\\w+\\b', tokenizer=None, use_idf=True,
        vocabulary=None)

tfidf_vec.vocabulary_
{'web': 1048,
 'designer': 278,
 'urban': 1017,
...}

text_tfidf
<617x1089 sparse matrix of type '<class 'numpy.float64'>'
	with 3172 stored elements in Compressed Sparse Row format>

```




```python

# Add in the rest of the parameters
def return_weights(vocab, original_vocab, vector, vector_index, top_n):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))

    # Let's transform that zipped dict into a series
    zipped_series = pd.Series({vocab[i]:zipped[i] for i in vector[vector_index].indices})

    # Let's sort the series to pull out the top n weighted words
    zipped_index = zipped_series.sort_values(ascending=False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]

# Print out the weighted words
print(return_weights(vocab, tfidf_vec.vocabulary_, text_tfidf, vector_index=8, top_n=3))
# [189, 942, 466]

```


### **4.3.2 Exploring text vectors, part 2**



 Using the function we wrote in the previous exercise, we’re going to extract the top words from each document in the text vector, return a list of the word indices, and use that list to filter the text vector down to those top words.





```

def words_to_filter(vocab, original_vocab, vector, top_n):
    filter_list = []
    for i in range(0, vector.shape[0]):

        # Here we'll call the function from the previous exercise, and extend the list we're creating
        filtered = return_weights(vocab, original_vocab, vector, i, top_n)
        filter_list.extend(filtered)
    # Return the list in a set, so we don't get duplicate word indices
    return set(filter_list)

# Call the function to get the list of word indices
filtered_words = words_to_filter(vocab, tfidf_vec.vocabulary_, text_tfidf, 3)

# By converting filtered_words back to a list, we can use it to filter the columns in the text vector
filtered_text = text_tfidf[:, list(filtered_words)]

filtered_text
<617x1008 sparse matrix of type '<class 'numpy.float64'>'
	with 2948 stored elements in Compressed Sparse Row format>

```


### **4.3.3 Training Naive Bayes with feature selection**



 Let’s re-run the Naive Bayes text classification model we ran at the end of chapter 3, with our selection choices from the previous exercise, on the
 `volunteer`
 dataset’s
 `title`
 and
 `category_desc`
 columns.





```python

# Split the dataset according to the class distribution of category_desc, using the filtered_text vector
train_X, test_X, train_y, test_y = train_test_split(filtered_text.toarray(), y, stratify=y)

# Fit the model to the training data
nb.fit(train_X, train_y)

# Print out the model's accuracy
print(nb.score(test_X,test_y))
# 0.567741935483871

```



 You can see that our accuracy score wasn’t that different from the score at the end of chapter 3. That’s okay; the
 `title`
 field is a very small text field, appropriate for demonstrating how filtering vectors works.





---


## **4.4 Dimensionality reduction**



![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/7-13.png?w=830)



### **4.4.1 Using PCA**



 Let’s apply PCA to the
 `wine`
 dataset, to see if we can get an increase in our model’s accuracy.





```

from sklearn.decomposition import PCA

# Set up PCA and the X vector for diminsionality reduction
pca = PCA()
wine_X = wine.drop("Type", axis=1)

# Apply PCA to the wine dataset X vector
transformed_X = pca.fit_transform(wine_X)

# Look at the percentage of variance explained by the different components
print(pca.explained_variance_ratio_)

```




```

[9.98091230e-01 1.73591562e-03 9.49589576e-05 5.02173562e-05
 1.23636847e-05 8.46213034e-06 2.80681456e-06 1.52308053e-06
 1.12783044e-06 7.21415811e-07 3.78060267e-07 2.12013755e-07
 8.25392788e-08]

```


### **4.4.2 Training a model with PCA**



 Now that we have run PCA on the
 `wine`
 dataset, let’s try training a model with it.





```python

# Split the transformed X and the y labels into training and test sets
X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(transformed_X,y)

# Fit knn to the training data
knn.fit(X_wine_train,y_wine_train)

# Score knn on the test data and print it out
knn.score(X_wine_test,y_wine_test)
# 0.6444444444444445

```


# **5. Putting it all together**
-------------------------------


## **5.1 UFOs and preprocessing**




```

ufo.head()
                 date               city state country      type  seconds  \
2 2002-11-21 05:45:00           clemmons    nc      us  triangle    300.0
4 2012-06-16 23:00:00          san diego    ca      us     light    600.0
7 2013-06-09 00:00:00  oakville (canada)    on      ca     light    120.0
8 2013-04-26 23:27:00              lacey    wa      us     light    120.0
9 2013-09-13 20:30:00           ben avon    pa      us    sphere    300.0

    length_of_time                                               desc  \
2  about 5 minutes  It was a large, triangular shaped flying ob...
4       10 minutes  Dancing lights that would fly around and then ...
7        2 minutes  Brilliant orange light or chinese lantern at o...
8        2 minutes  Bright red light moving north to north west fr...
9        5 minutes  North-east moving south-west. First 7 or so li...

     recorded        lat        long
2  12/23/2002  36.021389  -80.382222
4    7/4/2012  32.715278 -117.156389
7    7/3/2013  43.433333  -79.666667
8   5/15/2013  47.034444 -122.821944
9   9/30/2013  40.508056  -80.083333

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/8-12.png?w=977)

### **5.1.1 Checking column types**



 Take a look at the UFO dataset’s column types using the
 `dtypes`
 attribute. Two columns jump out for transformation: the seconds column, which is a numeric column but is being read in as
 `object`
 , and the
 `date`
 column, which can be transformed into the
 `datetime`
 type. That will make our feature engineering efforts easier later on.





```python

# Check the column types
print(ufo.dtypes)

# Change the type of seconds to float
ufo["seconds"] = ufo.seconds.astype('float')

# Change the date column to type datetime
ufo["date"] = pd.to_datetime(ufo['date'])

# Check the column types
print(ufo[['seconds','date']].dtypes)

```




```

date               object
city               object
state              object
country            object
type               object
seconds            object
length_of_time     object
desc               object
recorded           object
lat                object
long              float64
dtype: object
seconds           float64
date       datetime64[ns]
dtype: object

```


### **5.1.2 Dropping missing data**



 Let’s remove some of the rows where certain columns have missing values. We’re going to look at the
 `length_of_time`
 column, the
 `state`
 column, and the
 `type`
 column. If any of the values in these columns are missing, we’re going to drop the rows.





```python

# Check how many values are missing in the length_of_time, state, and type columns
print(ufo[['length_of_time', 'state', 'type']].isnull().sum())

# Keep only rows where length_of_time, state, and type are not null
ufo_no_missing = ufo[ufo['length_of_time'].notnull() &
          ufo['state'].notnull() &
          ufo['type'].notnull()]

# Print out the shape of the new dataset
print(ufo_no_missing.shape)

```




```

length_of_time    143
state             419
type              159
dtype: int64
(4283, 4)

```




---


## **5.2 Categorical variables and standardization**



![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/9-12.png?w=731)
![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/10-12.png?w=486)



### **5.2.1 Extracting numbers from strings**



 The
 `length_of_time`
 field in the UFO dataset is a text field that has the number of minutes within the string. Here, you’ll extract that number from that text field using regular expressions.





```

def return_minutes(time_string):

    # Use \d+ to grab digits
    pattern = re.compile(r"\d+")

    # Use match on the pattern and column
    num = re.match(pattern, time_string)
    if num is not None:
        return int(num.group(0))

# Apply the extraction to the length_of_time column
ufo["minutes"] = ufo["length_of_time"].apply(return_minutes)

# Take a look at the head of both of the columns
print(ufo[['length_of_time','minutes']].head())

```




```

    length_of_time  minutes
2  about 5 minutes      NaN
4       10 minutes     10.0
7        2 minutes      2.0
8        2 minutes      2.0
9        5 minutes      5.0

```


### **5.2.2 Identifying features for standardization**



 In this section, you’ll investigate the variance of columns in the UFO dataset to determine which features should be standardized. After taking a look at the variances of the
 `seconds`
 and
 `minutes`
 column, you’ll see that the variance of the
 `seconds`
 column is extremely high. Because
 `seconds`
 and
 `minutes`
 are related to each other (an issue we’ll deal with when we select features for modeling), let’s log normlize the
 `seconds`
 column.





```python

# Check the variance of the seconds and minutes columns
print(ufo[['seconds','minutes']].var())

# Log normalize the seconds column
ufo["seconds_log"] = np.log(ufo[['seconds']])

# Print out the variance of just the seconds_log column
print(ufo["seconds_log"].var())

```




```

seconds    424087.417474
minutes       117.546372
dtype: float64
1.1223923881183004

```




---


## **5.3 Engineering new features**



![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/11-11.png?w=995)

### **5.3.1 Encoding categorical variables**



 There are couple of columns in the UFO dataset that need to be encoded before they can be modeled through scikit-learn. You’ll do that transformation here, using both binary and one-hot encoding methods.





```python

# Use Pandas to encode us values as 1 and others as 0
ufo["country_enc"] = ufo["country"].apply(lambda x: 1 if x=='us' else 0)

# Print the number of unique type values
print(len(ufo['type'].unique()))
# 21

# Create a one-hot encoded set of the type values
type_set = pd.get_dummies(ufo['type'])

# Concatenate this set back to the ufo DataFrame
ufo = pd.concat([ufo, type_set], axis=1)

```


### **5.3.2 Features from dates**



 Another feature engineering task to perform is month and year extraction. Perform this task on the
 `date`
 column of the
 `ufo`
 dataset.





```python

# Look at the first 5 rows of the date column
print(ufo['date'].head())

# Extract the month from the date column
ufo["month"] = ufo["date"].apply(lambda x:x.month)

# Extract the year from the date column
ufo["year"] = ufo["date"].apply(lambda x:x.year)

# Take a look at the head of all three columns
print(ufo[['date','month','year']].head())

```




```

0   2002-11-21 05:45:00
1   2012-06-16 23:00:00
2   2013-06-09 00:00:00
3   2013-04-26 23:27:00
4   2013-09-13 20:30:00
Name: date, dtype: datetime64[ns]
                 date  month  year
0 2002-11-21 05:45:00     11  2002
1 2012-06-16 23:00:00      6  2012
2 2013-06-09 00:00:00      6  2013
3 2013-04-26 23:27:00      4  2013
4 2013-09-13 20:30:00      9  2013

```


### **5.3.3 Text vectorization**



 Let’s transform the
 `desc`
 column in the UFO dataset into tf/idf vectors, since there’s likely something we can learn from this field.





```python

# Take a look at the head of the desc field
print(ufo["desc"].head())

# Create the tfidf vectorizer object
vec = TfidfVectorizer()

# Use vec's fit_transform method on the desc field
desc_tfidf = vec.fit_transform(ufo["desc"])

# Look at the number of columns this creates
print(desc_tfidf.shape)

```




```

0    It was a large, triangular shaped flying ob...
1    Dancing lights that would fly around and then ...
2    Brilliant orange light or chinese lantern at o...
3    Bright red light moving north to north west fr...
4    North-east moving south-west. First 7 or so li...
Name: desc, dtype: object
(1866, 3422)

```




---


## **5.4 Feature selection and modeling**



![Desktop View]({{ site.baseurl }}/assets/datacamp/preprocessing-for-machine-learning-in-python/12-12.png?w=925)

### **5.4.1 Selecting the ideal dataset**



 Let’s get rid of some of the unnecessary features. Because we have an encoded country column,
 `country_enc`
 , keep it and drop other columns related to location:
 `city`
 ,
 `country`
 ,
 `lat`
 ,
 `long`
 ,
 `state`
 .




 We have columns related to
 `month`
 and
 `year`
 , so we don’t need the
 `date`
 or
 `recorded`
 columns.




 We vectorized
 `desc`
 , so we don’t need it anymore. For now we’ll keep
 `type`
 .




 We’ll keep
 `seconds_log`
 and drop
 `seconds`
 and
 `minutes`
 .




 Let’s also get rid of the
 `length_of_time`
 column, which is unnecessary after extracting
 `minutes`
 .





```python

# Check the correlation between the seconds, seconds_log, and minutes columns
print(ufo[['seconds','seconds_log','minutes']].corr())

# Make a list of features to drop
to_drop = ['city', 'country', 'date', 'desc', 'lat', 'length_of_time', 'long', 'minutes', 'recorded', 'seconds', 'state']

# Drop those features
ufo_dropped = ufo.drop(to_drop,axis=1)

# Let's also filter some words out of the text vector we created
filtered_words = words_to_filter(vocab, vec.vocabulary_, desc_tfidf, 4)

```




```

              seconds  seconds_log   minutes
seconds      1.000000     0.853371  0.980341
seconds_log  0.853371     1.000000  0.824493
minutes      0.980341     0.824493  1.000000

```


### **5.4.2 Modeling the UFO dataset, part 1**



 In this exercise, we’re going to build a k-nearest neighbor model to predict which country the UFO sighting took place in. Our
 `X`
 dataset has the log-normalized seconds column, the one-hot encoded type columns, as well as the month and year when the sighting took place. The
 `y`
 labels are the encoded country column, where 1 is
 `us`
 and 0 is
 `ca`
 .





```python

# Take a look at the features in the X set of data
print(X.columns)

# Split the X and y sets using train_test_split, setting stratify=y
train_X, test_X, train_y, test_y = train_test_split(X,y,stratify=y)

# Fit knn to the training sets
knn.fit(train_X,train_y)

# Print the score of knn on the test sets
print(knn.score(test_X,test_y))
# 0.8693790149892934

```




```

Index(['seconds_log', 'changing', 'chevron', 'cigar', 'circle', 'cone',
       'cross', 'cylinder', 'diamond', 'disk', 'egg', 'fireball', 'flash',
       'formation', 'light', 'other', 'oval', 'rectangle', 'sphere',
       'teardrop', 'triangle', 'unknown', 'month', 'year'],
      dtype='object')

```


### **5.4.3 Modeling the UFO dataset, part 2**



 Finally, let’s build a model using the text vector we created,
 `desc_tfidf`
 , using the
 `filtered_words`
 list to create a filtered text vector. Let’s see if we can predict the
 `type`
 of the sighting based on the text. We’ll use a Naive Bayes model for this.





```python

# Use the list of filtered words we created to filter the text vector
filtered_text = desc_tfidf[:, list(filtered_words)]

# Split the X and y sets using train_test_split, setting stratify=y
train_X, test_X, train_y, test_y = train_test_split(filtered_text.toarray(), y, stratify=y)

# Fit nb to the training sets
nb.fit(train_X,train_y)

# Print the score of nb on the test sets
print(nb.score(test_X,test_y))
# 0.16274089935760172

```



 As you can see, this model performs very poorly on this text data. This is a clear case where iteration would be necessary to figure out what subset of text improves the model, and if perhaps any of the other features are useful in predicting
 `type`
 .





---



 Thank you for reading and hope you’ve learned a lot.



