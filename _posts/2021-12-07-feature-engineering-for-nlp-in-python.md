---
title: Feature Engineering for NLP in Python
date: 2021-12-07 11:22:08 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Feature Engineering for NLP in Python
========================================







 This is the memo of the 13th course (23 courses in all) of ‘Machine Learning Scientist with Python’ skill track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/feature-engineering-for-nlp-in-python)**
 .



###
**Course Description**



 In this course, you will learn techniques that will allow you to extract useful information from text and process them into a format suitable for applying ML models. More specifically, you will learn about POS tagging, named entity recognition, readability scores, the n-gram and tf-idf models, and how to implement them using scikit-learn and spaCy. You will also learn to compute how similar two documents are to each other. In the process, you will predict the sentiment of movie reviews and build movie and Ted Talk recommenders. Following the course, you will be able to engineer critical features out of any text and solve some of the most challenging problems in data science!



###
**Table of contents**


* [Basic features and readability scores](https://datascience103579984.wordpress.com/2020/01/19/feature-engineering-for-nlp-in-python-from-datacamp/)
* [Text preprocessing, POS tagging and NER](https://datascience103579984.wordpress.com/2020/01/19/feature-engineering-for-nlp-in-python-from-datacamp/2/)
* [N-Gram models](https://datascience103579984.wordpress.com/2020/01/19/feature-engineering-for-nlp-in-python-from-datacamp/3/)
* [TF-IDF and similarity scores](https://datascience103579984.wordpress.com/2020/01/19/feature-engineering-for-nlp-in-python-from-datacamp/4/)





# **1. Basic features and readability scores**
---------------------------------------------


## **1.1 Introduction to NLP feature engineering**



![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/7-9.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/8-9.png?w=764)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/9-9.png?w=897)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/10-9.png?w=851)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/11-9.png?w=696)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/12-9.png?w=580)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/13-8.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/14-8.png?w=379)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/15-8.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/16-7.png?w=542)



### **1.1.1 Data format for ML algorithms**



 In this exercise, you have been given four dataframes
 `df1`
 ,
 `df2`
 ,
 `df3`
 and
 `df4`
 . The final column of each dataframe is the predictor variable and the rest of the columns are training features.




 Using the console, determine which dataframe is in a suitable format to be trained by a classifier.





```

df3
    feature 1  feature 2  feature 3  feature 4  feature 5  label
0           1         85         66         29          0      0
1           8        183         64          0          0      1
2           1         89         66         23         94      0
3           0        137         40         35        168      1
4           5        116         74          0          0      0
...

```


### **1.1.2 One-hot encoding**



 In the previous exercise, we encountered a dataframe
 `df1`
 which contained categorical features and therefore, was unsuitable for applying ML algorithms to.




 In this exercise, your task is to convert
 `df1`
 into a format that is suitable for machine learning.





```python

# Print the features of df1
print(df1.columns)

# Perform one-hot encoding
df1 = pd.get_dummies(df1, columns=['feature 5'])

# Print the new features of df1
print(df1.columns)

# Print first five rows of df1
print(df1.head())

```




```

Index(['feature 1', 'feature 2', 'feature 3', 'feature 4', 'feature 5', 'label'], dtype='object')
Index(['feature 1', 'feature 2', 'feature 3', 'feature 4', 'label', 'feature 5_female', 'feature 5_male'], dtype='object')

   feature 1  feature 2  feature 3  feature 4  label  feature 5_female  feature 5_male
0    29.0000          0          0   211.3375      1                 1               0
1     0.9167          1          2   151.5500      1                 0               1
2     2.0000          1          2   151.5500      0                 1               0
3    30.0000          1          2   151.5500      0                 0               1
4    25.0000          1          2   151.5500      0                 1               0

```




---


## **1.2 Basic feature extraction**



![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/1-11.png?w=653)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/2-11.png?w=612)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/3-10.png?w=744)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/4-10.png?w=931)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/5-10.png?w=821)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/6-10.png?w=1012)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/7-10.png?w=539)



### **1.2.1 Character count of Russian tweets**



 In this exercise, you have been given a dataframe
 `tweets`
 which contains some tweets associated with Russia’s Internet Research Agency and compiled by FiveThirtyEight.




 Your task is to create a new feature ‘char_count’ in
 `tweets`
 which computes the number of characters for each tweet. Also, compute the average length of each tweet. The tweets are available in the
 `content`
 feature of
 `tweets`
 .




*Be aware that this is real data from Twitter and as such there is always a risk that it may contain profanity or other offensive content (in this exercise, and any following exercises that also use real Twitter data).*





```

tweets
                                               content
0    LIVE STREAM VIDEO=> Donald Trump Rallies in Co...
1    Muslim Attacks NYPD Cops with Meat Cleaver. Me...
2    .@vfpatlas well that's a swella word there (di...
...

```




```python

# Create a feature char_count
tweets['char_count'] = tweets['content'].apply(len)

# Print the average character count
print(tweets['char_count'].mean())
# 103.462

```



 Great job! Notice that the average character count of these tweets is approximately 104, which is much higher than the overall average tweet length of around 40 characters. Depending on what you’re working on, this may be something worth investigating into. For your information, there is research that indicates that fake news articles tend to have longer titles! Therefore, even extremely basic features such as character counts can prove to be very useful in certain applications.



### **1.2.2 Word count of TED talks**



`ted`
 is a dataframe that contains the transcripts of 500 TED talks. Your job is to compute a new feature
 `word_count`
 which contains the approximate number of words for each talk. Consequently, you also need to compute the average word count of the talks. The transcripts are available as the
 `transcript`
 feature in
 `ted`
 .




 In order to complete this task, you will need to define a function
 `count_words`
 that takes in a string as an argument and returns the number of words in the string. You will then need to apply this function to the
 `transcript`
 feature of
 `ted`
 to create the new feature
 `word_count`
 and compute its mean.





```python

# Function that returns number of words in a string
def count_words(string):
	# Split the string into words
    words = string.split()

    # Return the number of words
    return len(words)

# Create a new feature word_count
ted['word_count'] = ted['transcript'].apply(count_words)

# Print the average word count of the talks
print(ted['word_count'].mean())
# 1987.1

```



 Amazing work! You now know how to compute the number of words in a given piece of text. Also, notice that the average length of a talk is close to 2000 words. You can use the
 `word_count`
 feature to compute its correlation with other variables such as number of views, number of comments, etc. and derive extremely interesting insights about TED.



### **1.2.3 Hashtags and mentions in Russian tweets**



 Let’s revisit the
 `tweets`
 dataframe containing the Russian tweets. In this exercise, you will compute the number of hashtags and mentions in each tweet by defining two functions
 `count_hashtags()`
 and
 `count_mentions()`
 respectively and applying them to the
 `content`
 feature of
 `tweets`
 .




 In case you don’t recall, the tweets are contained in the
 `content`
 feature of
 `tweets`
 .





```python

# Function that returns numner of hashtags in a string
def count_hashtags(string):
	# Split the string into words
    words = string.split()

    # Create a list of words that are hashtags
    hashtags = [word for word in words if word.startswith('#')]

    # Return number of hashtags
    return(len(hashtags))

# Create a feature hashtag_count and display distribution
tweets['hashtag_count'] = tweets['content'].apply(count_hashtags)
tweets['hashtag_count'].hist()
plt.title('Hashtag count distribution')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/8-10.png?w=1021)



```python

# Function that returns number of mentions in a string
def count_mentions(string):
	# Split the string into words
    words = string.split()

    # Create a list of words that are mentions
    mentions = [word for word in words if word.startswith('@')]

    # Return number of mentions
    return(len(mentions))

# Create a feature mention_count and display distribution
tweets['mention_count'] = tweets['content'].apply(count_mentions)
tweets['mention_count'].hist()
plt.title('Mention count distribution')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/9-10.png?w=1024)


 Excellent work! You now have a good grasp of how to compute various types of summary features. In the next lesson, we will learn about more advanced features that are capable of capturing more nuanced information beyond simple word and character counts.





---


## **1.3 Readability tests**



![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/10-10.png?w=996)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/11-10.png?w=764)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/12-10.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/13-9.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/14-9.png?w=1008)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/15-9.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/16-8.png?w=679)



### **1.3.1 Readability of ‘The Myth of Sisyphus’**



 In this exercise, you will compute the Flesch reading ease score for Albert Camus’ famous essay
 *The Myth of Sisyphus*
 . We will then interpret the value of this score as explained in the video and try to determine the reading level of the essay.




 The entire essay is in the form of a string and is available as
 `sisyphus_essay`
 .





```python

# Import Textatistic
from textatistic import Textatistic

# Compute the readability scores
readability_scores = Textatistic(sisyphus_essay).scores

# Print the flesch reading ease score
flesch = readability_scores['flesch_score']
print("The Flesch Reading Ease is %.2f" % (flesch))
# The Flesch Reading Ease is 81.67

```



 Excellent! You now know to compute the Flesch reading ease score for a given body of text. Notice that the score for this essay is approximately 81.67. This indicates that the essay is at the readability level of a 6th grade American student.



### **1.3.2 Readability of various publications**



 In this exercise, you have been given excerpts of articles from four publications. Your task is to compute the readability of these excerpts using the Gunning fog index and consequently, determine the relative difficulty of reading these publications.




 The excerpts are available as the following strings:



* `forbes`
 – An excerpt from an article from
 *Forbes*
 magazine on the Chinese social credit score system.
* `harvard_law`
 – An excerpt from a book review published in
 *Harvard Law Review*
 .
* `r_digest`
 – An excerpt from a
 *Reader’s Digest*
 article on flight turbulence.
* `time_kids`
 – An excerpt from an article on the ill effects of salt consumption published in
 *TIME for Kids*
 .




```python

# Import Textatistic
from textatistic import Textatistic

# List of excerpts
excerpts = [forbes, harvard_law, r_digest, time_kids]

# Loop through excerpts and compute gunning fog index
gunning_fog_scores = []
for excerpt in excerpts:
  readability_scores = Textatistic(excerpt).scores
  gunning_fog = readability_scores['gunningfog_score']
  gunning_fog_scores.append(gunning_fog)

# Print the gunning fog indices
print(gunning_fog_scores)
# [14.436002482929858, 20.735401069518716, 11.085587583148559, 5.926785009861934]

```



 Great job! You are now adept at computing readability scores for various pieces of text. Notice that the Harvard Law Review excerpt has the highest Gunning fog index; indicating that it can be comprehended only by readers who have graduated college. On the other hand, the Time for Kids article, intended for children, has a much lower fog index and can be comprehended by 5th grade students.







# **2. Text preprocessing, POS tagging and NER**
-----------------------------------------------


## **2.1 Tokenization and Lemmatization**



![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/1-12.png?w=395)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/2-12.png?w=883)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/3-11.png?w=891)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/4-11.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/5-11.png?w=983)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/6-11.png?w=1022)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/7-11.png?w=1024)



### **2.1.1 Tokenizing the Gettysburg Address**



 In this exercise, you will be tokenizing one of the most famous speeches of all time: the Gettysburg Address delivered by American President Abraham Lincoln during the American Civil War.




 The entire speech is available as a string named
 `gettysburg`
 .





```

gettysburg
"Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.
...

```




```

import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(gettysburg)

# Generate the tokens
tokens = [token.text for token in doc]
print(tokens)

```




```

['Four', 'score', 'and', 'seven', 'years', 'ago', 'our', 'fathers', 'brought', 'forth', 'on', 'this', 'continent', ',', 'a', 'new', 'nation', ...

```



 Excellent work! You now know how to tokenize a piece of text. In the next exercise, we will perform similar steps and conduct lemmatization.



### **2.1.2 Lemmatizing the Gettysburg address**



 In this exercise, we will perform lemmatization on the same
 `gettysburg`
 address from before.




 However, this time, we will also take a look at the speech, before and after lemmatization, and try to adjudge the kind of changes that take place to make the piece more machine friendly.





```

import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(gettysburg)

# Generate lemmas
lemmas = [token.lemma_ for token in doc]

# Convert lemmas into a string
print(' '.join(lemmas))

```




```

four score and seven year ago -PRON- father bring forth on this continent , a new nation , conceive in liberty , and dedicate to the proposition that all man be create equal .

```



 Excellent! You’re now proficient at performing lemmatization using spaCy. Observe the lemmatized version of the speech. It isn’t very readable to humans but it is in a much more convenient format for a machine to process.





---


## **2.2 Text cleaning**



![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/8-11.png?w=779)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/9-11.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/10-11.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/11-11.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/12-11.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/13-10.png?w=612)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/14-10.png?w=975)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/15-10.png?w=963)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/16-9.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/17-6.png?w=1024)



### **2.2.1 Cleaning a blog post**



 In this exercise, you have been given an excerpt from a blog post. Your task is to clean this text into a more machine friendly format. This will involve converting to lowercase, lemmatization and removing stopwords, punctuations and non-alphabetic characters.




 The excerpt is available as a string
 `blog`
 and has been printed to the console. The list of stopwords are available as
 `stopwords`
 .





```

Twenty-first-century politics has witnessed an alarming rise of populism in the U.S. and Europe. The first warning signs came with the UK Brexit Referendum vote in 2016 swinging in the way of Leave. This was followed by a stupendous victory by billionaire Donald Trump to become the 45th President of the United States in November 2016. Since then, Europe has seen a steady rise in populist and far-right parties that have capitalized on Europe’s Immigration Crisis to raise nationalist and anti-Europe sentiments. Some instances include Alternative for Germany (AfD) winning 12.6% of all seats and entering the Bundestag, thus upsetting Germany’s political order for the first time since the Second World War, the success of the Five Star Movement in Italy and the surge in popularity of neo-nazism and neo-fascism in countries such as Hungary, Czech Republic, Poland and Austria.

```




```python

# Load model and create Doc object
nlp = spacy.load('en_core_web_sm')
doc = nlp(blog)

# Generate lemmatized tokens
lemmas = [token.lemma_ for token in doc]

# Remove stopwords and non-alphabetic tokens
a_lemmas = [lemma for lemma in lemmas
            if lemma.isalpha() and lemma not in stopwords]

# Print string after text cleaning
print(' '.join(a_lemmas))

```




```

century politic witness alarming rise populism europe warning sign come uk brexit referendum vote swinging way leave follow stupendous victory billionaire donald trump president united states november europe steady rise populist far right party capitalize europe immigration crisis raise nationalist anti europe sentiment instance include alternative germany afd win seat enter bundestag upset germany political order time second world war success star movement italy surge popularity neo nazism neo fascism country hungary czech republic poland austria

```



 Great job! Take a look at the cleaned text; it is lowercased and devoid of numbers, punctuations and commonly used stopwords. Also, note that the word U.S. was present in the original text. Since it had periods in between, our text cleaning process completely removed it. This may not be ideal behavior. It is always advisable to use your custom functions in place of
 `isalpha()`
 for more nuanced cases.



### **2.2.2 Cleaning TED talks in a dataframe**



 In this exercise, we will revisit the TED Talks from the first chapter. You have been a given a dataframe
 `ted`
 consisting of 5 TED Talks. Your task is to clean these talks using techniques discussed earlier by writing a function
 `preprocess`
 and applying it to the
 `transcript`
 feature of the dataframe.




 The stopwords list is available as
 `stopwords`
 .





```python

# Function to preprocess text
def preprocess(text):
  	# Create Doc object
    doc = nlp(text, disable=['ner', 'parser'])
    # Generate lemmas
    lemmas = [token.lemma_ for token in doc]
    # Remove stopwords and non-alphabetic characters
    a_lemmas = [lemma for lemma in lemmas
            if lemma.isalpha() and lemma not in stopwords]

    return ' '.join(a_lemmas)

# Apply preprocess to ted['transcript']
ted['transcript'] = ted['transcript'].apply(preprocess)
print(ted['transcript'])

```




```

0     talk new lecture ted illusion create ted try r...
1     representation brain brain break left half log...
2     great honor today share digital universe creat...
...

```



 Excellent job! You have preprocessed all the TED talk transcripts contained in
 `ted`
 and it is now in a good shape to perform operations such as vectorization (as we will soon see how). You now have a good understanding of how text preprocessing works and why it is important. In the next lessons, we will move on to generating word level features for our texts.





---


## **2.3 Part-of-speech(POS) tagging**



![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/1-13.png?w=682)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/2-13.png?w=839)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/3-12.png?w=746)



### **2.3.1 POS tagging in Lord of the Flies**



 In this exercise, you will perform part-of-speech tagging on a famous passage from one of the most well-known novels of all time,
 *Lord of the Flies*
 , authored by William Golding.




 The passage is available as
 `lotf`
 and has already been printed to the console.





```

He found himself understanding the wearisomeness of this life, where every path was an improvisation and a considerable part of one’s waking life was spent watching one’s feet.

```




```python

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(lotf)

# Generate tokens and pos tags
pos = [(token.text, token.pos_) for token in doc]
print(pos)

```




```

[('He', 'PRON'), ('found', 'VERB'), ('himself', 'PRON'), ('understanding', 'VERB'), ('the', 'DET'), ('wearisomeness', 'NOUN'), ('of', 'ADP'), ('this', 'DET'), ('life', 'NOUN'), (',', 'PUNCT'), ('where', 'ADV'), ('every', 'DET'), ('path', 'NOUN'), ('was', 'VERB'), ('an', 'DET'), ('improvisation', 'NOUN'), ('and', 'CCONJ'), ('a', 'DET'), ('considerable', 'ADJ'), ('part', 'NOUN'), ('of', 'ADP'), ('one', 'NUM'), ('’s', 'PART'), ('waking', 'NOUN'), ('life', 'NOUN'), ('was', 'VERB'), ('spent', 'VERB'), ('watching', 'VERB'), ('one', 'PRON'), ('’s', 'PART'), ('feet', 'NOUN'), ('.', 'PUNCT')]

```



 Good job! Examine the various POS tags attached to each token and evaluate if they make intuitive sense to you. You will notice that they are indeed labelled correctly according to the standard rules of English grammar.



### **2.3.2 Counting nouns in a piece of text**



 In this exercise, we will write two functions,
 `nouns()`
 and
 `proper_nouns()`
 that will count the number of other nouns and proper nouns in a piece of text respectively.




 These functions will take in a piece of text and generate a list containing the POS tags for each word. It will then return the number of proper nouns/other nouns that the text contains. We will use these functions in the next exercise to generate interesting insights about fake news.




 The
 `en_core_web_sm`
 model has already been loaded as
 `nlp`
 in this exercise.





```

nlp = spacy.load('en_core_web_sm')

# Returns number of proper nouns
def proper_nouns(text, model=nlp):
  	# Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]

    # Return number of proper nouns
    return pos.count('PROPN')

print(proper_nouns("Abdul, Bill and Cathy went to the market to buy apples.", nlp))
# 3

```




```

nlp = spacy.load('en_core_web_sm')

# Returns number of other nouns
def nouns(text, model=nlp):
  	# Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]

    # Return number of other nouns
    return pos.count('NOUN')

print(nouns("Abdul, Bill and Cathy went to the market to buy apples.", nlp))
# 2

```



 Great job! You now know how to write functions that compute the number of instances of a particulat POS tag in a given piece of text. In the next exercise, we will use these functions to generate features from text in a dataframe.



### **2.3.3 Noun usage in fake news**



 In this exercise, you have been given a dataframe
 `headlines`
 that contains news headlines that are either fake or real. Your task is to generate two new features
 `num_propn`
 and
 `num_noun`
 that represent the number of proper nouns and other nouns contained in the
 `title`
 feature of
 `headlines`
 .




 Next, we will compute the mean number of proper nouns and other nouns used in fake and real news headlines and compare the values. If there is a remarkable difference, then there is a good chance that using the
 `num_propn`
 and
 `num_noun`
 features in fake news detectors will improve its performance.




 To accomplish this task, the functions
 `proper_nouns`
 and
 `nouns`
 that you had built in the previous exercise have already been made available to you.





```

headlines
    Unnamed: 0                                              title label
0            0                       You Can Smell Hillary’s Fear  FAKE
1            1  Watch The Exact Moment Paul Ryan Committed Pol...  FAKE
2            2        Kerry to go to Paris in gesture of sympathy  REAL
3            3  Bernie supporters on Twitter erupt in anger ag...  FAKE
4            4   The Battle of New York: Why This Primary Matters  REAL

```




```

headlines['num_propn'] = headlines['title'].apply(proper_nouns)
headlines['num_noun'] = headlines['title'].apply(nouns)

# Compute mean of proper nouns
real_propn = headlines[headlines['label'] == 'REAL']['num_propn'].mean()
fake_propn = headlines[headlines['label'] == 'FAKE']['num_propn'].mean()

# Compute mean of other nouns
real_noun = headlines[headlines['label'] == 'REAL']['num_noun'].mean()
fake_noun = headlines[headlines['label'] == 'FAKE']['num_noun'].mean()

# Print results
print("Mean no. of proper nouns in real and fake headlines are %.2f and %.2f respectively"%(real_propn, fake_propn))
print("Mean no. of other nouns in real and fake headlines are %.2f and %.2f respectively"%(real_noun, fake_noun))

```




```

Mean no. of proper nouns in real and fake headlines are 2.46 and 4.86 respectively
Mean no. of other nouns in real and fake headlines are 2.30 and 1.44 respectively

```



 Excellent work! You now know to construct features using POS tags information. Notice how the mean number of proper nouns is considerably higher for fake news than it is for real news. The opposite seems to be true in the case of other nouns. This fact can be put to great use in desgning fake news detectors.





---


## **2.4 Named entity recognition(NER)**



![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/4-12.png?w=439)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/5-12.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/6-12.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/7-12.png?w=1018)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/8-12.png?w=802)



### **2.4.1 Named entities in a sentence**



 In this exercise, we will identify and classify the labels of various named entities in a body of text using one of spaCy’s statistical models. We will also verify the veracity of these labels.





```python

# Load the required model
nlp = spacy.load('en_core_web_sm')

# Create a Doc instance
text = 'Sundar Pichai is the CEO of Google. Its headquarters is in Mountain View.'
doc = nlp(text)

# Print all named entities and their labels
for ent in doc.ents:
    print(ent.text, ent.label_)

```




```

Sundar Pichai ORG
Google ORG
Mountain View GPE

```



 Good job! Notice how the model correctly predicted the labels of Google and Mountain View but mislabeled Sundar Pichai as an organization. As discussed in the video, the predictions of the model depend strongly on the data it is trained on. It is possible to train spaCy models on your custom data. You will learn to do this in more advanced NLP courses.



### **2.4.2 Identifying people mentioned in a news article**



 In this exercise, you have been given an excerpt from a news article published in
 *TechCrunch*
 . Your task is to write a function
 `find_people`
 that identifies the names of people that have been mentioned in a particular piece of text. You will then use
 `find_people`
 to identify the people of interest in the article.




 The article is available as the string
 `tc`
 and has been printed to the console. The required spacy model has also been already loaded as
 `nlp`
 .





```

def find_persons(text):
  # Create Doc object
  doc = nlp(text)

  # Identify the persons
  persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']

  # Return persons
  return persons

print(find_persons(tc))
# ['Sheryl Sandberg', 'Mark Zuckerberg']

```



 Excellent work! The article was related to Facebook and our function correctly identified both the people mentioned. You can now see how NER could be used in a variety of applications. Publishers may use a technique like this to classify news articles by the people mentioned in them. A question answering system could also use something like this to answer questions such as ‘Who are the people mentioned in this passage?’. With this, we come to an end of this chapter. In the next, we will learn how to conduct vectorization on documents.







# **3. N-Gram models**
---------------------


## **3.1 Building a bag of words model**



![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/1-14.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/2-14.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/3-13.png?w=833)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/4-13.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/5-13.png?w=908)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/6-13.png?w=961)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/7-13.png?w=974)



### **3.1.1 Word vectors with a given vocabulary**



 You have been given a corpus of documents and you have computed the vocabulary of the corpus to be the following:
 ***V***
 :
 *a, an, and, but, can, come, evening, forever, go, i, men, may, on, the, women*




 Which of the following corresponds to the bag of words vector for the document “men may come and men may go but i go on forever”?





```

(0, 0, 1, 1, 0, 1, 0, 1, 2, 1, 2, 2, 1, 0, 0)

```



 Good job! That is, indeed, the correct answer. Each value in the vector corresponds to the frequency of the corresponding word in the vocabulary.



### **3.1.2 BoW model for movie taglines**



 In this exercise, you have been provided with a
 `corpus`
 of more than 7000 movie tag lines. Your job is to generate the bag of words representation
 `bow_matrix`
 for these taglines. For this exercise, we will ignore the text preprocessing step and generate
 `bow_matrix`
 directly.




 We will also investigate the shape of the resultant
 `bow_matrix`
 . The first five taglines in
 `corpus`
 have been printed to the console for you to examine.





```

corpus.shape
(7033,)

```




```python

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(corpus)

# Print the shape of bow_matrix
print(bow_matrix.shape)
# (7033, 6614)

```



 Excellent! You now know how to generate a bag of words representation for a given corpus of documents. Notice that the word vectors created have more than 6600 dimensions. However, most of these dimensions have a value of zero since most words do not occur in a particular tagline.



### **3.1.3 Analyzing dimensionality and preprocessing**



 In this exercise, you have been provided with a
 `lem_corpus`
 which contains the pre-processed versions of the movie taglines from the previous exercise. In other words, the taglines have been lowercased and lemmatized, and stopwords have been removed.




 Your job is to generate the bag of words representation
 `bow_lem_matrix`
 for these lemmatized taglines and compare its shape with that of
 `bow_matrix`
 obtained in the previous exercise. The first five lemmatized taglines in
 `lem_corpus`
 have been printed to the console for you to examine.





```python

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_lem_matrix = vectorizer.fit_transform(lem_corpus)

# Print the shape of bow_lem_matrix
print(bow_lem_matrix.shape)
# (6959, 5223)

```



 Good job! Notice how the number of features have reduced significantly from around 6600 to around 5223 for pre-processed movie taglines. The reduced number of dimensions on account of text preprocessing usually leads to better performance when conducting machine learning and it is a good idea to consider it. However, as mentioned in a previous lesson, the final decision always depends on the nature of the application.



### **3.1.4 Mapping feature indices with feature names**



 In the lesson video, we had seen that
 `CountVectorizer`
 doesn’t necessarily index the vocabulary in alphabetical order. In this exercise, we will learn to map each feature index to its corresponding feature name from the vocabulary.




 We will use the same three sentences on lions from the video. The sentences are available in a list named
 `corpus`
 and has already been printed to the console.





```

['The lion is the king of the jungle', 'Lions have lifespans of a decade', 'The lion is an endangered species']

```




```python

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(corpus)

# Convert bow_matrix into a DataFrame
bow_df = pd.DataFrame(bow_matrix.toarray())

# Map the column names to vocabulary
bow_df.columns = vectorizer.get_feature_names()

# Print bow_df
print(bow_df)

```




```

   an  decade  endangered  have  is  ...  lion  lions  of  species  the
0   0       0           0     0   1  ...     1      0   1        0    3
1   0       1           0     1   0  ...     0      1   1        0    0
2   1       0           1     0   1  ...     1      0   0        1    1

[3 rows x 13 columns]

```



 Great job! Observe that the column names refer to the token whose frequency is being recorded. Therefore, since the first column name is
 `an`
 , the first feature represents the number of times the word ‘an’ occurs in a particular sentence.
 `get_feature_names()`
 essentially gives us a list which represents the mapping of the feature indices to the feature name in the vocabulary.





---


## **3.2 Building a BoW Naive Bayes classifier**



![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/8-13.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/9-12.png?w=773)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/10-12.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/11-12.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/12-12.png?w=708)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/13-11.png?w=989)



### **3.2.1 BoW vectors for movie reviews**



 In this exercise, you have been given two pandas Series,
 `X_train`
 and
 `X_test`
 , which consist of movie reviews. They represent the training and the test review data respectively. Your task is to preprocess the reviews and generate BoW vectors for these two sets using
 `CountVectorizer`
 .




 Once we have generated the BoW vector matrices
 `X_train_bow`
 and
 `X_test_bow`
 , we will be in a very good position to apply a machine learning model to it and conduct sentiment analysis.





```python

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create a CountVectorizer object
vectorizer = CountVectorizer(lowercase=True, stop_words='english')

# Fit and transform X_train
X_train_bow = vectorizer.fit_transform(X_train)

# Transform X_test
X_test_bow = vectorizer.transform(X_test)

# Print shape of X_train_bow and X_test_bow
print(X_train_bow.shape)
print(X_test_bow.shape)

# (750, 8158)
# (250, 8158)

```



 Great job! You now have a good idea of preprocessing text and transforming them into their bag-of-words representation using
 `CountVectorizer`
 . In this exercise, you have set the
 `lowercase`
 argument to
 `True`
 . However, note that this is the default value of
 `lowercase`
 and passing it explicitly is not necessary. Also, note that both
 `X_train_bow`
 and
 `X_test_bow`
 have 8158 features. There were words present in
 `X_test`
 that were not in
 `X_train`
 .
 `CountVectorizer`
 chose to ignore them in order to ensure that the dimensions of both sets remain the same.



### **3.2.2 Predicting the sentiment of a movie review**



 In the previous exercise, you generated the bag-of-words representations for the training and test movie review data. In this exercise, we will use this model to train a Naive Bayes classifier that can detect the sentiment of a movie review and compute its accuracy. Note that since this is a binary classification problem, the model is only capable of classifying a review as either positive (1) or negative (0). It is incapable of detecting neutral reviews.




 In case you don’t recall, the training and test BoW vectors are available as
 `X_train_bow`
 and
 `X_test_bow`
 respectively. The corresponding labels are available as
 `y_train`
 and
 `y_test`
 respectively. Also, for you reference, the original movie review dataset is available as
 `df`
 .





```python

# Create a MultinomialNB object
clf = MultinomialNB()

# Fit the classifier
clf.fit(X_train_bow, y_train)

# Measure the accuracy
accuracy = clf.score(X_test_bow, y_test)
print("The accuracy of the classifier on the test set is %.3f" % accuracy)

# Predict the sentiment of a negative review
review = "The movie was terrible. The music was underwhelming and the acting mediocre."
prediction = clf.predict(vectorizer.transform([review]))[0]
print("The sentiment predicted by the classifier is %i" % (prediction))

```



 Excellent work! You have successfully performed basic sentiment analysis. Note that the accuracy of the classifier is 73.2%. Considering the fact that it was trained on only 750 reviews, this is reasonably good performance. The classifier also correctly predicts the sentiment of a mini negative review which we passed into it.





---


## **3.3 Building n-gram models**



![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/1-15.png?w=820)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/2-15.png?w=993)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/3-14.png?w=462)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/4-14.png?w=513)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/5-14.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/6-14.png?w=478)



### **3.3.1 n-gram models for movie tag lines**



 In this exercise, we have been provided with a
 `corpus`
 of more than 9000 movie tag lines. Our job is to generate n-gram models up to n equal to 1, n equal to 2 and n equal to 3 for this data and discover the number of features for each model.




 We will then compare the number of features generated for each model.





```python

# Generate n-grams upto n=1
vectorizer_ng1 = CountVectorizer(ngram_range=(1,1))
ng1 = vectorizer_ng1.fit_transform(corpus)

# Generate n-grams upto n=2
vectorizer_ng2 = CountVectorizer(ngram_range=(1,2))
ng2 = vectorizer_ng2.fit_transform(corpus)

# Generate n-grams upto n=3
vectorizer_ng3 = CountVectorizer(ngram_range=(1, 3))
ng3 = vectorizer_ng3.fit_transform(corpus)

# Print the number of features for each model
print("ng1, ng2 and ng3 have %i, %i and %i features respectively" % (ng1.shape[1], ng2.shape[1], ng3.shape[1]))
# ng1, ng2 and ng3 have 6614, 37100 and 76881 features respectively

```



 Good job! You now know how to generate n-gram models containing higher order n-grams. Notice that
 `ng2`
 has over 37,000 features whereas
 `ng3`
 has over 76,000 features. This is much greater than the 6,000 dimensions obtained for
 `ng1`
 . As the n-gram range increases, so does the number of features, leading to increased computational costs and a problem known as the curse of dimensionality.



### **3.3.2 Higher order n-grams for sentiment analysis**



 Similar to a previous exercise, we are going to build a classifier that can detect if the review of a particular movie is positive or negative. However, this time, we will use n-grams up to n=2 for the task.




 The n-gram training reviews are available as
 `X_train_ng`
 . The corresponding test reviews are available as
 `X_test_ng`
 . Finally, use
 `y_train`
 and
 `y_test`
 to access the training and test sentiment classes respectively.





```python

# Define an instance of MultinomialNB
clf_ng = MultinomialNB()

# Fit the classifier
clf_ng.fit(X_train_ng, y_train)

# Measure the accuracy
accuracy = clf_ng.score(X_test_ng, y_test)
print("The accuracy of the classifier on the test set is %.3f" % accuracy)

# Predict the sentiment of a negative review
review = "The movie was not good. The plot had several holes and the acting lacked panache."
prediction = clf_ng.predict(ng_vectorizer.transform([review]))[0]
print("The sentiment predicted by the classifier is %i" % (prediction))

```




```

The accuracy of the classifier on the test set is 0.758
The sentiment predicted by the classifier is 0

```



 Excellent job! You’re now adept at performing sentiment analysis using text. Notice how this classifier performs slightly better than the BoW version. Also, it succeeds at correctly identifying the sentiment of the mini-review as negative. In the next chapter, we will learn more complex methods of vectorizing textual data.



### **3.3.3 Comparing performance of n-gram models**



 You now know how to conduct sentiment analysis by converting text into various n-gram representations and feeding them to a classifier. In this exercise, we will conduct sentiment analysis for the same movie reviews from before using two n-gram models: unigrams and n-grams upto n equal to 3.




 We will then compare the performance using three criteria: accuracy of the model on the test set, time taken to execute the program and the number of features created when generating the n-gram representation.





```

start_time = time.time()
# Splitting the data into training and test sets
train_X, test_X, train_y, test_y = train_test_split(df['review'], df['sentiment'], test_size=0.5, random_state=42, stratify=df['sentiment'])

# Generating ngrams
vectorizer = CountVectorizer(ngram_range=(1,1))
train_X = vectorizer.fit_transform(train_X)
test_X = vectorizer.transform(test_X)

# Fit classifier
clf = MultinomialNB()
clf.fit(train_X, train_y)

# Print accuracy, time and number of dimensions
print("The program took %.3f seconds to complete. The accuracy on the test set is %.2f. The ngram representation had %i features." % (time.time() - start_time, clf.score(test_X, test_y), train_X.shape[1]))

# The program took 0.196 seconds to complete. The accuracy on the test set is 0.75. The ngram representation had 12347 features.

```




```

vectorizer = CountVectorizer(ngram_range=(1,3))
# The program took 2.933 seconds to complete. The accuracy on the test set is 0.77. The ngram representation had 178240 features.

```



 Amazing work! The program took around 0.2 seconds in the case of the unigram model and more than 10 times longer for the higher order n-gram model. The unigram model had over 12,000 features whereas the n-gram model for upto n=3 had over 178,000! Despite taking higher computation time and generating more features, the classifier only performs marginally better in the latter case, producing an accuracy of 77% in comparison to the 75% for the unigram model.







# **4. TF-IDF and similarity scores**
------------------------------------


## **4.1 Building tf-idf document vectors**



![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/7-14.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/8-14.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/9-13.png?w=878)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/10-13.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/11-13.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/12-13.png?w=928)



### **4.1.1 tf-idf weight of commonly occurring words**



 The word
 `bottle`
 occurs 5 times in a particular document
 `D`
 and also occurs in every document of the corpus. What is the tf-idf weight of
 `bottle`
 in
 `D`
 ?




**0**




 Correct! In fact, the tf-idf weight for
 `bottle`
 in every document will be 0. This is because the inverse document frequency is constant across documents in a corpus and since
 `bottle`
 occurs in every document, its value is log(1), which is 0.



### **4.1.2 tf-idf vectors for TED talks**



 In this exercise, you have been given a corpus
 `ted`
 which contains the transcripts of 500 TED Talks. Your task is to generate the tf-idf vectors for these talks.




 In a later lesson, we will use these vectors to generate recommendations of similar talks based on the transcript.





```python

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Generate matrix of word vectors
tfidf_matrix = vectorizer.fit_transform(ted)

# Print the shape of tfidf_matrix
print(tfidf_matrix.shape)
# (500, 29158)

```



 Good job! You now know how to generate tf-idf vectors for a given corpus of text. You can use these vectors to perform predictive modeling just like we did with
 `CountVectorizer`
 . In the next few lessons, we will see another extremely useful application of the vectorized form of documents: generating recommendations.





---


## **4.2 Cosine similarity**



![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/13-12.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/14-11.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/15-11.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/16-10.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/17-7.png?w=978)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/18-5.png?w=979)



### **4.2.1 Computing dot product**



 In this exercise, we will learn to compute the dot product between two vectors, A = (1, 3) and B = (-2, 2), using the
 `numpy`
 library. More specifically, we will use the
 `np.dot()`
 function to compute the dot product of two numpy arrays.





```python

# Initialize numpy vectors
A = np.array([1,3])
B = np.array([-2,2])

# Compute dot product
dot_prod = np.dot(A, B)

# Print dot product
print(dot_prod)
# 4

```



 Good job! The dot product of the two vectors is 1 * -2 + 3 * 2 = 4, which is indeed the output produced. We will not be using
 `np.dot()`
 too much in this course but it can prove to be a helpful function while computing dot products between two standalone vectors.



### **4.2.2 Cosine similarity matrix of a corpus**



 In this exercise, you have been given a
 `corpus`
 , which is a list containing five sentences. The
 `corpus`
 is printed in the console. You have to compute the cosine similarity matrix which contains the pairwise cosine similarity score for every pair of sentences (vectorized using tf-idf).




 Remember, the value corresponding to the ith row and jth column of a similarity matrix denotes the similarity score for the ith and jth vector.





```

corpus
['The sun is the largest celestial body in the solar system',
 'The solar system consists of the sun and eight revolving planets',
 'Ra was the Egyptian Sun God',
 'The Pyramids were the pinnacle of Egyptian architecture',
 'The quick brown fox jumps over the lazy dog']

```




```python

# Initialize an instance of tf-idf Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Generate the tf-idf vectors for the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Compute and print the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix,tfidf_matrix)
print(cosine_sim)

```




```

[[1.         0.36413198 0.18314713 0.18435251 0.16336438]
 [0.36413198 1.         0.15054075 0.21704584 0.11203887]
 [0.18314713 0.15054075 1.         0.21318602 0.07763512]
 [0.18435251 0.21704584 0.21318602 1.         0.12960089]
 [0.16336438 0.11203887 0.07763512 0.12960089 1.        ]]

```



 Great work! As you will see in a subsequent lesson, computing the cosine similarity matrix lies at the heart of many practical systems such as recommenders. From our similarity matrix, we see that the first and the second sentence are the most similar. Also the fifth sentence has, on average, the lowest pairwise cosine scores. This is intuitive as it contains entities that are not present in the other sentences.





---


## **4.3 Building a plot line based recommender**



![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/19-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/20-4.png?w=613)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/21-3.png?w=524)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/22-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/23-3.png?w=824)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/24-3.png?w=1020)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/25-2.png?w=904)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/26-2.png?w=1022)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/27-2.png?w=1024)



### **4.3.1 Comparing linear_kernel and cosine_similarity**



 In this exercise, you have been given
 `tfidf_matrix`
 which contains the tf-idf vectors of a thousand documents. Your task is to generate the cosine similarity matrix for these vectors first using
 `cosine_similarity`
 and then, using
 `linear_kernel`
 .




 We will then compare the computation times for both functions.





```python

# Record start time
start = time.time()

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Print cosine similarity matrix
print(cosine_sim)

# Print time taken
print("Time taken: %s seconds" %(time.time() - start))

```




```

[[1.         0.         0.         ... 0.         0.         0.        ]
 [0.         1.         0.         ... 0.         0.         0.        ]
 [0.         0.         1.         ... 0.         0.01418221 0.        ]
 ...
 [0.         0.         0.         ... 1.         0.01589009 0.        ]
 [0.         0.         0.01418221 ... 0.01589009 1.         0.        ]
 [0.         0.         0.         ... 0.         0.         1.        ]]
Time taken: 0.33341264724731445 seconds

```




```python

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

```



 Good job! Notice how both
 `linear_kernel`
 and
 `cosine_similarity`
 produced the same result. However,
 `linear_kernel`
 took a smaller amount of time to execute. When you’re working with a very large amount of data and your vectors are in the tf-idf representation, it is good practice to default to
 `linear_kernel`
 to improve performance. (NOTE: In case, you see
 `linear_kernel`
 taking more time, it’s because the dataset we’re dealing with is extremely small and Python’s
 `time`
 module is incapable of capture such minute time differences accurately)



### **4.3.2 Plot recommendation engine**



 In this exercise, we will build a recommendation engine that suggests movies based on similarity of plot lines. You have been given a
 `get_recommendations()`
 function that takes in the title of a movie, a similarity matrix and an
 `indices`
 series as its arguments and outputs a list of most similar movies.
 `indices`
 has already been provided to you.




 You have also been given a
 `movie_plots`
 Series that contains the plot lines of several movies. Your task is to generate a cosine similarity matrix for the tf-idf vectors of these plots.




 Consequently, we will check the potency of our engine by generating recommendations for one of my favorite movies, The Dark Knight Rises.





```python

# Initialize the TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(movie_plots)

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Generate recommendations
print(get_recommendations('The Dark Knight Rises', cosine_sim, indices))

```




```

1                              Batman Forever
2                                      Batman
3                              Batman Returns
8                  Batman: Under the Red Hood
9                            Batman: Year One
10    Batman: The Dark Knight Returns, Part 1
11    Batman: The Dark Knight Returns, Part 2
5                Batman: Mask of the Phantasm
7                               Batman Begins
4                              Batman & Robin
Name: title, dtype: object

```



 Congratulations! You’ve just built your very first recommendation system. Notice how the recommender correctly identifies
 `'The Dark Knight Rises'`
 as a Batman movie and recommends other Batman movies as a result. This sytem is, of course, very primitive and there are a host of ways in which it could be improved. One method would be to look at the cast, crew and genre in addition to the plot to generate recommendations. We will not be covering this in this course but you have all the tools necessary to accomplish this. Do give it a try!



### **4.3.3 The recommender function**



 In this exercise, we will build a recommender function
 `get_recommendations()`
 , as discussed in the lesson and the previous exercise. As we know, it takes in a title, a cosine similarity matrix, and a movie title and index mapping as arguments and outputs a list of 10 titles most similar to the original title (excluding the title itself).




 You have been given a dataset
 `metadata`
 that consists of the movie titles and overviews. The head of this dataset has been printed to console.





```

               title                                            tagline
938  Cinema Paradiso  A celebration of youth, friendship, and the ev...
630         Spy Hard  All the action. All the women. Half the intell...
682        Stonewall                    The fight for the right to love
514           Killer                    You only hurt the one you love.
365    Jason's Lyric                                   Love is courage.

```




```python

# Generate mapping between titles and index
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

def get_recommendations(title, cosine_sim, indices):
    # Get index of movie that matches title
    idx = indices[title]
    # Sort the movies based on the similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]

```



 Good job! With this recommender function in our toolkit, we are now in a very good place to build the rest of the components of our recommendation engine.



### **4.3.4 TED talk recommender**



 In this exercise, we will build a recommendation system that suggests TED Talks based on their transcripts. You have been given a
 `get_recommendations()`
 function that takes in the title of a talk, a similarity matrix and an
 `indices`
 series as its arguments, and outputs a list of most similar talks.
 `indices`
 has already been provided to you.




 You have also been given a
 `transcripts`
 series that contains the transcripts of around 500 TED talks. Your task is to generate a cosine similarity matrix for the tf-idf vectors of the talk transcripts.




 Consequently, we will generate recommendations for a talk titled ‘5 ways to kill your dreams’ by Brazilian entrepreneur Bel Pesce.





```

transcripts
0      I've noticed something interesting about socie...
1      Hetain Patel: (In Chinese)Yuyu Rau: Hi, I'm He...
2      (Music)Sophie Hawley-Weld: OK, you don't have ...

```




```python

# Initialize the TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(transcripts)

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix,tfidf_matrix)

# Generate recommendations
print(get_recommendations('5 ways to kill your dreams', cosine_sim, indices))

```




```

453             Success is a continuous journey
157                        Why we do what we do
494                   How to find work you love
149          My journey into movies that matter
447                        One Laptop per Child
230             How to get your ideas to spread
497         Plug into your hard-wired happiness
495    Why you will fail to have a great career
179             Be suspicious of simple stories
53                          To upgrade is human
Name: title, dtype: object

```



 Excellent work! You have successfully built a TED talk recommender. This recommender works surprisingly well despite being trained only on a small subset of TED talks. In fact, three of the talks recommended by our system is also recommended by the official TED website as talks to watch next after
 `'5 ways to kill your dreams'`
 !





---


## **4.4 Beyond n-grams: word embeddings**



![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/28-1.png?w=976)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/29.png?w=910)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/30.png?w=907)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/31.png?w=883)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/32.png?w=666)



### **4.4.1 Generating word vectors**



 In this exercise, we will generate the pairwise similarity scores of all the words in a sentence. The sentence is available as
 `sent`
 and has been printed to the console for your convenience.





```

sent
'I like apples and oranges'

```




```python

# Create the doc object
doc = nlp(sent)

# Compute pairwise similarity scores
for token1 in doc:
  for token2 in doc:
    print(token1.text, token2.text, token1.similarity(token2))

```




```

I I 1.0
I like 0.023032807
I apples 0.10175116
I and 0.047492094
I oranges 0.10894456
like I 0.023032807
like like 1.0
like apples 0.015370452
like and 0.189293
like oranges 0.021943133
apples I 0.10175116
apples like 0.015370452
apples apples 1.0
apples and -0.17736834
apples oranges 0.6315578
and I 0.047492094
and like 0.189293
and apples -0.17736834
and and 1.0
and oranges 0.018627528
oranges I 0.10894456
oranges like 0.021943133
oranges apples 0.6315578
oranges and 0.018627528
oranges oranges 1.0

```



 Good job! Notice how the words
 `'apples'`
 and
 `'oranges'`
 have the highest pairwaise similarity score. This is expected as they are both fruits and are more related to each other than any other pair of words.



### **4.4.2 Computing similarity of Pink Floyd songs**



 In this final exercise, you have been given lyrics of three songs by the British band Pink Floyd, namely ‘High Hopes’, ‘Hey You’ and ‘Mother’. The lyrics to these songs are available as
 `hopes`
 ,
 `hey`
 and
 `mother`
 respectively.




 Your task is to compute the pairwise similarity between
 `mother`
 and
 `hopes`
 , and
 `mother`
 and
 `hey`
 .





```

mother
 "\nMother do you think they'll drop the bomb?\nMother do you think they'll like this song?\nMother do you think they'll try to ...

```




```python

# Create Doc objects
mother_doc = nlp(mother)
hopes_doc = nlp(hopes)
hey_doc = nlp(hey)

# Print similarity between mother and hopes
print(mother_doc.similarity(hopes_doc))
# 0.6006234924640204

# Print similarity between mother and hey
print(mother_doc.similarity(hey_doc))
# 0.9135920924498578

```



 Excellent work! Notice that ‘Mother’ and ‘Hey You’ have a similarity score of 0.9 whereas ‘Mother’ and ‘High Hopes’ has a score of only 0.6. This is probably because ‘Mother’ and ‘Hey You’ were both songs from the same album ‘The Wall’ and were penned by Roger Waters. On the other hand, ‘High Hopes’ was a part of the album ‘Division Bell’ with lyrics by David Gilmour and his wife, Penny Samson. Treat yourself by listening to these songs. They’re some of the best!





---


## **4.5 Final thoughts**



![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/33.png?w=784)
![Desktop View]({{ site.baseurl }}/assets/datacamp/feature-engineering-for-nlp-in-python/34.png?w=547)





---



 Thank you for reading and hope you’ve learned a lot.



