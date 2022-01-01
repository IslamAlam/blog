---
title: Introduction to Natural Language Processing in Python
date: 2021-12-07 11:22:10 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Introduction to Natural Language Processing in Python
==========================================================







 This is the memo of the 12th course (23 courses in all) of ‘Machine Learning Scientist with Python’ skill track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/natural-language-processing-fundamentals-in-python)**
 .




 reference:
 **[Natural Language Toolkit](https://www.nltk.org/)**



###
**Course Description**



 In this course, you’ll learn natural language processing (NLP) basics, such as how to identify and separate words, how to extract topics in a text, and how to build your own fake news classifier. You’ll also learn how to use basic libraries such as NLTK, alongside libraries which utilize deep learning to solve common NLP problems. This course will give you the foundation to process and parse text as you move forward in your Python learning.



###
**Table of contents**


* [Regular expressions & word tokenization](https://datascience103579984.wordpress.com/2020/01/14/introduction-to-natural-language-processing-in-python-from-datacamp/)
* [Simple topic identification](https://datascience103579984.wordpress.com/2020/01/14/introduction-to-natural-language-processing-in-python-from-datacamp/2/)
* [Named-entity recognition](https://datascience103579984.wordpress.com/2020/01/14/introduction-to-natural-language-processing-in-python-from-datacamp/3/)
* [Building a “fake news” classifier](https://datascience103579984.wordpress.com/2020/01/14/introduction-to-natural-language-processing-in-python-from-datacamp/4/)





# **1. Regular expressions & word tokenization**
-----------------------------------------------


## **1.1 Introduction to regular expressions**



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/1-6.png?w=954)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/2-6.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/3-6.png?w=780)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/4-6.png?w=1024)



### **1.1.1 Which pattern?**



 Which of the following Regex patterns results in the following text?





```

>>> my_string = "Let's write RegEx!"
>>> re.findall(PATTERN, my_string)
['Let', 's', 'write', 'RegEx']


```



 In the IPython Shell, try replacing
 `PATTERN`
 with one of the below options and observe the resulting output. The
 `re`
 module has been pre-imported for you and
 `my_string`
 is available in your namespace.





```

PATTERN = r"\w+"

In [2]: re.findall(PATTERN, my_string)
Out[2]: ['Let', 's', 'write', 'RegEx']

```


### **1.1.2 Practicing regular expressions: re.split() and re.findall()**



 Now you’ll get a chance to write some regular expressions to match digits, strings and non-alphanumeric characters. Take a look at
 `my_string`
 first by printing it in the IPython Shell, to determine how you might best match the different steps.




 Note: It’s important to prefix your regex patterns with
 `r`
 to ensure that your patterns are interpreted in the way you want them to. Else, you may encounter problems to do with escape sequences in strings. For example,
 `"\n"`
 in Python is used to indicate a new line, but if you use the
 `r`
 prefix, it will be interpreted as the raw string
 `"\n"`
 – that is, the character
 `"\"`
 followed by the character
 `"n"`
 – and not as a new line.




 The regular expression module
 `re`
 has already been imported for you.




*Remember from the video that the syntax for the regex library is to always to pass the
 **pattern first**
 , and then the
 **string second**
 .*





```

my_string
"Let's write RegEx!  Won't that be fun?  I sure think so.  Can you find 4 sentences?  Or perhaps, all 19 words?"

```




```python

# Write a pattern to match sentence endings: sentence_endings
sentence_endings = r"[.?!]"

# Split my_string on sentence endings and print the result
print(re.split(sentence_endings, my_string))
# ["Let's write RegEx", "  Won't that be fun", '  I sure think so', '  Can you find 4 sentences', '  Or perhaps, all 19 words', '']


# Find all capitalized words in my_string and print the result
capitalized_words = r"[A-Z]\w+"
print(re.findall(capitalized_words, my_string))
# ['Let', 'RegEx', 'Won', 'Can', 'Or']


# Split my_string on spaces and print the result
spaces = r"\s+"
print(re.split(spaces, my_string))
# ["Let's", 'write', 'RegEx!', "Won't", 'that', 'be', 'fun?', 'I', 'sure', 'think', 'so.', 'Can', 'you', 'find', '4', 'sentences?', 'Or', 'perhaps,', 'all', '19', 'words?']


# Find all digits in my_string and print the result
digits = r"\d+"
print(re.findall(digits, my_string))
# ['4', '19']

```




---


## **1.2 Introduction to tokenization**



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/5-6.png?w=985)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/6-6.png?w=593)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/7-6.png?w=755)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/8-6.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/9-6.png?w=790)



### **1.2.1 Word tokenization with NLTK**



 Here, you’ll be using the first scene of Monty Python’s Holy Grail, which has been pre-loaded as
 `scene_one`
 . Feel free to check it out in the IPython Shell!




 Your job in this exercise is to utilize
 `word_tokenize`
 and
 `sent_tokenize`
 from
 `nltk.tokenize`
 to tokenize both words and sentences from Python strings – in this case, the first scene of Monty Python’s Holy Grail.





```

scene_one
"SCENE 1: [wind] [clop clop clop] \nKING ARTHUR: Whoa there!  [clop clop clop] \nSOLDIER #1: Halt!  Who goes there?\nARTHUR: It is I, Arthur, son of
...
creeper!\nSOLDIER #1: What, held under the dorsal guiding feathers?\nSOLDIER #2: Well, why not?\n"

```




```python

# Import necessary modules
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# Split scene_one into sentences: sentences
sentences = sent_tokenize(scene_one)

# Use word_tokenize to tokenize the fourth sentence: tokenized_sent
tokenized_sent = word_tokenize(sentences[3])

# Make a set of unique tokens in the entire scene: unique_tokens
unique_tokens = set(word_tokenize(scene_one))

# Print the unique tokens result
print(unique_tokens)

```




```

{'ridden', 'bird', 'King', 'winter', 'right', 'under', 'needs', 'them', 'with', 'use', 'Mercea', 'simple', 'No', '!', 'Ridden', 'Pendragon',
...
'minute', 'Whoa', '...', "'m", '[', '#', 'will', "'ve", 'an', 'In', 'interested', 'England', "'re"}

```



 Excellent! Tokenization is fundamental to NLP, and you’ll end up using it a lot in text mining and information retrieval projects.



### **1.2.2 More regex with re.search()**



 In this exercise, you’ll utilize
 `re.search()`
 and
 `re.match()`
 to find specific tokens. Both
 `search`
 and
 `match`
 expect regex patterns, similar to those you defined in an earlier exercise. You’ll apply these regex library methods to the same Monty Python text from the
 `nltk`
 corpora.




 You have both
 `scene_one`
 and
 `sentences`
 available from the last exercise; now you can use them with
 `re.search()`
 and
 `re.match()`
 to extract and match more text.





```python

# Search for the first occurrence of "coconuts" in scene_one: match
match = re.search("coconuts", scene_one)

# Print the start and end indexes of match
print(match.start(), match.end())
# 580 588


# Write a regular expression to search for anything in square brackets: pattern1
pattern1 = r"\[.*]"

# Use re.search to find the first text in square brackets
print(re.search(pattern1, scene_one))
# <_sre.SRE_Match object; span=(9, 32), match='[wind] [clop clop clop]'>


# Find the script notation at the beginning of the fourth sentence and print it
pattern2 = r"[\w\s]+:"
print(re.match(pattern2, sentences[3]))
# <_sre.SRE_Match object; span=(0, 7), match='ARTHUR:'>

```



 Fantastic work! Now that you’re familiar with the basics of tokenization and regular expressions, it’s time to learn about more advanced tokenization.





---


## **1.3 Advanced tokenization with NLTK and regex**



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/1-7.png?w=814)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/2-7.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/3-7.png?w=857)



### **1.3.1 Choosing a tokenizer**



 Given the following string, which of the below patterns is the best tokenizer? If possible, you want to retain sentence punctuation as separate tokens, but have
 `'#1'`
 remain a single token.





```

my_string = "SOLDIER #1: Found them? In Mercea? The coconut's tropical!"

```



 The string is available in your workspace as
 `my_string`
 , and the patterns have been pre-loaded as
 `pattern1`
 ,
 `pattern2`
 ,
 `pattern3`
 , and
 `pattern4`
 , respectively.




 Additionally,
 `regexp_tokenize`
 has been imported from
 `nltk.tokenize`
 . You can use
 `regexp_tokenize(string, pattern)`
 with
 `my_string`
 and one of the patterns as arguments to experiment for yourself and see which is the best tokenizer.





```

my_string
# "SOLDIER #1: Found them? In Mercea? The coconut's tropical!"

pattern2
# '(\\w+|#\\d|\\?|!)'

```




```

regexp_tokenize(my_string, pattern2)
['SOLDIER',
 '#1',
 'Found',
 'them',
 '?',
 'In',
 'Mercea',
 '?',
 'The',
 'coconut',
 's',
 'tropical',
 '!']

```


### **1.3.2 Regex with NLTK tokenization**



 Twitter is a frequently used source for NLP text and tasks. In this exercise, you’ll build a more complex tokenizer for tweets with hashtags and mentions using
 `nltk`
 and regex. The
 `nltk.tokenize.TweetTokenizer`
 class gives you some extra methods and attributes for parsing tweets.




 Here, you’re given some example tweets to parse using both
 `TweetTokenizer`
 and
 `regexp_tokenize`
 from the
 `nltk.tokenize`
 module. These example tweets have been pre-loaded into the variable
 `tweets`
 . Feel free to explore it in the IPython Shell!




*Unlike the syntax for the regex library, with
 `nltk_tokenize()`
 you pass the pattern as the
 **second**
 argument.*





```

tweets[0]
'This is the best #nlp exercise ive found online! #python'

```




```python

# Import the necessary modules
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import TweetTokenizer

# Define a regex pattern to find hashtags: pattern1
pattern1 = r"#\w+"
# Use the pattern on the first tweet in the tweets list
hashtags = regexp_tokenize(tweets[0], pattern1)
print(hashtags)
['#nlp', '#python']

```




```

tweets[-1]
'Thanks @datacamp 🙂 #nlp #python'

```




```python

# Import the necessary modules
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import TweetTokenizer
# Write a pattern that matches both mentions (@) and hashtags
pattern2 = r"([@#]\w+)"
# Use the pattern on the last tweet in the tweets list
mentions_hashtags = regexp_tokenize(tweets[-1], pattern2)
print(mentions_hashtags)
# ['@datacamp', '#nlp', '#python']

```




```

tweets
['This is the best #nlp exercise ive found online! #python',
 '#NLP is super fun! ❤ #learning',
 'Thanks @datacamp 🙂 #nlp #python']

```




```python

# Import the necessary modules
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import TweetTokenizer
# Use the TweetTokenizer to tokenize all tweets into one list
tknzr = TweetTokenizer()
all_tokens = [tknzr.tokenize(t) for t in tweets]
print(all_tokens)

```




```

[['This', 'is', 'the', 'best', '#nlp', 'exercise', 'ive', 'found', 'online', '!', '#python'], ['#NLP', 'is', 'super', 'fun', '!', '<3', '#learning'], ['Thanks', '@datacamp', ':)', '#nlp', '#python']]

```


### **1.3.3 Non-ascii tokenization**



 In this exercise, you’ll practice advanced tokenization by tokenizing some non-ascii based text. You’ll be using German with emoji!




 Here, you have access to a string called
 `german_text`
 , which has been printed for you in the Shell. Notice the emoji and the German characters!




 The following modules have been pre-imported from
 `nltk.tokenize`
 :
 `regexp_tokenize`
 and
 `word_tokenize`
 .




 Unicode ranges for emoji are:




`('\U0001F300'-'\U0001F5FF')`
 ,
 `('\U0001F600-\U0001F64F')`
 ,
 `('\U0001F680-\U0001F6FF')`
 , and
 `('\u2600'-\u26FF-\u2700-\u27BF')`
 .





```

german_text
'Wann gehen wir Pizza essen? 🍕 Und fährst du mit Über? 🚕'

```




```python

# Tokenize and print all words in german_text
all_words = word_tokenize(german_text)
print(all_words)
['Wann', 'gehen', 'wir', 'Pizza', 'essen', '?', '🍕', 'Und', 'fährst', 'du', 'mit', 'Über', '?', '🚕']

# Tokenize and print only capital words
capital_words = r"[A-ZÜ]\w+"
print(regexp_tokenize(german_text, capital_words))
['Wann', 'Pizza', 'Und', 'Über']


# Tokenize and print only emoji
emoji = "['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"
print(regexp_tokenize(german_text, emoji))
['🍕', '🚕']


```


## **1.4 Charting word length with NLTK**



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/4-7.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/5-7.png?w=896)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/6-7.png?w=539)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/7-7.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/8-7.png?w=861)




 Charting practice
===================



 Try using your new skills to find and chart the number of words per line in the script using
 `matplotlib`
 . The Holy Grail script is loaded for you, and you need to use regex to find the words per line.




 Using list comprehensions here will speed up your computations. For example:
 `my_lines = [tokenize(l) for l in lines]`
 will call a function
 `tokenize`
 on each line in the list
 `lines`
 . The new transformed list will be saved in the
 `my_lines`
 variable.




 You have access to the entire script in the variable
 `holy_grail`
 . Go for it!





```

holy_grail
"SCENE 1: [wind] [clop clop clop] \nKING ARTHUR: Whoa there!  [clop clop clop] \nSOLDIER #1: Halt!  Who goes there?\nARTHUR: It is I, Arthur,
...
 along.\nINSPECTOR: Everything? [squeak] \nOFFICER #1: All right, sonny.  That's enough.  Just pack that in. [crash] \nCAMERAMAN: Christ!\n"

```




```python

# Split the script into lines: lines
lines = holy_grail.split('\n')

# Replace all script lines for speaker
pattern = "[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?:"
lines = [re.sub(pattern, '', l) for l in lines]

# Tokenize each line: tokenized_lines
tokenized_lines = [regexp_tokenize(s, '\w+') for s in lines]

# Make a frequency list of lengths: line_num_words
line_num_words = [len(t_line) for t_line in tokenized_lines]

# Plot a histogram of the line lengths
plt.hist(line_num_words)

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/9-7.png?w=1024)



# **2. Simple topic identification**
-----------------------------------


## **2.1 Word counts with bag-of-words**



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/10-6.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/11-6.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/12-6.png?w=802)



### **2.1.1 Bag-of-words picker**



 It’s time for a quick check on your understanding of bag-of-words. Which of the below options, with basic
 `nltk`
 tokenization, map the bag-of-words for the following text?




 “The cat is in the box. The cat box.”




**(‘The’, 2), (‘box’, 2), (‘.’, 2), (‘cat’, 2), (‘is’, 1), (‘in’, 1), (‘the’, 1)**



### **2.1.2 Building a Counter with bag-of-words**



 In this exercise, you’ll build your first (in this course) bag-of-words counter using a Wikipedia article, which has been pre-loaded as
 `article`
 . Try doing the bag-of-words without looking at the full article text, and guessing what the topic is! If you’d like to peek at the title at the end, we’ve included it as
 `article_title`
 . Note that this article text has had very little preprocessing from the raw Wikipedia database entry.




`word_tokenize`
 has been imported for you.





```python

# Import Counter
from collections import Counter

# Tokenize the article: tokens
tokens = word_tokenize(article)

# Convert the tokens into lowercase: lower_tokens
lower_tokens = [t.lower() for t in tokens]

# Create a Counter with the lowercase tokens: bow_simple
bow_simple = Counter(lower_tokens)

# Print the 10 most common tokens
print(bow_simple.most_common(10))
# [(',', 151), ('the', 150), ('.', 89), ('of', 81), ("''", 68), ('to', 63), ('a', 60), ('in', 44), ('and', 41), ('debugging', 40)]

```




---


## **2.2 Simple text preprocessing**



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/13-6.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/14-6.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/15-6.png?w=789)



### **2.2.1 Text preprocessing steps**



 Which of the following are useful text preprocessing steps?




**Lemmatization, lowercasing, removing unwanted tokens.**



### **2.2.2 Text preprocessing practice**



 Now, it’s your turn to apply the techniques you’ve learned to help clean up text for better NLP results. You’ll need to remove stop words and non-alphabetic characters, lemmatize, and perform a new bag-of-words on your cleaned text.




 You start with the same tokens you created in the last exercise:
 `lower_tokens`
 . You also have the
 `Counter`
 class imported.





```python

# Import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer

# Retain alphabetic words: alpha_only
alpha_only = [t for t in lower_tokens if t.isalpha()]

# Remove all stop words: no_stops
no_stops = [t for t in alpha_only if t not in english_stops]

# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Lemmatize all tokens into a new list: lemmatized
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

# Create the bag-of-words: bow
bow = Counter(lemmatized)

# Print the 10 most common tokens
print(bow.most_common(10))

```




```

[('debugging', 40), ('system', 25), ('software', 16), ('bug', 16), ('problem', 15), ('tool', 15), ('computer', 14), ('process', 13), ('term', 13), ('used', 12)]

```




---


## **2.3 Introduction to gensim**



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/1-8.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/2-8.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/3-8.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/4-8.png?w=898)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/5-8.png?w=1024)



### **2.3.1 What are word vectors?**



 What are word vectors and how do they help with NLP?




**Word vectors are multi-dimensional mathematical representations of words created using deep learning methods. They give us insight into relationships between words in a corpus.**



### **2.3.2 Creating and querying a corpus with gensim**



 It’s time to apply the methods you learned in the previous video to create your first
 `gensim`
 dictionary and corpus!




 You’ll use these data structures to investigate word trends and potential interesting topics in your document set. To get started, we have imported a few additional messy articles from Wikipedia, which were preprocessed by lowercasing all words, tokenizing them, and removing stop words and punctuation. These were then stored in a list of document tokens called
 `articles`
 . You’ll need to do some light preprocessing and then generate the
 `gensim`
 dictionary and corpus.





```python

# Import Dictionary
from gensim.corpora.dictionary import Dictionary

# Create a Dictionary from the articles: dictionary
dictionary = Dictionary(articles)

# Select the id for "computer": computer_id
computer_id = dictionary.token2id.get("computer")

# Use computer_id with the dictionary to print the word
print(dictionary.get(computer_id))
# computer


# Create a MmCorpus: corpus
corpus = [dictionary.doc2bow(article) for article in articles]

# Print the first 10 word ids with their frequency counts from the fifth document
print(corpus[4][:10])
# [(0, 88), (23, 11), (24, 2), (39, 1), (41, 2), (55, 22), (56, 1), (57, 1), (58, 1), (59, 3)]

```


### **2.3.3 Gensim bag-of-words**



 Now, you’ll use your new
 `gensim`
 corpus and dictionary to see the most common terms per document and across all documents. You can use your dictionary to look up the terms. Take a guess at what the topics are and feel free to explore more documents in the IPython Shell!




 You have access to the
 `dictionary`
 and
 `corpus`
 objects you created in the previous exercise, as well as the Python
 `defaultdict`
 and
 `itertools`
 to help with the creation of intermediate data structures for analysis.



* `defaultdict`
 allows us to initialize a dictionary that will assign a default value to non-existent keys. By supplying the argument
 `int`
 , we are able to ensure that any non-existent keys are automatically assigned a default value of
 `0`
 . This makes it ideal for storing the counts of words in this exercise.
* `itertools.chain.from_iterable()`
 allows us to iterate through a set of sequences as if they were one continuous sequence. Using this function, we can easily iterate through our
 `corpus`
 object (which is a list of lists).



 The fifth document from
 `corpus`
 is stored in the variable
 `doc`
 , which has been sorted in descending order.





```python

# Save the fifth document: doc
doc = corpus[4]

# Sort the doc for frequency: bow_doc
bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)

# Print the top 5 words of the document alongside the count
for word_id, word_count in bow_doc[:5]:
    print(dictionary.get(word_id), word_count)

# Create the defaultdict: total_word_count
total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count

```




```

engineering 91
'' 88
reverse 71
software 51
cite 26

total_word_count
defaultdict(int,
            {0: 1042,
             1: 1,
             2: 1,
...
             997: 1,
             998: 1,
             999: 22,
             ...})

```




```python

# Create a sorted list from the defaultdict: sorted_word_count
sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True)

# Print the top 5 words across all documents alongside the count
for word_id, word_count in sorted_word_count[:5]:
    print(dictionary.get(word_id), word_count)

```




```

'' 1042
computer 594
software 450
`` 345
cite 322

```




---


## **2.4 Tf-idf with gensim**



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/6-8.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/7-8.png?w=971)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/8-8.png?w=705)



### **2.4.1 What is tf-idf?**



 You want to calculate the tf-idf weight for the word
 `"computer"`
 , which appears five times in a document containing 100 words. Given a corpus containing 200 documents, with 20 documents mentioning the word
 `"computer"`
 , tf-idf can be calculated by multiplying term frequency with inverse document frequency.




 Term frequency = percentage share of the word compared to all tokens in the document Inverse document frequency = logarithm of the total number of documents in a corpora divided by the number of documents containing the term




 Which of the below options is correct?




**(5 / 100) * log(200 / 20)**



### **2.4.2 Tf-idf with Wikipedia**



 Now it’s your turn to determine new significant terms for your corpus by applying
 `gensim`
 ‘s tf-idf. You will again have access to the same corpus and dictionary objects you created in the previous exercises –
 `dictionary`
 ,
 `corpus`
 , and
 `doc`
 . Will tf-idf make for more interesting results on the document level?




`TfidfModel`
 has been imported for you from
 `gensim.models.tfidfmodel`
 .





```python

# Create a new TfidfModel using the corpus: tfidf
tfidf = TfidfModel(corpus)

# Calculate the tfidf weights of doc: tfidf_weights
tfidf_weights = tfidf[doc]

# Print the first five weights
print(tfidf_weights[:5])
# [(24, 0.0022836332291091273), (39, 0.0043409401554717324), (41, 0.008681880310943465), (55, 0.011988285029371418), (56, 0.005482756770026296)]

```




```

corpus
[[(0, 50),
  (1, 1),
  (2, 1),
...
  (10325, 1),
  (10326, 1),
  (10327, 1)]]

tfidf
<gensim.models.tfidfmodel.TfidfModel at 0x7f10f5755978>

doc
[(0, 88),
 (23, 11),
 (24, 2),
...
 (3627, 1),
 (3628, 2),
 ...]


tfidf_weights
[(24, 0.0022836332291091273),
 (39, 0.0043409401554717324),
 (41, 0.008681880310943465),
 (55, 0.011988285029371418),
...

```




```python

# Sort the weights from highest to lowest: sorted_tfidf_weights
sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)

# Print the top 5 weighted words
for term_id, weight in sorted_tfidf_weights[:5]:
    print(dictionary.get(term_id), weight)

reverse 0.4884961428651127
infringement 0.18674529210288995
engineering 0.16395041814479536
interoperability 0.12449686140192663
reverse-engineered 0.12449686140192663

```




```

sorted_tfidf_weights
[(1535, 0.4884961428651127),
 (3796, 0.18674529210288995),
 (350, 0.16395041814479536),
 (3804, 0.12449686140192663),
...

```




# **3. Named-entity recognition**
--------------------------------


## **3.1 Named Entity Recognition**



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/9-8.png?w=949)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/10-8.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/11-8.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/12-8.png?w=995)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/13-7.png?w=532)



### **3.1.1 NER with NLTK**



 You’re now going to have some fun with named-entity recognition! A scraped news article has been pre-loaded into your workspace. Your task is to use
 `nltk`
 to find the named entities in this article.




 What might the article be about, given the names you found?




 Along with
 `nltk`
 ,
 `sent_tokenize`
 and
 `word_tokenize`
 from
 `nltk.tokenize`
 have been pre-imported.





```python

# Tokenize the article into sentences: sentences
sentences = sent_tokenize(article)

# Tokenize each sentence into words: token_sentences
token_sentences = [word_tokenize(sent) for sent in sentences]

# Tag each tokenized sentence into parts of speech: pos_sentences
pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences]

# Create the named entity chunks: chunked_sentences
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)

# Test for stems of the tree with 'NE' tags
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, "label") and chunk.label() == "NE":
            print(chunk)

```




```

(NE Uber/NNP)
(NE Beyond/NN)
(NE Apple/NNP)
(NE Uber/NNP)
(NE Uber/NNP)
(NE Travis/NNP Kalanick/NNP)
(NE Tim/NNP Cook/NNP)
(NE Apple/NNP)
(NE Silicon/NNP Valley/NNP)
(NE CEO/NNP)
(NE Yahoo/NNP)
(NE Marissa/NNP Mayer/NNP)

```


## **3.1.2 Charting practice**



 In this exercise, you’ll use some extracted named entities and their groupings from a series of newspaper articles to chart the diversity of named entity types in the articles.




 You’ll use a
 `defaultdict`
 called
 `ner_categories`
 , with keys representing every named entity group type, and values to count the number of each different named entity type. You have a chunked sentence list called
 `chunked_sentences`
 similar to the last exercise, but this time with non-binary category names.




 You can use
 `hasattr()`
 to determine if each chunk has a
 `'label'`
 and then simply use the chunk’s
 `.label()`
 method as the dictionary key.





```

type(chunked_sentences)
list

chunked_sentences
[Tree('S', [('\ufeffImage', 'NN'), ('copyright', 'NN'), Tree('ORGANIZATION', [('EPA', 'NNP'), ('Image', 'NNP')]), ('caption', 'NN'), ('Uber', 'NNP'), ('has', 'VBZ'), ('been', 'VBN'), ('criticised', 'VBN'), ('many', 'JJ'), ('times', 'NNS'), ('over', 'IN'), ('the', 'DT'), ('way', 'NN'), ('it', 'PRP'), ('runs', 'VBZ'), ('its', 'PRP$'), ('business', 'NN'), ('Ride-sharing', 'JJ'), ('firm', 'NN'), ('Uber', 'NNP'), ('is', 'VBZ'), ('facing', 'VBG'), ('a', 'DT'), ('criminal', 'JJ'), ('investigation', 'NN'), ('by', 'IN'), ('the', 'DT'), Tree('GPE', [('US', 'JJ')]), ('government', 'NN'), ('.', '.')]),

```




```python

# Create the defaultdict: ner_categories
ner_categories = defaultdict(int)

# Create the nested for loop
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, 'label'):
            ner_categories[chunk.label()] += 1

# Create a list from the dictionary keys for the chart labels: labels
labels = list(ner_categories.keys())

labels
# ['ORGANIZATION', 'GPE', 'PERSON', 'LOCATION', 'FACILITY']

```




```python

# Create a list of the values: values
values = [ner_categories.get(v) for v in labels]

# Create the pie chart
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)

# Display the chart
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/14-7.png?w=1024)

## **3.1.3 Stanford library with NLTK**



 When using the Stanford library with NLTK, what is needed to get started?




**NLTK, the Stanford Java Libraries and some environment variables to help with integration.**





---



**3.2 Introduction to SpaCy**
------------------------------



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/15-7.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/16-6.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/17-5.png?w=844)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/18-4.png?w=888)



### **3.2.1 Comparing NLTK with spaCy NER**



 Using the same text you used in the first exercise of this chapter, you’ll now see the results using spaCy’s NER annotator. How will they compare?




 The article has been pre-loaded as
 `article`
 . To minimize execution times, you’ll be asked to specify the keyword arguments
 `tagger=False, parser=False, matcher=False`
 when loading the spaCy model, because you only care about the
 `entity`
 in this exercise.





```python

# Import spacy
import spacy

# Instantiate the English model: nlp
nlp = spacy.load('en',tagger=False, parser=False, matcher=False)

# Create a new document: doc
doc = nlp(article)

# Print all of the found entities and their labels
for ent in doc.ents:
    print(ent.label_, ent.text)


```




```

ORG Uber
ORG Uber
ORG Apple
ORG Uber
ORG Uber
PERSON Travis Kalanick
ORG Uber
PERSON Tim Cook
ORG Apple
CARDINAL Millions
ORG Uber
GPE drivers’
LOC Silicon Valley’s
ORG Yahoo
PERSON Marissa Mayer
MONEY $186m

```


### **3.2.2 spaCy NER Categories**



 Which are the
 *extra*
 categories that
 `spacy`
 uses compared to
 `nltk`
 in its named-entity recognition?




**NORP, CARDINAL, MONEY, WORK
 *OF*
 ART, LANGUAGE, EVENT**





---


## **3.3 Multilingual NER with polyglot**



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/19-3.png?w=1004)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/20-3.png?w=873)



####
 3.3.1 French NER with polyglot I



 In this exercise and the next, you’ll use the
 `polyglot`
 library to identify French entities. The library functions slightly differently than
 `spacy`
 , so you’ll use a few of the new things you learned in the last video to display the named entity text and category.




 You have access to the full article string in
 `article`
 . Additionally, the
 `Text`
 class of
 `polyglot`
 has been imported from
 `polyglot.text`
 .





```python

# Create a new text object using Polyglot's Text class: txt
txt = Text(article)

# Print each of the entities found
for ent in txt.entities:
    print(ent)

# Print the type of ent
print(type(ent))


```




```

['Charles', 'Cuvelliez']
['Charles', 'Cuvelliez']
['Bruxelles']
['l’IA']
['Julien', 'Maldonato']
['Deloitte']
['Ethiquement']
['l’IA']
['.']

<class 'polyglot.text.Chunk'>

```


### **3.3.2 French NER with polyglot II**



 Here, you’ll complete the work you began in the previous exercise.




 Your task is to use a list comprehension to create a list of tuples, in which the first element is the entity tag, and the second element is the full string of the entity text.





```python

# Create the list of tuples: entities
entities = [(ent.tag, ' '.join(ent)) for ent in txt.entities]

# Print entities
print(entities)

```




```

[('I-PER', 'Charles Cuvelliez'), ('I-PER', 'Charles Cuvelliez'), ('I-ORG', 'Bruxelles'), ('I-PER', 'l’IA'), ('I-PER', 'Julien Maldonato'), ('I-ORG', 'Deloitte'), ('I-PER', 'Ethiquement'), ('I-LOC', 'l’IA'), ('I-PER', '.')]

```


### **3.3.3 Spanish NER with polyglot**



 You’ll continue your exploration of
 `polyglot`
 now with some Spanish annotation. This article is not written by a newspaper, so it is your first example of a more blog-like text. How do you think that might compare when finding entities?




 The
 `Text`
 object has been created as
 `txt`
 , and each entity has been printed, as you can see in the IPython Shell.




 Your specific task is to determine how many of the entities contain the words
 `"Márquez"`
 or
 `"Gabo"`
 – these refer to the same person in different ways!




`txt.entities`
 is available.





```python

# Calculate the proportion of txt.entities that
# contains 'Márquez' or 'Gabo': prop_ggm
count = 0
for ent in txt.entities:
    if ("Márquez" in ent) or ("Gabo" in ent):
        count += 1

prop_ggm = count/len(txt.entities)
print(prop_ggm)
# 0.29591836734693877

```




# **4. Building a “fake news” classifier**
-----------------------------------------


## **4.1 Classifying fake news using supervised learning with NLP**



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/21-2.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/22-2.png?w=881)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/23-2.png?w=983)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/24-2.png?w=911)



### **4.1.1 Which possible features?**



 Which of the following are possible features for a text classification problem?



* **Number of words in a document.**
* **Specific named entities.**
* **Language.**




---


## **4.2 Building word count vectors with scikit-learn**



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/1-9.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/2-9.png?w=849)



### **4.2.1 CountVectorizer for text classification**



 It’s time to begin building your text classifier! The
 [data](https://s3.amazonaws.com/assets.datacamp.com/production/course_3629/fake_or_real_news.csv)
 has been loaded into a DataFrame called
 `df`
 . Explore it in the IPython Shell to investigate what columns you can use. The
 `.head()`
 method is particularly informative.




 In this exercise, you’ll use
 `pandas`
 alongside scikit-learn to create a sparse text vectorizer you can use to train and test a simple supervised model. To begin, you’ll set up a
 `CountVectorizer`
 and investigate some of its features.





```

print(df.head())
   Unnamed: 0                                              title  \
0        8476                       You Can Smell Hillary’s Fear
1       10294  Watch The Exact Moment Paul Ryan Committed Pol...
2        3608        Kerry to go to Paris in gesture of sympathy
3       10142  Bernie supporters on Twitter erupt in anger ag...
4         875   The Battle of New York: Why This Primary Matters

                                                text label
0  Daniel Greenfield, a Shillman Journalism Fello...  FAKE
1  Google Pinterest Digg Linkedin Reddit Stumbleu...  FAKE
2  U.S. Secretary of State John F. Kerry said Mon...  REAL
3  — Kaydee King (@KaydeeKing) November 9, 2016 T...  FAKE
4  It's primary day in New York and front-runners...  REAL

```




```python

# Import the necessary modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Print the head of df
print(df.head())

# Create a series to store the labels: y
y = df.label

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'],y,test_size=0.33,random_state=53)

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words='english')

# Transform the training data using only the 'text' column values: count_train
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test data using only the 'text' column values: count_test
count_test = count_vectorizer.transform(X_test)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])
# ['00', '000', '0000', '00000031', '000035', '00006', '0001', '0001pt', '000ft', '000km']

```


### **4.2.2 TfidfVectorizer for text classification**



 Similar to the sparse
 `CountVectorizer`
 created in the previous exercise, you’ll work on creating tf-idf vectors for your documents. You’ll set up a
 `TfidfVectorizer`
 and investigate some of its features.




 In this exercise, you’ll use
 `pandas`
 and
 `sklearn`
 along with the same
 `X_train`
 ,
 `y_train`
 and
 `X_test`
 ,
 `y_test`
 DataFrames and Series you created in the last exercise.





```python

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform the training data: tfidf_train
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data: tfidf_test
tfidf_test = tfidf_vectorizer.transform(X_test)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train.A[:5])


```




```

['00', '000', '001', '008s', '00am', '00pm', '01', '01am', '02', '024']
[[0.         0.01928563 0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 [0.         0.02895055 0.         ... 0.         0.         0.        ]
 [0.         0.03056734 0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]]

```


### **4.2.3 Inspecting the vectors**



 To get a better idea of how the vectors work, you’ll investigate them by converting them into
 `pandas`
 DataFrames.




 Here, you’ll use the same data structures you created in the previous two exercises (
 `count_train`
 ,
 `count_vectorizer`
 ,
 `tfidf_train`
 ,
 `tfidf_vectorizer`
 ) as well as
 `pandas`
 , which is imported as
 `pd`
 .





```python

# Create the CountVectorizer DataFrame: count_df
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

# Print the head of tfidf_df
print(tfidf_df.head())


# Calculate the difference in columns: difference
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)
# set()

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))
# False


```




```

print(count_df.head())
   000  00am  0600  10  100  107  11  110  1100  12    ...      younger  \
0    0     0     0   0    0    0   0    0     0   0    ...            0
1    0     0     0   3    0    0   0    0     0   0    ...            0
2    0     0     0   0    0    0   0    0     0   0    ...            0
3    0     0     0   0    0    0   0    0     0   0    ...            1
4    0     0     0   0    0    0   0    0     0   0    ...            0

   youth  youths  youtube  ypg  yuan  zawahiri  zeitung  zero  zerohedge
0      0       0        0    0     0         0        0     1          0
1      0       0        0    0     0         0        0     0          0
2      0       0        0    0     0         0        0     0          0
3      0       0        0    0     0         0        0     0          0
4      0       0        0    0     0         0        0     0          0

[5 rows x 5111 columns]


print(tfidf_df.head())
   000  00am  0600        10  100  107   11  110  1100   12    ...      \
0  0.0   0.0   0.0  0.000000  0.0  0.0  0.0  0.0   0.0  0.0    ...
1  0.0   0.0   0.0  0.105636  0.0  0.0  0.0  0.0   0.0  0.0    ...
2  0.0   0.0   0.0  0.000000  0.0  0.0  0.0  0.0   0.0  0.0    ...
3  0.0   0.0   0.0  0.000000  0.0  0.0  0.0  0.0   0.0  0.0    ...
4  0.0   0.0   0.0  0.000000  0.0  0.0  0.0  0.0   0.0  0.0    ...

    younger  youth  youths  youtube  ypg  yuan  zawahiri  zeitung      zero  \
0  0.000000    0.0     0.0      0.0  0.0   0.0       0.0      0.0  0.033579
1  0.000000    0.0     0.0      0.0  0.0   0.0       0.0      0.0  0.000000
2  0.000000    0.0     0.0      0.0  0.0   0.0       0.0      0.0  0.000000
3  0.015175    0.0     0.0      0.0  0.0   0.0       0.0      0.0  0.000000
4  0.000000    0.0     0.0      0.0  0.0   0.0       0.0      0.0  0.000000

   zerohedge
0        0.0
1        0.0
2        0.0
3        0.0
4        0.0

[5 rows x 5111 columns]

```




---


## **4.3 Training and testing a classification model with scikit-learn**



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/1-10.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/2-10.png?w=721)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/3-9.png?w=780)



### **4.3.1 Text classification models**



 Which of the below is the most reasonable model to use when training a new supervised model using text vector data?




**Naive Bayes**



### **4.3.2 Training and testing the “fake news” model with CountVectorizer**



 Now it’s your turn to train the “fake news” model using the features you identified and extracted. In this first exercise you’ll train and test a Naive Bayes model using the
 `CountVectorizer`
 data.




 The training and test sets have been created, and
 `count_vectorizer`
 ,
 `count_train`
 , and
 `count_test`
 have been computed.





```python

# Import the necessary modules
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(count_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test,pred)
print(score)
# 0.893352462936394

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test,pred,labels=['FAKE', 'REAL'])
print(cm)
#[[ 865  143]
 [  80 1003]]

```


### **4.3.3 Training and testing the “fake news” model with TfidfVectorizer**



 Now that you have evaluated the model using the
 `CountVectorizer`
 , you’ll do the same using the
 `TfidfVectorizer`
 with a Naive Bayes model.




 The training and test sets have been created, and
 `tfidf_vectorizer`
 ,
 `tfidf_train`
 , and
 `tfidf_test`
 have been computed. Additionally,
 `MultinomialNB`
 and
 `metrics`
 have been imported from, respectively,
 `sklearn.naive_bayes`
 and
 `sklearn`
 .





```python

# Calculate the accuracy score and confusion matrix of
# Multinomial Naive Bayes classifier predictions trained on
# tfidf_train, y_train and tested against tfidf_test and
# y_test

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(tfidf_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(tfidf_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test,pred)
print(score)
# 0.8565279770444764

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test,pred,labels=['FAKE', 'REAL'])
print(cm)
#[[ 739  269]
 [  31 1052]]

```



 Fantastic fake detection! The model correctly identifies fake news about 86% of the time. That’s a great start, but for a real world situation, you’d need to improve the score.





---


## **4.4 Simple NLP, complex problems**



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/4-9.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/5-9.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-natural-language-processing-in-python/6-9.png?w=1024)



### **4.4.1 Improving the model**



 What are possible next steps you could take to improve the model?



* Tweaking alpha levels.
* **Trying a new classification model.**
* **Training on a larger dataset.**
* **Improving text preprocessing.**


### **4.4.2 Improving your model**



 Your job in this exercise is to test a few different alpha levels using the
 `Tfidf`
 vectors to determine if there is a better performing combination.




 The training and test sets have been created, and
 `tfidf_vectorizer`
 ,
 `tfidf_train`
 , and
 `tfidf_test`
 have been computed.





```python

# Create the list of alphas: alphas
alphas = np.arange(0,1,0.1)

# Define train_and_predict()
def train_and_predict(alpha):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=alpha)
    # Fit to the training data
    nb_classifier.fit(tfidf_train,y_train)
    # Predict the labels: pred
    pred = nb_classifier.predict(tfidf_test)
    # Compute accuracy: score
    score = metrics.accuracy_score(y_test,pred)
    return score

# Iterate over the alphas and print the corresponding score
for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()


```




```

Alpha:  0.0
Score:  0.8813964610234337

Alpha:  0.1
Score:  0.8976566236250598

Alpha:  0.2
Score:  0.8938307030129125

Alpha:  0.30000000000000004
Score:  0.8900047824007652

Alpha:  0.4
Score:  0.8857006217120995

Alpha:  0.5
Score:  0.8842659014825442

Alpha:  0.6000000000000001
Score:  0.874701099952176

Alpha:  0.7000000000000001
Score:  0.8703969392635102

Alpha:  0.8
Score:  0.8660927785748446

Alpha:  0.9
Score:  0.8589191774270684

```


### **4.4.3 Inspecting your model**



 Now that you have built a “fake news” classifier, you’ll investigate what it has learned. You can map the important vector weights back to actual words using some simple inspection techniques.




 You have your well performing tfidf Naive Bayes classifier available as
 `nb_classifier`
 , and the vectors as
 `tfidf_vectorizer`
 .





```python

# Get the class labels: class_labels
class_labels = nb_classifier.classes_

# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])

```




```

FAKE [(-12.641778440826338, '0000'), (-12.641778440826338, '000035'), (-12.641778440826338, '0001'), (-12.641778440826338, '0001pt'), (-12.641778440826338, '000km'), (-12.641778440826338, '0011'), (-12.641778440826338, '006s'), (-12.641778440826338, '007'), (-12.641778440826338, '007s'), (-12.641778440826338, '008s'), (-12.641778440826338, '0099'), (-12.641778440826338, '00am'), (-12.641778440826338, '00p'), (-12.641778440826338, '00pm'), (-12.641778440826338, '014'), (-12.641778440826338, '015'), (-12.641778440826338, '018'), (-12.641778440826338, '01am'), (-12.641778440826338, '020'), (-12.641778440826338, '023')]

REAL [(-6.790929954967984, 'states'), (-6.765360557845786, 'rubio'), (-6.751044290367751, 'voters'), (-6.701050756752027, 'house'), (-6.695547793099875, 'republicans'), (-6.6701912490429685, 'bush'), (-6.661945235816139, 'percent'), (-6.589623788689862, 'people'), (-6.559670340096453, 'new'), (-6.489892292073901, 'party'), (-6.452319082422527, 'cruz'), (-6.452076515575875, 'state'), (-6.397696648238072, 'republican'), (-6.376343060363355, 'campaign'), (-6.324397735392007, 'president'), (-6.2546017970213645, 'sanders'), (-6.144621899738043, 'obama'), (-5.756817248152807, 'clinton'), (-5.596085785733112, 'said'), (-5.357523914504495, 'trump')]

```




---



 Thank you for reading and hope you’ve learned a lot.



