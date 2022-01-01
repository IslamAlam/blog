---
title: Unsupervised Learning in Python
date: 2021-12-07 11:22:07 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Unsupervised Learning in Python
==================================







 This is the memo of the 23th course of ‘Data Scientist with Python’ track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/unsupervised-learning-in-python)**
 .





---



# **1. Clustering for dataset exploration**
------------------------------------------




## **1.1 Unsupervised learning**


####
**How many clusters?**



 You are given an array
 `points`
 of size 300×2, where each row gives the (x, y) co-ordinates of a point on a map. Make a scatter plot of these points, and use the scatter plot to guess how many clusters there are.





```

points[:3]
array([[ 0.06544649, -0.76866376],
       [-1.52901547, -0.42953079],
       [ 1.70993371,  0.69885253]])

xs=points[:,0]
ys=points[:,1]

plt.scatter(xs,ys)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture3.png?w=1024)

####
**Clustering 2D points**



 From the scatter plot of the previous exercise, you saw that the points seem to separate into 3 clusters. You’ll now create a KMeans model to find 3 clusters, and fit it to the data points from the previous exercise. After the model has been fit, you’ll obtain the cluster labels for some new points using the
 `.predict()`
 method.





```python

# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)

[1 2 0 1 2 1 2 2 2 0 1 2 2 0 0 2 0 0 2 2 0 2 1 2 1 0 2 0 0 1 1 2 2 2 0 1 2
...
 0 2 2 1]


```



 You’ve successfully performed k-Means clustering and predicted the labels of new points. But it is not easy to inspect the clustering by just looking at the printed labels. A visualization would be far more useful. In the next exercise, you’ll inspect your clustering with a scatter plot!



####
**Inspect your clustering**




```python

# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c=labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture4.png?w=1024)


 The clustering looks great! But how can you be sure that 3 clusters is the correct choice? In other words, how can you evaluate the quality of a clustering?





---


## **1.2 Evaluating a clustering**


####
**How many clusters of grain?**




```

samples
array([[15.26  , 14.84  ,  0.871 , ...,  3.312 ,  2.221 ,  5.22  ],
       ...,
       [12.3   , 13.34  ,  0.8684, ...,  2.974 ,  5.637 ,  5.063 ]])

```




```

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(samples)

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture5.png?w=647)


 The inertia decreases very slowly from 3 clusters to 4, so it looks like 3 clusters would be a good choice for this data.



####
**Evaluating the grain clustering**



 In the previous exercise, you observed from the inertia plot that 3 is a good number of clusters for the grain data. In fact, the grain samples come from a mix of 3 different grain varieties: “Kama”, “Rosa” and “Canadian”. In this exercise, cluster the grain samples into three clusters, and compare the clusters to the grain varieties using a cross-tabulation.




 You have the array
 `samples`
 of grain samples, and a list
 `varieties`
 giving the grain variety for each sample.





```

varieties
['Kama wheat',
 'Kama wheat',
...
 'Canadian wheat',
 'Canadian wheat']

```




```python

# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)


```




```

varieties  Canadian wheat  Kama wheat  Rosa wheat
labels
0                       0           1          60
1                      68           9           0
2                       2          60          10

```



 The cross-tabulation shows that the 3 varieties of grain separate really well into 3 clusters. But depending on the type of data you are working with, the clustering may not always be this good. Is there anything you can do in such situations to improve your clustering?





---


## **1.3 Transforming features for better clusterings**


####
**Scaling fish data for clustering**



 You are given an array
 `samples`
 giving measurements of fish. Each row represents an individual fish. The measurements, such as weight in grams, length in centimeters, and the percentage ratio of height to length, have very different scales. In order to cluster this data effectively, you’ll need to standardize these features first. In this exercise, you’ll build a pipeline to standardize and cluster the data.




 These fish measurement data were sourced from the
 [Journal of Statistics Education](http://ww2.amstat.org/publications/jse/jse_data_archive.htm)
 .





```python

# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)


```


####
**Clustering the fish data**




```

samples
array([[ 242. ,   23.2,   25.4,   30. ,   38.4,   13.4],
       [ 290. ,   24. ,   26.3,   31.2,   40. ,   13.8],
       [ 340. ,   23.9,   26.5,   31.1,   39.8,   15.1],
...

```




```python

# Import pandas
import pandas as pd

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels':labels, 'species':species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['species'])

# Display ct
print(ct)


```




```

species  Bream  Pike  Roach  Smelt
labels
0            0     0      0     13
1           33     0      1      0
2            0    17      0      0
3            1     0     19      1

```



 It looks like the fish data separates really well into 4 clusters!



####
**Clustering stocks using KMeans**



 In this exercise, you’ll cluster companies using their daily stock price movements (i.e. the dollar difference between the closing and opening prices for each trading day). You are given a NumPy array
 `movements`
 of daily price movements from 2010 to 2015 (obtained from Yahoo! Finance), where each row corresponds to a company, and each column corresponds to a trading day.




 Some stocks are more expensive than others. To account for this, include a
 `Normalizer`
 at the beginning of your pipeline. The Normalizer will separately transform each company’s stock price to a relative scale before the clustering begins.




 Note that
 `Normalizer()`
 is different to
 `StandardScaler()`
 , which you used in the previous exercise. While
 `StandardScaler()`
 standardizes
 **features**
 (such as the features of the fish data from the previous exercise) by removing the mean and scaling to unit variance,
 `Normalizer()`
 rescales
 **each sample**
 – here, each company’s stock price – independently of the other.





```

movements
array([[ 5.8000000e-01, -2.2000500e-01, -3.4099980e+00, ...,
        -5.3599620e+00,  8.4001900e-01, -1.9589981e+01],
       ...,
       [ 1.5999900e-01,  1.0001000e-02,  0.0000000e+00, ...,
        -6.0001000e-02,  2.5999800e-01,  9.9998000e-02]])

```




```python

# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)


```



 Now that your pipeline has been set up, you can find out which stocks move together in the next exercise!



####
**Which stocks move together?**



 In the previous exercise, you clustered companies by their daily stock price movements. So which company have stock prices that tend to change in the same way? You’ll now inspect the cluster labels from your clustering to find out.





```python

# Import pandas
import pandas as pd

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))


```




```

                             companies  labels
59                               Yahoo       0
15                                Ford       0
35                            Navistar       0
26                      JPMorgan Chase       1
16                   General Electrics       1
58                               Xerox       1
11                               Cisco       1
18                       Goldman Sachs       1
20                          Home Depot       1
5                      Bank of America       1
3                     American express       1
55                         Wells Fargo       1
1                                  AIG       1
38                               Pepsi       2
...

```



 In the next chapter, you’ll learn about how to communicate results such as this through visualizations.





---



# **2. Visualization with hierarchical clustering and t-SNE**
------------------------------------------------------------


###
**Visualizing hierarchies**


####
**Hierarchical clustering of the grain data**



 Use the
 `linkage()`
 function to obtain a hierarchical clustering of the grain samples, and use
 `dendrogram()`
 to visualize the result.





```python

# Perform the necessary imports
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Calculate the linkage: mergings
mergings = linkage(samples, method='complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture6.png?w=1024)


 Dendrograms are a great way to illustrate the arrangement of the clusters produced by hierarchical clustering.



####
**Hierarchies of stocks**




```python

# Import normalize
from sklearn.preprocessing import normalize

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements, method='complete')

# Plot the dendrogram
dendrogram(mergings, labels=companies, leaf_rotation=90, leaf_font_size=6)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture7.png?w=1024)



---


###
**Cluster labels in hierarchical clustering**


####
**Different linkage, different hierarchical clustering!**



 Perform a hierarchical clustering of the voting countries with
 `'single'`
 linkage, and compare the resulting dendrogram with the one in the video. Different linkage, different hierarchical clustering!





```python

# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Calculate the linkage: mergings
mergings = linkage(samples, method='single')

# Plot the dendrogram
dendrogram(mergings, labels=country_names, leaf_rotation=90, leaf_font_size=6)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture8.png?w=1024)


 As you can see, performing single linkage hierarchical clustering produces a different dendrogram!



###
**Extracting the cluster labels**



 Use the
 `fcluster()`
 function to extract the cluster labels for this intermediate clustering, and compare the labels with the grain varieties using a cross-tabulation.





```python

# Perform the necessary imports
import pandas as pd
from scipy.cluster.hierarchy import fcluster

# Use fcluster to extract labels: labels
labels = fcluster(mergings, 6, criterion='distance')

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)


```




```

varieties  Canadian wheat  Kama wheat  Rosa wheat
labels
1                      14           3           0
2                       0           0          14
3                       0          11           0

```



 You’ve now mastered the fundamentals of k-Means and agglomerative hierarchical clustering. Next, you’ll learn about t-SNE, which is a powerful tool for visualizing high dimensional data.





---


###
**t-SNE for 2-dimensional maps**


####
**t-SNE visualization of grain dataset**



[t-distributed stochastic neighbor embedding](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)




 In this exercise, you’ll apply t-SNE to the grain samples data and inspect the resulting t-SNE features using a scatter plot.





```

samples[:3]
array([[15.26  , 14.84  ,  0.871 ,  5.763 ,  3.312 ,  2.221 ,  5.22  ],
       [14.88  , 14.57  ,  0.8811,  5.554 ,  3.333 ,  1.018 ,  4.956 ],
       [14.29  , 14.09  ,  0.905 ,  5.291 ,  3.337 ,  2.699 ,  4.825 ]])

 variety_numbers[:3]
 [1, 1, 1]

```




```python

# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=200)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(samples)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs, ys, c=variety_numbers)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture.png?w=1024)


 As you can see, the t-SNE visualization manages to separate the 3 varieties of grain samples.



####
**A t-SNE map of the stock market**



 t-SNE provides great visualizations when the individual samples can be labeled. In this exercise, you’ll apply t-SNE to the company stock price data. A scatter plot of the resulting t-SNE features, labeled by the company names, gives you a map of the stock market!





```python

# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=50)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs, ys, alpha=0.5)

# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture1.png?w=1024)


 It’s visualizations such as this that make t-SNE such a powerful tool for extracting quick insights from high dimensional data.





---



# **3. Decorrelating your data and dimension reduction**
-------------------------------------------------------


###
**Visualizing the PCA(**
 principal component analysis
 **) transformation**


####
**Correlated data in nature**



 You are given an array
 `grains`
 giving the width and length of samples of grain. You suspect that width and length will be correlated. To confirm this, make a scatter plot of width vs length and measure their Pearson correlation.





```python

# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Assign the 0th column of grains: width
width = grains[:,0]

# Assign the 1st column of grains: length
length = grains[:,1]

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.xlabel('grains width')
plt.ylabel('grains length')

plt.show()

# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width, length)

# Display the correlation
print(correlation)

#  0.8604149377143467

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture2.png?w=1024)


 The width and length of the grain samples are highly correlated.



####
**Decorrelating the grain measurements with PCA**



 You’ll use PCA to decorrelate these measurements, then plot the decorrelated points and measure their Pearson correlation.





```python

# Import PCA
from sklearn.decomposition import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.xlabel('pca_features 0')
plt.ylabel('pca_features 1')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture3-1.png?w=1024)

###
**Intrinsic dimension**


####
**The first principal component**



 The first principal component of the data is the direction in which the data varies the most. In this exercise, your job is to use PCA to find the first principal component of the length and width measurements of the grain samples, and represent it as an arrow on the scatter plot.





```python

# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])

# Add labels
plt.xlabel('width')
plt.ylabel('length')

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture4-1.png?w=1024)


 This is the direction in which the grain data varies the most.



####
**Variance of the PCA features**



 The fish dataset is 6-dimensional. But what is its
 *intrinsic*
 dimension? Make a plot of the variances of the PCA features to find out. As before,
 `samples`
 is a 2D array, where each row represents a fish. You’ll need to standardize the features first.





```

samples[:3]
array([[242. ,  23.2,  25.4,  30. ,  38.4,  13.4],
       [290. ,  24. ,  26.3,  31.2,  40. ,  13.8],
       [340. ,  23.9,  26.5,  31.1,  39.8,  15.1]])

```




```python

# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

# Fit the pipeline to 'samples'
pipeline.fit(samples)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture5-1.png?w=1024)


 It looks like PCA features 0 and 1 have significant variance.





---


###
**Dimension reduction with PCA**



 Use PCA for dimensionality reduction of the fish measurements, retaining only the 2 most important components.





```

scaled_samples[:3]
array([[-0.50109735, -0.36878558, -0.34323399, -0.23781518,  1.0032125 ,
         0.25373964],
       [-0.37434344, -0.29750241, -0.26893461, -0.14634781,  1.15869615,
         0.44376493],
       [-0.24230812, -0.30641281, -0.25242364, -0.15397009,  1.13926069,
         1.0613471 ]])

```




```python

# Import PCA
from sklearn.decomposition import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)

(85, 2)

```




```

pca_features[:3]
array([[-0.57640502, -0.94649159],
       [-0.36852393, -1.17103598],
       [-0.28028168, -1.59709224]])

```



 You’ve successfully reduced the dimensionality from 6 to 2.



####
**A tf-idf word-frequency array**



 In this exercise, you’ll create a tf-idf word frequency array for a toy collection of documents. For this, use the
 `TfidfVectorizer`
 from sklearn. It transforms a list of documents into a word frequency array, which it outputs as a csr_matrix. It has
 `fit()`
 and
 `transform()`
 methods like other sklearn objects.




**[term frequency–inverse document frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)**





```

documents
['cats say meow', 'dogs say woof', 'dogs chase cats']

```




```python

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer()

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names()

# Print words
print(words)

```




```

[[0.51785612 0.         0.         0.68091856 0.51785612 0.        ]
 [0.         0.         0.51785612 0.         0.51785612 0.68091856]
 [0.51785612 0.68091856 0.51785612 0.         0.         0.        ]]
['cats', 'chase', 'dogs', 'meow', 'say', 'woof']

```




---


####
**Clustering Wikipedia part I**



 You saw in the video that
 `TruncatedSVD`
 is able to perform PCA on sparse arrays in csr_matrix format, such as word-frequency arrays. Combine your knowledge of TruncatedSVD and k-means to cluster some popular pages from Wikipedia. In this exercise, build the pipeline. In the next exercise, you’ll apply it to the word-frequency array of some Wikipedia articles.




 Create a Pipeline object consisting of a TruncatedSVD followed by KMeans. (This time, we’ve precomputed the word-frequency matrix for you, so there’s no need for a TfidfVectorizer).




 The Wikipedia dataset you will be working with was obtained from
 [here](https://blog.lateral.io/2015/06/the-unknown-perils-of-mining-wikipedia/)
 .





```python

# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd, kmeans)


```


####
**Clustering Wikipedia part II**



 You are given an array
 `articles`
 of tf-idf word-frequencies of some popular Wikipedia articles, and a list
 `titles`
 of their titles. Use your pipeline to cluster the Wikipedia articles.





```

type(articles)
scipy.sparse.csr.csr_matrix

articles
<60x13125 sparse matrix of type '<class 'numpy.float64'>'
	with 42091 stored elements in Compressed Sparse Row format>

 titles[:3]
['HTTP 404', 'Alexa Internet', 'Internet Explorer']

df.shape
(13125, 60)

df.head(1)
   HTTP 404  Alexa Internet  Internet Explorer  HTTP cookie  Google Search  \
0       0.0             0.0                0.0          0.0            0.0

   Tumblr  Hypertext Transfer Protocol  Social search  Firefox  LinkedIn  \
0     0.0                          0.0            0.0      0.0       0.0

      ...       Chad Kroeger  Nate Ruess  The Wanted  Stevie Nicks  \
0     ...                0.0         0.0         0.0      0.008878

   Arctic Monkeys  Black Sabbath  Skrillex  Red Hot Chili Peppers  Sepsis  \
0             0.0            0.0  0.049502                    0.0     0.0

   Adam Levine
0          0.0

[1 rows x 60 columns]

```




```python

# Import pandas
import pandas as pd

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values('label'))


```




```

                                          article  label
59                                    Adam Levine      0
57                          Red Hot Chili Peppers      0
56                                       Skrillex      0
55                                  Black Sabbath      0
54                                 Arctic Monkeys      0
53                                   Stevie Nicks      0
52                                     The Wanted      0
51                                     Nate Ruess      0
50                                   Chad Kroeger      0
58                                         Sepsis      0
30                  France national football team      1
31                              Cristiano Ronaldo      1
32                                   Arsenal F.C.      1
33                                 Radamel Falcao      1
37                                       Football      1
35                Colombia national football team      1
36              2014 FIFA World Cup qualification      1
38                                         Neymar      1
39                                  Franck Ribéry      1
34                             Zlatan Ibrahimović      1
26                                     Mila Kunis      2
28                                  Anne Hathaway      2
27                                 Dakota Fanning      2
25                                  Russell Crowe      2
29                               Jennifer Aniston      2
23                           Catherine Zeta-Jones      2
22                              Denzel Washington      2
21                             Michael Fassbender      2
20                                 Angelina Jolie      2
24                                   Jessica Biel      2

```




---



# **4. Discovering interpretable features**
------------------------------------------


###
**Non-negative matrix factorization (NMF)**


####
**NMF applied to Wikipedia articles**




```

articles
<60x13125 sparse matrix of type '<class 'numpy.float64'>'
	with 42091 stored elements in Compressed Sparse Row format>

print(articles)
  (0, 16)	0.024688249778400003
  (0, 32)	0.0239370711117
  (0, 33)	0.0210896267411
  (0, 137)	0.012295430569100001

```




```python

# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
print(nmf_features)


```




```

[[0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 4.40531625e-01]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 5.66689786e-01]
...
 [3.78224082e-01 1.43978958e-02 0.00000000e+00 9.84935436e-02
  1.35904928e-02 0.00000000e+00]]

```



 these NMF features don’t make much sense at this point, but you will explore them in the next exercise!



####
**NMF features of the Wikipedia articles**



 When investigating the features, notice that for both actors, the NMF feature 3 has by far the highest value. This means that both articles are reconstructed using mainly the 3rd NMF component.





```python

# Import pandas
import pandas as pd

# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features, index=titles)

# Print the row for 'Anne Hathaway'
print(df.loc['Anne Hathaway'])

# Print the row for 'Denzel Washington'
print(df.loc['Denzel Washington'])


```




```

0    0.003845
1    0.000000
2    0.000000
3    0.575711
4    0.000000
5    0.000000
Name: Anne Hathaway, dtype: float64

0    0.000000
1    0.005601
2    0.000000
3    0.422380
4    0.000000
5    0.000000
Name: Denzel Washington, dtype: float64


df.head()
                          0    1    2    3    4         5
HTTP 404           0.000000  0.0  0.0  0.0  0.0  0.440465
Alexa Internet     0.000000  0.0  0.0  0.0  0.0  0.566605
Internet Explorer  0.003821  0.0  0.0  0.0  0.0  0.398646
HTTP cookie        0.000000  0.0  0.0  0.0  0.0  0.381740
Google Search      0.000000  0.0  0.0  0.0  0.0  0.485517

```



 Notice that for both actors, the NMF feature 3 has by far the highest value. This means that both articles are reconstructed using mainly the 3rd NMF component. Because NMF components represent topics (for instance, acting!).



###
**NMF learns interpretable parts**


####
**NMF learns topics of documents**



 When NMF is applied to documents, the components correspond to topics of documents, and the NMF features reconstruct the documents from the topics.




 Verify this for yourself for the NMF model that you built earlier using the Wikipedia articles. Previously, you saw that the 3rd NMF feature value was high for the articles about actors Anne Hathaway and Denzel Washington.




 In this exercise, identify the topic of the corresponding NMF component.





```

model.components_
array([[1.13754523e-02, 1.20974422e-03, 0.00000000e+00, ...,
        0.00000000e+00, 4.23594130e-04, 0.00000000e+00],
       [0.00000000e+00, 9.57177268e-06, 5.66343849e-03, ...,
        2.81289906e-03, 2.97179984e-04, 0.00000000e+00],
       [0.00000000e+00, 8.30814049e-06, 0.00000000e+00, ...,
        0.00000000e+00, 1.43192324e-04, 0.00000000e+00],
       [4.14811200e-03, 0.00000000e+00, 3.05595648e-03, ...,
        1.74191620e-03, 6.71969911e-03, 0.00000000e+00],
       [0.00000000e+00, 5.68399302e-04, 4.91797182e-03, ...,
        1.91632504e-04, 1.35146218e-03, 0.00000000e+00],
       [1.38501597e-04, 0.00000000e+00, 8.74840829e-03, ...,
        2.40081634e-03, 1.68211026e-03, 0.00000000e+00]])

words[:3]
['aaron', 'abandon', 'abandoned']

```




```python

# Import pandas
import pandas as pd

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_, columns=words)

# Print the shape of the DataFrame
print(components_df.shape)
# (6, 13125)

# Select row 3: component
component = components_df.iloc[3,:]

# Print result of nlargest
print(component.nlargest())


film       0.627877
award      0.253131
starred    0.245284
role       0.211451
actress    0.186398
Name: 3, dtype: float64

```




```

components_df
      aaron   abandon  abandoned  abandoning  abandonment  abbas  abbey  \
0  0.011375  0.001210   0.000000    0.001739     0.000136    0.0    0.0
1  0.000000  0.000010   0.005663    0.000000     0.000002    0.0    0.0
2  0.000000  0.000008   0.000000    0.000000     0.004692    0.0    0.0
3  0.004148  0.000000   0.003056    0.000000     0.000614    0.0    0.0
4  0.000000  0.000568   0.004918    0.000000     0.000000    0.0    0.0
5  0.000139  0.000000   0.008748    0.000000     0.000185    0.0    0.0

   abbreviated  abbreviation       abc ...    zealand  zenith  zeppelin  \
0     0.002463  2.445684e-07  0.000834 ...   0.025781     0.0  0.008324
1     0.000566  5.002620e-04  0.000000 ...   0.008106     0.0  0.000000
2     0.000758  1.604283e-05  0.000000 ...   0.008730     0.0  0.000000
3     0.002436  8.143270e-05  0.003985 ...   0.012594     0.0  0.000000
4     0.000089  4.259695e-05  0.000000 ...   0.001809     0.0  0.000000
5     0.008629  1.530385e-05  0.000000 ...   0.000000     0.0  0.000000

       zero  zeus  zimbabwe  zinc      zone     zones  zoo
0  0.000000   0.0       0.0   0.0  0.000000  0.000424  0.0
1  0.001710   0.0       0.0   0.0  0.002813  0.000297  0.0
2  0.001317   0.0       0.0   0.0  0.000000  0.000143  0.0
3  0.000000   0.0       0.0   0.0  0.001742  0.006720  0.0
4  0.000017   0.0       0.0   0.0  0.000192  0.001351  0.0
5  0.000000   0.0       0.0   0.0  0.002401  0.001682  0.0

[6 rows x 13125 columns]

```




---


####
**Explore the LED digits dataset**



 In the following exercises, you’ll use NMF to decompose grayscale images into their commonly occurring patterns.





```

samples
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]])

```




```python

# Import pyplot
from matplotlib import pyplot as plt

# Select the 0th row: digit
digit = samples[0,:]

# Print digit
print(digit)

[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.
 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0.
 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0.]

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape(13,8)

# Print bitmap
print(bitmap)
[[0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 1. 1. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0.]]

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture6-1.png?w=849)


 You’ll explore this dataset further in the next exercise and see for yourself how NMF can learn the parts of images.



####
**NMF learns the parts of images**



 Now use what you’ve learned about NMF to decompose the digits dataset. You are again given the digit images as a 2D array
 `samples`
 . This time, you are also provided with a function
 `show_as_image()`
 that displays the image encoded by any 1D array:





```

def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

```



 After you are done, take a moment to look through the plots and notice how NMF has expressed the digit as a sum of the components!





```python

# Import NMF
from sklearn.decomposition import NMF

# Create an NMF model: model
model = NMF(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Assign the 0th row of features: digit_features
digit_features = features[0,:]

# Print digit_features
print(digit_features)

```




```

features
Out[3]:
array([[4.76823559e-01, 0.00000000e+00, 0.00000000e+00, 5.90605054e-01,
        4.81559442e-01, 0.00000000e+00, 7.37557191e-16],
...
       [0.00000000e+00, 0.00000000e+00, 5.21027460e-01, 0.00000000e+00,
        4.81559442e-01, 4.93832117e-01, 0.00000000e+00]])

```


![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture16-1.png?w=470)
![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture15-1.png?w=470)
![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture14-1.png?w=479)
![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture13-1.png?w=465)
![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture12-1.png?w=471)
![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture11-1.png?w=466)
![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture10-1.png?w=470)




---


####
**PCA doesn’t learn parts**



 Unlike NMF, PCA
 *doesn’t*
 learn the parts of things. Its components do not correspond to topics (in the case of documents) or to parts of images, when trained on images. Verify this for yourself by inspecting the components of a PCA model fit to the dataset of LED digit images from the previous exercise.





```python

# Import PCA
from sklearn.decomposition import PCA

# Create a PCA instance: model
model = PCA(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)


```


![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture26.png?w=346)
![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture25.png?w=346)
![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture24.png?w=350)
![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture23.png?w=350)
![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture22.png?w=353)
![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture21.png?w=348)
![Desktop View]({{ site.baseurl }}/assets/datacamp/unsupervised-learning-in-python/capture20.png?w=352)



 Notice that the components of PCA do not represent meaningful parts of images of LED digits!



###
**Building recommender systems using NMF**


####
**Which articles are similar to ‘Cristiano Ronaldo’?**



 You learned how to use NMF features and the cosine similarity to find similar articles. Apply this to your NMF model for popular Wikipedia articles, by finding the articles most similar to the article about the footballer Cristiano Ronaldo.





```python

# Perform the necessary imports
import pandas as pd
from sklearn.preprocessing import normalize

# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=titles)

# Select the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo',:]

# Compute the dot products: similarities
similarities = df.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())

```




```

Cristiano Ronaldo                1.000000
Franck Ribéry                    0.999972
Radamel Falcao                   0.999942
Zlatan Ibrahimović               0.999942
France national football team    0.999923
dtype: float64


df.head()
                          0    1    2    3    4         5
HTTP 404           0.000000  0.0  0.0  0.0  0.0  1.000000
Alexa Internet     0.000000  0.0  0.0  0.0  0.0  1.000000
Internet Explorer  0.009583  0.0  0.0  0.0  0.0  0.999954
HTTP cookie        0.000000  0.0  0.0  0.0  0.0  1.000000
Google Search      0.000000  0.0  0.0  0.0  0.0  1.000000


article
0    0.002523
1    0.999942
2    0.000859
3    0.010274
4    0.001947
5    0.000724
Name: Cristiano Ronaldo, dtype: float64

similarities
Out[15]:
HTTP 404                                         0.000724
Alexa Internet                                   0.000724
Internet Explorer                                0.000748
...
France national football team                    0.999923
Cristiano Ronaldo                                1.000000
Arsenal F.C.                                     0.997739
Radamel Falcao                                   0.999942
Zlatan Ibrahimović                               0.999942
Colombia national football team                  0.999897
2014 FIFA World Cup qualification                0.998443
Football                                         0.974915
Neymar                                           0.999021
Franck Ribéry                                    0.999972
...
Sepsis                                           0.041880
Adam Levine                                      0.041873
dtype: float64

```



 Although you may need to know a little about football (or soccer, depending on where you’re from!) to be able to evaluate for yourself the quality of the computed similarities!





---


####
**Recommend musical artists part I**



 In this exercise and the next, you’ll use what you’ve learned about NMF to recommend popular music artists! You are given a sparse array
 `artists`
 whose rows correspond to artists and whose column correspond to users. The entries give the number of times each artist was listened to by each user.




 In this exercise, build a pipeline and transform the array into normalized NMF features. The first step in the pipeline,
 `MaxAbsScaler`
 , transforms the data so that all users have the same influence on the model, regardless of how many different artists they’ve listened to. In the next exercise, you’ll use the resulting normalized NMF features for recommendation!





```

 artists
<111x500 sparse matrix of type '<class 'numpy.float64'>'
	with 2894 stored elements in Compressed Sparse Row format>

print(artists)
  (0, 2)	105.0
  (0, 15)	165.0
  (0, 20)	91.0

```




```python

# Perform the necessary imports
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components=20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler, nmf, normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)


```


####
**Recommend musical artists part II**



 Suppose you were a big fan of Bruce Springsteen – which other musicial artists might you like? Use your NMF features from the previous exercise and the cosine similarity to find similar musical artists.





```python

# Import pandas
import pandas as pd

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print(similarities.nlargest())


```




```

Bruce Springsteen    1.000000
Neil Young           0.955896
Van Morrison         0.872452
Leonard Cohen        0.864763
Bob Dylan            0.859047
dtype: float64

```



 The End.


 Thank you for reading.



