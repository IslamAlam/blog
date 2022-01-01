---
title: Statistical Thinking in Python (Part 1)
date: 2021-12-07 11:22:11 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Statistical Thinking in Python (Part 1)
==========================================







 This is the memo of the 1st course (5 courses in all) of ‘Statistics Fundamentals with Python’ skill track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/statistical-thinking-in-python-part-1)**
 .



###
**Table of contents**


1. Graphical exploratory data analysis
2. Quantitative exploratory data analysis
3. Thinking probabilistically– Discrete variables
4. Thinking probabilistically– Continuous variables





# **1. Graphical exploratory data analysis**
-------------------------------------------


###
 Introduction to exploratory data analysis


####
 Tukey’s comments on EDA


* Exploratory data analysis is detective work.
* There is no excuse for failing to plot and look.
* The greatest value of a picture is that it forces us to notice what we never expected to see.
* It is important to understand what you can do before you learn how to measure how well you seem to have done it.


####
 Advantages of graphical EDA


* It often involves converting tabular data into graphical form.
* If done well, graphical representations can allow for more rapid interpretation of data.
* There is no excuse for neglecting to do graphical EDA.


###
 Plotting a histogram


####
 Plotting a histogram of iris data




```

versicolor_petal_length
array([4.7, 4.5, 4.9, 4. , 4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4. ,
       4.7, 3.6, 4.4, 4.5, 4.1, 4.5, 3.9, 4.8, 4. , 4.9, 4.7, 4.3, 4.4,
       4.8, 5. , 4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1,
       4. , 4.4, 4.6, 4. , 3.3, 4.2, 4.2, 4.2, 4.3, 3. , 4.1])

```




```python

# Import plotting modules
import matplotlib.pyplot as plt
import seaborn as sns

# Set default Seaborn style
sns.set()

# Plot histogram of versicolor petal lengths
plt.hist(versicolor_petal_length)

# Show histogram
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture-12.png)

####
 Axis labels!




```python

# Plot histogram of versicolor petal lengths
_ = plt.hist(versicolor_petal_length)

# Label axes
plt.xlabel('petal length (cm)')
plt.ylabel('count')

# Show histogram
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture1-11.png)

####
 Adjusting the number of bins in a histogram



 The “square root rule” is a commonly-used rule of thumb for choosing number of bins: choose the number of bins to be the square root of the number of samples.





```python

# Import numpy
import numpy as np

# Compute number of data points: n_data
n_data = len(versicolor_petal_length)

# Number of bins is the square root of number of data points: n_bins
n_bins = np.sqrt(n_data)

# Convert number of bins to integer: n_bins
n_bins = int(n_bins)

# Plot the histogram
_ = plt.hist(versicolor_petal_length, bins=n_bins)

# Label axes
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('count')

# Show histogram
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture2-10.png)

###
 Plotting all of your data: Bee swarm plots


####
 Bee swarm plot




```python

# Create bee swarm plot with Seaborn's default settings
sns.swarmplot(x='species', y = 'petal length (cm)', data=df)

# Label the axes
plt.xlabel('species')
plt.ylabel('petal length (cm)')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture3-9.png)

####
 Interpreting a bee swarm plot



*I. virginica*
 petals tend to be the longest, and
 *I. setosa*
 petals tend to be the shortest of the three species.



###
 Plotting all of your data: Empirical cumulative distribution functions (ECDF)


####
 Computing the ECDF




```

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n

    return x, y

```


####
 Plotting the ECDF




```python

# Compute ECDF for versicolor data: x_vers, y_vers
x_vers, y_vers = ecdf(versicolor_petal_length)

# Generate plot
plt.plot(x_vers, y_vers, marker = '.', linestyle = 'none')

# Label the axes
plt.xlabel('petal length (cm)')
plt.ylabel('ECDF')

# Display the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture4-9.png)

####
 Comparison of ECDFs




```python

# Compute ECDFs
x_set, y_set = ecdf(setosa_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)
x_virg, y_virg = ecdf(virginica_petal_length)

# Plot all ECDFs on the same plot
plt.plot(x_set, y_set, marker = '.', linestyle = 'none')
plt.plot(x_vers, y_vers, marker = '.', linestyle = 'none')
plt.plot(x_virg, y_virg, marker = '.', linestyle = 'none')


# Annotate the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture5-7.png)



# **2. Quantitative exploratory data analysis**
----------------------------------------------


###
 Introduction to summary statistics: The sample mean and median


####
 Means and medians



 An outlier can significantly affect the value of the mean, but not the median.



####
 Computing means




```python

# Compute the mean: mean_length_vers
mean_length_vers = np.mean(versicolor_petal_length)

# Print the result with some nice formatting
print('I. versicolor:', mean_length_vers, 'cm')
I. versicolor: 4.26 cm

```


###
 Percentiles, outliers, and box plots


####
 Computing percentiles




```python

# Specify array of percentiles: percentiles
percentiles = np.array([2.5, 25, 50, 75, 97.5])

# Compute percentiles: ptiles_vers
ptiles_vers = np.percentile(versicolor_petal_length, percentiles)

# Print the result
print(ptiles_vers)
[3.3    4.     4.35   4.6    4.9775]

```


####
 Comparing percentiles to ECDF




```python

# Plot the ECDF
_ = plt.plot(x_vers, y_vers, '.')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Overlay percentiles as red diamonds.
_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red',
         linestyle='none')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture-13.png)

####
 Box-and-whisker plot




```python

# Create box plot with Seaborn's default settings
sns.boxplot(x='species', y='petal length (cm)', data=df)

# Label the axes
plt.xlabel('species')
plt.ylabel('petal length (cm)')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture1-12.png)

###
 Variance and standard deviation


####
 Computing the variance




```python

# Array of differences to mean: differences
differences = versicolor_petal_length - np.mean(versicolor_petal_length)

# Square the differences: diff_sq
diff_sq = differences ** 2

# Compute the mean square difference: variance_explicit
variance_explicit = np.mean(diff_sq)

# Compute the variance using NumPy: variance_np
variance_np = np.var(versicolor_petal_length)

# Print the results
print(variance_explicit, variance_np)
0.21640000000000004 0.21640000000000004

```


####
 The standard deviation and the variance




```python

# Compute the variance: variance
variance = np.var(versicolor_petal_length)

# Print the square root of the variance
print(variance ** 0.5)

# Print the standard deviation
print(np.std(versicolor_petal_length))

0.4651881339845203
0.4651881339845203

```


###
 Covariance and Pearson correlation coefficient



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture2-11.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture3-10.png)

####
 Scatter plots




```python

# Make a scatter plot
plt.plot(versicolor_petal_length, versicolor_petal_width, marker='.', linestyle='none')


# Label the axes
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')


# Show the result
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture4-10.png)

####
 Variance and covariance by looking



 Consider four scatter plots of x-y data, appearing to the right. Which has, respectively,



* the highest variance in the variable x, d
* the highest covariance, c
* negative covariance, b



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture5-8.png)

####
 Computing the covariance




```python

# Compute the covariance matrix: covariance_matrix
covariance_matrix = np.cov(versicolor_petal_length, versicolor_petal_width)


# Print covariance matrix
print(covariance_matrix)

# Extract covariance of length and width of petals: petal_cov
petal_cov = covariance_matrix[0,1]

# Print the length/width covariance
print(petal_cov)


[[0.22081633 0.07310204]
 [0.07310204 0.03910612]]
0.07310204081632653

```


####
 Computing the Pearson correlation coefficient




```

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0,1]

# Compute Pearson correlation coefficient for I. versicolor: r
r = pearson_r(versicolor_petal_length , versicolor_petal_width)

# Print the result
print(r)
0.7866680885228169

```




# **3. Thinking probabilistically– Discrete variables**
------------------------------------------------------


###
 Probabilistic logic and statistical inference


####
 What is the goal of statistical inference?



 Why do we do statistical inference?



* To draw probabilistic conclusions about what we might expect if we collected the same data again.
* To draw actionable conclusions from data.
* To draw more general conclusions from relatively few data or observations.


####
 Why do we use the language of probability?



 Why we use probabilistic language in statistical inference?



* Probability provides a measure of uncertainty.
* Data are almost never exactly the same when acquired again, and probability allows us to say how much we expect them to vary.


###
 Random number generators and hacker statistics


###
 Generating random numbers using the np.random module




```python

# Seed the random number generator
np.random.seed(42)

# Initialize random numbers: random_numbers
random_numbers = np.empty(100000)

# Generate random numbers by looping over range(100000)
for i in range(100000):
    random_numbers[i] = np.random.random()

# Plot a histogram
_ = plt.hist(random_numbers)

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture-14.png)

####
 The np.random module and Bernoulli trials




```python

# Seed random number generator
np.random.seed(42)

# Initialize the number of defaults: n_defaults
n_defaults = np.empty(1000)

# Compute the number of defaults
for i in range(1000):
    n_defaults[i] = perform_bernoulli_trials(100, 0.05)

# Plot the histogram with default number of bins; label your axes
_ = plt.hist(n_defaults, normed=True)
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('probability')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture1-13.png)

####
 Will the bank fail?




```python

# Compute ECDF: x, y
x, y = ecdf(n_defaults)

# Plot the ECDF with labeled axes
plt.plot(x, y, marker = '.', linestyle = 'none')
plt.xlabel('number of defaults out of 100')
plt.ylabel('CDF')

# Show the plot
plt.show()

# Compute the number of 100-loan simulations with 10 or more defaults: n_lose_money
n_lose_money = np.sum(n_defaults >= 10)

# Compute and print probability of losing money
print('Probability of losing money =', n_lose_money / len(n_defaults))

Probability of losing money = 0.022


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture2-12.png)

###
 Probability distributions and stories: The Binomial distribution


####
 Sampling out of the Binomial distribution




```python

# Take 10,000 samples out of the binomial distribution: n_defaults
n_defaults = np.random.binomial(n=100, p=0.05, size=10000)

# Compute CDF: x, y
x,y = ecdf(n_defaults)

# Plot the CDF with axis labels
plt.plot(x, y, marker='.', linestyle='none')
plt.xlabel('number of defaults out of 100 loans')
plt.ylabel('CDF')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture3-11.png)

####
 Plotting the Binomial PMF(probability mass function)




```python

# Compute bin edges: bins
bins = np.arange(0, max(n_defaults) + 1.5) - 0.5

# Generate histogram
plt.hist(n_defaults, normed=True, bins=bins)

# Label axes
plt.xlabel('number of defaults out of 100 loans')
plt.ylabel('PMF')


# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture4-11.png)

###
 Poisson processes and the Poisson distribution


####
 Relationship between Binomial and Poisson distributions




```python

# Draw 10,000 samples out of Poisson distribution: samples_poisson
samples_poisson = np.random.poisson(10, size=10000)

# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))

# Specify values of n and p to consider for Binomial: n, p
n = [20, 100, 1000]
p = [0.5, 0.1, 0.01]


# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = np.random.binomial(n[i], p[i], size=10000)

    # Print results
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))


Poisson:      10.0186 3.144813832327758

n = 20 Binom: 9.9637 2.2163443572694206
n = 100 Binom: 9.9947 3.0135812433050484
n = 1000 Binom: 9.9985 3.139378561116833

```


####
 How many no-hitters in a season?



 In baseball, a no-hitter is a game in which a pitcher does not allow the other team to get a hit. This is a rare event, and since the beginning of the so-called modern era of baseball (starting in 1901), there have only been 251 of them through the 2015 season in over 200,000 games. The ECDF of the number of no-hitters in a season is shown to the below. Which probability distribution would be appropriate to describe the number of no-hitters we would expect in a given season?




 Both Binomial and Poisson, though Poisson is easier to model and compute.


 When we have rare events (low p, high n), the Binomial distribution is Poisson. This has a single parameter, the mean number of successes per time interval, in our case the mean number of no-hitters per season.




![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture5-9.png)

###
 Was 2015 anomalous?




```python

# Draw 10,000 samples out of Poisson distribution: n_nohitters
n_nohitters = np.random.poisson(251/115, size=10000)

# Compute number of samples that are seven or greater: n_large
n_large = np.sum(n_nohitters >= 7)

# Compute probability of getting seven or more: p_large
p_large = n_large / 10000

# Print the result
print('Probability of seven or more no-hitters:', p_large)

Probability of seven or more no-hitters: 0.0067

```




# **4. Thinking probabilistically– Continuous variables**
--------------------------------------------------------


###
 Probability density functions


####
 Interpreting PDFs



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture-15.png)


 x is more likely than not greater than 10.




 The probability is given by the
 *area under the PDF*
 , and there is more area to the left of 10 than to the right.



####
 Interpreting CDFs



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture1-14.png)


 Above is the CDF corresponding to the PDF. Using the CDF, what is the probability that x is greater than 10?




 0.25


 The value of the CDF at x = 10 is 0.75, so the probability that x < 10 is 0.75. Thus, the probability that x > 10 is 0.25.



###
 Introduction to the Normal distribution


####
 The Normal PDF(Probability density functions)




```python

# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
samples_std1  = np.random.normal(20, 1, size=100000)
samples_std3  = np.random.normal(20, 3, size=100000)
samples_std10 = np.random.normal(20, 10, size=100000)


# Make histograms
plt.hist(samples_std1, normed=True, histtype='step', bins=100)
plt.hist(samples_std3, normed=True, histtype='step', bins=100)
plt.hist(samples_std10, normed=True, histtype='step', bins=100)


# Make a legend, set limits and show plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
plt.ylim(-0.01, 0.42)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture2-13.png)

####
 The Normal CDF




```python

# Generate CDFs
x_std1, y_std1   = ecdf(samples_std1)
x_std3, y_std3   = ecdf(samples_std3)
x_std10, y_std10 = ecdf(samples_std10)

# Plot CDFs
plt.plot(x_std1, y_std1, marker = '.', linestyle = 'none')
plt.plot(x_std3, y_std3, marker = '.', linestyle = 'none')
plt.plot(x_std10, y_std10, marker = '.', linestyle = 'none')


# Make a legend and show the plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture3-12.png)


 The CDFs all pass through the mean at the 50th percentile; the mean and median of a Normal distribution are equal. The width of the CDF varies with the standard deviation.



###
 The Normal distribution: Properties and warnings


####
 Gauss and the 10 Deutschmark banknote



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture4-12.png)

[source](https://forgottenbucks.com/german-mark/)



 What are the mean and standard deviation, respectively, of the Normal distribution that was on the 10 Deutschmark banknote ?




 mean = 3, std = 1



####
 Are the Belmont Stakes results Normally distributed?



 data souce:
 [Belmont_Stakes](https://en.wikipedia.org/wiki/Belmont_Stakes)





```

belmont_no_outliers
array([148.51, 146.65, 148.52, 150.7 , 150.42, 150.88, 151.57, 147.54,
       149.65, 148.74, 147.86, 148.75, 147.5 , 148.26, 149.71, 146.56,
       151.19, 147.88, 149.16, 148.82, 148.96, 152.02, 146.82, 149.97,
       146.13, 148.1 , 147.2 , 146.  , 146.4 , 148.2 , 149.8 , 147.  ,
       147.2 , 147.8 , 148.2 , 149.  , 149.8 , 148.6 , 146.8 , 149.6 ,
       149.  , 148.2 , 149.2 , 148.  , 150.4 , 148.8 , 147.2 , 148.8 ,
       149.6 , 148.4 , 148.4 , 150.2 , 148.8 , 149.2 , 149.2 , 148.4 ,
       150.2 , 146.6 , 149.8 , 149.  , 150.8 , 148.6 , 150.2 , 149.  ,
       148.6 , 150.2 , 148.2 , 149.4 , 150.8 , 150.2 , 152.2 , 148.2 ,
       149.2 , 151.  , 149.6 , 149.6 , 149.4 , 148.6 , 150.  , 150.6 ,
       149.2 , 152.6 , 152.8 , 149.6 , 151.6 , 152.8 , 153.2 , 152.4 ,
       152.2 ])

```




```python

# Compute mean and standard deviation: mu, sigma
mu = np.mean(belmont_no_outliers)
sigma = np.std(belmont_no_outliers)

# Sample out of a normal distribution with this mu and sigma: samples
samples = np.random.normal(mu, sigma, size=10000)


# Get the CDF of the samples and of the data
x_theor, y_theor = ecdf(samples)
x, y = ecdf(belmont_no_outliers)


# Plot the CDFs and show the plot
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('Belmont winning time (sec.)')
_ = plt.ylabel('CDF')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture5-10.png)

####
 What are the chances of a horse matching or beating Secretariat’s record?




```python

# Take a million samples out of the Normal distribution: samples
samples = np.random.normal(mu, sigma, size=1000000)

# Compute the fraction that are faster than 144 seconds: prob
prob = np.sum(samples < 144) / 1000000

# Print the result
print('Probability of beating Secretariat:', prob)

Probability of beating Secretariat: 0.000635

```


###
 The Exponential distribution


####
 Matching a story and a distribution



 How might we expect the time between Major League no-hitters to be distributed? Be careful here: a few exercises ago, we considered the probability distribution for the number of no-hitters in a season. Now, we are looking at the probability distribution of the
 *time between*
 no hitters.




 Exponential



####
 Waiting for the next Secretariat



 Unfortunately, Justin was not alive when Secretariat ran the Belmont in 1973. Do you think he will get to see a performance like that? To answer this, you are interested in how many years you would expect to wait until you see another performance like Secretariat’s. How is the waiting time until the next performance as good or better than Secretariat’s distributed?




 Exponential: A horse as fast as Secretariat is a rare event, which can be modeled as a Poisson process, and the waiting time between arrivals of a Poisson process is Exponentially distributed.




**The Exponential distribution describes the waiting times between rare events, and Secretariat is
 *rare*
 !**



####
 If you have a story, you can simulate it!




```

def successive_poisson(tau1, tau2, size=1):
    """Compute time for arrival of 2 successive Poisson processes."""
    # Draw samples out of first exponential distribution: t1
    t1 = np.random.exponential(tau1, size)

    # Draw samples out of second exponential distribution: t2
    t2 = np.random.exponential(tau2, size)

    return t1 + t2

```


####
 Distribution of no-hitters and cycles



 In baseball,
 **hitting for the cycle**
 is the accomplishment of one batter hitting a single, a double, a triple, and a home run in the same game.




 Now, you’ll use your sampling function to compute the waiting time to observe a no-hitter and hitting of the cycle. The mean waiting time for a no-hitter is 764 games, and the mean waiting time for hitting the cycle is 715 games.





```python

# Draw samples of waiting times: waiting_times
waiting_times = successive_poisson(764, 715, size=100000)

# Make the histogram
plt.hist(waiting_times, bins=100, normed=True, histtype='step')


# Label axes
plt.xlabel('total waiting time (games)')
plt.ylabel('PDF')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture6-7.png)



```

x, y = ecdf(waiting_times)
plt.plot(x, y)
plt.xlabel('total waiting time (games)')
plt.ylabel('CDF')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-1)/capture7-8.png)

