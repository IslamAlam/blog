---
title: Statistical Thinking in Python (Part 2)
date: 2021-12-07 11:22:12 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Statistical Thinking in Python (Part 2)
==========================================







 Parameter estimation by optimization
--------------------------------------


###
 Optimal parameters


####
 How often do we get no-hitters?



 The number of games played between each no-hitter in the modern era (1901-2015) of Major League Baseball is stored in the array
 `nohitter_times`
 .




 If you assume that no-hitters are described as a Poisson process, then the time between no-hitters is Exponentially distributed. As you have seen, the Exponential distribution has a single parameter, which we will call ττ, the typical interval time. The value of the parameter ττ that makes the exponential distribution best match the data is the mean interval time (where time is in units of number of games) between no-hitters.




 Compute the value of this parameter from the data. Then, use
 `np.random.exponential()`
 to “repeat” the history of Major League Baseball by drawing inter-no-hitter times from an exponential distribution with the ττ you found and plot the histogram as an approximation to the PDF.







```python

# Seed random number generator
np.random.seed(42)

# Compute mean no-hitter time: tau
tau = np.mean(nohitter_times)

# Draw out of an exponential distribution with parameter tau: inter_nohitter_time
inter_nohitter_time = np.random.exponential(tau, 100000)

# Plot the PDF and label axes
_ = plt.hist(inter_nohitter_time,
             bins=50, normed=True, histtype='step')
_ = plt.xlabel('Games between no-hitters')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture-16.png)


 We see the typical shape of the Exponential distribution, going from a maximum at 0 and decaying to the right.



####
 Do the data follow our story?




```python

# Create an ECDF from real data: x, y
x, y = ecdf(nohitter_times)

# Create a CDF from theoretical samples: x_theor, y_theor
x_theor, y_theor = ecdf(inter_nohitter_time)

# Overlay the plots
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')

# Margins and axis labels
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture1-15.png)


 It looks like no-hitters in the modern era of Major League Baseball are Exponentially distributed. Based on the story of the Exponential distribution, this suggests that they are a random process; when a no-hitter will happen is independent of when the last no-hitter was.



####
 How is this parameter optimal?




```python

# Plot the theoretical CDFs
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# Take samples with half tau: samples_half
samples_half = np.random.exponential(tau/2, size=10000)

# Take samples with double tau: samples_double
samples_double = np.random.exponential(2*tau, size=10000)

# Generate CDFs from these samples
x_half, y_half = ecdf(samples_half)
x_double, y_double = ecdf(samples_double)

# Plot these CDFs as lines
_ = plt.plot(x_half, y_half)
_ = plt.plot(x_double, y_double)

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture2-14.png)

 red = half, purple = double



 Notice how the value of tau given by the mean matches the data best. In this way, tau is an optimal parameter.



###
 Linear regression by least squares


####
 EDA of literacy/fertility data




```python

# Plot the illiteracy rate versus fertility
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')

# Set the margins and label axes
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')

# Show the plot
plt.show()

# Show the Pearson correlation coefficient
print(pearson_r(illiteracy, fertility))
0.8041324026815344

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture3-13.png)


 You can see the correlation between illiteracy and fertility by eye, and by the substantial Pearson correlation coefficient of 0.8. It is difficult to resolve in the scatter plot, but there are many points around near-zero illiteracy and about 1.8 children/woman.



####
 Linear regression




```python

# Plot the illiteracy rate versus fertility
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')

# Perform a linear regression using np.polyfit(): a, b
a, b = np.polyfit(illiteracy, fertility, deg=1)

# Print the results to the screen
print('slope =', a, 'children per woman / percent illiterate')
print('intercept =', b, 'children per woman')

# Make theoretical line to plot
x = np.array([0, 100])
y = a * x + b

# Add regression line to your plot
_ = plt.plot(x, y)

# Draw the plot
plt.show()


# slope = 0.04979854809063423 children per woman / percent illiterate
# intercept = 1.888050610636557 children per woman


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture4-13.png)

####
 How is it optimal?




```python

# Specify slopes to consider: a_vals
a_vals = np.linspace(0, 0.1, 200)

# Initialize sum of square of residuals: rss
rss = np.empty_like(a_vals)

# Compute sum of square of residuals for each value of a_vals
for i, a in enumerate(a_vals):
    rss[i] = np.sum((fertility - a*illiteracy - b)**2)

# Plot the RSS
plt.plot(a_vals, rss, '-')
plt.xlabel('slope (children per woman / percent illiterate)')
plt.ylabel('sum of square of residuals')

plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture5-11.png)


 Notice that the minimum on the plot, that is the value of the slope that gives the minimum sum of the square of the residuals, is the same value you got when performing the regression.



###
 The importance of EDA: Anscombe’s quartet



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture6-8.png)

####
 The importance of EDA



 Why should exploratory data analysis be the first step in an analysis of data (after getting your data imported and cleaned, of course)?



* You can be protected from misinterpretation of the type demonstrated by Anscombe’s quartet.
* EDA provides a good starting point for planning the rest of your analysis.
* EDA is not really any more difficult than any of the subsequent analysis, so there is no excuse for not exploring the data.


####
 Linear regression on appropriate Anscombe data




```python

# Perform linear regression: a, b
a, b = np.polyfit(x, y, deg=1)

# Print the slope and intercept
print(a, b)

# Generate theoretical x and y data: x_theor, y_theor
x_theor = np.array([3, 15])
y_theor = a * x_theor + b

# Plot the Anscombe data and theoretical line
_ = plt.plot(x, y, marker = '.', linestyle = 'none')
_ = plt.plot(x_theor, y_theor)

# Label the axes
plt.xlabel('x')
plt.ylabel('y')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture7-9.png)

####
 Linear regression on all Anscombe data




```

for i in range(4):
    plt.subplot(2,2,i+1)

    # plot the scatter plot
    plt.plot(anscombe_x[i], anscombe_y[i], marker = '.', linestyle = 'none')

    # plot the regression line
    a, b = np.polyfit(anscombe_x[i], anscombe_y[i], deg=1)
    x_theor = np.array([np.min(anscombe_x[i]), np.max(anscombe_x[i])])
    y_theor = a * x_theor + b
    plt.plot(x_theor, y_theor)

    # add label
    plt.xlabel('x' + str(i+1))
    plt.ylabel('y' + str(i+1))

plt.show()

# slope1: 0.5000909090909095 intercept: 3.000090909090909
# slope2: 0.5000000000000004 intercept: 3.0009090909090896
# slope3: 0.4997272727272731 intercept: 3.0024545454545453
# slope4: 0.4999090909090908 intercept: 3.0017272727272735

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture8-6.png)

###
 Generating bootstrap replicates


####
 Getting the terminology down



 If we have a data set with n repeated measurements, a
 **bootstrap sample**
 is an array of length n that was drawn from the original data with replacement.




**Bootstrap replicate**
 is a single value of a statistic computed from a bootstrap sample.



####
 Visualizing bootstrap samples

 np.random.choice()




```

for _ in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(rainfall, size=len(rainfall))

    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)

# Compute and plot ECDF from original data
x, y = ecdf(rainfall)
_ = plt.plot(x, y, marker='.')

# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('yearly rainfall (mm)')
_ = plt.ylabel('ECDF')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture9-5.png)


 Bootstrap confidence intervals
--------------------------------


####
 Generating many bootstrap replicates




```

def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates

```


####
 Bootstrap replicates of the mean and the SEM (
 **standard error of the mean**
 )



 In fact, it can be shown theoretically that under not-too-restrictive conditions, the value of the mean will always be Normally distributed. (This does not hold in general, just for the mean and a few other statistics.)




 The standard deviation of this distribution, called the
 **standard error of the mean**
 , or SEM, is given by the standard deviation of the data divided by the square root of the number of data points. I.e., for a data set,
 `sem = np.std(data) / np.sqrt(len(data))`
 . Using hacker statistics, you get this same result without the need to derive it, but you will verify this result from your bootstrap replicates.





```python

# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates = draw_bs_reps(rainfall, np.mean, size=10000)

# Compute and print SEM
sem = np.std(rainfall) / np.sqrt(len(rainfall))
print(sem)

# Compute and print standard deviation of bootstrap replicates
bs_std = np.std(bs_replicates)
print(bs_std)

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('mean annual rainfall (mm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()

# 10.51054915050619
# 10.465927071184412

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture10-5.png)

####
 Confidence intervals of rainfall data



 Use the bootstrap replicates you just generated to compute the 95% confidence interval. That is, give the 2.5th and 97.5th percentile of your bootstrap replicates stored as
 `bs_replicates`
 . What is the 95% confidence interval?





```

np.percentile(bs_replicates,2.5)
779.7699248120301

np.percentile(bs_replicates,97.5)
820.950432330827

```


####
 Bootstrap replicates of other statistics




```

def draw_bs_reps(data, func, size=1):
    return np.array([bootstrap_replicate_1d(data, func) for _ in range(size)])

# Generate 10,000 bootstrap replicates of the variance: bs_replicates
bs_replicates = draw_bs_reps(rainfall, np.var, size=10000)

# Put the variance in units of square centimeters
bs_replicates /= 100

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('variance of annual rainfall (sq. cm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture11-5.png)


 This is not normally distributed, as it has a longer tail to the right. Note that you can also compute a confidence interval on the variance, or any other statistic, using
 `np.percentile()`
 with your bootstrap replicates.



####
 Confidence interval on the rate of no-hitters




```python

# Draw bootstrap replicates of the mean no-hitter time (equal to tau): bs_replicates
bs_replicates = draw_bs_reps(nohitter_times, np.mean, size=10000)

# Compute the 95% confidence interval: conf_int
conf_int = np.percentile(bs_replicates, [2.5, 97.5])

# Print the confidence interval
print('95% confidence interval =', conf_int, 'games')

# Plot the histogram of the replicates
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel(r'$\tau$ (games)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()

# 95% confidence interval = [660.67280876 871.63077689] games

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture12-4.png)


 This gives you an estimate of what the typical time between no-hitters is. It could be anywhere between 660 and 870 games.



###
 Pairs bootstrap


####
 A function to do pairs bootstrap




```

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, deg=1)

    return bs_slope_reps, bs_intercept_reps


```


####
 Pairs bootstrap of literacy/fertility data




```python

# Generate replicates of slope and intercept using pairs bootstrap
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(illiteracy, fertility, size=1000)

# Compute and print 95% CI for slope
print(np.percentile(bs_slope_reps, [2.5, 97.5]))

# Plot the histogram
_ = plt.hist(bs_slope_reps, bins=50, normed=True)
_ = plt.xlabel('slope')
_ = plt.ylabel('PDF')
plt.show()

# [0.04378061 0.0551616 ]

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture13-4.png)

####
 Plotting bootstrap regressions




```python

# Generate array of x-values for bootstrap lines: x
x = np.array([0, 100])

# Plot the bootstrap lines
for i in range(100):
    _ = plt.plot(x,
                 bs_slope_reps[i]*x + bs_intercept_reps[i],
                 linewidth=0.5, alpha=0.2, color='red')

# Plot the data
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')

# Label axes, set the margins, and show the plot
_ = plt.xlabel('illiteracy')
_ = plt.ylabel('fertility')
plt.margins(0.02)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture14-3.png)


 Introduction to hypothesis testing
------------------------------------


###
 Formulating and simulating a hypothesis



**Null hypothesis**
 : another name for the hypothesis you are testing




**Permutation**
 : random reordering of entries in an array



####
 Generating a permutation sample

 np.random.permutation()



 Permutation sampling is a great way to simulate the hypothesis that two variables have identical probability distributions. This is often a hypothesis you want to test, so in this exercise, you will write a function to generate a permutation sample from two data sets.





```

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

```


####
 Visualizing permutation sampling




```

for _ in range(50):
    # Generate permutation samples
    perm_sample_1, perm_sample_2 = permutation_sample(rain_june, rain_november)


    # Compute ECDFs
    x_1, y_1 = ecdf(perm_sample_1)
    x_2, y_2 = ecdf(perm_sample_2)

    # Plot ECDFs of permutation sample
    _ = plt.plot(x_1, y_1, marker='.', linestyle='none',
                 color='red', alpha=0.02)
    _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                 color='blue', alpha=0.02)

# Create and plot ECDFs from original data
x_1, y_1 = ecdf(rain_june)
x_2, y_2 = ecdf(rain_november)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('monthly rainfall (mm)')
_ = plt.ylabel('ECDF')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture1-16.png)


 Notice that the permutation samples ECDFs overlap and give a purple haze. None of the ECDFs from the permutation samples overlap with the observed data, suggesting that the hypothesis is not commensurate with the data. June and November rainfall are not identically distributed.



###
 Test statistics and p-values


####
 Test statistics



 When performing hypothesis tests, your choice of test statistic should be pertinent to the question you are seeking to answer in your hypothesis test.


 The most important thing to consider is:
 **What are you asking?**



####
 What is a p-value?



 The p-value is generally a measure of the probability of observing a test statistic equally or more extreme than the one you observed, given that the null hypothesis is true.



####
 Generating permutation replicates




```python

# In most circumstances, func will be a function you write yourself.
def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates

```


####
 Look before you leap: EDA before hypothesis testing



 Kleinteich and Gorb (
 *Sci. Rep.*
 ,
 **4**
 , 5225, 2014) performed an interesting experiment with South American horned frogs. They held a plate connected to a force transducer, along with a bait fly, in front of them. They then measured the impact force and adhesive force of the frog’s tongue when it struck the target.




 Frog A is an adult and Frog B is a juvenile. The researchers measured the impact force of 20 strikes for each frog. In the next exercise, we will test the hypothesis that the two frogs have the same distribution of impact forces. But, remember, it is important to do EDA first! Let’s make a bee swarm plot for the data. They are stored in a Pandas data frame,
 `df`
 , where column
 `ID`
 is the identity of the frog and column
 `impact_force`
 is the impact force in Newtons (N).





```

df.head()
   ID  impact_force
20  A         1.612
21  A         0.605
22  A         0.327
23  A         0.946
24  A         0.541

```




```python

# Make bee swarm plot
_ = sns.swarmplot(x='ID', y='impact_force', data=df)

# Label axes
_ = plt.xlabel('frog')
_ = plt.ylabel('impact force (N)')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture2-15.png)


 Eyeballing it, it does not look like they come from the same distribution. Frog A, the adult, has three or four very hard strikes, and Frog B, the juvenile, has a couple weak ones. However, it is possible that with only 20 samples it might be too difficult to tell if they have difference distributions, so we should proceed with the hypothesis test.



####
 Permutation test on frog data



 The average strike force of Frog A was 0.71 Newtons (N), and that of Frog B was 0.42 N for a difference of 0.29 N. It is possible the frogs strike with the same force and this observed difference was by chance. You will compute the probability of getting at least a 0.29 N difference in mean strike force under the hypothesis that the distributions of strike forces for the two frogs are identical. We use a permutation test with a test statistic of the difference of means to test this hypothesis.





```

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff

# Compute difference of mean impact force from experiment: empirical_diff_means
empirical_diff_means = diff_of_means(force_a, force_b)

# Draw 10,000 permutation replicates: perm_replicates
perm_replicates = draw_perm_reps(force_a, force_b,
                                 diff_of_means, size=10000)

# Compute p-value: p
p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

# Print the result
print('p-value =', p)

# p-value = 0.0063
# p-value = 0.63%

```



 The p-value tells you that there is about a 0.6% chance that you would get the difference of means observed in the experiment if frogs were exactly the same.




 A p-value below 0.01 is typically said to be “statistically significant,” but: warning! warning! warning! You have computed a p-value; it is a number. I encourage you not to distill it to a yes-or-no phrase. p = 0.006 and p = 0.000000006 are both said to be “statistically significant,” but they are definitely not the same!



###
 Bootstrap hypothesis tests



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture3-14.png)

####
 A one-sample bootstrap hypothesis test



 Another juvenile frog was studied, Frog C, and you want to see if Frog B and Frog C have similar impact forces. Unfortunately, you do not have Frog C’s impact forces available, but you know they have a mean of 0.55 N. Because you don’t have the original data, you cannot do a permutation test, and you cannot assess the hypothesis that the forces from Frog B and Frog C come from the same distribution. You will therefore test another, less restrictive hypothesis: The mean strike force of Frog B is equal to that of Frog C.




 To set up the bootstrap hypothesis test, you will take the mean as our test statistic. Remember, your goal is to calculate the probability of getting a mean impact force less than or equal to what was observed for Frog B
 *if the hypothesis that the true mean of Frog B’s impact forces is equal to that of Frog C is true*
 . You first translate all of the data of Frog B such that the mean is 0.55 N. This involves adding the mean force of Frog C and subtracting the mean force of Frog B from each measurement of Frog B. This leaves other properties of Frog B’s distribution, such as the variance, unchanged.





```python

# Make an array of translated impact forces: translated_force_b
translated_force_b = force_b - np.mean(force_b) + 0.55

# Take bootstrap replicates of Frog B's translated impact forces: bs_replicates
bs_replicates = draw_bs_reps(translated_force_b, np.mean, 10000)

# Compute fraction of replicates that are less than the observed Frog B force: p
p = np.sum(bs_replicates <= np.mean(force_b)) / 10000

# Print the p-value
print('p = ', p)

# p =  0.0046
# p = 0.46%

```



 The low p-value suggests that the null hypothesis that Frog B and Frog C have the same mean impact force is false.



####
 A two-sample bootstrap hypothesis test for difference of means



 We now want to test the hypothesis that Frog A and Frog B have the same mean impact force, but not necessarily the same distribution, which is also impossible with a permutation test.




 To do the two-sample bootstrap test, we shift
 *both*
 arrays to have the same mean, since we are simulating the hypothesis that their means are, in fact, equal. We then draw bootstrap samples out of the shifted arrays and compute the difference in means. This constitutes a bootstrap replicate, and we generate many of them. The p-value is the fraction of replicates with a difference in means greater than or equal to what was observed.





```python

# Compute mean of all forces: mean_force
mean_force = np.mean(forces_concat)

# Generate shifted arrays
force_a_shifted = force_a - np.mean(force_a) + mean_force
force_b_shifted = force_b - np.mean(force_b) + mean_force

# Compute 10,000 bootstrap replicates from shifted arrays
bs_replicates_a = draw_bs_reps(force_a_shifted, np.mean, size=10000)
bs_replicates_b = draw_bs_reps(force_b_shifted, np.mean, size=10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_replicates_a - bs_replicates_b

# Compute and print p-value: p
p = np.sum(bs_replicates >= empirical_diff_means) / 10000
print('p-value =', p)

# p-value = 0.0043
# p-value = 0.43%

```



 You got a similar result as when you did the permutation test. Nonetheless, remember that it is important to carefully think about what question you want to ask. Are you only interested in the mean impact force, or in the distribution of impact forces?




 Hypothesis test examples
--------------------------


###
 A/B testing


####
 The vote for the Civil Rights Act in 1964



 The Civil Rights Act of 1964 was one of the most important pieces of legislation ever passed in the USA. Excluding “present” and “abstain” votes, 153 House Democrats and 136 Republicans voted yea. However, 91 Democrats and 35 Republicans voted nay. Did party affiliation make a difference in the vote?




 To answer this question, you will evaluate the hypothesis that the party of a House member has no bearing on his or her vote. You will use the fraction of Democrats voting in favor as your test statistic and evaluate the probability of observing a fraction of Democrats voting in favor at least as small as the observed fraction of 153/244. (That’s right, at least as
 *small*
 as. In 1964, it was the
 *Democrats*
 who were less progressive on civil rights issues.) To do this, permute the party labels of the House voters and then arbitrarily divide them into “Democrats” and “Republicans” and compute the fraction of Democrats voting yea.





```python

# Construct arrays of data: dems, reps
dems = np.array([True] * 153 + [False] * 91)
reps = np.array([True] * 136 + [False] * 35)

def frac_yea_dems(dems, reps):
    """Compute fraction of Democrat yea votes."""
    frac = np.sum(dems) / len(dems)
    return frac

# Acquire permutation samples: perm_replicates
perm_replicates = draw_perm_reps(dems, reps, frac_yea_dems, size=10000)

# Compute and print p-value: p
p = np.sum(perm_replicates <= 153/244) / len(perm_replicates)
print('p-value =', p)

# p-value = 0.0002
# p-value = 0.02%

```



 This small p-value suggests that party identity had a lot to do with the voting. Importantly, the South had a higher fraction of Democrat representatives, and consequently also a more racist bias.



####
 What is equivalent?



 You have experience matching a stories to probability distributions. Similarly, you use the same procedure for two different A/B tests if their stories match. In the Civil Rights Act example you just did, you performed an A/B test on voting data, which has a Yes/No type of outcome for each subject (in that case, a voter). Which of the following situations involving testing by a web-based company has an equivalent set up for an A/B test as the one you just did with the Civil Rights Act of 1964?




 You measure the number of people who click on an ad on your company’s website before and after changing its color.




 The “Democrats” are those who view the ad before the color change, and the “Republicans” are those who view it after.



####
 A time-on-website analog



 It turns out that you already did a hypothesis test analogous to an A/B test where you are interested in how much time is spent on the website before and after an ad campaign. The frog tongue force (a continuous quantity like time on the website) is an analog. “Before” = Frog A and “after” = Frog B. Let’s practice this again with something that actually is a before/after scenario.




 We return to the no-hitter data set. In 1920, Major League Baseball implemented important rule changes that ended the so-called dead ball era. Importantly, the pitcher was no longer allowed to spit on or scuff the ball, an activity that greatly favors pitchers. In this problem you will perform an A/B test to determine if these rule changes resulted in a slower rate of no-hitters (i.e., longer average time between no-hitters) using the difference in mean inter-no-hitter time as your test statistic. The inter-no-hitter times for the respective eras are stored in the arrays
 `nht_dead`
 and
 `nht_live`
 , where “nht” is meant to stand for “no-hitter time.”





```

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates



# Compute the observed difference in mean inter-no-hitter times: nht_diff_obs
nht_diff_obs = diff_of_means(nht_dead, nht_live)

# Acquire 10,000 permutation replicates of difference in mean no-hitter time: perm_replicates
perm_replicates = draw_perm_reps(nht_dead, nht_live, diff_of_means, size=10000)

# Compute and print the p-value: p
p = np.sum(perm_replicates <= nht_diff_obs) /len(perm_replicates)
print('p-val =', p)

p-val = 0.0001

```



 Your p-value is 0.0001, which means that only one out of your 10,000 replicates had a result as extreme as the actual difference between the dead ball and live ball eras. This suggests strong statistical significance. Watch out, though, you could very well have gotten zero replicates that were as extreme as the observed value. This just means that the p-value is quite small, almost certainly smaller than 0.001.



####
 What should you have done first?



 That was a nice hypothesis test you just did to check out whether the rule changes in 1920 changed the rate of no-hitters. But what
 *should*
 you have done with the data first?




 Performed EDA, perhaps plotting the ECDFs of inter-no-hitter times in the dead ball and live ball eras.




 Always a good idea to do first! I encourage you to go ahead and plot the ECDFs right now. You will see by eye that the null hypothesis that the distributions are the same is almost certainly not true.





```python

# Create and plot ECDFs
x_1, y_1 = ecdf(nht_dead)
x_2, y_2 = ecdf(nht_live)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('Games between no-hitters')
_ = plt.ylabel('ECDF')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture1-17.png)

###
 Test of correlation



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture2-16.png)

####
 Simulating a null hypothesis concerning correlation



 The observed correlation between female illiteracy and fertility in the data set of 162 countries may just be by chance; the fertility of a given country may actually be totally independent of its illiteracy. You will test this null hypothesis in the next exercise.




 To do the test, you need to simulate the data assuming the null hypothesis is true. Of the following choices, which is the best way to to do it?




 Answer: Do a permutation test: Permute the illiteracy values but leave the fertility values fixed to generate a new set of (illiteracy, fertility) data.




 This exactly simulates the null hypothesis and does so more efficiently than the last option. It is exact because it uses all data and eliminates any correlation because which illiteracy value pairs to which fertility value is shuffled.




 Last option: Do a permutation test: Permute both the illiteracy and fertility values to generate a new set of (illiteracy, fertility data). This exactly simulates the null hypothesis and does so more efficiently than the last option. It is exact because it uses all data and eliminates any correlation because which illiteracy value pairs to which fertility value is shuffled.



####
 Hypothesis test on Pearson correlation



 The observed correlation between female illiteracy and fertility may just be by chance; the fertility of a given country may actually be totally independent of its illiteracy. You will test this hypothesis. To do so, permute the illiteracy values but leave the fertility values fixed. This simulates the hypothesis that they are totally independent of each other. For each permutation, compute the Pearson correlation coefficient and assess how many of your permutation replicates have a Pearson correlation coefficient greater than the observed one.





```

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0,1]

# Compute observed correlation: r_obs
r_obs = pearson_r(illiteracy, fertility)

# r_obs = 0.8041324026815344

# Initialize permutation replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute illiteracy measurments: illiteracy_permuted
    illiteracy_permuted = np.random.permutation(illiteracy)

    # Compute Pearson correlation
    perm_replicates[i] = pearson_r(illiteracy_permuted, fertility)

# Compute p-value: p
p = np.sum(perm_replicates >= r_obs) /len(perm_replicates)
print('p-val =', p)

# p-val = 0.0

```



 You got a p-value of zero. In hacker statistics, this means that your p-value is very low, since you never got a single replicate in the 10,000 you took that had a Pearson correlation greater than the observed one. You could try increasing the number of replicates you take to continue to move the upper bound on your p-value lower and lower.



####
 Do neonicotinoid insecticides have unintended consequences?



 As a final exercise in hypothesis testing before we put everything together in our case study in the next chapter, you will investigate the effects of neonicotinoid insecticides on bee reproduction. These insecticides are very widely used in the United States to combat aphids and other pests that damage plants.




 In a recent study, Straub, et al. (
 [*Proc. Roy. Soc. B*
 , 2016](http://dx.doi.org/10.1098/rspb.2016.0506)
 ) investigated the effects of neonicotinoids on the sperm of pollinating bees. In this and the next exercise, you will study how the pesticide treatment affected the count of live sperm per half milliliter of semen.




 First, we will do EDA, as usual. Plot ECDFs of the alive sperm count for untreated bees (stored in the Numpy array
 `control`
 ) and bees treated with pesticide (stored in the Numpy array
 `treated`
 ).





```python

# Compute x,y values for ECDFs
x_control, y_control = ecdf(control)
x_treated, y_treated = ecdf(treated)

# Plot the ECDFs
plt.plot(x_control, y_control, marker='.', linestyle='none')
plt.plot(x_treated, y_treated, marker='.', linestyle='none')

# Set the margins
plt.margins(0.02)

# Add a legend
plt.legend(('control', 'treated'), loc='lower right')

# Label axes and show plot
plt.xlabel('millions of alive sperm per mL')
plt.ylabel('ECDF')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture3-15.png)


 The ECDFs show a pretty clear difference between the treatment and control; treated bees have fewer alive sperm. Let’s now do a hypothesis test in the next exercise.



####
 Bootstrap hypothesis test on bee sperm counts



 Now, you will test the following hypothesis:




**On average, male bees treated with neonicotinoid insecticide have the same number of active sperm per milliliter of semen than do untreated male bees.**




 You will use the difference of means as your test statistic.





```

def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates


# Compute the difference in mean sperm count: diff_means
diff_means = np.mean(control) - np.mean(treated)

# Compute mean of pooled data: mean_count
mean_count = np.mean(np.concatenate((control, treated)))

# Generate shifted data sets
control_shifted = control - np.mean(control) + mean_count
treated_shifted = treated - np.mean(treated) + mean_count

# Generate bootstrap replicates
bs_reps_control = draw_bs_reps(control_shifted,
                       np.mean, size=10000)
bs_reps_treated = draw_bs_reps(treated_shifted,
                       np.mean, size=10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_reps_control - bs_reps_treated

# Compute and print p-value: p
p = np.sum(bs_replicates >= np.mean(control) - np.mean(treated)) \
            / len(bs_replicates)
print('p-value =', p)

# p-value = 0.0

```



 The p-value is small, most likely less than 0.0001, since you never saw a bootstrap replicated with a difference of means at least as extreme as what was observed. In fact, when I did the calculation with 10 million replicates, I got a p-value of
 `2e-05`




 Putting it all together: a case study
---------------------------------------


###
 Finch beaks and the need for statistics



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture4-14.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture5-12.png)

####
 EDA of beak depths of Darwin’s finches



 For your first foray into the Darwin finch data, you will study how the beak depth (the distance, top to bottom, of a closed beak) of the finch species
 *Geospiza scandens*
 has changed over time. The Grants have noticed some changes of beak geometry depending on the types of seeds available on the island, and they also noticed that there was some interbreeding with another major species on Daphne Major,
 *Geospiza fortis*
 . These effects can lead to changes in the species over time.




 In the next few problems, you will look at the beak depth of
 *G. scandens*
 on Daphne Major in 1975 and in 2012. To start with, let’s plot all of the beak depth measurements in 1975 and 2012 in a bee swarm plot.




 The data are stored in a pandas DataFrame called
 `df`
 with columns
 `'year'`
 and
 `'beak_depth'`
 . The units of beak depth are millimeters (mm).





```

df.head()
   beak_depth  year
0         8.4  1975
1         8.8  1975
2         8.4  1975
3         8.0  1975
4         7.9  1975

```




```python

# Create bee swarm plot
_ = sns.swarmplot('year', 'beak_depth', data=df)

# Label the axes
_ = plt.xlabel('year')
_ = plt.ylabel('beak depth (mm)')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture6-9.png)


 It is kind of hard to see if there is a clear difference between the 1975 and 2012 data set. Eyeballing it, it appears as though the mean of the 2012 data set might be slightly higher, and it might have a bigger variance.



####
 ECDFs of beak depths



 While bee swarm plots are useful, we found that ECDFs are often even better when doing EDA. Plot the ECDFs for the 1975 and 2012 beak depth measurements on the same plot.





```python

# Compute ECDFs
x_1975, y_1975 = ecdf(bd_1975)
x_2012, y_2012 = ecdf(bd_2012)

# Plot the ECDFs
_ = plt.plot(x_1975, y_1975, marker='.', linestyle='none')
_ = plt.plot(x_2012, y_2012, marker='.', linestyle='none')

# Set margins
plt.margins(0.02)

# Add axis labels and legend
_ = plt.xlabel('beak depth (mm)')
_ = plt.ylabel('ECDF')
_ = plt.legend(('1975', '2012'), loc='lower right')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture7-10.png)


 The differences are much clearer in the ECDF. The mean is larger in the 2012 data, and the variance does appear larger as well.



####
 Parameter estimates of beak depths



 Estimate the
 *difference*
 of the mean beak depth of the
 *G. scandens*
 samples from 1975 and 2012 and report a 95% confidence interval.





```

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates


# Compute the difference of the sample means: mean_diff
mean_diff = np.mean(bd_2012) - np.mean(bd_1975)

# Get bootstrap replicates of means
bs_replicates_1975 = draw_bs_reps(bd_1975, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(bd_2012, np.mean, size=10000)

# Compute samples of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute 95% confidence interval: conf_int
conf_int = np.percentile(bs_diff_replicates, [2.5, 97.5])

# Print the results
print('difference of means =', mean_diff, 'mm')
print('95% confidence interval =', conf_int, 'mm')

# difference of means = 0.22622047244094645 mm
# 95% confidence interval = [0.05633521 0.39190544] mm

```


####
 Hypothesis test: Are beaks deeper in 2012?



 Your plot of the ECDF and determination of the confidence interval make it pretty clear that the beaks of
 *G. scandens*
 on Daphne Major have gotten deeper. But is it possible that this effect is just due to random chance? In other words, what is the probability that we would get the observed difference in mean beak depth if the means were the same?




 Be careful! The hypothesis we are testing is
 *not*
 that the beak depths come from the same distribution. For that we could use a permutation test.
 **The hypothesis is that the means are equal.**
 To perform this hypothesis test, we need to shift the two data sets so that they have the same mean and then use bootstrap sampling to compute the difference of means.





```python

# Compute mean of combined data set: combined_mean
combined_mean = np.mean(np.concatenate((bd_1975, bd_2012)))

# Shift the samples
# why shift the mean?
# to make np.mean(bd_1975_shifted) - np.mean(bd_2012_shifted) = 0 #1
# why make #1 = 0?
# because our hypothesis is "beak depth are the same in 1975 and 2012"
bd_1975_shifted = bd_1975 - np.mean(bd_1975) + combined_mean
bd_2012_shifted = bd_2012 - np.mean(bd_2012) + combined_mean

# Get bootstrap replicates of shifted data sets
bs_replicates_1975 = draw_bs_reps(bd_1975_shifted, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(bd_2012_shifted, np.mean, size=10000)

# Compute replicates of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute the p-value
p = np.sum(bs_diff_replicates >= mean_diff) / len(bs_diff_replicates)

# Print p-value
print('p =', p)

# p = 0.0034
# p = 0.34%

```



 We get a p-value of 0.0034, which suggests that there is a statistically significant difference. But remember: it is very important to know how different they are! In the previous exercise, you got a difference of 0.2 mm between the means. You should combine this with the statistical significance. Changing by 0.2 mm in 37 years is substantial by evolutionary standards. If it kept changing at that rate, the beak depth would double in only 400 years.



###
 Variation of beak shapes


####
 EDA of beak length and depth



 The beak length data are stored as
 `bl_1975`
 and
 `bl_2012`
 , again with units of millimeters (mm). You still have the beak depth data stored in
 `bd_1975`
 and
 `bd_2012`
 . Make scatter plots of beak depth (y-axis) versus beak length (x-axis) for the 1975 and 2012 specimens.





```python

# Make scatter plot of 1975 data
_ = plt.plot(bl_1975, bd_1975, marker='.',
             linestyle='None', color='blue', alpha=0.5)

# Make scatter plot of 2012 data
_ = plt.plot(bl_2012, bd_2012, marker='.',
            linestyle='None', color='red', alpha=0.5)

# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture8-7.png)


 In looking at the plot, we see that beaks got deeper (the red points are higher up in the y-direction), but not really longer. If anything, they got a bit shorter, since the red dots are to the left of the blue dots. So, it does not look like the beaks kept the same shape; they became shorter and deeper.



####
 Linear regressions



 Perform a linear regression for both the 1975 and 2012 data. Then, perform pairs bootstrap estimates for the regression parameters. Report 95% confidence intervals on the slope and intercept of the regression line.





```

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, deg=1)

    return bs_slope_reps, bs_intercept_reps


# Compute the linear regressions
slope_1975, intercept_1975 = np.polyfit(bl_1975, bd_1975, deg=1)
slope_2012, intercept_2012 = np.polyfit(bl_2012, bd_2012, deg=1)

# Perform pairs bootstrap for the linear regressions
bs_slope_reps_1975, bs_intercept_reps_1975 = draw_bs_pairs_linreg(bl_1975, bd_1975, size=1000)
bs_slope_reps_2012, bs_intercept_reps_2012 = draw_bs_pairs_linreg(bl_2012, bd_2012, size=1000)

# Compute confidence intervals of slopes
slope_conf_int_1975 = np.percentile(bs_slope_reps_1975, [2.5, 97.5])
slope_conf_int_2012 = np.percentile(bs_slope_reps_2012, [2.5, 97.5])
intercept_conf_int_1975 = np.percentile(bs_intercept_reps_1975, [2.5, 97.5])
intercept_conf_int_2012 = np.percentile(bs_intercept_reps_2012, [2.5, 97.5])


# Print the results
print('1975: slope =', slope_1975,
      'conf int =', slope_conf_int_1975)
print('1975: intercept =', intercept_1975,
      'conf int =', intercept_conf_int_1975)
print('2012: slope =', slope_2012,
      'conf int =', slope_conf_int_2012)
print('2012: intercept =', intercept_2012,
      'conf int =', intercept_conf_int_2012)

#   1975: slope = 0.4652051691605937 conf int = [0.33851226 0.59306491]
#   1975: intercept = 2.3908752365842263 conf int = [0.64892945 4.18037063]
#   2012: slope = 0.462630358835313 conf int = [0.33137479 0.60695527]
#   2012: intercept = 2.977247498236019 conf int = [1.06792753 4.70599387]

```



 It looks like they have the same slope, but different intercepts.



####
 Displaying the linear regression results




```python

# Make scatter plot of 1975 data
_ = plt.plot(bl_1975, bd_1975, marker='.',
             linestyle='none', color='blue', alpha=0.5)

# Make scatter plot of 2012 data
_ = plt.plot(bl_2012, bd_2012, marker='.',
             linestyle='none', color='red', alpha=0.5)

# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')

# Generate x-values for bootstrap lines: x
x = np.array([10, 17])

# Plot the bootstrap lines
for i in range(100):
    plt.plot(x, bs_slope_reps_1975[i] * x + bs_intercept_reps_1975[i],
             linewidth=0.5, alpha=0.2, color='blue')
    plt.plot(x, bs_slope_reps_2012[i] * x + bs_intercept_reps_2012[i],
             linewidth=0.5, alpha=0.2, color='red')

# Draw the plot again
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture9-6.png)

####
 Beak length to depth ratio



 The linear regressions showed interesting information about the beak geometry. The slope was the same in 1975 and 2012, suggesting that for every millimeter gained in beak length, the birds gained about half a millimeter in depth in both years. However, if we are interested in the shape of the beak, we want to compare the
 *ratio*
 of beak length to beak depth. Let’s make that comparison.





```python

# Compute length-to-depth ratios
ratio_1975 = bl_1975 / bd_1975
ratio_2012 = bl_2012 / bd_2012

# Compute means
mean_ratio_1975 = np.mean(ratio_1975)
mean_ratio_2012 = np.mean(ratio_2012)

# Generate bootstrap replicates of the means
bs_replicates_1975 = draw_bs_reps(ratio_1975, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(ratio_2012, np.mean, size=10000)

# Compute the 99% confidence intervals
conf_int_1975 = np.percentile(bs_replicates_1975, [0.5, 99.5])
conf_int_2012 = np.percentile(bs_replicates_2012, [0.5, 99.5])

# Print the results
print('1975: mean ratio =', mean_ratio_1975,
      'conf int =', conf_int_1975)
print('2012: mean ratio =', mean_ratio_2012,
      'conf int =', conf_int_2012)

# 1975: mean ratio = 1.5788823771858533 conf int = [1.55668803 1.60073509]
# 2012: mean ratio = 1.4658342276847767 conf int = [1.44363932 1.48729149]

```


####
 How different is the ratio?



 In the previous exercise, you computed the mean beak length to depth ratio with 99% confidence intervals for 1975 and for 2012. The results of that calculation are shown graphically in the plot accompanying this problem. In addition to these results, what would you say about the ratio of beak length to depth?




![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture10-6.png)


 The mean beak length-to-depth ratio decreased by about 0.1, or 7%, from 1975 to 2012. The 99% confidence intervals are not even close to overlapping, so this is a real change. The beak shape changed.



###
 Calculation of heritability


####
 EDA of heritability



 The array
 `bd_parent_scandens`
 contains the average beak depth (in mm) of two parents of the species
 `G. scandens`
 . The array
 `bd_offspring_scandens`
 contains the average beak depth of the offspring of the respective parents. The arrays
 `bd_parent_fortis`
 and
 `bd_offspring_fortis`
 contain the same information about measurements from
 *G. fortis*
 birds.




 Make a scatter plot of the average offspring beak depth (y-axis) versus average parental beak depth (x-axis) for both species. Use the
 `alpha=0.5`
 keyword argument to help you see overlapping points.





```python

# Make scatter plots
_ = plt.plot(bd_parent_fortis, bd_offspring_fortis,
             marker='.', linestyle='none', color='blue', alpha=0.5)
_ = plt.plot(bd_parent_scandens, bd_offspring_scandens,
             marker='.', linestyle='none', color='red', alpha=0.5)

# Label axes
_ = plt.xlabel('parental beak depth (mm)')
_ = plt.ylabel('offspring beak depth (mm)')

# Add legend
_ = plt.legend(('G. fortis', 'G. scandens'), loc='lower right')

# Show plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture11-6.png)


 It appears as though there is a stronger correlation in
 *G. fortis*
 than than in
 *G. scandens*
 . This suggests that beak depth is more strongly inherited in
 *G. fortis*
 . We’ll quantify this correlation next.



####
 Correlation of offspring and parental data



 In an effort to quantify the correlation between offspring and parent beak depths, we would like to compute statistics, such as the Pearson correlation coefficient, between parents and offspring. To get confidence intervals on this, we need to do a pairs bootstrap.




 You have
 [already written](https://campus.datacamp.com/courses/statistical-thinking-in-python-part-2/bootstrap-confidence-intervals?ex=12)
 a function to do pairs bootstrap to get estimates for parameters derived from linear regression. Your task in this exercise is to make a new function with call signature
 `draw_bs_pairs(x, y, func, size=1)`
 that performs pairs bootstrap and computes a single statistic on pairs samples defined. The statistic of interest is computed by calling
 `func(bs_x, bs_y)`
 . In the next exercise, you will use
 `pearson_r`
 for
 `func`
 .





```

def draw_bs_pairs(x, y, func, size=1):
    """Perform pairs bootstrap for a single statistic."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_replicates[i] = func(bs_x, bs_y)

    return bs_replicates


```


####
 Pearson correlation of offspring and parental data



 The Pearson correlation coefficient seems like a useful measure of how strongly the beak depth of parents are inherited by their offspring. Compute the Pearson correlation coefficient between parental and offspring beak depths for
 *G. scandens*
 . Do the same for
 *G. fortis*
 . Then, use the function you wrote in the last exercise to compute a 95% confidence interval using pairs bootstrap.





```

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0,1]


# Compute the Pearson correlation coefficients
r_scandens = pearson_r(bd_parent_scandens, bd_offspring_scandens)
r_fortis = pearson_r(bd_parent_fortis, bd_offspring_fortis)

# Acquire 1000 bootstrap replicates of Pearson r
bs_replicates_scandens = draw_bs_pairs(bd_parent_scandens, bd_offspring_scandens, pearson_r, size=1000)

bs_replicates_fortis = draw_bs_pairs(bd_parent_fortis, bd_offspring_fortis, pearson_r, size=1000)


# Compute 95% confidence intervals
conf_int_scandens = np.percentile(bs_replicates_scandens, [2.5, 97.5])
conf_int_fortis = np.percentile(bs_replicates_fortis, [2.5, 97.5])

# Print results
print('G. scandens:', r_scandens, conf_int_scandens)
print('G. fortis:', r_fortis, conf_int_fortis)

#    G. scandens: 0.4117063629401258 [0.26564228 0.54388972]
#    G. fortis: 0.7283412395518487 [0.6694112  0.77840616]

```



 It is clear from the confidence intervals that beak depth of the offspring of
 *G. fortis*
 parents is more strongly correlated with their offspring than their
 *G. scandens*
 counterparts.



####
 Measuring heritability



 Remember that the Pearson correlation coefficient is the ratio of the covariance to the geometric mean of the variances of the two data sets. This is a measure of the correlation between parents and offspring, but might not be the best estimate of heritability. If we stop and think, it makes more sense to define heritability as the ratio of the covariance between parent and offspring to the
 *variance of the parents alone*
 . In this exercise, you will estimate the heritability and perform a pairs bootstrap calculation to get the 95% confidence interval.




 This exercise highlights a very important point. Statistical inference (and data analysis in general) is not a plug-n-chug enterprise. You need to think carefully about the questions you are seeking to answer with your data and analyze them appropriately. If you are interested in how heritable traits are, the quantity we defined as the heritability is more apt than the off-the-shelf statistic, the Pearson correlation coefficient.





```

def heritability(parents, offspring):
    """Compute the heritability from parent and offspring samples."""
    covariance_matrix = np.cov(parents, offspring)
    return covariance_matrix[0,1] / covariance_matrix[0,0]

# Compute the heritability
heritability_scandens = heritability(bd_parent_scandens, bd_offspring_scandens)
heritability_fortis = heritability(bd_parent_fortis, bd_offspring_fortis)

# Acquire 1000 bootstrap replicates of heritability
replicates_scandens = draw_bs_pairs(
        bd_parent_scandens, bd_offspring_scandens, heritability, size=1000)

replicates_fortis = draw_bs_pairs(
        bd_parent_fortis, bd_offspring_fortis, heritability, size=1000)


# Compute 95% confidence intervals
conf_int_scandens = np.percentile(replicates_scandens, [2.5, 97.5])
conf_int_fortis = np.percentile(replicates_fortis, [2.5, 97.5])

# Print results
print('G. scandens:', heritability_scandens, conf_int_scandens)
print('G. fortis:', heritability_fortis, conf_int_fortis)


#   G. scandens: 0.5485340868685982 [0.34395487 0.75638267]
#   G. fortis: 0.7229051911438159 [0.64655013 0.79688342]

```



 Here again, we see that
 *G. fortis*
 has stronger heritability than
 *G. scandens*
 . This suggests that the traits of
 *G. fortis*
 may be strongly incorporated into
 *G. scandens*
 by introgressive hybridization.



####
 Is beak depth heritable at all in G. scandens?



 The heritability of beak depth in
 *G. scandens*
 seems low. It could be that this observed heritability was just achieved by chance and
 **beak depth is actually not really heritable in the species**
 . You will test that hypothesis here. To do this, you will do a pairs permutation test.





```python

# Initialize array of replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute parent beak depths
    bd_parent_permuted = np.random.permutation(bd_parent_scandens)
    perm_replicates[i] = heritability(bd_parent_permuted,
                                      bd_offspring_scandens)

# Compute p-value: p
p = np.sum(perm_replicates >= heritability_scandens) / len(perm_replicates)

# Print the p-value
print('p-val =', p)

# p-val = 0.0

```



 You get a p-value of zero, which means that none of the 10,000 permutation pairs replicates you drew had a heritability high enough to match that which was observed. This strongly suggests that beak depth is heritable in
 *G. scandens*
 , just not as much as in
 *G. fortis*
 . If you like, you can plot a histogram of the heritability replicates to get a feel for how extreme of a value of heritability you might expect by chance.





```

plt.hist(perm_replicates)
plt.axvline(x=heritability_scandens, color = 'red')
plt.text(heritability_scandens, 1500, 'heritability_scandens', ha='center', va='center',rotation='vertical', backgroundcolor='white')
plt.show()

```




 Parameter estimation by optimization
--------------------------------------


###
 Optimal parameters


####
 How often do we get no-hitters?



 The number of games played between each no-hitter in the modern era (1901-2015) of Major League Baseball is stored in the array
 `nohitter_times`
 .




 If you assume that no-hitters are described as a Poisson process, then the time between no-hitters is Exponentially distributed. As you have seen, the Exponential distribution has a single parameter, which we will call ττ, the typical interval time. The value of the parameter ττ that makes the exponential distribution best match the data is the mean interval time (where time is in units of number of games) between no-hitters.




 Compute the value of this parameter from the data. Then, use
 `np.random.exponential()`
 to “repeat” the history of Major League Baseball by drawing inter-no-hitter times from an exponential distribution with the ττ you found and plot the histogram as an approximation to the PDF.







```python

# Seed random number generator
np.random.seed(42)

# Compute mean no-hitter time: tau
tau = np.mean(nohitter_times)

# Draw out of an exponential distribution with parameter tau: inter_nohitter_time
inter_nohitter_time = np.random.exponential(tau, 100000)

# Plot the PDF and label axes
_ = plt.hist(inter_nohitter_time,
             bins=50, normed=True, histtype='step')
_ = plt.xlabel('Games between no-hitters')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture-16.png)


 We see the typical shape of the Exponential distribution, going from a maximum at 0 and decaying to the right.



####
 Do the data follow our story?




```python

# Create an ECDF from real data: x, y
x, y = ecdf(nohitter_times)

# Create a CDF from theoretical samples: x_theor, y_theor
x_theor, y_theor = ecdf(inter_nohitter_time)

# Overlay the plots
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')

# Margins and axis labels
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture1-15.png)


 It looks like no-hitters in the modern era of Major League Baseball are Exponentially distributed. Based on the story of the Exponential distribution, this suggests that they are a random process; when a no-hitter will happen is independent of when the last no-hitter was.



####
 How is this parameter optimal?




```python

# Plot the theoretical CDFs
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# Take samples with half tau: samples_half
samples_half = np.random.exponential(tau/2, size=10000)

# Take samples with double tau: samples_double
samples_double = np.random.exponential(2*tau, size=10000)

# Generate CDFs from these samples
x_half, y_half = ecdf(samples_half)
x_double, y_double = ecdf(samples_double)

# Plot these CDFs as lines
_ = plt.plot(x_half, y_half)
_ = plt.plot(x_double, y_double)

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture2-14.png)

 red = half, purple = double



 Notice how the value of tau given by the mean matches the data best. In this way, tau is an optimal parameter.



###
 Linear regression by least squares


####
 EDA of literacy/fertility data




```python

# Plot the illiteracy rate versus fertility
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')

# Set the margins and label axes
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')

# Show the plot
plt.show()

# Show the Pearson correlation coefficient
print(pearson_r(illiteracy, fertility))
0.8041324026815344

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture3-13.png)


 You can see the correlation between illiteracy and fertility by eye, and by the substantial Pearson correlation coefficient of 0.8. It is difficult to resolve in the scatter plot, but there are many points around near-zero illiteracy and about 1.8 children/woman.



####
 Linear regression




```python

# Plot the illiteracy rate versus fertility
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')

# Perform a linear regression using np.polyfit(): a, b
a, b = np.polyfit(illiteracy, fertility, deg=1)

# Print the results to the screen
print('slope =', a, 'children per woman / percent illiterate')
print('intercept =', b, 'children per woman')

# Make theoretical line to plot
x = np.array([0, 100])
y = a * x + b

# Add regression line to your plot
_ = plt.plot(x, y)

# Draw the plot
plt.show()


# slope = 0.04979854809063423 children per woman / percent illiterate
# intercept = 1.888050610636557 children per woman


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture4-13.png)

####
 How is it optimal?




```python

# Specify slopes to consider: a_vals
a_vals = np.linspace(0, 0.1, 200)

# Initialize sum of square of residuals: rss
rss = np.empty_like(a_vals)

# Compute sum of square of residuals for each value of a_vals
for i, a in enumerate(a_vals):
    rss[i] = np.sum((fertility - a*illiteracy - b)**2)

# Plot the RSS
plt.plot(a_vals, rss, '-')
plt.xlabel('slope (children per woman / percent illiterate)')
plt.ylabel('sum of square of residuals')

plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture5-11.png)


 Notice that the minimum on the plot, that is the value of the slope that gives the minimum sum of the square of the residuals, is the same value you got when performing the regression.



###
 The importance of EDA: Anscombe’s quartet



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture6-8.png)

####
 The importance of EDA



 Why should exploratory data analysis be the first step in an analysis of data (after getting your data imported and cleaned, of course)?



* You can be protected from misinterpretation of the type demonstrated by Anscombe’s quartet.
* EDA provides a good starting point for planning the rest of your analysis.
* EDA is not really any more difficult than any of the subsequent analysis, so there is no excuse for not exploring the data.


####
 Linear regression on appropriate Anscombe data




```python

# Perform linear regression: a, b
a, b = np.polyfit(x, y, deg=1)

# Print the slope and intercept
print(a, b)

# Generate theoretical x and y data: x_theor, y_theor
x_theor = np.array([3, 15])
y_theor = a * x_theor + b

# Plot the Anscombe data and theoretical line
_ = plt.plot(x, y, marker = '.', linestyle = 'none')
_ = plt.plot(x_theor, y_theor)

# Label the axes
plt.xlabel('x')
plt.ylabel('y')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture7-9.png)

####
 Linear regression on all Anscombe data




```

for i in range(4):
    plt.subplot(2,2,i+1)

    # plot the scatter plot
    plt.plot(anscombe_x[i], anscombe_y[i], marker = '.', linestyle = 'none')

    # plot the regression line
    a, b = np.polyfit(anscombe_x[i], anscombe_y[i], deg=1)
    x_theor = np.array([np.min(anscombe_x[i]), np.max(anscombe_x[i])])
    y_theor = a * x_theor + b
    plt.plot(x_theor, y_theor)

    # add label
    plt.xlabel('x' + str(i+1))
    plt.ylabel('y' + str(i+1))

plt.show()

# slope1: 0.5000909090909095 intercept: 3.000090909090909
# slope2: 0.5000000000000004 intercept: 3.0009090909090896
# slope3: 0.4997272727272731 intercept: 3.0024545454545453
# slope4: 0.4999090909090908 intercept: 3.0017272727272735

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture8-6.png)

###
 Generating bootstrap replicates


####
 Getting the terminology down



 If we have a data set with n repeated measurements, a
 **bootstrap sample**
 is an array of length n that was drawn from the original data with replacement.




**Bootstrap replicate**
 is a single value of a statistic computed from a bootstrap sample.



####
 Visualizing bootstrap samples

 np.random.choice()




```

for _ in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(rainfall, size=len(rainfall))

    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)

# Compute and plot ECDF from original data
x, y = ecdf(rainfall)
_ = plt.plot(x, y, marker='.')

# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('yearly rainfall (mm)')
_ = plt.ylabel('ECDF')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture9-5.png)


 Bootstrap confidence intervals
--------------------------------


####
 Generating many bootstrap replicates




```

def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates

```


####
 Bootstrap replicates of the mean and the SEM (
 **standard error of the mean**
 )



 In fact, it can be shown theoretically that under not-too-restrictive conditions, the value of the mean will always be Normally distributed. (This does not hold in general, just for the mean and a few other statistics.)




 The standard deviation of this distribution, called the
 **standard error of the mean**
 , or SEM, is given by the standard deviation of the data divided by the square root of the number of data points. I.e., for a data set,
 `sem = np.std(data) / np.sqrt(len(data))`
 . Using hacker statistics, you get this same result without the need to derive it, but you will verify this result from your bootstrap replicates.





```python

# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates = draw_bs_reps(rainfall, np.mean, size=10000)

# Compute and print SEM
sem = np.std(rainfall) / np.sqrt(len(rainfall))
print(sem)

# Compute and print standard deviation of bootstrap replicates
bs_std = np.std(bs_replicates)
print(bs_std)

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('mean annual rainfall (mm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()

# 10.51054915050619
# 10.465927071184412

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture10-5.png)

####
 Confidence intervals of rainfall data



 Use the bootstrap replicates you just generated to compute the 95% confidence interval. That is, give the 2.5th and 97.5th percentile of your bootstrap replicates stored as
 `bs_replicates`
 . What is the 95% confidence interval?





```

np.percentile(bs_replicates,2.5)
779.7699248120301

np.percentile(bs_replicates,97.5)
820.950432330827

```


####
 Bootstrap replicates of other statistics




```

def draw_bs_reps(data, func, size=1):
    return np.array([bootstrap_replicate_1d(data, func) for _ in range(size)])

# Generate 10,000 bootstrap replicates of the variance: bs_replicates
bs_replicates = draw_bs_reps(rainfall, np.var, size=10000)

# Put the variance in units of square centimeters
bs_replicates /= 100

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('variance of annual rainfall (sq. cm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture11-5.png)


 This is not normally distributed, as it has a longer tail to the right. Note that you can also compute a confidence interval on the variance, or any other statistic, using
 `np.percentile()`
 with your bootstrap replicates.



####
 Confidence interval on the rate of no-hitters




```python

# Draw bootstrap replicates of the mean no-hitter time (equal to tau): bs_replicates
bs_replicates = draw_bs_reps(nohitter_times, np.mean, size=10000)

# Compute the 95% confidence interval: conf_int
conf_int = np.percentile(bs_replicates, [2.5, 97.5])

# Print the confidence interval
print('95% confidence interval =', conf_int, 'games')

# Plot the histogram of the replicates
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel(r'$\tau$ (games)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()

# 95% confidence interval = [660.67280876 871.63077689] games

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture12-4.png)


 This gives you an estimate of what the typical time between no-hitters is. It could be anywhere between 660 and 870 games.



###
 Pairs bootstrap


####
 A function to do pairs bootstrap




```

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, deg=1)

    return bs_slope_reps, bs_intercept_reps


```


####
 Pairs bootstrap of literacy/fertility data




```python

# Generate replicates of slope and intercept using pairs bootstrap
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(illiteracy, fertility, size=1000)

# Compute and print 95% CI for slope
print(np.percentile(bs_slope_reps, [2.5, 97.5]))

# Plot the histogram
_ = plt.hist(bs_slope_reps, bins=50, normed=True)
_ = plt.xlabel('slope')
_ = plt.ylabel('PDF')
plt.show()

# [0.04378061 0.0551616 ]

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture13-4.png)

####
 Plotting bootstrap regressions




```python

# Generate array of x-values for bootstrap lines: x
x = np.array([0, 100])

# Plot the bootstrap lines
for i in range(100):
    _ = plt.plot(x,
                 bs_slope_reps[i]*x + bs_intercept_reps[i],
                 linewidth=0.5, alpha=0.2, color='red')

# Plot the data
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')

# Label axes, set the margins, and show the plot
_ = plt.xlabel('illiteracy')
_ = plt.ylabel('fertility')
plt.margins(0.02)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture14-3.png)


 Introduction to hypothesis testing
------------------------------------


###
 Formulating and simulating a hypothesis



**Null hypothesis**
 : another name for the hypothesis you are testing




**Permutation**
 : random reordering of entries in an array



####
 Generating a permutation sample

 np.random.permutation()



 Permutation sampling is a great way to simulate the hypothesis that two variables have identical probability distributions. This is often a hypothesis you want to test, so in this exercise, you will write a function to generate a permutation sample from two data sets.





```

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

```


####
 Visualizing permutation sampling




```

for _ in range(50):
    # Generate permutation samples
    perm_sample_1, perm_sample_2 = permutation_sample(rain_june, rain_november)


    # Compute ECDFs
    x_1, y_1 = ecdf(perm_sample_1)
    x_2, y_2 = ecdf(perm_sample_2)

    # Plot ECDFs of permutation sample
    _ = plt.plot(x_1, y_1, marker='.', linestyle='none',
                 color='red', alpha=0.02)
    _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                 color='blue', alpha=0.02)

# Create and plot ECDFs from original data
x_1, y_1 = ecdf(rain_june)
x_2, y_2 = ecdf(rain_november)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('monthly rainfall (mm)')
_ = plt.ylabel('ECDF')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture1-16.png)


 Notice that the permutation samples ECDFs overlap and give a purple haze. None of the ECDFs from the permutation samples overlap with the observed data, suggesting that the hypothesis is not commensurate with the data. June and November rainfall are not identically distributed.



###
 Test statistics and p-values


####
 Test statistics



 When performing hypothesis tests, your choice of test statistic should be pertinent to the question you are seeking to answer in your hypothesis test.


 The most important thing to consider is:
 **What are you asking?**



####
 What is a p-value?



 The p-value is generally a measure of the probability of observing a test statistic equally or more extreme than the one you observed, given that the null hypothesis is true.



####
 Generating permutation replicates




```python

# In most circumstances, func will be a function you write yourself.
def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates

```


####
 Look before you leap: EDA before hypothesis testing



 Kleinteich and Gorb (
 *Sci. Rep.*
 ,
 **4**
 , 5225, 2014) performed an interesting experiment with South American horned frogs. They held a plate connected to a force transducer, along with a bait fly, in front of them. They then measured the impact force and adhesive force of the frog’s tongue when it struck the target.




 Frog A is an adult and Frog B is a juvenile. The researchers measured the impact force of 20 strikes for each frog. In the next exercise, we will test the hypothesis that the two frogs have the same distribution of impact forces. But, remember, it is important to do EDA first! Let’s make a bee swarm plot for the data. They are stored in a Pandas data frame,
 `df`
 , where column
 `ID`
 is the identity of the frog and column
 `impact_force`
 is the impact force in Newtons (N).





```

df.head()
   ID  impact_force
20  A         1.612
21  A         0.605
22  A         0.327
23  A         0.946
24  A         0.541

```




```python

# Make bee swarm plot
_ = sns.swarmplot(x='ID', y='impact_force', data=df)

# Label axes
_ = plt.xlabel('frog')
_ = plt.ylabel('impact force (N)')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture2-15.png)


 Eyeballing it, it does not look like they come from the same distribution. Frog A, the adult, has three or four very hard strikes, and Frog B, the juvenile, has a couple weak ones. However, it is possible that with only 20 samples it might be too difficult to tell if they have difference distributions, so we should proceed with the hypothesis test.



####
 Permutation test on frog data



 The average strike force of Frog A was 0.71 Newtons (N), and that of Frog B was 0.42 N for a difference of 0.29 N. It is possible the frogs strike with the same force and this observed difference was by chance. You will compute the probability of getting at least a 0.29 N difference in mean strike force under the hypothesis that the distributions of strike forces for the two frogs are identical. We use a permutation test with a test statistic of the difference of means to test this hypothesis.





```

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff

# Compute difference of mean impact force from experiment: empirical_diff_means
empirical_diff_means = diff_of_means(force_a, force_b)

# Draw 10,000 permutation replicates: perm_replicates
perm_replicates = draw_perm_reps(force_a, force_b,
                                 diff_of_means, size=10000)

# Compute p-value: p
p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

# Print the result
print('p-value =', p)

# p-value = 0.0063
# p-value = 0.63%

```



 The p-value tells you that there is about a 0.6% chance that you would get the difference of means observed in the experiment if frogs were exactly the same.




 A p-value below 0.01 is typically said to be “statistically significant,” but: warning! warning! warning! You have computed a p-value; it is a number. I encourage you not to distill it to a yes-or-no phrase. p = 0.006 and p = 0.000000006 are both said to be “statistically significant,” but they are definitely not the same!



###
 Bootstrap hypothesis tests



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture3-14.png)

####
 A one-sample bootstrap hypothesis test



 Another juvenile frog was studied, Frog C, and you want to see if Frog B and Frog C have similar impact forces. Unfortunately, you do not have Frog C’s impact forces available, but you know they have a mean of 0.55 N. Because you don’t have the original data, you cannot do a permutation test, and you cannot assess the hypothesis that the forces from Frog B and Frog C come from the same distribution. You will therefore test another, less restrictive hypothesis: The mean strike force of Frog B is equal to that of Frog C.




 To set up the bootstrap hypothesis test, you will take the mean as our test statistic. Remember, your goal is to calculate the probability of getting a mean impact force less than or equal to what was observed for Frog B
 *if the hypothesis that the true mean of Frog B’s impact forces is equal to that of Frog C is true*
 . You first translate all of the data of Frog B such that the mean is 0.55 N. This involves adding the mean force of Frog C and subtracting the mean force of Frog B from each measurement of Frog B. This leaves other properties of Frog B’s distribution, such as the variance, unchanged.





```python

# Make an array of translated impact forces: translated_force_b
translated_force_b = force_b - np.mean(force_b) + 0.55

# Take bootstrap replicates of Frog B's translated impact forces: bs_replicates
bs_replicates = draw_bs_reps(translated_force_b, np.mean, 10000)

# Compute fraction of replicates that are less than the observed Frog B force: p
p = np.sum(bs_replicates <= np.mean(force_b)) / 10000

# Print the p-value
print('p = ', p)

# p =  0.0046
# p = 0.46%

```



 The low p-value suggests that the null hypothesis that Frog B and Frog C have the same mean impact force is false.



####
 A two-sample bootstrap hypothesis test for difference of means



 We now want to test the hypothesis that Frog A and Frog B have the same mean impact force, but not necessarily the same distribution, which is also impossible with a permutation test.




 To do the two-sample bootstrap test, we shift
 *both*
 arrays to have the same mean, since we are simulating the hypothesis that their means are, in fact, equal. We then draw bootstrap samples out of the shifted arrays and compute the difference in means. This constitutes a bootstrap replicate, and we generate many of them. The p-value is the fraction of replicates with a difference in means greater than or equal to what was observed.





```python

# Compute mean of all forces: mean_force
mean_force = np.mean(forces_concat)

# Generate shifted arrays
force_a_shifted = force_a - np.mean(force_a) + mean_force
force_b_shifted = force_b - np.mean(force_b) + mean_force

# Compute 10,000 bootstrap replicates from shifted arrays
bs_replicates_a = draw_bs_reps(force_a_shifted, np.mean, size=10000)
bs_replicates_b = draw_bs_reps(force_b_shifted, np.mean, size=10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_replicates_a - bs_replicates_b

# Compute and print p-value: p
p = np.sum(bs_replicates >= empirical_diff_means) / 10000
print('p-value =', p)

# p-value = 0.0043
# p-value = 0.43%

```



 You got a similar result as when you did the permutation test. Nonetheless, remember that it is important to carefully think about what question you want to ask. Are you only interested in the mean impact force, or in the distribution of impact forces?




 Hypothesis test examples
--------------------------


###
 A/B testing


####
 The vote for the Civil Rights Act in 1964



 The Civil Rights Act of 1964 was one of the most important pieces of legislation ever passed in the USA. Excluding “present” and “abstain” votes, 153 House Democrats and 136 Republicans voted yea. However, 91 Democrats and 35 Republicans voted nay. Did party affiliation make a difference in the vote?




 To answer this question, you will evaluate the hypothesis that the party of a House member has no bearing on his or her vote. You will use the fraction of Democrats voting in favor as your test statistic and evaluate the probability of observing a fraction of Democrats voting in favor at least as small as the observed fraction of 153/244. (That’s right, at least as
 *small*
 as. In 1964, it was the
 *Democrats*
 who were less progressive on civil rights issues.) To do this, permute the party labels of the House voters and then arbitrarily divide them into “Democrats” and “Republicans” and compute the fraction of Democrats voting yea.





```python

# Construct arrays of data: dems, reps
dems = np.array([True] * 153 + [False] * 91)
reps = np.array([True] * 136 + [False] * 35)

def frac_yea_dems(dems, reps):
    """Compute fraction of Democrat yea votes."""
    frac = np.sum(dems) / len(dems)
    return frac

# Acquire permutation samples: perm_replicates
perm_replicates = draw_perm_reps(dems, reps, frac_yea_dems, size=10000)

# Compute and print p-value: p
p = np.sum(perm_replicates <= 153/244) / len(perm_replicates)
print('p-value =', p)

# p-value = 0.0002
# p-value = 0.02%

```



 This small p-value suggests that party identity had a lot to do with the voting. Importantly, the South had a higher fraction of Democrat representatives, and consequently also a more racist bias.



####
 What is equivalent?



 You have experience matching a stories to probability distributions. Similarly, you use the same procedure for two different A/B tests if their stories match. In the Civil Rights Act example you just did, you performed an A/B test on voting data, which has a Yes/No type of outcome for each subject (in that case, a voter). Which of the following situations involving testing by a web-based company has an equivalent set up for an A/B test as the one you just did with the Civil Rights Act of 1964?




 You measure the number of people who click on an ad on your company’s website before and after changing its color.




 The “Democrats” are those who view the ad before the color change, and the “Republicans” are those who view it after.



####
 A time-on-website analog



 It turns out that you already did a hypothesis test analogous to an A/B test where you are interested in how much time is spent on the website before and after an ad campaign. The frog tongue force (a continuous quantity like time on the website) is an analog. “Before” = Frog A and “after” = Frog B. Let’s practice this again with something that actually is a before/after scenario.




 We return to the no-hitter data set. In 1920, Major League Baseball implemented important rule changes that ended the so-called dead ball era. Importantly, the pitcher was no longer allowed to spit on or scuff the ball, an activity that greatly favors pitchers. In this problem you will perform an A/B test to determine if these rule changes resulted in a slower rate of no-hitters (i.e., longer average time between no-hitters) using the difference in mean inter-no-hitter time as your test statistic. The inter-no-hitter times for the respective eras are stored in the arrays
 `nht_dead`
 and
 `nht_live`
 , where “nht” is meant to stand for “no-hitter time.”





```

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates



# Compute the observed difference in mean inter-no-hitter times: nht_diff_obs
nht_diff_obs = diff_of_means(nht_dead, nht_live)

# Acquire 10,000 permutation replicates of difference in mean no-hitter time: perm_replicates
perm_replicates = draw_perm_reps(nht_dead, nht_live, diff_of_means, size=10000)

# Compute and print the p-value: p
p = np.sum(perm_replicates <= nht_diff_obs) /len(perm_replicates)
print('p-val =', p)

p-val = 0.0001

```



 Your p-value is 0.0001, which means that only one out of your 10,000 replicates had a result as extreme as the actual difference between the dead ball and live ball eras. This suggests strong statistical significance. Watch out, though, you could very well have gotten zero replicates that were as extreme as the observed value. This just means that the p-value is quite small, almost certainly smaller than 0.001.



####
 What should you have done first?



 That was a nice hypothesis test you just did to check out whether the rule changes in 1920 changed the rate of no-hitters. But what
 *should*
 you have done with the data first?




 Performed EDA, perhaps plotting the ECDFs of inter-no-hitter times in the dead ball and live ball eras.




 Always a good idea to do first! I encourage you to go ahead and plot the ECDFs right now. You will see by eye that the null hypothesis that the distributions are the same is almost certainly not true.





```python

# Create and plot ECDFs
x_1, y_1 = ecdf(nht_dead)
x_2, y_2 = ecdf(nht_live)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('Games between no-hitters')
_ = plt.ylabel('ECDF')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture1-17.png)

###
 Test of correlation



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture2-16.png)

####
 Simulating a null hypothesis concerning correlation



 The observed correlation between female illiteracy and fertility in the data set of 162 countries may just be by chance; the fertility of a given country may actually be totally independent of its illiteracy. You will test this null hypothesis in the next exercise.




 To do the test, you need to simulate the data assuming the null hypothesis is true. Of the following choices, which is the best way to to do it?




 Answer: Do a permutation test: Permute the illiteracy values but leave the fertility values fixed to generate a new set of (illiteracy, fertility) data.




 This exactly simulates the null hypothesis and does so more efficiently than the last option. It is exact because it uses all data and eliminates any correlation because which illiteracy value pairs to which fertility value is shuffled.




 Last option: Do a permutation test: Permute both the illiteracy and fertility values to generate a new set of (illiteracy, fertility data). This exactly simulates the null hypothesis and does so more efficiently than the last option. It is exact because it uses all data and eliminates any correlation because which illiteracy value pairs to which fertility value is shuffled.



####
 Hypothesis test on Pearson correlation



 The observed correlation between female illiteracy and fertility may just be by chance; the fertility of a given country may actually be totally independent of its illiteracy. You will test this hypothesis. To do so, permute the illiteracy values but leave the fertility values fixed. This simulates the hypothesis that they are totally independent of each other. For each permutation, compute the Pearson correlation coefficient and assess how many of your permutation replicates have a Pearson correlation coefficient greater than the observed one.





```

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0,1]

# Compute observed correlation: r_obs
r_obs = pearson_r(illiteracy, fertility)

# r_obs = 0.8041324026815344

# Initialize permutation replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute illiteracy measurments: illiteracy_permuted
    illiteracy_permuted = np.random.permutation(illiteracy)

    # Compute Pearson correlation
    perm_replicates[i] = pearson_r(illiteracy_permuted, fertility)

# Compute p-value: p
p = np.sum(perm_replicates >= r_obs) /len(perm_replicates)
print('p-val =', p)

# p-val = 0.0

```



 You got a p-value of zero. In hacker statistics, this means that your p-value is very low, since you never got a single replicate in the 10,000 you took that had a Pearson correlation greater than the observed one. You could try increasing the number of replicates you take to continue to move the upper bound on your p-value lower and lower.



####
 Do neonicotinoid insecticides have unintended consequences?



 As a final exercise in hypothesis testing before we put everything together in our case study in the next chapter, you will investigate the effects of neonicotinoid insecticides on bee reproduction. These insecticides are very widely used in the United States to combat aphids and other pests that damage plants.




 In a recent study, Straub, et al. (
 [*Proc. Roy. Soc. B*
 , 2016](http://dx.doi.org/10.1098/rspb.2016.0506)
 ) investigated the effects of neonicotinoids on the sperm of pollinating bees. In this and the next exercise, you will study how the pesticide treatment affected the count of live sperm per half milliliter of semen.




 First, we will do EDA, as usual. Plot ECDFs of the alive sperm count for untreated bees (stored in the Numpy array
 `control`
 ) and bees treated with pesticide (stored in the Numpy array
 `treated`
 ).





```python

# Compute x,y values for ECDFs
x_control, y_control = ecdf(control)
x_treated, y_treated = ecdf(treated)

# Plot the ECDFs
plt.plot(x_control, y_control, marker='.', linestyle='none')
plt.plot(x_treated, y_treated, marker='.', linestyle='none')

# Set the margins
plt.margins(0.02)

# Add a legend
plt.legend(('control', 'treated'), loc='lower right')

# Label axes and show plot
plt.xlabel('millions of alive sperm per mL')
plt.ylabel('ECDF')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture3-15.png)


 The ECDFs show a pretty clear difference between the treatment and control; treated bees have fewer alive sperm. Let’s now do a hypothesis test in the next exercise.



####
 Bootstrap hypothesis test on bee sperm counts



 Now, you will test the following hypothesis:




**On average, male bees treated with neonicotinoid insecticide have the same number of active sperm per milliliter of semen than do untreated male bees.**




 You will use the difference of means as your test statistic.





```

def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates


# Compute the difference in mean sperm count: diff_means
diff_means = np.mean(control) - np.mean(treated)

# Compute mean of pooled data: mean_count
mean_count = np.mean(np.concatenate((control, treated)))

# Generate shifted data sets
control_shifted = control - np.mean(control) + mean_count
treated_shifted = treated - np.mean(treated) + mean_count

# Generate bootstrap replicates
bs_reps_control = draw_bs_reps(control_shifted,
                       np.mean, size=10000)
bs_reps_treated = draw_bs_reps(treated_shifted,
                       np.mean, size=10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_reps_control - bs_reps_treated

# Compute and print p-value: p
p = np.sum(bs_replicates >= np.mean(control) - np.mean(treated)) \
            / len(bs_replicates)
print('p-value =', p)

# p-value = 0.0

```



 The p-value is small, most likely less than 0.0001, since you never saw a bootstrap replicated with a difference of means at least as extreme as what was observed. In fact, when I did the calculation with 10 million replicates, I got a p-value of
 `2e-05`




 Putting it all together: a case study
---------------------------------------


###
 Finch beaks and the need for statistics



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture4-14.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture5-12.png)

####
 EDA of beak depths of Darwin’s finches



 For your first foray into the Darwin finch data, you will study how the beak depth (the distance, top to bottom, of a closed beak) of the finch species
 *Geospiza scandens*
 has changed over time. The Grants have noticed some changes of beak geometry depending on the types of seeds available on the island, and they also noticed that there was some interbreeding with another major species on Daphne Major,
 *Geospiza fortis*
 . These effects can lead to changes in the species over time.




 In the next few problems, you will look at the beak depth of
 *G. scandens*
 on Daphne Major in 1975 and in 2012. To start with, let’s plot all of the beak depth measurements in 1975 and 2012 in a bee swarm plot.




 The data are stored in a pandas DataFrame called
 `df`
 with columns
 `'year'`
 and
 `'beak_depth'`
 . The units of beak depth are millimeters (mm).





```

df.head()
   beak_depth  year
0         8.4  1975
1         8.8  1975
2         8.4  1975
3         8.0  1975
4         7.9  1975

```




```python

# Create bee swarm plot
_ = sns.swarmplot('year', 'beak_depth', data=df)

# Label the axes
_ = plt.xlabel('year')
_ = plt.ylabel('beak depth (mm)')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture6-9.png)


 It is kind of hard to see if there is a clear difference between the 1975 and 2012 data set. Eyeballing it, it appears as though the mean of the 2012 data set might be slightly higher, and it might have a bigger variance.



####
 ECDFs of beak depths



 While bee swarm plots are useful, we found that ECDFs are often even better when doing EDA. Plot the ECDFs for the 1975 and 2012 beak depth measurements on the same plot.





```python

# Compute ECDFs
x_1975, y_1975 = ecdf(bd_1975)
x_2012, y_2012 = ecdf(bd_2012)

# Plot the ECDFs
_ = plt.plot(x_1975, y_1975, marker='.', linestyle='none')
_ = plt.plot(x_2012, y_2012, marker='.', linestyle='none')

# Set margins
plt.margins(0.02)

# Add axis labels and legend
_ = plt.xlabel('beak depth (mm)')
_ = plt.ylabel('ECDF')
_ = plt.legend(('1975', '2012'), loc='lower right')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture7-10.png)


 The differences are much clearer in the ECDF. The mean is larger in the 2012 data, and the variance does appear larger as well.



####
 Parameter estimates of beak depths



 Estimate the
 *difference*
 of the mean beak depth of the
 *G. scandens*
 samples from 1975 and 2012 and report a 95% confidence interval.





```

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates


# Compute the difference of the sample means: mean_diff
mean_diff = np.mean(bd_2012) - np.mean(bd_1975)

# Get bootstrap replicates of means
bs_replicates_1975 = draw_bs_reps(bd_1975, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(bd_2012, np.mean, size=10000)

# Compute samples of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute 95% confidence interval: conf_int
conf_int = np.percentile(bs_diff_replicates, [2.5, 97.5])

# Print the results
print('difference of means =', mean_diff, 'mm')
print('95% confidence interval =', conf_int, 'mm')

# difference of means = 0.22622047244094645 mm
# 95% confidence interval = [0.05633521 0.39190544] mm

```


####
 Hypothesis test: Are beaks deeper in 2012?



 Your plot of the ECDF and determination of the confidence interval make it pretty clear that the beaks of
 *G. scandens*
 on Daphne Major have gotten deeper. But is it possible that this effect is just due to random chance? In other words, what is the probability that we would get the observed difference in mean beak depth if the means were the same?




 Be careful! The hypothesis we are testing is
 *not*
 that the beak depths come from the same distribution. For that we could use a permutation test.
 **The hypothesis is that the means are equal.**
 To perform this hypothesis test, we need to shift the two data sets so that they have the same mean and then use bootstrap sampling to compute the difference of means.





```python

# Compute mean of combined data set: combined_mean
combined_mean = np.mean(np.concatenate((bd_1975, bd_2012)))

# Shift the samples
# why shift the mean?
# to make np.mean(bd_1975_shifted) - np.mean(bd_2012_shifted) = 0 #1
# why make #1 = 0?
# because our hypothesis is "beak depth are the same in 1975 and 2012"
bd_1975_shifted = bd_1975 - np.mean(bd_1975) + combined_mean
bd_2012_shifted = bd_2012 - np.mean(bd_2012) + combined_mean

# Get bootstrap replicates of shifted data sets
bs_replicates_1975 = draw_bs_reps(bd_1975_shifted, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(bd_2012_shifted, np.mean, size=10000)

# Compute replicates of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute the p-value
p = np.sum(bs_diff_replicates >= mean_diff) / len(bs_diff_replicates)

# Print p-value
print('p =', p)

# p = 0.0034
# p = 0.34%

```



 We get a p-value of 0.0034, which suggests that there is a statistically significant difference. But remember: it is very important to know how different they are! In the previous exercise, you got a difference of 0.2 mm between the means. You should combine this with the statistical significance. Changing by 0.2 mm in 37 years is substantial by evolutionary standards. If it kept changing at that rate, the beak depth would double in only 400 years.



###
 Variation of beak shapes


####
 EDA of beak length and depth



 The beak length data are stored as
 `bl_1975`
 and
 `bl_2012`
 , again with units of millimeters (mm). You still have the beak depth data stored in
 `bd_1975`
 and
 `bd_2012`
 . Make scatter plots of beak depth (y-axis) versus beak length (x-axis) for the 1975 and 2012 specimens.





```python

# Make scatter plot of 1975 data
_ = plt.plot(bl_1975, bd_1975, marker='.',
             linestyle='None', color='blue', alpha=0.5)

# Make scatter plot of 2012 data
_ = plt.plot(bl_2012, bd_2012, marker='.',
            linestyle='None', color='red', alpha=0.5)

# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture8-7.png)


 In looking at the plot, we see that beaks got deeper (the red points are higher up in the y-direction), but not really longer. If anything, they got a bit shorter, since the red dots are to the left of the blue dots. So, it does not look like the beaks kept the same shape; they became shorter and deeper.



####
 Linear regressions



 Perform a linear regression for both the 1975 and 2012 data. Then, perform pairs bootstrap estimates for the regression parameters. Report 95% confidence intervals on the slope and intercept of the regression line.





```

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, deg=1)

    return bs_slope_reps, bs_intercept_reps


# Compute the linear regressions
slope_1975, intercept_1975 = np.polyfit(bl_1975, bd_1975, deg=1)
slope_2012, intercept_2012 = np.polyfit(bl_2012, bd_2012, deg=1)

# Perform pairs bootstrap for the linear regressions
bs_slope_reps_1975, bs_intercept_reps_1975 = draw_bs_pairs_linreg(bl_1975, bd_1975, size=1000)
bs_slope_reps_2012, bs_intercept_reps_2012 = draw_bs_pairs_linreg(bl_2012, bd_2012, size=1000)

# Compute confidence intervals of slopes
slope_conf_int_1975 = np.percentile(bs_slope_reps_1975, [2.5, 97.5])
slope_conf_int_2012 = np.percentile(bs_slope_reps_2012, [2.5, 97.5])
intercept_conf_int_1975 = np.percentile(bs_intercept_reps_1975, [2.5, 97.5])
intercept_conf_int_2012 = np.percentile(bs_intercept_reps_2012, [2.5, 97.5])


# Print the results
print('1975: slope =', slope_1975,
      'conf int =', slope_conf_int_1975)
print('1975: intercept =', intercept_1975,
      'conf int =', intercept_conf_int_1975)
print('2012: slope =', slope_2012,
      'conf int =', slope_conf_int_2012)
print('2012: intercept =', intercept_2012,
      'conf int =', intercept_conf_int_2012)

#   1975: slope = 0.4652051691605937 conf int = [0.33851226 0.59306491]
#   1975: intercept = 2.3908752365842263 conf int = [0.64892945 4.18037063]
#   2012: slope = 0.462630358835313 conf int = [0.33137479 0.60695527]
#   2012: intercept = 2.977247498236019 conf int = [1.06792753 4.70599387]

```



 It looks like they have the same slope, but different intercepts.



####
 Displaying the linear regression results




```python

# Make scatter plot of 1975 data
_ = plt.plot(bl_1975, bd_1975, marker='.',
             linestyle='none', color='blue', alpha=0.5)

# Make scatter plot of 2012 data
_ = plt.plot(bl_2012, bd_2012, marker='.',
             linestyle='none', color='red', alpha=0.5)

# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')

# Generate x-values for bootstrap lines: x
x = np.array([10, 17])

# Plot the bootstrap lines
for i in range(100):
    plt.plot(x, bs_slope_reps_1975[i] * x + bs_intercept_reps_1975[i],
             linewidth=0.5, alpha=0.2, color='blue')
    plt.plot(x, bs_slope_reps_2012[i] * x + bs_intercept_reps_2012[i],
             linewidth=0.5, alpha=0.2, color='red')

# Draw the plot again
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture9-6.png)

####
 Beak length to depth ratio



 The linear regressions showed interesting information about the beak geometry. The slope was the same in 1975 and 2012, suggesting that for every millimeter gained in beak length, the birds gained about half a millimeter in depth in both years. However, if we are interested in the shape of the beak, we want to compare the
 *ratio*
 of beak length to beak depth. Let’s make that comparison.





```python

# Compute length-to-depth ratios
ratio_1975 = bl_1975 / bd_1975
ratio_2012 = bl_2012 / bd_2012

# Compute means
mean_ratio_1975 = np.mean(ratio_1975)
mean_ratio_2012 = np.mean(ratio_2012)

# Generate bootstrap replicates of the means
bs_replicates_1975 = draw_bs_reps(ratio_1975, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(ratio_2012, np.mean, size=10000)

# Compute the 99% confidence intervals
conf_int_1975 = np.percentile(bs_replicates_1975, [0.5, 99.5])
conf_int_2012 = np.percentile(bs_replicates_2012, [0.5, 99.5])

# Print the results
print('1975: mean ratio =', mean_ratio_1975,
      'conf int =', conf_int_1975)
print('2012: mean ratio =', mean_ratio_2012,
      'conf int =', conf_int_2012)

# 1975: mean ratio = 1.5788823771858533 conf int = [1.55668803 1.60073509]
# 2012: mean ratio = 1.4658342276847767 conf int = [1.44363932 1.48729149]

```


####
 How different is the ratio?



 In the previous exercise, you computed the mean beak length to depth ratio with 99% confidence intervals for 1975 and for 2012. The results of that calculation are shown graphically in the plot accompanying this problem. In addition to these results, what would you say about the ratio of beak length to depth?




![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture10-6.png)


 The mean beak length-to-depth ratio decreased by about 0.1, or 7%, from 1975 to 2012. The 99% confidence intervals are not even close to overlapping, so this is a real change. The beak shape changed.



###
 Calculation of heritability


####
 EDA of heritability



 The array
 `bd_parent_scandens`
 contains the average beak depth (in mm) of two parents of the species
 `G. scandens`
 . The array
 `bd_offspring_scandens`
 contains the average beak depth of the offspring of the respective parents. The arrays
 `bd_parent_fortis`
 and
 `bd_offspring_fortis`
 contain the same information about measurements from
 *G. fortis*
 birds.




 Make a scatter plot of the average offspring beak depth (y-axis) versus average parental beak depth (x-axis) for both species. Use the
 `alpha=0.5`
 keyword argument to help you see overlapping points.





```python

# Make scatter plots
_ = plt.plot(bd_parent_fortis, bd_offspring_fortis,
             marker='.', linestyle='none', color='blue', alpha=0.5)
_ = plt.plot(bd_parent_scandens, bd_offspring_scandens,
             marker='.', linestyle='none', color='red', alpha=0.5)

# Label axes
_ = plt.xlabel('parental beak depth (mm)')
_ = plt.ylabel('offspring beak depth (mm)')

# Add legend
_ = plt.legend(('G. fortis', 'G. scandens'), loc='lower right')

# Show plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture11-6.png)


 It appears as though there is a stronger correlation in
 *G. fortis*
 than than in
 *G. scandens*
 . This suggests that beak depth is more strongly inherited in
 *G. fortis*
 . We’ll quantify this correlation next.



####
 Correlation of offspring and parental data



 In an effort to quantify the correlation between offspring and parent beak depths, we would like to compute statistics, such as the Pearson correlation coefficient, between parents and offspring. To get confidence intervals on this, we need to do a pairs bootstrap.




 You have
 [already written](https://campus.datacamp.com/courses/statistical-thinking-in-python-part-2/bootstrap-confidence-intervals?ex=12)
 a function to do pairs bootstrap to get estimates for parameters derived from linear regression. Your task in this exercise is to make a new function with call signature
 `draw_bs_pairs(x, y, func, size=1)`
 that performs pairs bootstrap and computes a single statistic on pairs samples defined. The statistic of interest is computed by calling
 `func(bs_x, bs_y)`
 . In the next exercise, you will use
 `pearson_r`
 for
 `func`
 .





```

def draw_bs_pairs(x, y, func, size=1):
    """Perform pairs bootstrap for a single statistic."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_replicates[i] = func(bs_x, bs_y)

    return bs_replicates


```


####
 Pearson correlation of offspring and parental data



 The Pearson correlation coefficient seems like a useful measure of how strongly the beak depth of parents are inherited by their offspring. Compute the Pearson correlation coefficient between parental and offspring beak depths for
 *G. scandens*
 . Do the same for
 *G. fortis*
 . Then, use the function you wrote in the last exercise to compute a 95% confidence interval using pairs bootstrap.





```

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0,1]


# Compute the Pearson correlation coefficients
r_scandens = pearson_r(bd_parent_scandens, bd_offspring_scandens)
r_fortis = pearson_r(bd_parent_fortis, bd_offspring_fortis)

# Acquire 1000 bootstrap replicates of Pearson r
bs_replicates_scandens = draw_bs_pairs(bd_parent_scandens, bd_offspring_scandens, pearson_r, size=1000)

bs_replicates_fortis = draw_bs_pairs(bd_parent_fortis, bd_offspring_fortis, pearson_r, size=1000)


# Compute 95% confidence intervals
conf_int_scandens = np.percentile(bs_replicates_scandens, [2.5, 97.5])
conf_int_fortis = np.percentile(bs_replicates_fortis, [2.5, 97.5])

# Print results
print('G. scandens:', r_scandens, conf_int_scandens)
print('G. fortis:', r_fortis, conf_int_fortis)

#    G. scandens: 0.4117063629401258 [0.26564228 0.54388972]
#    G. fortis: 0.7283412395518487 [0.6694112  0.77840616]

```



 It is clear from the confidence intervals that beak depth of the offspring of
 *G. fortis*
 parents is more strongly correlated with their offspring than their
 *G. scandens*
 counterparts.



####
 Measuring heritability



 Remember that the Pearson correlation coefficient is the ratio of the covariance to the geometric mean of the variances of the two data sets. This is a measure of the correlation between parents and offspring, but might not be the best estimate of heritability. If we stop and think, it makes more sense to define heritability as the ratio of the covariance between parent and offspring to the
 *variance of the parents alone*
 . In this exercise, you will estimate the heritability and perform a pairs bootstrap calculation to get the 95% confidence interval.




 This exercise highlights a very important point. Statistical inference (and data analysis in general) is not a plug-n-chug enterprise. You need to think carefully about the questions you are seeking to answer with your data and analyze them appropriately. If you are interested in how heritable traits are, the quantity we defined as the heritability is more apt than the off-the-shelf statistic, the Pearson correlation coefficient.





```

def heritability(parents, offspring):
    """Compute the heritability from parent and offspring samples."""
    covariance_matrix = np.cov(parents, offspring)
    return covariance_matrix[0,1] / covariance_matrix[0,0]

# Compute the heritability
heritability_scandens = heritability(bd_parent_scandens, bd_offspring_scandens)
heritability_fortis = heritability(bd_parent_fortis, bd_offspring_fortis)

# Acquire 1000 bootstrap replicates of heritability
replicates_scandens = draw_bs_pairs(
        bd_parent_scandens, bd_offspring_scandens, heritability, size=1000)

replicates_fortis = draw_bs_pairs(
        bd_parent_fortis, bd_offspring_fortis, heritability, size=1000)


# Compute 95% confidence intervals
conf_int_scandens = np.percentile(replicates_scandens, [2.5, 97.5])
conf_int_fortis = np.percentile(replicates_fortis, [2.5, 97.5])

# Print results
print('G. scandens:', heritability_scandens, conf_int_scandens)
print('G. fortis:', heritability_fortis, conf_int_fortis)


#   G. scandens: 0.5485340868685982 [0.34395487 0.75638267]
#   G. fortis: 0.7229051911438159 [0.64655013 0.79688342]

```



 Here again, we see that
 *G. fortis*
 has stronger heritability than
 *G. scandens*
 . This suggests that the traits of
 *G. fortis*
 may be strongly incorporated into
 *G. scandens*
 by introgressive hybridization.



####
 Is beak depth heritable at all in G. scandens?



 The heritability of beak depth in
 *G. scandens*
 seems low. It could be that this observed heritability was just achieved by chance and
 **beak depth is actually not really heritable in the species**
 . You will test that hypothesis here. To do this, you will do a pairs permutation test.





```python

# Initialize array of replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute parent beak depths
    bd_parent_permuted = np.random.permutation(bd_parent_scandens)
    perm_replicates[i] = heritability(bd_parent_permuted,
                                      bd_offspring_scandens)

# Compute p-value: p
p = np.sum(perm_replicates >= heritability_scandens) / len(perm_replicates)

# Print the p-value
print('p-val =', p)

# p-val = 0.0

```



 You get a p-value of zero, which means that none of the 10,000 permutation pairs replicates you drew had a heritability high enough to match that which was observed. This strongly suggests that beak depth is heritable in
 *G. scandens*
 , just not as much as in
 *G. fortis*
 . If you like, you can plot a histogram of the heritability replicates to get a feel for how extreme of a value of heritability you might expect by chance.





```

plt.hist(perm_replicates)
plt.axvline(x=heritability_scandens, color = 'red')
plt.text(heritability_scandens, 1500, 'heritability_scandens', ha='center', va='center',rotation='vertical', backgroundcolor='white')
plt.show()

```




 Parameter estimation by optimization
--------------------------------------


###
 Optimal parameters


####
 How often do we get no-hitters?



 The number of games played between each no-hitter in the modern era (1901-2015) of Major League Baseball is stored in the array
 `nohitter_times`
 .




 If you assume that no-hitters are described as a Poisson process, then the time between no-hitters is Exponentially distributed. As you have seen, the Exponential distribution has a single parameter, which we will call ττ, the typical interval time. The value of the parameter ττ that makes the exponential distribution best match the data is the mean interval time (where time is in units of number of games) between no-hitters.




 Compute the value of this parameter from the data. Then, use
 `np.random.exponential()`
 to “repeat” the history of Major League Baseball by drawing inter-no-hitter times from an exponential distribution with the ττ you found and plot the histogram as an approximation to the PDF.







```python

# Seed random number generator
np.random.seed(42)

# Compute mean no-hitter time: tau
tau = np.mean(nohitter_times)

# Draw out of an exponential distribution with parameter tau: inter_nohitter_time
inter_nohitter_time = np.random.exponential(tau, 100000)

# Plot the PDF and label axes
_ = plt.hist(inter_nohitter_time,
             bins=50, normed=True, histtype='step')
_ = plt.xlabel('Games between no-hitters')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture-16.png)


 We see the typical shape of the Exponential distribution, going from a maximum at 0 and decaying to the right.



####
 Do the data follow our story?




```python

# Create an ECDF from real data: x, y
x, y = ecdf(nohitter_times)

# Create a CDF from theoretical samples: x_theor, y_theor
x_theor, y_theor = ecdf(inter_nohitter_time)

# Overlay the plots
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')

# Margins and axis labels
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture1-15.png)


 It looks like no-hitters in the modern era of Major League Baseball are Exponentially distributed. Based on the story of the Exponential distribution, this suggests that they are a random process; when a no-hitter will happen is independent of when the last no-hitter was.



####
 How is this parameter optimal?




```python

# Plot the theoretical CDFs
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# Take samples with half tau: samples_half
samples_half = np.random.exponential(tau/2, size=10000)

# Take samples with double tau: samples_double
samples_double = np.random.exponential(2*tau, size=10000)

# Generate CDFs from these samples
x_half, y_half = ecdf(samples_half)
x_double, y_double = ecdf(samples_double)

# Plot these CDFs as lines
_ = plt.plot(x_half, y_half)
_ = plt.plot(x_double, y_double)

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture2-14.png)

 red = half, purple = double



 Notice how the value of tau given by the mean matches the data best. In this way, tau is an optimal parameter.



###
 Linear regression by least squares


####
 EDA of literacy/fertility data




```python

# Plot the illiteracy rate versus fertility
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')

# Set the margins and label axes
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')

# Show the plot
plt.show()

# Show the Pearson correlation coefficient
print(pearson_r(illiteracy, fertility))
0.8041324026815344

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture3-13.png)


 You can see the correlation between illiteracy and fertility by eye, and by the substantial Pearson correlation coefficient of 0.8. It is difficult to resolve in the scatter plot, but there are many points around near-zero illiteracy and about 1.8 children/woman.



####
 Linear regression




```python

# Plot the illiteracy rate versus fertility
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')

# Perform a linear regression using np.polyfit(): a, b
a, b = np.polyfit(illiteracy, fertility, deg=1)

# Print the results to the screen
print('slope =', a, 'children per woman / percent illiterate')
print('intercept =', b, 'children per woman')

# Make theoretical line to plot
x = np.array([0, 100])
y = a * x + b

# Add regression line to your plot
_ = plt.plot(x, y)

# Draw the plot
plt.show()


# slope = 0.04979854809063423 children per woman / percent illiterate
# intercept = 1.888050610636557 children per woman


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture4-13.png)

####
 How is it optimal?




```python

# Specify slopes to consider: a_vals
a_vals = np.linspace(0, 0.1, 200)

# Initialize sum of square of residuals: rss
rss = np.empty_like(a_vals)

# Compute sum of square of residuals for each value of a_vals
for i, a in enumerate(a_vals):
    rss[i] = np.sum((fertility - a*illiteracy - b)**2)

# Plot the RSS
plt.plot(a_vals, rss, '-')
plt.xlabel('slope (children per woman / percent illiterate)')
plt.ylabel('sum of square of residuals')

plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture5-11.png)


 Notice that the minimum on the plot, that is the value of the slope that gives the minimum sum of the square of the residuals, is the same value you got when performing the regression.



###
 The importance of EDA: Anscombe’s quartet



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture6-8.png)

####
 The importance of EDA



 Why should exploratory data analysis be the first step in an analysis of data (after getting your data imported and cleaned, of course)?



* You can be protected from misinterpretation of the type demonstrated by Anscombe’s quartet.
* EDA provides a good starting point for planning the rest of your analysis.
* EDA is not really any more difficult than any of the subsequent analysis, so there is no excuse for not exploring the data.


####
 Linear regression on appropriate Anscombe data




```python

# Perform linear regression: a, b
a, b = np.polyfit(x, y, deg=1)

# Print the slope and intercept
print(a, b)

# Generate theoretical x and y data: x_theor, y_theor
x_theor = np.array([3, 15])
y_theor = a * x_theor + b

# Plot the Anscombe data and theoretical line
_ = plt.plot(x, y, marker = '.', linestyle = 'none')
_ = plt.plot(x_theor, y_theor)

# Label the axes
plt.xlabel('x')
plt.ylabel('y')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture7-9.png)

####
 Linear regression on all Anscombe data




```

for i in range(4):
    plt.subplot(2,2,i+1)

    # plot the scatter plot
    plt.plot(anscombe_x[i], anscombe_y[i], marker = '.', linestyle = 'none')

    # plot the regression line
    a, b = np.polyfit(anscombe_x[i], anscombe_y[i], deg=1)
    x_theor = np.array([np.min(anscombe_x[i]), np.max(anscombe_x[i])])
    y_theor = a * x_theor + b
    plt.plot(x_theor, y_theor)

    # add label
    plt.xlabel('x' + str(i+1))
    plt.ylabel('y' + str(i+1))

plt.show()

# slope1: 0.5000909090909095 intercept: 3.000090909090909
# slope2: 0.5000000000000004 intercept: 3.0009090909090896
# slope3: 0.4997272727272731 intercept: 3.0024545454545453
# slope4: 0.4999090909090908 intercept: 3.0017272727272735

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture8-6.png)

###
 Generating bootstrap replicates


####
 Getting the terminology down



 If we have a data set with n repeated measurements, a
 **bootstrap sample**
 is an array of length n that was drawn from the original data with replacement.




**Bootstrap replicate**
 is a single value of a statistic computed from a bootstrap sample.



####
 Visualizing bootstrap samples

 np.random.choice()




```

for _ in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(rainfall, size=len(rainfall))

    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)

# Compute and plot ECDF from original data
x, y = ecdf(rainfall)
_ = plt.plot(x, y, marker='.')

# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('yearly rainfall (mm)')
_ = plt.ylabel('ECDF')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture9-5.png)


 Bootstrap confidence intervals
--------------------------------


####
 Generating many bootstrap replicates




```

def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates

```


####
 Bootstrap replicates of the mean and the SEM (
 **standard error of the mean**
 )



 In fact, it can be shown theoretically that under not-too-restrictive conditions, the value of the mean will always be Normally distributed. (This does not hold in general, just for the mean and a few other statistics.)




 The standard deviation of this distribution, called the
 **standard error of the mean**
 , or SEM, is given by the standard deviation of the data divided by the square root of the number of data points. I.e., for a data set,
 `sem = np.std(data) / np.sqrt(len(data))`
 . Using hacker statistics, you get this same result without the need to derive it, but you will verify this result from your bootstrap replicates.





```python

# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates = draw_bs_reps(rainfall, np.mean, size=10000)

# Compute and print SEM
sem = np.std(rainfall) / np.sqrt(len(rainfall))
print(sem)

# Compute and print standard deviation of bootstrap replicates
bs_std = np.std(bs_replicates)
print(bs_std)

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('mean annual rainfall (mm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()

# 10.51054915050619
# 10.465927071184412

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture10-5.png)

####
 Confidence intervals of rainfall data



 Use the bootstrap replicates you just generated to compute the 95% confidence interval. That is, give the 2.5th and 97.5th percentile of your bootstrap replicates stored as
 `bs_replicates`
 . What is the 95% confidence interval?





```

np.percentile(bs_replicates,2.5)
779.7699248120301

np.percentile(bs_replicates,97.5)
820.950432330827

```


####
 Bootstrap replicates of other statistics




```

def draw_bs_reps(data, func, size=1):
    return np.array([bootstrap_replicate_1d(data, func) for _ in range(size)])

# Generate 10,000 bootstrap replicates of the variance: bs_replicates
bs_replicates = draw_bs_reps(rainfall, np.var, size=10000)

# Put the variance in units of square centimeters
bs_replicates /= 100

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('variance of annual rainfall (sq. cm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture11-5.png)


 This is not normally distributed, as it has a longer tail to the right. Note that you can also compute a confidence interval on the variance, or any other statistic, using
 `np.percentile()`
 with your bootstrap replicates.



####
 Confidence interval on the rate of no-hitters




```python

# Draw bootstrap replicates of the mean no-hitter time (equal to tau): bs_replicates
bs_replicates = draw_bs_reps(nohitter_times, np.mean, size=10000)

# Compute the 95% confidence interval: conf_int
conf_int = np.percentile(bs_replicates, [2.5, 97.5])

# Print the confidence interval
print('95% confidence interval =', conf_int, 'games')

# Plot the histogram of the replicates
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel(r'$\tau$ (games)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()

# 95% confidence interval = [660.67280876 871.63077689] games

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture12-4.png)


 This gives you an estimate of what the typical time between no-hitters is. It could be anywhere between 660 and 870 games.



###
 Pairs bootstrap


####
 A function to do pairs bootstrap




```

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, deg=1)

    return bs_slope_reps, bs_intercept_reps


```


####
 Pairs bootstrap of literacy/fertility data




```python

# Generate replicates of slope and intercept using pairs bootstrap
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(illiteracy, fertility, size=1000)

# Compute and print 95% CI for slope
print(np.percentile(bs_slope_reps, [2.5, 97.5]))

# Plot the histogram
_ = plt.hist(bs_slope_reps, bins=50, normed=True)
_ = plt.xlabel('slope')
_ = plt.ylabel('PDF')
plt.show()

# [0.04378061 0.0551616 ]

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture13-4.png)

####
 Plotting bootstrap regressions




```python

# Generate array of x-values for bootstrap lines: x
x = np.array([0, 100])

# Plot the bootstrap lines
for i in range(100):
    _ = plt.plot(x,
                 bs_slope_reps[i]*x + bs_intercept_reps[i],
                 linewidth=0.5, alpha=0.2, color='red')

# Plot the data
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')

# Label axes, set the margins, and show the plot
_ = plt.xlabel('illiteracy')
_ = plt.ylabel('fertility')
plt.margins(0.02)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture14-3.png)


 Introduction to hypothesis testing
------------------------------------


###
 Formulating and simulating a hypothesis



**Null hypothesis**
 : another name for the hypothesis you are testing




**Permutation**
 : random reordering of entries in an array



####
 Generating a permutation sample

 np.random.permutation()



 Permutation sampling is a great way to simulate the hypothesis that two variables have identical probability distributions. This is often a hypothesis you want to test, so in this exercise, you will write a function to generate a permutation sample from two data sets.





```

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

```


####
 Visualizing permutation sampling




```

for _ in range(50):
    # Generate permutation samples
    perm_sample_1, perm_sample_2 = permutation_sample(rain_june, rain_november)


    # Compute ECDFs
    x_1, y_1 = ecdf(perm_sample_1)
    x_2, y_2 = ecdf(perm_sample_2)

    # Plot ECDFs of permutation sample
    _ = plt.plot(x_1, y_1, marker='.', linestyle='none',
                 color='red', alpha=0.02)
    _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                 color='blue', alpha=0.02)

# Create and plot ECDFs from original data
x_1, y_1 = ecdf(rain_june)
x_2, y_2 = ecdf(rain_november)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('monthly rainfall (mm)')
_ = plt.ylabel('ECDF')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture1-16.png)


 Notice that the permutation samples ECDFs overlap and give a purple haze. None of the ECDFs from the permutation samples overlap with the observed data, suggesting that the hypothesis is not commensurate with the data. June and November rainfall are not identically distributed.



###
 Test statistics and p-values


####
 Test statistics



 When performing hypothesis tests, your choice of test statistic should be pertinent to the question you are seeking to answer in your hypothesis test.


 The most important thing to consider is:
 **What are you asking?**



####
 What is a p-value?



 The p-value is generally a measure of the probability of observing a test statistic equally or more extreme than the one you observed, given that the null hypothesis is true.



####
 Generating permutation replicates




```python

# In most circumstances, func will be a function you write yourself.
def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates

```


####
 Look before you leap: EDA before hypothesis testing



 Kleinteich and Gorb (
 *Sci. Rep.*
 ,
 **4**
 , 5225, 2014) performed an interesting experiment with South American horned frogs. They held a plate connected to a force transducer, along with a bait fly, in front of them. They then measured the impact force and adhesive force of the frog’s tongue when it struck the target.




 Frog A is an adult and Frog B is a juvenile. The researchers measured the impact force of 20 strikes for each frog. In the next exercise, we will test the hypothesis that the two frogs have the same distribution of impact forces. But, remember, it is important to do EDA first! Let’s make a bee swarm plot for the data. They are stored in a Pandas data frame,
 `df`
 , where column
 `ID`
 is the identity of the frog and column
 `impact_force`
 is the impact force in Newtons (N).





```

df.head()
   ID  impact_force
20  A         1.612
21  A         0.605
22  A         0.327
23  A         0.946
24  A         0.541

```




```python

# Make bee swarm plot
_ = sns.swarmplot(x='ID', y='impact_force', data=df)

# Label axes
_ = plt.xlabel('frog')
_ = plt.ylabel('impact force (N)')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture2-15.png)


 Eyeballing it, it does not look like they come from the same distribution. Frog A, the adult, has three or four very hard strikes, and Frog B, the juvenile, has a couple weak ones. However, it is possible that with only 20 samples it might be too difficult to tell if they have difference distributions, so we should proceed with the hypothesis test.



####
 Permutation test on frog data



 The average strike force of Frog A was 0.71 Newtons (N), and that of Frog B was 0.42 N for a difference of 0.29 N. It is possible the frogs strike with the same force and this observed difference was by chance. You will compute the probability of getting at least a 0.29 N difference in mean strike force under the hypothesis that the distributions of strike forces for the two frogs are identical. We use a permutation test with a test statistic of the difference of means to test this hypothesis.





```

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff

# Compute difference of mean impact force from experiment: empirical_diff_means
empirical_diff_means = diff_of_means(force_a, force_b)

# Draw 10,000 permutation replicates: perm_replicates
perm_replicates = draw_perm_reps(force_a, force_b,
                                 diff_of_means, size=10000)

# Compute p-value: p
p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

# Print the result
print('p-value =', p)

# p-value = 0.0063
# p-value = 0.63%

```



 The p-value tells you that there is about a 0.6% chance that you would get the difference of means observed in the experiment if frogs were exactly the same.




 A p-value below 0.01 is typically said to be “statistically significant,” but: warning! warning! warning! You have computed a p-value; it is a number. I encourage you not to distill it to a yes-or-no phrase. p = 0.006 and p = 0.000000006 are both said to be “statistically significant,” but they are definitely not the same!



###
 Bootstrap hypothesis tests



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture3-14.png)

####
 A one-sample bootstrap hypothesis test



 Another juvenile frog was studied, Frog C, and you want to see if Frog B and Frog C have similar impact forces. Unfortunately, you do not have Frog C’s impact forces available, but you know they have a mean of 0.55 N. Because you don’t have the original data, you cannot do a permutation test, and you cannot assess the hypothesis that the forces from Frog B and Frog C come from the same distribution. You will therefore test another, less restrictive hypothesis: The mean strike force of Frog B is equal to that of Frog C.




 To set up the bootstrap hypothesis test, you will take the mean as our test statistic. Remember, your goal is to calculate the probability of getting a mean impact force less than or equal to what was observed for Frog B
 *if the hypothesis that the true mean of Frog B’s impact forces is equal to that of Frog C is true*
 . You first translate all of the data of Frog B such that the mean is 0.55 N. This involves adding the mean force of Frog C and subtracting the mean force of Frog B from each measurement of Frog B. This leaves other properties of Frog B’s distribution, such as the variance, unchanged.





```python

# Make an array of translated impact forces: translated_force_b
translated_force_b = force_b - np.mean(force_b) + 0.55

# Take bootstrap replicates of Frog B's translated impact forces: bs_replicates
bs_replicates = draw_bs_reps(translated_force_b, np.mean, 10000)

# Compute fraction of replicates that are less than the observed Frog B force: p
p = np.sum(bs_replicates <= np.mean(force_b)) / 10000

# Print the p-value
print('p = ', p)

# p =  0.0046
# p = 0.46%

```



 The low p-value suggests that the null hypothesis that Frog B and Frog C have the same mean impact force is false.



####
 A two-sample bootstrap hypothesis test for difference of means



 We now want to test the hypothesis that Frog A and Frog B have the same mean impact force, but not necessarily the same distribution, which is also impossible with a permutation test.




 To do the two-sample bootstrap test, we shift
 *both*
 arrays to have the same mean, since we are simulating the hypothesis that their means are, in fact, equal. We then draw bootstrap samples out of the shifted arrays and compute the difference in means. This constitutes a bootstrap replicate, and we generate many of them. The p-value is the fraction of replicates with a difference in means greater than or equal to what was observed.





```python

# Compute mean of all forces: mean_force
mean_force = np.mean(forces_concat)

# Generate shifted arrays
force_a_shifted = force_a - np.mean(force_a) + mean_force
force_b_shifted = force_b - np.mean(force_b) + mean_force

# Compute 10,000 bootstrap replicates from shifted arrays
bs_replicates_a = draw_bs_reps(force_a_shifted, np.mean, size=10000)
bs_replicates_b = draw_bs_reps(force_b_shifted, np.mean, size=10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_replicates_a - bs_replicates_b

# Compute and print p-value: p
p = np.sum(bs_replicates >= empirical_diff_means) / 10000
print('p-value =', p)

# p-value = 0.0043
# p-value = 0.43%

```



 You got a similar result as when you did the permutation test. Nonetheless, remember that it is important to carefully think about what question you want to ask. Are you only interested in the mean impact force, or in the distribution of impact forces?




 Hypothesis test examples
--------------------------


###
 A/B testing


####
 The vote for the Civil Rights Act in 1964



 The Civil Rights Act of 1964 was one of the most important pieces of legislation ever passed in the USA. Excluding “present” and “abstain” votes, 153 House Democrats and 136 Republicans voted yea. However, 91 Democrats and 35 Republicans voted nay. Did party affiliation make a difference in the vote?




 To answer this question, you will evaluate the hypothesis that the party of a House member has no bearing on his or her vote. You will use the fraction of Democrats voting in favor as your test statistic and evaluate the probability of observing a fraction of Democrats voting in favor at least as small as the observed fraction of 153/244. (That’s right, at least as
 *small*
 as. In 1964, it was the
 *Democrats*
 who were less progressive on civil rights issues.) To do this, permute the party labels of the House voters and then arbitrarily divide them into “Democrats” and “Republicans” and compute the fraction of Democrats voting yea.





```python

# Construct arrays of data: dems, reps
dems = np.array([True] * 153 + [False] * 91)
reps = np.array([True] * 136 + [False] * 35)

def frac_yea_dems(dems, reps):
    """Compute fraction of Democrat yea votes."""
    frac = np.sum(dems) / len(dems)
    return frac

# Acquire permutation samples: perm_replicates
perm_replicates = draw_perm_reps(dems, reps, frac_yea_dems, size=10000)

# Compute and print p-value: p
p = np.sum(perm_replicates <= 153/244) / len(perm_replicates)
print('p-value =', p)

# p-value = 0.0002
# p-value = 0.02%

```



 This small p-value suggests that party identity had a lot to do with the voting. Importantly, the South had a higher fraction of Democrat representatives, and consequently also a more racist bias.



####
 What is equivalent?



 You have experience matching a stories to probability distributions. Similarly, you use the same procedure for two different A/B tests if their stories match. In the Civil Rights Act example you just did, you performed an A/B test on voting data, which has a Yes/No type of outcome for each subject (in that case, a voter). Which of the following situations involving testing by a web-based company has an equivalent set up for an A/B test as the one you just did with the Civil Rights Act of 1964?




 You measure the number of people who click on an ad on your company’s website before and after changing its color.




 The “Democrats” are those who view the ad before the color change, and the “Republicans” are those who view it after.



####
 A time-on-website analog



 It turns out that you already did a hypothesis test analogous to an A/B test where you are interested in how much time is spent on the website before and after an ad campaign. The frog tongue force (a continuous quantity like time on the website) is an analog. “Before” = Frog A and “after” = Frog B. Let’s practice this again with something that actually is a before/after scenario.




 We return to the no-hitter data set. In 1920, Major League Baseball implemented important rule changes that ended the so-called dead ball era. Importantly, the pitcher was no longer allowed to spit on or scuff the ball, an activity that greatly favors pitchers. In this problem you will perform an A/B test to determine if these rule changes resulted in a slower rate of no-hitters (i.e., longer average time between no-hitters) using the difference in mean inter-no-hitter time as your test statistic. The inter-no-hitter times for the respective eras are stored in the arrays
 `nht_dead`
 and
 `nht_live`
 , where “nht” is meant to stand for “no-hitter time.”





```

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates



# Compute the observed difference in mean inter-no-hitter times: nht_diff_obs
nht_diff_obs = diff_of_means(nht_dead, nht_live)

# Acquire 10,000 permutation replicates of difference in mean no-hitter time: perm_replicates
perm_replicates = draw_perm_reps(nht_dead, nht_live, diff_of_means, size=10000)

# Compute and print the p-value: p
p = np.sum(perm_replicates <= nht_diff_obs) /len(perm_replicates)
print('p-val =', p)

p-val = 0.0001

```



 Your p-value is 0.0001, which means that only one out of your 10,000 replicates had a result as extreme as the actual difference between the dead ball and live ball eras. This suggests strong statistical significance. Watch out, though, you could very well have gotten zero replicates that were as extreme as the observed value. This just means that the p-value is quite small, almost certainly smaller than 0.001.



####
 What should you have done first?



 That was a nice hypothesis test you just did to check out whether the rule changes in 1920 changed the rate of no-hitters. But what
 *should*
 you have done with the data first?




 Performed EDA, perhaps plotting the ECDFs of inter-no-hitter times in the dead ball and live ball eras.




 Always a good idea to do first! I encourage you to go ahead and plot the ECDFs right now. You will see by eye that the null hypothesis that the distributions are the same is almost certainly not true.





```python

# Create and plot ECDFs
x_1, y_1 = ecdf(nht_dead)
x_2, y_2 = ecdf(nht_live)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('Games between no-hitters')
_ = plt.ylabel('ECDF')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture1-17.png)

###
 Test of correlation



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture2-16.png)

####
 Simulating a null hypothesis concerning correlation



 The observed correlation between female illiteracy and fertility in the data set of 162 countries may just be by chance; the fertility of a given country may actually be totally independent of its illiteracy. You will test this null hypothesis in the next exercise.




 To do the test, you need to simulate the data assuming the null hypothesis is true. Of the following choices, which is the best way to to do it?




 Answer: Do a permutation test: Permute the illiteracy values but leave the fertility values fixed to generate a new set of (illiteracy, fertility) data.




 This exactly simulates the null hypothesis and does so more efficiently than the last option. It is exact because it uses all data and eliminates any correlation because which illiteracy value pairs to which fertility value is shuffled.




 Last option: Do a permutation test: Permute both the illiteracy and fertility values to generate a new set of (illiteracy, fertility data). This exactly simulates the null hypothesis and does so more efficiently than the last option. It is exact because it uses all data and eliminates any correlation because which illiteracy value pairs to which fertility value is shuffled.



####
 Hypothesis test on Pearson correlation



 The observed correlation between female illiteracy and fertility may just be by chance; the fertility of a given country may actually be totally independent of its illiteracy. You will test this hypothesis. To do so, permute the illiteracy values but leave the fertility values fixed. This simulates the hypothesis that they are totally independent of each other. For each permutation, compute the Pearson correlation coefficient and assess how many of your permutation replicates have a Pearson correlation coefficient greater than the observed one.





```

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0,1]

# Compute observed correlation: r_obs
r_obs = pearson_r(illiteracy, fertility)

# r_obs = 0.8041324026815344

# Initialize permutation replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute illiteracy measurments: illiteracy_permuted
    illiteracy_permuted = np.random.permutation(illiteracy)

    # Compute Pearson correlation
    perm_replicates[i] = pearson_r(illiteracy_permuted, fertility)

# Compute p-value: p
p = np.sum(perm_replicates >= r_obs) /len(perm_replicates)
print('p-val =', p)

# p-val = 0.0

```



 You got a p-value of zero. In hacker statistics, this means that your p-value is very low, since you never got a single replicate in the 10,000 you took that had a Pearson correlation greater than the observed one. You could try increasing the number of replicates you take to continue to move the upper bound on your p-value lower and lower.



####
 Do neonicotinoid insecticides have unintended consequences?



 As a final exercise in hypothesis testing before we put everything together in our case study in the next chapter, you will investigate the effects of neonicotinoid insecticides on bee reproduction. These insecticides are very widely used in the United States to combat aphids and other pests that damage plants.




 In a recent study, Straub, et al. (
 [*Proc. Roy. Soc. B*
 , 2016](http://dx.doi.org/10.1098/rspb.2016.0506)
 ) investigated the effects of neonicotinoids on the sperm of pollinating bees. In this and the next exercise, you will study how the pesticide treatment affected the count of live sperm per half milliliter of semen.




 First, we will do EDA, as usual. Plot ECDFs of the alive sperm count for untreated bees (stored in the Numpy array
 `control`
 ) and bees treated with pesticide (stored in the Numpy array
 `treated`
 ).





```python

# Compute x,y values for ECDFs
x_control, y_control = ecdf(control)
x_treated, y_treated = ecdf(treated)

# Plot the ECDFs
plt.plot(x_control, y_control, marker='.', linestyle='none')
plt.plot(x_treated, y_treated, marker='.', linestyle='none')

# Set the margins
plt.margins(0.02)

# Add a legend
plt.legend(('control', 'treated'), loc='lower right')

# Label axes and show plot
plt.xlabel('millions of alive sperm per mL')
plt.ylabel('ECDF')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture3-15.png)


 The ECDFs show a pretty clear difference between the treatment and control; treated bees have fewer alive sperm. Let’s now do a hypothesis test in the next exercise.



####
 Bootstrap hypothesis test on bee sperm counts



 Now, you will test the following hypothesis:




**On average, male bees treated with neonicotinoid insecticide have the same number of active sperm per milliliter of semen than do untreated male bees.**




 You will use the difference of means as your test statistic.





```

def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates


# Compute the difference in mean sperm count: diff_means
diff_means = np.mean(control) - np.mean(treated)

# Compute mean of pooled data: mean_count
mean_count = np.mean(np.concatenate((control, treated)))

# Generate shifted data sets
control_shifted = control - np.mean(control) + mean_count
treated_shifted = treated - np.mean(treated) + mean_count

# Generate bootstrap replicates
bs_reps_control = draw_bs_reps(control_shifted,
                       np.mean, size=10000)
bs_reps_treated = draw_bs_reps(treated_shifted,
                       np.mean, size=10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_reps_control - bs_reps_treated

# Compute and print p-value: p
p = np.sum(bs_replicates >= np.mean(control) - np.mean(treated)) \
            / len(bs_replicates)
print('p-value =', p)

# p-value = 0.0

```



 The p-value is small, most likely less than 0.0001, since you never saw a bootstrap replicated with a difference of means at least as extreme as what was observed. In fact, when I did the calculation with 10 million replicates, I got a p-value of
 `2e-05`




 Putting it all together: a case study
---------------------------------------


###
 Finch beaks and the need for statistics



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture4-14.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture5-12.png)

####
 EDA of beak depths of Darwin’s finches



 For your first foray into the Darwin finch data, you will study how the beak depth (the distance, top to bottom, of a closed beak) of the finch species
 *Geospiza scandens*
 has changed over time. The Grants have noticed some changes of beak geometry depending on the types of seeds available on the island, and they also noticed that there was some interbreeding with another major species on Daphne Major,
 *Geospiza fortis*
 . These effects can lead to changes in the species over time.




 In the next few problems, you will look at the beak depth of
 *G. scandens*
 on Daphne Major in 1975 and in 2012. To start with, let’s plot all of the beak depth measurements in 1975 and 2012 in a bee swarm plot.




 The data are stored in a pandas DataFrame called
 `df`
 with columns
 `'year'`
 and
 `'beak_depth'`
 . The units of beak depth are millimeters (mm).





```

df.head()
   beak_depth  year
0         8.4  1975
1         8.8  1975
2         8.4  1975
3         8.0  1975
4         7.9  1975

```




```python

# Create bee swarm plot
_ = sns.swarmplot('year', 'beak_depth', data=df)

# Label the axes
_ = plt.xlabel('year')
_ = plt.ylabel('beak depth (mm)')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture6-9.png)


 It is kind of hard to see if there is a clear difference between the 1975 and 2012 data set. Eyeballing it, it appears as though the mean of the 2012 data set might be slightly higher, and it might have a bigger variance.



####
 ECDFs of beak depths



 While bee swarm plots are useful, we found that ECDFs are often even better when doing EDA. Plot the ECDFs for the 1975 and 2012 beak depth measurements on the same plot.





```python

# Compute ECDFs
x_1975, y_1975 = ecdf(bd_1975)
x_2012, y_2012 = ecdf(bd_2012)

# Plot the ECDFs
_ = plt.plot(x_1975, y_1975, marker='.', linestyle='none')
_ = plt.plot(x_2012, y_2012, marker='.', linestyle='none')

# Set margins
plt.margins(0.02)

# Add axis labels and legend
_ = plt.xlabel('beak depth (mm)')
_ = plt.ylabel('ECDF')
_ = plt.legend(('1975', '2012'), loc='lower right')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture7-10.png)


 The differences are much clearer in the ECDF. The mean is larger in the 2012 data, and the variance does appear larger as well.



####
 Parameter estimates of beak depths



 Estimate the
 *difference*
 of the mean beak depth of the
 *G. scandens*
 samples from 1975 and 2012 and report a 95% confidence interval.





```

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates


# Compute the difference of the sample means: mean_diff
mean_diff = np.mean(bd_2012) - np.mean(bd_1975)

# Get bootstrap replicates of means
bs_replicates_1975 = draw_bs_reps(bd_1975, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(bd_2012, np.mean, size=10000)

# Compute samples of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute 95% confidence interval: conf_int
conf_int = np.percentile(bs_diff_replicates, [2.5, 97.5])

# Print the results
print('difference of means =', mean_diff, 'mm')
print('95% confidence interval =', conf_int, 'mm')

# difference of means = 0.22622047244094645 mm
# 95% confidence interval = [0.05633521 0.39190544] mm

```


####
 Hypothesis test: Are beaks deeper in 2012?



 Your plot of the ECDF and determination of the confidence interval make it pretty clear that the beaks of
 *G. scandens*
 on Daphne Major have gotten deeper. But is it possible that this effect is just due to random chance? In other words, what is the probability that we would get the observed difference in mean beak depth if the means were the same?




 Be careful! The hypothesis we are testing is
 *not*
 that the beak depths come from the same distribution. For that we could use a permutation test.
 **The hypothesis is that the means are equal.**
 To perform this hypothesis test, we need to shift the two data sets so that they have the same mean and then use bootstrap sampling to compute the difference of means.





```python

# Compute mean of combined data set: combined_mean
combined_mean = np.mean(np.concatenate((bd_1975, bd_2012)))

# Shift the samples
# why shift the mean?
# to make np.mean(bd_1975_shifted) - np.mean(bd_2012_shifted) = 0 #1
# why make #1 = 0?
# because our hypothesis is "beak depth are the same in 1975 and 2012"
bd_1975_shifted = bd_1975 - np.mean(bd_1975) + combined_mean
bd_2012_shifted = bd_2012 - np.mean(bd_2012) + combined_mean

# Get bootstrap replicates of shifted data sets
bs_replicates_1975 = draw_bs_reps(bd_1975_shifted, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(bd_2012_shifted, np.mean, size=10000)

# Compute replicates of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute the p-value
p = np.sum(bs_diff_replicates >= mean_diff) / len(bs_diff_replicates)

# Print p-value
print('p =', p)

# p = 0.0034
# p = 0.34%

```



 We get a p-value of 0.0034, which suggests that there is a statistically significant difference. But remember: it is very important to know how different they are! In the previous exercise, you got a difference of 0.2 mm between the means. You should combine this with the statistical significance. Changing by 0.2 mm in 37 years is substantial by evolutionary standards. If it kept changing at that rate, the beak depth would double in only 400 years.



###
 Variation of beak shapes


####
 EDA of beak length and depth



 The beak length data are stored as
 `bl_1975`
 and
 `bl_2012`
 , again with units of millimeters (mm). You still have the beak depth data stored in
 `bd_1975`
 and
 `bd_2012`
 . Make scatter plots of beak depth (y-axis) versus beak length (x-axis) for the 1975 and 2012 specimens.





```python

# Make scatter plot of 1975 data
_ = plt.plot(bl_1975, bd_1975, marker='.',
             linestyle='None', color='blue', alpha=0.5)

# Make scatter plot of 2012 data
_ = plt.plot(bl_2012, bd_2012, marker='.',
            linestyle='None', color='red', alpha=0.5)

# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture8-7.png)


 In looking at the plot, we see that beaks got deeper (the red points are higher up in the y-direction), but not really longer. If anything, they got a bit shorter, since the red dots are to the left of the blue dots. So, it does not look like the beaks kept the same shape; they became shorter and deeper.



####
 Linear regressions



 Perform a linear regression for both the 1975 and 2012 data. Then, perform pairs bootstrap estimates for the regression parameters. Report 95% confidence intervals on the slope and intercept of the regression line.





```

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, deg=1)

    return bs_slope_reps, bs_intercept_reps


# Compute the linear regressions
slope_1975, intercept_1975 = np.polyfit(bl_1975, bd_1975, deg=1)
slope_2012, intercept_2012 = np.polyfit(bl_2012, bd_2012, deg=1)

# Perform pairs bootstrap for the linear regressions
bs_slope_reps_1975, bs_intercept_reps_1975 = draw_bs_pairs_linreg(bl_1975, bd_1975, size=1000)
bs_slope_reps_2012, bs_intercept_reps_2012 = draw_bs_pairs_linreg(bl_2012, bd_2012, size=1000)

# Compute confidence intervals of slopes
slope_conf_int_1975 = np.percentile(bs_slope_reps_1975, [2.5, 97.5])
slope_conf_int_2012 = np.percentile(bs_slope_reps_2012, [2.5, 97.5])
intercept_conf_int_1975 = np.percentile(bs_intercept_reps_1975, [2.5, 97.5])
intercept_conf_int_2012 = np.percentile(bs_intercept_reps_2012, [2.5, 97.5])


# Print the results
print('1975: slope =', slope_1975,
      'conf int =', slope_conf_int_1975)
print('1975: intercept =', intercept_1975,
      'conf int =', intercept_conf_int_1975)
print('2012: slope =', slope_2012,
      'conf int =', slope_conf_int_2012)
print('2012: intercept =', intercept_2012,
      'conf int =', intercept_conf_int_2012)

#   1975: slope = 0.4652051691605937 conf int = [0.33851226 0.59306491]
#   1975: intercept = 2.3908752365842263 conf int = [0.64892945 4.18037063]
#   2012: slope = 0.462630358835313 conf int = [0.33137479 0.60695527]
#   2012: intercept = 2.977247498236019 conf int = [1.06792753 4.70599387]

```



 It looks like they have the same slope, but different intercepts.



####
 Displaying the linear regression results




```python

# Make scatter plot of 1975 data
_ = plt.plot(bl_1975, bd_1975, marker='.',
             linestyle='none', color='blue', alpha=0.5)

# Make scatter plot of 2012 data
_ = plt.plot(bl_2012, bd_2012, marker='.',
             linestyle='none', color='red', alpha=0.5)

# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')

# Generate x-values for bootstrap lines: x
x = np.array([10, 17])

# Plot the bootstrap lines
for i in range(100):
    plt.plot(x, bs_slope_reps_1975[i] * x + bs_intercept_reps_1975[i],
             linewidth=0.5, alpha=0.2, color='blue')
    plt.plot(x, bs_slope_reps_2012[i] * x + bs_intercept_reps_2012[i],
             linewidth=0.5, alpha=0.2, color='red')

# Draw the plot again
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture9-6.png)

####
 Beak length to depth ratio



 The linear regressions showed interesting information about the beak geometry. The slope was the same in 1975 and 2012, suggesting that for every millimeter gained in beak length, the birds gained about half a millimeter in depth in both years. However, if we are interested in the shape of the beak, we want to compare the
 *ratio*
 of beak length to beak depth. Let’s make that comparison.





```python

# Compute length-to-depth ratios
ratio_1975 = bl_1975 / bd_1975
ratio_2012 = bl_2012 / bd_2012

# Compute means
mean_ratio_1975 = np.mean(ratio_1975)
mean_ratio_2012 = np.mean(ratio_2012)

# Generate bootstrap replicates of the means
bs_replicates_1975 = draw_bs_reps(ratio_1975, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(ratio_2012, np.mean, size=10000)

# Compute the 99% confidence intervals
conf_int_1975 = np.percentile(bs_replicates_1975, [0.5, 99.5])
conf_int_2012 = np.percentile(bs_replicates_2012, [0.5, 99.5])

# Print the results
print('1975: mean ratio =', mean_ratio_1975,
      'conf int =', conf_int_1975)
print('2012: mean ratio =', mean_ratio_2012,
      'conf int =', conf_int_2012)

# 1975: mean ratio = 1.5788823771858533 conf int = [1.55668803 1.60073509]
# 2012: mean ratio = 1.4658342276847767 conf int = [1.44363932 1.48729149]

```


####
 How different is the ratio?



 In the previous exercise, you computed the mean beak length to depth ratio with 99% confidence intervals for 1975 and for 2012. The results of that calculation are shown graphically in the plot accompanying this problem. In addition to these results, what would you say about the ratio of beak length to depth?




![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture10-6.png)


 The mean beak length-to-depth ratio decreased by about 0.1, or 7%, from 1975 to 2012. The 99% confidence intervals are not even close to overlapping, so this is a real change. The beak shape changed.



###
 Calculation of heritability


####
 EDA of heritability



 The array
 `bd_parent_scandens`
 contains the average beak depth (in mm) of two parents of the species
 `G. scandens`
 . The array
 `bd_offspring_scandens`
 contains the average beak depth of the offspring of the respective parents. The arrays
 `bd_parent_fortis`
 and
 `bd_offspring_fortis`
 contain the same information about measurements from
 *G. fortis*
 birds.




 Make a scatter plot of the average offspring beak depth (y-axis) versus average parental beak depth (x-axis) for both species. Use the
 `alpha=0.5`
 keyword argument to help you see overlapping points.





```python

# Make scatter plots
_ = plt.plot(bd_parent_fortis, bd_offspring_fortis,
             marker='.', linestyle='none', color='blue', alpha=0.5)
_ = plt.plot(bd_parent_scandens, bd_offspring_scandens,
             marker='.', linestyle='none', color='red', alpha=0.5)

# Label axes
_ = plt.xlabel('parental beak depth (mm)')
_ = plt.ylabel('offspring beak depth (mm)')

# Add legend
_ = plt.legend(('G. fortis', 'G. scandens'), loc='lower right')

# Show plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture11-6.png)


 It appears as though there is a stronger correlation in
 *G. fortis*
 than than in
 *G. scandens*
 . This suggests that beak depth is more strongly inherited in
 *G. fortis*
 . We’ll quantify this correlation next.



####
 Correlation of offspring and parental data



 In an effort to quantify the correlation between offspring and parent beak depths, we would like to compute statistics, such as the Pearson correlation coefficient, between parents and offspring. To get confidence intervals on this, we need to do a pairs bootstrap.




 You have
 [already written](https://campus.datacamp.com/courses/statistical-thinking-in-python-part-2/bootstrap-confidence-intervals?ex=12)
 a function to do pairs bootstrap to get estimates for parameters derived from linear regression. Your task in this exercise is to make a new function with call signature
 `draw_bs_pairs(x, y, func, size=1)`
 that performs pairs bootstrap and computes a single statistic on pairs samples defined. The statistic of interest is computed by calling
 `func(bs_x, bs_y)`
 . In the next exercise, you will use
 `pearson_r`
 for
 `func`
 .





```

def draw_bs_pairs(x, y, func, size=1):
    """Perform pairs bootstrap for a single statistic."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_replicates[i] = func(bs_x, bs_y)

    return bs_replicates


```


####
 Pearson correlation of offspring and parental data



 The Pearson correlation coefficient seems like a useful measure of how strongly the beak depth of parents are inherited by their offspring. Compute the Pearson correlation coefficient between parental and offspring beak depths for
 *G. scandens*
 . Do the same for
 *G. fortis*
 . Then, use the function you wrote in the last exercise to compute a 95% confidence interval using pairs bootstrap.





```

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0,1]


# Compute the Pearson correlation coefficients
r_scandens = pearson_r(bd_parent_scandens, bd_offspring_scandens)
r_fortis = pearson_r(bd_parent_fortis, bd_offspring_fortis)

# Acquire 1000 bootstrap replicates of Pearson r
bs_replicates_scandens = draw_bs_pairs(bd_parent_scandens, bd_offspring_scandens, pearson_r, size=1000)

bs_replicates_fortis = draw_bs_pairs(bd_parent_fortis, bd_offspring_fortis, pearson_r, size=1000)


# Compute 95% confidence intervals
conf_int_scandens = np.percentile(bs_replicates_scandens, [2.5, 97.5])
conf_int_fortis = np.percentile(bs_replicates_fortis, [2.5, 97.5])

# Print results
print('G. scandens:', r_scandens, conf_int_scandens)
print('G. fortis:', r_fortis, conf_int_fortis)

#    G. scandens: 0.4117063629401258 [0.26564228 0.54388972]
#    G. fortis: 0.7283412395518487 [0.6694112  0.77840616]

```



 It is clear from the confidence intervals that beak depth of the offspring of
 *G. fortis*
 parents is more strongly correlated with their offspring than their
 *G. scandens*
 counterparts.



####
 Measuring heritability



 Remember that the Pearson correlation coefficient is the ratio of the covariance to the geometric mean of the variances of the two data sets. This is a measure of the correlation between parents and offspring, but might not be the best estimate of heritability. If we stop and think, it makes more sense to define heritability as the ratio of the covariance between parent and offspring to the
 *variance of the parents alone*
 . In this exercise, you will estimate the heritability and perform a pairs bootstrap calculation to get the 95% confidence interval.




 This exercise highlights a very important point. Statistical inference (and data analysis in general) is not a plug-n-chug enterprise. You need to think carefully about the questions you are seeking to answer with your data and analyze them appropriately. If you are interested in how heritable traits are, the quantity we defined as the heritability is more apt than the off-the-shelf statistic, the Pearson correlation coefficient.





```

def heritability(parents, offspring):
    """Compute the heritability from parent and offspring samples."""
    covariance_matrix = np.cov(parents, offspring)
    return covariance_matrix[0,1] / covariance_matrix[0,0]

# Compute the heritability
heritability_scandens = heritability(bd_parent_scandens, bd_offspring_scandens)
heritability_fortis = heritability(bd_parent_fortis, bd_offspring_fortis)

# Acquire 1000 bootstrap replicates of heritability
replicates_scandens = draw_bs_pairs(
        bd_parent_scandens, bd_offspring_scandens, heritability, size=1000)

replicates_fortis = draw_bs_pairs(
        bd_parent_fortis, bd_offspring_fortis, heritability, size=1000)


# Compute 95% confidence intervals
conf_int_scandens = np.percentile(replicates_scandens, [2.5, 97.5])
conf_int_fortis = np.percentile(replicates_fortis, [2.5, 97.5])

# Print results
print('G. scandens:', heritability_scandens, conf_int_scandens)
print('G. fortis:', heritability_fortis, conf_int_fortis)


#   G. scandens: 0.5485340868685982 [0.34395487 0.75638267]
#   G. fortis: 0.7229051911438159 [0.64655013 0.79688342]

```



 Here again, we see that
 *G. fortis*
 has stronger heritability than
 *G. scandens*
 . This suggests that the traits of
 *G. fortis*
 may be strongly incorporated into
 *G. scandens*
 by introgressive hybridization.



####
 Is beak depth heritable at all in G. scandens?



 The heritability of beak depth in
 *G. scandens*
 seems low. It could be that this observed heritability was just achieved by chance and
 **beak depth is actually not really heritable in the species**
 . You will test that hypothesis here. To do this, you will do a pairs permutation test.





```python

# Initialize array of replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute parent beak depths
    bd_parent_permuted = np.random.permutation(bd_parent_scandens)
    perm_replicates[i] = heritability(bd_parent_permuted,
                                      bd_offspring_scandens)

# Compute p-value: p
p = np.sum(perm_replicates >= heritability_scandens) / len(perm_replicates)

# Print the p-value
print('p-val =', p)

# p-val = 0.0

```



 You get a p-value of zero, which means that none of the 10,000 permutation pairs replicates you drew had a heritability high enough to match that which was observed. This strongly suggests that beak depth is heritable in
 *G. scandens*
 , just not as much as in
 *G. fortis*
 . If you like, you can plot a histogram of the heritability replicates to get a feel for how extreme of a value of heritability you might expect by chance.





```

plt.hist(perm_replicates)
plt.axvline(x=heritability_scandens, color = 'red')
plt.text(heritability_scandens, 1500, 'heritability_scandens', ha='center', va='center',rotation='vertical', backgroundcolor='white')
plt.show()

```




 Parameter estimation by optimization
--------------------------------------


###
 Optimal parameters


####
 How often do we get no-hitters?



 The number of games played between each no-hitter in the modern era (1901-2015) of Major League Baseball is stored in the array
 `nohitter_times`
 .




 If you assume that no-hitters are described as a Poisson process, then the time between no-hitters is Exponentially distributed. As you have seen, the Exponential distribution has a single parameter, which we will call ττ, the typical interval time. The value of the parameter ττ that makes the exponential distribution best match the data is the mean interval time (where time is in units of number of games) between no-hitters.




 Compute the value of this parameter from the data. Then, use
 `np.random.exponential()`
 to “repeat” the history of Major League Baseball by drawing inter-no-hitter times from an exponential distribution with the ττ you found and plot the histogram as an approximation to the PDF.







```python

# Seed random number generator
np.random.seed(42)

# Compute mean no-hitter time: tau
tau = np.mean(nohitter_times)

# Draw out of an exponential distribution with parameter tau: inter_nohitter_time
inter_nohitter_time = np.random.exponential(tau, 100000)

# Plot the PDF and label axes
_ = plt.hist(inter_nohitter_time,
             bins=50, normed=True, histtype='step')
_ = plt.xlabel('Games between no-hitters')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture-16.png)


 We see the typical shape of the Exponential distribution, going from a maximum at 0 and decaying to the right.



####
 Do the data follow our story?




```python

# Create an ECDF from real data: x, y
x, y = ecdf(nohitter_times)

# Create a CDF from theoretical samples: x_theor, y_theor
x_theor, y_theor = ecdf(inter_nohitter_time)

# Overlay the plots
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')

# Margins and axis labels
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture1-15.png)


 It looks like no-hitters in the modern era of Major League Baseball are Exponentially distributed. Based on the story of the Exponential distribution, this suggests that they are a random process; when a no-hitter will happen is independent of when the last no-hitter was.



####
 How is this parameter optimal?




```python

# Plot the theoretical CDFs
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# Take samples with half tau: samples_half
samples_half = np.random.exponential(tau/2, size=10000)

# Take samples with double tau: samples_double
samples_double = np.random.exponential(2*tau, size=10000)

# Generate CDFs from these samples
x_half, y_half = ecdf(samples_half)
x_double, y_double = ecdf(samples_double)

# Plot these CDFs as lines
_ = plt.plot(x_half, y_half)
_ = plt.plot(x_double, y_double)

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture2-14.png)

 red = half, purple = double



 Notice how the value of tau given by the mean matches the data best. In this way, tau is an optimal parameter.



###
 Linear regression by least squares


####
 EDA of literacy/fertility data




```python

# Plot the illiteracy rate versus fertility
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')

# Set the margins and label axes
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')

# Show the plot
plt.show()

# Show the Pearson correlation coefficient
print(pearson_r(illiteracy, fertility))
0.8041324026815344

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture3-13.png)


 You can see the correlation between illiteracy and fertility by eye, and by the substantial Pearson correlation coefficient of 0.8. It is difficult to resolve in the scatter plot, but there are many points around near-zero illiteracy and about 1.8 children/woman.



####
 Linear regression




```python

# Plot the illiteracy rate versus fertility
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')

# Perform a linear regression using np.polyfit(): a, b
a, b = np.polyfit(illiteracy, fertility, deg=1)

# Print the results to the screen
print('slope =', a, 'children per woman / percent illiterate')
print('intercept =', b, 'children per woman')

# Make theoretical line to plot
x = np.array([0, 100])
y = a * x + b

# Add regression line to your plot
_ = plt.plot(x, y)

# Draw the plot
plt.show()


# slope = 0.04979854809063423 children per woman / percent illiterate
# intercept = 1.888050610636557 children per woman


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture4-13.png)

####
 How is it optimal?




```python

# Specify slopes to consider: a_vals
a_vals = np.linspace(0, 0.1, 200)

# Initialize sum of square of residuals: rss
rss = np.empty_like(a_vals)

# Compute sum of square of residuals for each value of a_vals
for i, a in enumerate(a_vals):
    rss[i] = np.sum((fertility - a*illiteracy - b)**2)

# Plot the RSS
plt.plot(a_vals, rss, '-')
plt.xlabel('slope (children per woman / percent illiterate)')
plt.ylabel('sum of square of residuals')

plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture5-11.png)


 Notice that the minimum on the plot, that is the value of the slope that gives the minimum sum of the square of the residuals, is the same value you got when performing the regression.



###
 The importance of EDA: Anscombe’s quartet



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture6-8.png)

####
 The importance of EDA



 Why should exploratory data analysis be the first step in an analysis of data (after getting your data imported and cleaned, of course)?



* You can be protected from misinterpretation of the type demonstrated by Anscombe’s quartet.
* EDA provides a good starting point for planning the rest of your analysis.
* EDA is not really any more difficult than any of the subsequent analysis, so there is no excuse for not exploring the data.


####
 Linear regression on appropriate Anscombe data




```python

# Perform linear regression: a, b
a, b = np.polyfit(x, y, deg=1)

# Print the slope and intercept
print(a, b)

# Generate theoretical x and y data: x_theor, y_theor
x_theor = np.array([3, 15])
y_theor = a * x_theor + b

# Plot the Anscombe data and theoretical line
_ = plt.plot(x, y, marker = '.', linestyle = 'none')
_ = plt.plot(x_theor, y_theor)

# Label the axes
plt.xlabel('x')
plt.ylabel('y')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture7-9.png)

####
 Linear regression on all Anscombe data




```

for i in range(4):
    plt.subplot(2,2,i+1)

    # plot the scatter plot
    plt.plot(anscombe_x[i], anscombe_y[i], marker = '.', linestyle = 'none')

    # plot the regression line
    a, b = np.polyfit(anscombe_x[i], anscombe_y[i], deg=1)
    x_theor = np.array([np.min(anscombe_x[i]), np.max(anscombe_x[i])])
    y_theor = a * x_theor + b
    plt.plot(x_theor, y_theor)

    # add label
    plt.xlabel('x' + str(i+1))
    plt.ylabel('y' + str(i+1))

plt.show()

# slope1: 0.5000909090909095 intercept: 3.000090909090909
# slope2: 0.5000000000000004 intercept: 3.0009090909090896
# slope3: 0.4997272727272731 intercept: 3.0024545454545453
# slope4: 0.4999090909090908 intercept: 3.0017272727272735

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture8-6.png)

###
 Generating bootstrap replicates


####
 Getting the terminology down



 If we have a data set with n repeated measurements, a
 **bootstrap sample**
 is an array of length n that was drawn from the original data with replacement.




**Bootstrap replicate**
 is a single value of a statistic computed from a bootstrap sample.



####
 Visualizing bootstrap samples

 np.random.choice()




```

for _ in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(rainfall, size=len(rainfall))

    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)

# Compute and plot ECDF from original data
x, y = ecdf(rainfall)
_ = plt.plot(x, y, marker='.')

# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('yearly rainfall (mm)')
_ = plt.ylabel('ECDF')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture9-5.png)


 Bootstrap confidence intervals
--------------------------------


####
 Generating many bootstrap replicates




```

def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates

```


####
 Bootstrap replicates of the mean and the SEM (
 **standard error of the mean**
 )



 In fact, it can be shown theoretically that under not-too-restrictive conditions, the value of the mean will always be Normally distributed. (This does not hold in general, just for the mean and a few other statistics.)




 The standard deviation of this distribution, called the
 **standard error of the mean**
 , or SEM, is given by the standard deviation of the data divided by the square root of the number of data points. I.e., for a data set,
 `sem = np.std(data) / np.sqrt(len(data))`
 . Using hacker statistics, you get this same result without the need to derive it, but you will verify this result from your bootstrap replicates.





```python

# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates = draw_bs_reps(rainfall, np.mean, size=10000)

# Compute and print SEM
sem = np.std(rainfall) / np.sqrt(len(rainfall))
print(sem)

# Compute and print standard deviation of bootstrap replicates
bs_std = np.std(bs_replicates)
print(bs_std)

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('mean annual rainfall (mm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()

# 10.51054915050619
# 10.465927071184412

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture10-5.png)

####
 Confidence intervals of rainfall data



 Use the bootstrap replicates you just generated to compute the 95% confidence interval. That is, give the 2.5th and 97.5th percentile of your bootstrap replicates stored as
 `bs_replicates`
 . What is the 95% confidence interval?





```

np.percentile(bs_replicates,2.5)
779.7699248120301

np.percentile(bs_replicates,97.5)
820.950432330827

```


####
 Bootstrap replicates of other statistics




```

def draw_bs_reps(data, func, size=1):
    return np.array([bootstrap_replicate_1d(data, func) for _ in range(size)])

# Generate 10,000 bootstrap replicates of the variance: bs_replicates
bs_replicates = draw_bs_reps(rainfall, np.var, size=10000)

# Put the variance in units of square centimeters
bs_replicates /= 100

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('variance of annual rainfall (sq. cm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture11-5.png)


 This is not normally distributed, as it has a longer tail to the right. Note that you can also compute a confidence interval on the variance, or any other statistic, using
 `np.percentile()`
 with your bootstrap replicates.



####
 Confidence interval on the rate of no-hitters




```python

# Draw bootstrap replicates of the mean no-hitter time (equal to tau): bs_replicates
bs_replicates = draw_bs_reps(nohitter_times, np.mean, size=10000)

# Compute the 95% confidence interval: conf_int
conf_int = np.percentile(bs_replicates, [2.5, 97.5])

# Print the confidence interval
print('95% confidence interval =', conf_int, 'games')

# Plot the histogram of the replicates
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel(r'$\tau$ (games)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()

# 95% confidence interval = [660.67280876 871.63077689] games

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture12-4.png)


 This gives you an estimate of what the typical time between no-hitters is. It could be anywhere between 660 and 870 games.



###
 Pairs bootstrap


####
 A function to do pairs bootstrap




```

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, deg=1)

    return bs_slope_reps, bs_intercept_reps


```


####
 Pairs bootstrap of literacy/fertility data




```python

# Generate replicates of slope and intercept using pairs bootstrap
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(illiteracy, fertility, size=1000)

# Compute and print 95% CI for slope
print(np.percentile(bs_slope_reps, [2.5, 97.5]))

# Plot the histogram
_ = plt.hist(bs_slope_reps, bins=50, normed=True)
_ = plt.xlabel('slope')
_ = plt.ylabel('PDF')
plt.show()

# [0.04378061 0.0551616 ]

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture13-4.png)

####
 Plotting bootstrap regressions




```python

# Generate array of x-values for bootstrap lines: x
x = np.array([0, 100])

# Plot the bootstrap lines
for i in range(100):
    _ = plt.plot(x,
                 bs_slope_reps[i]*x + bs_intercept_reps[i],
                 linewidth=0.5, alpha=0.2, color='red')

# Plot the data
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')

# Label axes, set the margins, and show the plot
_ = plt.xlabel('illiteracy')
_ = plt.ylabel('fertility')
plt.margins(0.02)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture14-3.png)


 Introduction to hypothesis testing
------------------------------------


###
 Formulating and simulating a hypothesis



**Null hypothesis**
 : another name for the hypothesis you are testing




**Permutation**
 : random reordering of entries in an array



####
 Generating a permutation sample

 np.random.permutation()



 Permutation sampling is a great way to simulate the hypothesis that two variables have identical probability distributions. This is often a hypothesis you want to test, so in this exercise, you will write a function to generate a permutation sample from two data sets.





```

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

```


####
 Visualizing permutation sampling




```

for _ in range(50):
    # Generate permutation samples
    perm_sample_1, perm_sample_2 = permutation_sample(rain_june, rain_november)


    # Compute ECDFs
    x_1, y_1 = ecdf(perm_sample_1)
    x_2, y_2 = ecdf(perm_sample_2)

    # Plot ECDFs of permutation sample
    _ = plt.plot(x_1, y_1, marker='.', linestyle='none',
                 color='red', alpha=0.02)
    _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                 color='blue', alpha=0.02)

# Create and plot ECDFs from original data
x_1, y_1 = ecdf(rain_june)
x_2, y_2 = ecdf(rain_november)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('monthly rainfall (mm)')
_ = plt.ylabel('ECDF')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture1-16.png)


 Notice that the permutation samples ECDFs overlap and give a purple haze. None of the ECDFs from the permutation samples overlap with the observed data, suggesting that the hypothesis is not commensurate with the data. June and November rainfall are not identically distributed.



###
 Test statistics and p-values


####
 Test statistics



 When performing hypothesis tests, your choice of test statistic should be pertinent to the question you are seeking to answer in your hypothesis test.


 The most important thing to consider is:
 **What are you asking?**



####
 What is a p-value?



 The p-value is generally a measure of the probability of observing a test statistic equally or more extreme than the one you observed, given that the null hypothesis is true.



####
 Generating permutation replicates




```python

# In most circumstances, func will be a function you write yourself.
def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates

```


####
 Look before you leap: EDA before hypothesis testing



 Kleinteich and Gorb (
 *Sci. Rep.*
 ,
 **4**
 , 5225, 2014) performed an interesting experiment with South American horned frogs. They held a plate connected to a force transducer, along with a bait fly, in front of them. They then measured the impact force and adhesive force of the frog’s tongue when it struck the target.




 Frog A is an adult and Frog B is a juvenile. The researchers measured the impact force of 20 strikes for each frog. In the next exercise, we will test the hypothesis that the two frogs have the same distribution of impact forces. But, remember, it is important to do EDA first! Let’s make a bee swarm plot for the data. They are stored in a Pandas data frame,
 `df`
 , where column
 `ID`
 is the identity of the frog and column
 `impact_force`
 is the impact force in Newtons (N).





```

df.head()
   ID  impact_force
20  A         1.612
21  A         0.605
22  A         0.327
23  A         0.946
24  A         0.541

```




```python

# Make bee swarm plot
_ = sns.swarmplot(x='ID', y='impact_force', data=df)

# Label axes
_ = plt.xlabel('frog')
_ = plt.ylabel('impact force (N)')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture2-15.png)


 Eyeballing it, it does not look like they come from the same distribution. Frog A, the adult, has three or four very hard strikes, and Frog B, the juvenile, has a couple weak ones. However, it is possible that with only 20 samples it might be too difficult to tell if they have difference distributions, so we should proceed with the hypothesis test.



####
 Permutation test on frog data



 The average strike force of Frog A was 0.71 Newtons (N), and that of Frog B was 0.42 N for a difference of 0.29 N. It is possible the frogs strike with the same force and this observed difference was by chance. You will compute the probability of getting at least a 0.29 N difference in mean strike force under the hypothesis that the distributions of strike forces for the two frogs are identical. We use a permutation test with a test statistic of the difference of means to test this hypothesis.





```

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff

# Compute difference of mean impact force from experiment: empirical_diff_means
empirical_diff_means = diff_of_means(force_a, force_b)

# Draw 10,000 permutation replicates: perm_replicates
perm_replicates = draw_perm_reps(force_a, force_b,
                                 diff_of_means, size=10000)

# Compute p-value: p
p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

# Print the result
print('p-value =', p)

# p-value = 0.0063
# p-value = 0.63%

```



 The p-value tells you that there is about a 0.6% chance that you would get the difference of means observed in the experiment if frogs were exactly the same.




 A p-value below 0.01 is typically said to be “statistically significant,” but: warning! warning! warning! You have computed a p-value; it is a number. I encourage you not to distill it to a yes-or-no phrase. p = 0.006 and p = 0.000000006 are both said to be “statistically significant,” but they are definitely not the same!



###
 Bootstrap hypothesis tests



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture3-14.png)

####
 A one-sample bootstrap hypothesis test



 Another juvenile frog was studied, Frog C, and you want to see if Frog B and Frog C have similar impact forces. Unfortunately, you do not have Frog C’s impact forces available, but you know they have a mean of 0.55 N. Because you don’t have the original data, you cannot do a permutation test, and you cannot assess the hypothesis that the forces from Frog B and Frog C come from the same distribution. You will therefore test another, less restrictive hypothesis: The mean strike force of Frog B is equal to that of Frog C.




 To set up the bootstrap hypothesis test, you will take the mean as our test statistic. Remember, your goal is to calculate the probability of getting a mean impact force less than or equal to what was observed for Frog B
 *if the hypothesis that the true mean of Frog B’s impact forces is equal to that of Frog C is true*
 . You first translate all of the data of Frog B such that the mean is 0.55 N. This involves adding the mean force of Frog C and subtracting the mean force of Frog B from each measurement of Frog B. This leaves other properties of Frog B’s distribution, such as the variance, unchanged.





```python

# Make an array of translated impact forces: translated_force_b
translated_force_b = force_b - np.mean(force_b) + 0.55

# Take bootstrap replicates of Frog B's translated impact forces: bs_replicates
bs_replicates = draw_bs_reps(translated_force_b, np.mean, 10000)

# Compute fraction of replicates that are less than the observed Frog B force: p
p = np.sum(bs_replicates <= np.mean(force_b)) / 10000

# Print the p-value
print('p = ', p)

# p =  0.0046
# p = 0.46%

```



 The low p-value suggests that the null hypothesis that Frog B and Frog C have the same mean impact force is false.



####
 A two-sample bootstrap hypothesis test for difference of means



 We now want to test the hypothesis that Frog A and Frog B have the same mean impact force, but not necessarily the same distribution, which is also impossible with a permutation test.




 To do the two-sample bootstrap test, we shift
 *both*
 arrays to have the same mean, since we are simulating the hypothesis that their means are, in fact, equal. We then draw bootstrap samples out of the shifted arrays and compute the difference in means. This constitutes a bootstrap replicate, and we generate many of them. The p-value is the fraction of replicates with a difference in means greater than or equal to what was observed.





```python

# Compute mean of all forces: mean_force
mean_force = np.mean(forces_concat)

# Generate shifted arrays
force_a_shifted = force_a - np.mean(force_a) + mean_force
force_b_shifted = force_b - np.mean(force_b) + mean_force

# Compute 10,000 bootstrap replicates from shifted arrays
bs_replicates_a = draw_bs_reps(force_a_shifted, np.mean, size=10000)
bs_replicates_b = draw_bs_reps(force_b_shifted, np.mean, size=10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_replicates_a - bs_replicates_b

# Compute and print p-value: p
p = np.sum(bs_replicates >= empirical_diff_means) / 10000
print('p-value =', p)

# p-value = 0.0043
# p-value = 0.43%

```



 You got a similar result as when you did the permutation test. Nonetheless, remember that it is important to carefully think about what question you want to ask. Are you only interested in the mean impact force, or in the distribution of impact forces?




 Hypothesis test examples
--------------------------


###
 A/B testing


####
 The vote for the Civil Rights Act in 1964



 The Civil Rights Act of 1964 was one of the most important pieces of legislation ever passed in the USA. Excluding “present” and “abstain” votes, 153 House Democrats and 136 Republicans voted yea. However, 91 Democrats and 35 Republicans voted nay. Did party affiliation make a difference in the vote?




 To answer this question, you will evaluate the hypothesis that the party of a House member has no bearing on his or her vote. You will use the fraction of Democrats voting in favor as your test statistic and evaluate the probability of observing a fraction of Democrats voting in favor at least as small as the observed fraction of 153/244. (That’s right, at least as
 *small*
 as. In 1964, it was the
 *Democrats*
 who were less progressive on civil rights issues.) To do this, permute the party labels of the House voters and then arbitrarily divide them into “Democrats” and “Republicans” and compute the fraction of Democrats voting yea.





```python

# Construct arrays of data: dems, reps
dems = np.array([True] * 153 + [False] * 91)
reps = np.array([True] * 136 + [False] * 35)

def frac_yea_dems(dems, reps):
    """Compute fraction of Democrat yea votes."""
    frac = np.sum(dems) / len(dems)
    return frac

# Acquire permutation samples: perm_replicates
perm_replicates = draw_perm_reps(dems, reps, frac_yea_dems, size=10000)

# Compute and print p-value: p
p = np.sum(perm_replicates <= 153/244) / len(perm_replicates)
print('p-value =', p)

# p-value = 0.0002
# p-value = 0.02%

```



 This small p-value suggests that party identity had a lot to do with the voting. Importantly, the South had a higher fraction of Democrat representatives, and consequently also a more racist bias.



####
 What is equivalent?



 You have experience matching a stories to probability distributions. Similarly, you use the same procedure for two different A/B tests if their stories match. In the Civil Rights Act example you just did, you performed an A/B test on voting data, which has a Yes/No type of outcome for each subject (in that case, a voter). Which of the following situations involving testing by a web-based company has an equivalent set up for an A/B test as the one you just did with the Civil Rights Act of 1964?




 You measure the number of people who click on an ad on your company’s website before and after changing its color.




 The “Democrats” are those who view the ad before the color change, and the “Republicans” are those who view it after.



####
 A time-on-website analog



 It turns out that you already did a hypothesis test analogous to an A/B test where you are interested in how much time is spent on the website before and after an ad campaign. The frog tongue force (a continuous quantity like time on the website) is an analog. “Before” = Frog A and “after” = Frog B. Let’s practice this again with something that actually is a before/after scenario.




 We return to the no-hitter data set. In 1920, Major League Baseball implemented important rule changes that ended the so-called dead ball era. Importantly, the pitcher was no longer allowed to spit on or scuff the ball, an activity that greatly favors pitchers. In this problem you will perform an A/B test to determine if these rule changes resulted in a slower rate of no-hitters (i.e., longer average time between no-hitters) using the difference in mean inter-no-hitter time as your test statistic. The inter-no-hitter times for the respective eras are stored in the arrays
 `nht_dead`
 and
 `nht_live`
 , where “nht” is meant to stand for “no-hitter time.”





```

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates



# Compute the observed difference in mean inter-no-hitter times: nht_diff_obs
nht_diff_obs = diff_of_means(nht_dead, nht_live)

# Acquire 10,000 permutation replicates of difference in mean no-hitter time: perm_replicates
perm_replicates = draw_perm_reps(nht_dead, nht_live, diff_of_means, size=10000)

# Compute and print the p-value: p
p = np.sum(perm_replicates <= nht_diff_obs) /len(perm_replicates)
print('p-val =', p)

p-val = 0.0001

```



 Your p-value is 0.0001, which means that only one out of your 10,000 replicates had a result as extreme as the actual difference between the dead ball and live ball eras. This suggests strong statistical significance. Watch out, though, you could very well have gotten zero replicates that were as extreme as the observed value. This just means that the p-value is quite small, almost certainly smaller than 0.001.



####
 What should you have done first?



 That was a nice hypothesis test you just did to check out whether the rule changes in 1920 changed the rate of no-hitters. But what
 *should*
 you have done with the data first?




 Performed EDA, perhaps plotting the ECDFs of inter-no-hitter times in the dead ball and live ball eras.




 Always a good idea to do first! I encourage you to go ahead and plot the ECDFs right now. You will see by eye that the null hypothesis that the distributions are the same is almost certainly not true.





```python

# Create and plot ECDFs
x_1, y_1 = ecdf(nht_dead)
x_2, y_2 = ecdf(nht_live)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('Games between no-hitters')
_ = plt.ylabel('ECDF')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture1-17.png)

###
 Test of correlation



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture2-16.png)

####
 Simulating a null hypothesis concerning correlation



 The observed correlation between female illiteracy and fertility in the data set of 162 countries may just be by chance; the fertility of a given country may actually be totally independent of its illiteracy. You will test this null hypothesis in the next exercise.




 To do the test, you need to simulate the data assuming the null hypothesis is true. Of the following choices, which is the best way to to do it?




 Answer: Do a permutation test: Permute the illiteracy values but leave the fertility values fixed to generate a new set of (illiteracy, fertility) data.




 This exactly simulates the null hypothesis and does so more efficiently than the last option. It is exact because it uses all data and eliminates any correlation because which illiteracy value pairs to which fertility value is shuffled.




 Last option: Do a permutation test: Permute both the illiteracy and fertility values to generate a new set of (illiteracy, fertility data). This exactly simulates the null hypothesis and does so more efficiently than the last option. It is exact because it uses all data and eliminates any correlation because which illiteracy value pairs to which fertility value is shuffled.



####
 Hypothesis test on Pearson correlation



 The observed correlation between female illiteracy and fertility may just be by chance; the fertility of a given country may actually be totally independent of its illiteracy. You will test this hypothesis. To do so, permute the illiteracy values but leave the fertility values fixed. This simulates the hypothesis that they are totally independent of each other. For each permutation, compute the Pearson correlation coefficient and assess how many of your permutation replicates have a Pearson correlation coefficient greater than the observed one.





```

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0,1]

# Compute observed correlation: r_obs
r_obs = pearson_r(illiteracy, fertility)

# r_obs = 0.8041324026815344

# Initialize permutation replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute illiteracy measurments: illiteracy_permuted
    illiteracy_permuted = np.random.permutation(illiteracy)

    # Compute Pearson correlation
    perm_replicates[i] = pearson_r(illiteracy_permuted, fertility)

# Compute p-value: p
p = np.sum(perm_replicates >= r_obs) /len(perm_replicates)
print('p-val =', p)

# p-val = 0.0

```



 You got a p-value of zero. In hacker statistics, this means that your p-value is very low, since you never got a single replicate in the 10,000 you took that had a Pearson correlation greater than the observed one. You could try increasing the number of replicates you take to continue to move the upper bound on your p-value lower and lower.



####
 Do neonicotinoid insecticides have unintended consequences?



 As a final exercise in hypothesis testing before we put everything together in our case study in the next chapter, you will investigate the effects of neonicotinoid insecticides on bee reproduction. These insecticides are very widely used in the United States to combat aphids and other pests that damage plants.




 In a recent study, Straub, et al. (
 [*Proc. Roy. Soc. B*
 , 2016](http://dx.doi.org/10.1098/rspb.2016.0506)
 ) investigated the effects of neonicotinoids on the sperm of pollinating bees. In this and the next exercise, you will study how the pesticide treatment affected the count of live sperm per half milliliter of semen.




 First, we will do EDA, as usual. Plot ECDFs of the alive sperm count for untreated bees (stored in the Numpy array
 `control`
 ) and bees treated with pesticide (stored in the Numpy array
 `treated`
 ).





```python

# Compute x,y values for ECDFs
x_control, y_control = ecdf(control)
x_treated, y_treated = ecdf(treated)

# Plot the ECDFs
plt.plot(x_control, y_control, marker='.', linestyle='none')
plt.plot(x_treated, y_treated, marker='.', linestyle='none')

# Set the margins
plt.margins(0.02)

# Add a legend
plt.legend(('control', 'treated'), loc='lower right')

# Label axes and show plot
plt.xlabel('millions of alive sperm per mL')
plt.ylabel('ECDF')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture3-15.png)


 The ECDFs show a pretty clear difference between the treatment and control; treated bees have fewer alive sperm. Let’s now do a hypothesis test in the next exercise.



####
 Bootstrap hypothesis test on bee sperm counts



 Now, you will test the following hypothesis:




**On average, male bees treated with neonicotinoid insecticide have the same number of active sperm per milliliter of semen than do untreated male bees.**




 You will use the difference of means as your test statistic.





```

def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates


# Compute the difference in mean sperm count: diff_means
diff_means = np.mean(control) - np.mean(treated)

# Compute mean of pooled data: mean_count
mean_count = np.mean(np.concatenate((control, treated)))

# Generate shifted data sets
control_shifted = control - np.mean(control) + mean_count
treated_shifted = treated - np.mean(treated) + mean_count

# Generate bootstrap replicates
bs_reps_control = draw_bs_reps(control_shifted,
                       np.mean, size=10000)
bs_reps_treated = draw_bs_reps(treated_shifted,
                       np.mean, size=10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_reps_control - bs_reps_treated

# Compute and print p-value: p
p = np.sum(bs_replicates >= np.mean(control) - np.mean(treated)) \
            / len(bs_replicates)
print('p-value =', p)

# p-value = 0.0

```



 The p-value is small, most likely less than 0.0001, since you never saw a bootstrap replicated with a difference of means at least as extreme as what was observed. In fact, when I did the calculation with 10 million replicates, I got a p-value of
 `2e-05`




 Putting it all together: a case study
---------------------------------------


###
 Finch beaks and the need for statistics



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture4-14.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture5-12.png)

####
 EDA of beak depths of Darwin’s finches



 For your first foray into the Darwin finch data, you will study how the beak depth (the distance, top to bottom, of a closed beak) of the finch species
 *Geospiza scandens*
 has changed over time. The Grants have noticed some changes of beak geometry depending on the types of seeds available on the island, and they also noticed that there was some interbreeding with another major species on Daphne Major,
 *Geospiza fortis*
 . These effects can lead to changes in the species over time.




 In the next few problems, you will look at the beak depth of
 *G. scandens*
 on Daphne Major in 1975 and in 2012. To start with, let’s plot all of the beak depth measurements in 1975 and 2012 in a bee swarm plot.




 The data are stored in a pandas DataFrame called
 `df`
 with columns
 `'year'`
 and
 `'beak_depth'`
 . The units of beak depth are millimeters (mm).





```

df.head()
   beak_depth  year
0         8.4  1975
1         8.8  1975
2         8.4  1975
3         8.0  1975
4         7.9  1975

```




```python

# Create bee swarm plot
_ = sns.swarmplot('year', 'beak_depth', data=df)

# Label the axes
_ = plt.xlabel('year')
_ = plt.ylabel('beak depth (mm)')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture6-9.png)


 It is kind of hard to see if there is a clear difference between the 1975 and 2012 data set. Eyeballing it, it appears as though the mean of the 2012 data set might be slightly higher, and it might have a bigger variance.



####
 ECDFs of beak depths



 While bee swarm plots are useful, we found that ECDFs are often even better when doing EDA. Plot the ECDFs for the 1975 and 2012 beak depth measurements on the same plot.





```python

# Compute ECDFs
x_1975, y_1975 = ecdf(bd_1975)
x_2012, y_2012 = ecdf(bd_2012)

# Plot the ECDFs
_ = plt.plot(x_1975, y_1975, marker='.', linestyle='none')
_ = plt.plot(x_2012, y_2012, marker='.', linestyle='none')

# Set margins
plt.margins(0.02)

# Add axis labels and legend
_ = plt.xlabel('beak depth (mm)')
_ = plt.ylabel('ECDF')
_ = plt.legend(('1975', '2012'), loc='lower right')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture7-10.png)


 The differences are much clearer in the ECDF. The mean is larger in the 2012 data, and the variance does appear larger as well.



####
 Parameter estimates of beak depths



 Estimate the
 *difference*
 of the mean beak depth of the
 *G. scandens*
 samples from 1975 and 2012 and report a 95% confidence interval.





```

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates


# Compute the difference of the sample means: mean_diff
mean_diff = np.mean(bd_2012) - np.mean(bd_1975)

# Get bootstrap replicates of means
bs_replicates_1975 = draw_bs_reps(bd_1975, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(bd_2012, np.mean, size=10000)

# Compute samples of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute 95% confidence interval: conf_int
conf_int = np.percentile(bs_diff_replicates, [2.5, 97.5])

# Print the results
print('difference of means =', mean_diff, 'mm')
print('95% confidence interval =', conf_int, 'mm')

# difference of means = 0.22622047244094645 mm
# 95% confidence interval = [0.05633521 0.39190544] mm

```


####
 Hypothesis test: Are beaks deeper in 2012?



 Your plot of the ECDF and determination of the confidence interval make it pretty clear that the beaks of
 *G. scandens*
 on Daphne Major have gotten deeper. But is it possible that this effect is just due to random chance? In other words, what is the probability that we would get the observed difference in mean beak depth if the means were the same?




 Be careful! The hypothesis we are testing is
 *not*
 that the beak depths come from the same distribution. For that we could use a permutation test.
 **The hypothesis is that the means are equal.**
 To perform this hypothesis test, we need to shift the two data sets so that they have the same mean and then use bootstrap sampling to compute the difference of means.





```python

# Compute mean of combined data set: combined_mean
combined_mean = np.mean(np.concatenate((bd_1975, bd_2012)))

# Shift the samples
# why shift the mean?
# to make np.mean(bd_1975_shifted) - np.mean(bd_2012_shifted) = 0 #1
# why make #1 = 0?
# because our hypothesis is "beak depth are the same in 1975 and 2012"
bd_1975_shifted = bd_1975 - np.mean(bd_1975) + combined_mean
bd_2012_shifted = bd_2012 - np.mean(bd_2012) + combined_mean

# Get bootstrap replicates of shifted data sets
bs_replicates_1975 = draw_bs_reps(bd_1975_shifted, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(bd_2012_shifted, np.mean, size=10000)

# Compute replicates of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute the p-value
p = np.sum(bs_diff_replicates >= mean_diff) / len(bs_diff_replicates)

# Print p-value
print('p =', p)

# p = 0.0034
# p = 0.34%

```



 We get a p-value of 0.0034, which suggests that there is a statistically significant difference. But remember: it is very important to know how different they are! In the previous exercise, you got a difference of 0.2 mm between the means. You should combine this with the statistical significance. Changing by 0.2 mm in 37 years is substantial by evolutionary standards. If it kept changing at that rate, the beak depth would double in only 400 years.



###
 Variation of beak shapes


####
 EDA of beak length and depth



 The beak length data are stored as
 `bl_1975`
 and
 `bl_2012`
 , again with units of millimeters (mm). You still have the beak depth data stored in
 `bd_1975`
 and
 `bd_2012`
 . Make scatter plots of beak depth (y-axis) versus beak length (x-axis) for the 1975 and 2012 specimens.





```python

# Make scatter plot of 1975 data
_ = plt.plot(bl_1975, bd_1975, marker='.',
             linestyle='None', color='blue', alpha=0.5)

# Make scatter plot of 2012 data
_ = plt.plot(bl_2012, bd_2012, marker='.',
            linestyle='None', color='red', alpha=0.5)

# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture8-7.png)


 In looking at the plot, we see that beaks got deeper (the red points are higher up in the y-direction), but not really longer. If anything, they got a bit shorter, since the red dots are to the left of the blue dots. So, it does not look like the beaks kept the same shape; they became shorter and deeper.



####
 Linear regressions



 Perform a linear regression for both the 1975 and 2012 data. Then, perform pairs bootstrap estimates for the regression parameters. Report 95% confidence intervals on the slope and intercept of the regression line.





```

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, deg=1)

    return bs_slope_reps, bs_intercept_reps


# Compute the linear regressions
slope_1975, intercept_1975 = np.polyfit(bl_1975, bd_1975, deg=1)
slope_2012, intercept_2012 = np.polyfit(bl_2012, bd_2012, deg=1)

# Perform pairs bootstrap for the linear regressions
bs_slope_reps_1975, bs_intercept_reps_1975 = draw_bs_pairs_linreg(bl_1975, bd_1975, size=1000)
bs_slope_reps_2012, bs_intercept_reps_2012 = draw_bs_pairs_linreg(bl_2012, bd_2012, size=1000)

# Compute confidence intervals of slopes
slope_conf_int_1975 = np.percentile(bs_slope_reps_1975, [2.5, 97.5])
slope_conf_int_2012 = np.percentile(bs_slope_reps_2012, [2.5, 97.5])
intercept_conf_int_1975 = np.percentile(bs_intercept_reps_1975, [2.5, 97.5])
intercept_conf_int_2012 = np.percentile(bs_intercept_reps_2012, [2.5, 97.5])


# Print the results
print('1975: slope =', slope_1975,
      'conf int =', slope_conf_int_1975)
print('1975: intercept =', intercept_1975,
      'conf int =', intercept_conf_int_1975)
print('2012: slope =', slope_2012,
      'conf int =', slope_conf_int_2012)
print('2012: intercept =', intercept_2012,
      'conf int =', intercept_conf_int_2012)

#   1975: slope = 0.4652051691605937 conf int = [0.33851226 0.59306491]
#   1975: intercept = 2.3908752365842263 conf int = [0.64892945 4.18037063]
#   2012: slope = 0.462630358835313 conf int = [0.33137479 0.60695527]
#   2012: intercept = 2.977247498236019 conf int = [1.06792753 4.70599387]

```



 It looks like they have the same slope, but different intercepts.



####
 Displaying the linear regression results




```python

# Make scatter plot of 1975 data
_ = plt.plot(bl_1975, bd_1975, marker='.',
             linestyle='none', color='blue', alpha=0.5)

# Make scatter plot of 2012 data
_ = plt.plot(bl_2012, bd_2012, marker='.',
             linestyle='none', color='red', alpha=0.5)

# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')

# Generate x-values for bootstrap lines: x
x = np.array([10, 17])

# Plot the bootstrap lines
for i in range(100):
    plt.plot(x, bs_slope_reps_1975[i] * x + bs_intercept_reps_1975[i],
             linewidth=0.5, alpha=0.2, color='blue')
    plt.plot(x, bs_slope_reps_2012[i] * x + bs_intercept_reps_2012[i],
             linewidth=0.5, alpha=0.2, color='red')

# Draw the plot again
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture9-6.png)

####
 Beak length to depth ratio



 The linear regressions showed interesting information about the beak geometry. The slope was the same in 1975 and 2012, suggesting that for every millimeter gained in beak length, the birds gained about half a millimeter in depth in both years. However, if we are interested in the shape of the beak, we want to compare the
 *ratio*
 of beak length to beak depth. Let’s make that comparison.





```python

# Compute length-to-depth ratios
ratio_1975 = bl_1975 / bd_1975
ratio_2012 = bl_2012 / bd_2012

# Compute means
mean_ratio_1975 = np.mean(ratio_1975)
mean_ratio_2012 = np.mean(ratio_2012)

# Generate bootstrap replicates of the means
bs_replicates_1975 = draw_bs_reps(ratio_1975, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(ratio_2012, np.mean, size=10000)

# Compute the 99% confidence intervals
conf_int_1975 = np.percentile(bs_replicates_1975, [0.5, 99.5])
conf_int_2012 = np.percentile(bs_replicates_2012, [0.5, 99.5])

# Print the results
print('1975: mean ratio =', mean_ratio_1975,
      'conf int =', conf_int_1975)
print('2012: mean ratio =', mean_ratio_2012,
      'conf int =', conf_int_2012)

# 1975: mean ratio = 1.5788823771858533 conf int = [1.55668803 1.60073509]
# 2012: mean ratio = 1.4658342276847767 conf int = [1.44363932 1.48729149]

```


####
 How different is the ratio?



 In the previous exercise, you computed the mean beak length to depth ratio with 99% confidence intervals for 1975 and for 2012. The results of that calculation are shown graphically in the plot accompanying this problem. In addition to these results, what would you say about the ratio of beak length to depth?




![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture10-6.png)


 The mean beak length-to-depth ratio decreased by about 0.1, or 7%, from 1975 to 2012. The 99% confidence intervals are not even close to overlapping, so this is a real change. The beak shape changed.



###
 Calculation of heritability


####
 EDA of heritability



 The array
 `bd_parent_scandens`
 contains the average beak depth (in mm) of two parents of the species
 `G. scandens`
 . The array
 `bd_offspring_scandens`
 contains the average beak depth of the offspring of the respective parents. The arrays
 `bd_parent_fortis`
 and
 `bd_offspring_fortis`
 contain the same information about measurements from
 *G. fortis*
 birds.




 Make a scatter plot of the average offspring beak depth (y-axis) versus average parental beak depth (x-axis) for both species. Use the
 `alpha=0.5`
 keyword argument to help you see overlapping points.





```python

# Make scatter plots
_ = plt.plot(bd_parent_fortis, bd_offspring_fortis,
             marker='.', linestyle='none', color='blue', alpha=0.5)
_ = plt.plot(bd_parent_scandens, bd_offspring_scandens,
             marker='.', linestyle='none', color='red', alpha=0.5)

# Label axes
_ = plt.xlabel('parental beak depth (mm)')
_ = plt.ylabel('offspring beak depth (mm)')

# Add legend
_ = plt.legend(('G. fortis', 'G. scandens'), loc='lower right')

# Show plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture11-6.png)


 It appears as though there is a stronger correlation in
 *G. fortis*
 than than in
 *G. scandens*
 . This suggests that beak depth is more strongly inherited in
 *G. fortis*
 . We’ll quantify this correlation next.



####
 Correlation of offspring and parental data



 In an effort to quantify the correlation between offspring and parent beak depths, we would like to compute statistics, such as the Pearson correlation coefficient, between parents and offspring. To get confidence intervals on this, we need to do a pairs bootstrap.




 You have
 [already written](https://campus.datacamp.com/courses/statistical-thinking-in-python-part-2/bootstrap-confidence-intervals?ex=12)
 a function to do pairs bootstrap to get estimates for parameters derived from linear regression. Your task in this exercise is to make a new function with call signature
 `draw_bs_pairs(x, y, func, size=1)`
 that performs pairs bootstrap and computes a single statistic on pairs samples defined. The statistic of interest is computed by calling
 `func(bs_x, bs_y)`
 . In the next exercise, you will use
 `pearson_r`
 for
 `func`
 .





```

def draw_bs_pairs(x, y, func, size=1):
    """Perform pairs bootstrap for a single statistic."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_replicates[i] = func(bs_x, bs_y)

    return bs_replicates


```


####
 Pearson correlation of offspring and parental data



 The Pearson correlation coefficient seems like a useful measure of how strongly the beak depth of parents are inherited by their offspring. Compute the Pearson correlation coefficient between parental and offspring beak depths for
 *G. scandens*
 . Do the same for
 *G. fortis*
 . Then, use the function you wrote in the last exercise to compute a 95% confidence interval using pairs bootstrap.





```

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0,1]


# Compute the Pearson correlation coefficients
r_scandens = pearson_r(bd_parent_scandens, bd_offspring_scandens)
r_fortis = pearson_r(bd_parent_fortis, bd_offspring_fortis)

# Acquire 1000 bootstrap replicates of Pearson r
bs_replicates_scandens = draw_bs_pairs(bd_parent_scandens, bd_offspring_scandens, pearson_r, size=1000)

bs_replicates_fortis = draw_bs_pairs(bd_parent_fortis, bd_offspring_fortis, pearson_r, size=1000)


# Compute 95% confidence intervals
conf_int_scandens = np.percentile(bs_replicates_scandens, [2.5, 97.5])
conf_int_fortis = np.percentile(bs_replicates_fortis, [2.5, 97.5])

# Print results
print('G. scandens:', r_scandens, conf_int_scandens)
print('G. fortis:', r_fortis, conf_int_fortis)

#    G. scandens: 0.4117063629401258 [0.26564228 0.54388972]
#    G. fortis: 0.7283412395518487 [0.6694112  0.77840616]

```



 It is clear from the confidence intervals that beak depth of the offspring of
 *G. fortis*
 parents is more strongly correlated with their offspring than their
 *G. scandens*
 counterparts.



####
 Measuring heritability



 Remember that the Pearson correlation coefficient is the ratio of the covariance to the geometric mean of the variances of the two data sets. This is a measure of the correlation between parents and offspring, but might not be the best estimate of heritability. If we stop and think, it makes more sense to define heritability as the ratio of the covariance between parent and offspring to the
 *variance of the parents alone*
 . In this exercise, you will estimate the heritability and perform a pairs bootstrap calculation to get the 95% confidence interval.




 This exercise highlights a very important point. Statistical inference (and data analysis in general) is not a plug-n-chug enterprise. You need to think carefully about the questions you are seeking to answer with your data and analyze them appropriately. If you are interested in how heritable traits are, the quantity we defined as the heritability is more apt than the off-the-shelf statistic, the Pearson correlation coefficient.





```

def heritability(parents, offspring):
    """Compute the heritability from parent and offspring samples."""
    covariance_matrix = np.cov(parents, offspring)
    return covariance_matrix[0,1] / covariance_matrix[0,0]

# Compute the heritability
heritability_scandens = heritability(bd_parent_scandens, bd_offspring_scandens)
heritability_fortis = heritability(bd_parent_fortis, bd_offspring_fortis)

# Acquire 1000 bootstrap replicates of heritability
replicates_scandens = draw_bs_pairs(
        bd_parent_scandens, bd_offspring_scandens, heritability, size=1000)

replicates_fortis = draw_bs_pairs(
        bd_parent_fortis, bd_offspring_fortis, heritability, size=1000)


# Compute 95% confidence intervals
conf_int_scandens = np.percentile(replicates_scandens, [2.5, 97.5])
conf_int_fortis = np.percentile(replicates_fortis, [2.5, 97.5])

# Print results
print('G. scandens:', heritability_scandens, conf_int_scandens)
print('G. fortis:', heritability_fortis, conf_int_fortis)


#   G. scandens: 0.5485340868685982 [0.34395487 0.75638267]
#   G. fortis: 0.7229051911438159 [0.64655013 0.79688342]

```



 Here again, we see that
 *G. fortis*
 has stronger heritability than
 *G. scandens*
 . This suggests that the traits of
 *G. fortis*
 may be strongly incorporated into
 *G. scandens*
 by introgressive hybridization.



####
 Is beak depth heritable at all in G. scandens?



 The heritability of beak depth in
 *G. scandens*
 seems low. It could be that this observed heritability was just achieved by chance and
 **beak depth is actually not really heritable in the species**
 . You will test that hypothesis here. To do this, you will do a pairs permutation test.





```python

# Initialize array of replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute parent beak depths
    bd_parent_permuted = np.random.permutation(bd_parent_scandens)
    perm_replicates[i] = heritability(bd_parent_permuted,
                                      bd_offspring_scandens)

# Compute p-value: p
p = np.sum(perm_replicates >= heritability_scandens) / len(perm_replicates)

# Print the p-value
print('p-val =', p)

# p-val = 0.0

```



 You get a p-value of zero, which means that none of the 10,000 permutation pairs replicates you drew had a heritability high enough to match that which was observed. This strongly suggests that beak depth is heritable in
 *G. scandens*
 , just not as much as in
 *G. fortis*
 . If you like, you can plot a histogram of the heritability replicates to get a feel for how extreme of a value of heritability you might expect by chance.





```

plt.hist(perm_replicates)
plt.axvline(x=heritability_scandens, color = 'red')
plt.text(heritability_scandens, 1500, 'heritability_scandens', ha='center', va='center',rotation='vertical', backgroundcolor='white')
plt.show()

```




 Parameter estimation by optimization
--------------------------------------


###
 Optimal parameters


####
 How often do we get no-hitters?



 The number of games played between each no-hitter in the modern era (1901-2015) of Major League Baseball is stored in the array
 `nohitter_times`
 .




 If you assume that no-hitters are described as a Poisson process, then the time between no-hitters is Exponentially distributed. As you have seen, the Exponential distribution has a single parameter, which we will call ττ, the typical interval time. The value of the parameter ττ that makes the exponential distribution best match the data is the mean interval time (where time is in units of number of games) between no-hitters.




 Compute the value of this parameter from the data. Then, use
 `np.random.exponential()`
 to “repeat” the history of Major League Baseball by drawing inter-no-hitter times from an exponential distribution with the ττ you found and plot the histogram as an approximation to the PDF.







```python

# Seed random number generator
np.random.seed(42)

# Compute mean no-hitter time: tau
tau = np.mean(nohitter_times)

# Draw out of an exponential distribution with parameter tau: inter_nohitter_time
inter_nohitter_time = np.random.exponential(tau, 100000)

# Plot the PDF and label axes
_ = plt.hist(inter_nohitter_time,
             bins=50, normed=True, histtype='step')
_ = plt.xlabel('Games between no-hitters')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture-16.png)


 We see the typical shape of the Exponential distribution, going from a maximum at 0 and decaying to the right.



####
 Do the data follow our story?




```python

# Create an ECDF from real data: x, y
x, y = ecdf(nohitter_times)

# Create a CDF from theoretical samples: x_theor, y_theor
x_theor, y_theor = ecdf(inter_nohitter_time)

# Overlay the plots
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')

# Margins and axis labels
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture1-15.png)


 It looks like no-hitters in the modern era of Major League Baseball are Exponentially distributed. Based on the story of the Exponential distribution, this suggests that they are a random process; when a no-hitter will happen is independent of when the last no-hitter was.



####
 How is this parameter optimal?




```python

# Plot the theoretical CDFs
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# Take samples with half tau: samples_half
samples_half = np.random.exponential(tau/2, size=10000)

# Take samples with double tau: samples_double
samples_double = np.random.exponential(2*tau, size=10000)

# Generate CDFs from these samples
x_half, y_half = ecdf(samples_half)
x_double, y_double = ecdf(samples_double)

# Plot these CDFs as lines
_ = plt.plot(x_half, y_half)
_ = plt.plot(x_double, y_double)

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture2-14.png)

 red = half, purple = double



 Notice how the value of tau given by the mean matches the data best. In this way, tau is an optimal parameter.



###
 Linear regression by least squares


####
 EDA of literacy/fertility data




```python

# Plot the illiteracy rate versus fertility
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')

# Set the margins and label axes
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')

# Show the plot
plt.show()

# Show the Pearson correlation coefficient
print(pearson_r(illiteracy, fertility))
0.8041324026815344

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture3-13.png)


 You can see the correlation between illiteracy and fertility by eye, and by the substantial Pearson correlation coefficient of 0.8. It is difficult to resolve in the scatter plot, but there are many points around near-zero illiteracy and about 1.8 children/woman.



####
 Linear regression




```python

# Plot the illiteracy rate versus fertility
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')

# Perform a linear regression using np.polyfit(): a, b
a, b = np.polyfit(illiteracy, fertility, deg=1)

# Print the results to the screen
print('slope =', a, 'children per woman / percent illiterate')
print('intercept =', b, 'children per woman')

# Make theoretical line to plot
x = np.array([0, 100])
y = a * x + b

# Add regression line to your plot
_ = plt.plot(x, y)

# Draw the plot
plt.show()


# slope = 0.04979854809063423 children per woman / percent illiterate
# intercept = 1.888050610636557 children per woman


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture4-13.png)

####
 How is it optimal?




```python

# Specify slopes to consider: a_vals
a_vals = np.linspace(0, 0.1, 200)

# Initialize sum of square of residuals: rss
rss = np.empty_like(a_vals)

# Compute sum of square of residuals for each value of a_vals
for i, a in enumerate(a_vals):
    rss[i] = np.sum((fertility - a*illiteracy - b)**2)

# Plot the RSS
plt.plot(a_vals, rss, '-')
plt.xlabel('slope (children per woman / percent illiterate)')
plt.ylabel('sum of square of residuals')

plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture5-11.png)


 Notice that the minimum on the plot, that is the value of the slope that gives the minimum sum of the square of the residuals, is the same value you got when performing the regression.



###
 The importance of EDA: Anscombe’s quartet



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture6-8.png)

####
 The importance of EDA



 Why should exploratory data analysis be the first step in an analysis of data (after getting your data imported and cleaned, of course)?



* You can be protected from misinterpretation of the type demonstrated by Anscombe’s quartet.
* EDA provides a good starting point for planning the rest of your analysis.
* EDA is not really any more difficult than any of the subsequent analysis, so there is no excuse for not exploring the data.


####
 Linear regression on appropriate Anscombe data




```python

# Perform linear regression: a, b
a, b = np.polyfit(x, y, deg=1)

# Print the slope and intercept
print(a, b)

# Generate theoretical x and y data: x_theor, y_theor
x_theor = np.array([3, 15])
y_theor = a * x_theor + b

# Plot the Anscombe data and theoretical line
_ = plt.plot(x, y, marker = '.', linestyle = 'none')
_ = plt.plot(x_theor, y_theor)

# Label the axes
plt.xlabel('x')
plt.ylabel('y')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture7-9.png)

####
 Linear regression on all Anscombe data




```

for i in range(4):
    plt.subplot(2,2,i+1)

    # plot the scatter plot
    plt.plot(anscombe_x[i], anscombe_y[i], marker = '.', linestyle = 'none')

    # plot the regression line
    a, b = np.polyfit(anscombe_x[i], anscombe_y[i], deg=1)
    x_theor = np.array([np.min(anscombe_x[i]), np.max(anscombe_x[i])])
    y_theor = a * x_theor + b
    plt.plot(x_theor, y_theor)

    # add label
    plt.xlabel('x' + str(i+1))
    plt.ylabel('y' + str(i+1))

plt.show()

# slope1: 0.5000909090909095 intercept: 3.000090909090909
# slope2: 0.5000000000000004 intercept: 3.0009090909090896
# slope3: 0.4997272727272731 intercept: 3.0024545454545453
# slope4: 0.4999090909090908 intercept: 3.0017272727272735

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture8-6.png)

###
 Generating bootstrap replicates


####
 Getting the terminology down



 If we have a data set with n repeated measurements, a
 **bootstrap sample**
 is an array of length n that was drawn from the original data with replacement.




**Bootstrap replicate**
 is a single value of a statistic computed from a bootstrap sample.



####
 Visualizing bootstrap samples

 np.random.choice()




```

for _ in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(rainfall, size=len(rainfall))

    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)

# Compute and plot ECDF from original data
x, y = ecdf(rainfall)
_ = plt.plot(x, y, marker='.')

# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('yearly rainfall (mm)')
_ = plt.ylabel('ECDF')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture9-5.png)


 Bootstrap confidence intervals
--------------------------------


####
 Generating many bootstrap replicates




```

def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates

```


####
 Bootstrap replicates of the mean and the SEM (
 **standard error of the mean**
 )



 In fact, it can be shown theoretically that under not-too-restrictive conditions, the value of the mean will always be Normally distributed. (This does not hold in general, just for the mean and a few other statistics.)




 The standard deviation of this distribution, called the
 **standard error of the mean**
 , or SEM, is given by the standard deviation of the data divided by the square root of the number of data points. I.e., for a data set,
 `sem = np.std(data) / np.sqrt(len(data))`
 . Using hacker statistics, you get this same result without the need to derive it, but you will verify this result from your bootstrap replicates.





```python

# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates = draw_bs_reps(rainfall, np.mean, size=10000)

# Compute and print SEM
sem = np.std(rainfall) / np.sqrt(len(rainfall))
print(sem)

# Compute and print standard deviation of bootstrap replicates
bs_std = np.std(bs_replicates)
print(bs_std)

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('mean annual rainfall (mm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()

# 10.51054915050619
# 10.465927071184412

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture10-5.png)

####
 Confidence intervals of rainfall data



 Use the bootstrap replicates you just generated to compute the 95% confidence interval. That is, give the 2.5th and 97.5th percentile of your bootstrap replicates stored as
 `bs_replicates`
 . What is the 95% confidence interval?





```

np.percentile(bs_replicates,2.5)
779.7699248120301

np.percentile(bs_replicates,97.5)
820.950432330827

```


####
 Bootstrap replicates of other statistics




```

def draw_bs_reps(data, func, size=1):
    return np.array([bootstrap_replicate_1d(data, func) for _ in range(size)])

# Generate 10,000 bootstrap replicates of the variance: bs_replicates
bs_replicates = draw_bs_reps(rainfall, np.var, size=10000)

# Put the variance in units of square centimeters
bs_replicates /= 100

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('variance of annual rainfall (sq. cm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture11-5.png)


 This is not normally distributed, as it has a longer tail to the right. Note that you can also compute a confidence interval on the variance, or any other statistic, using
 `np.percentile()`
 with your bootstrap replicates.



####
 Confidence interval on the rate of no-hitters




```python

# Draw bootstrap replicates of the mean no-hitter time (equal to tau): bs_replicates
bs_replicates = draw_bs_reps(nohitter_times, np.mean, size=10000)

# Compute the 95% confidence interval: conf_int
conf_int = np.percentile(bs_replicates, [2.5, 97.5])

# Print the confidence interval
print('95% confidence interval =', conf_int, 'games')

# Plot the histogram of the replicates
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel(r'$\tau$ (games)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()

# 95% confidence interval = [660.67280876 871.63077689] games

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture12-4.png)


 This gives you an estimate of what the typical time between no-hitters is. It could be anywhere between 660 and 870 games.



###
 Pairs bootstrap


####
 A function to do pairs bootstrap




```

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, deg=1)

    return bs_slope_reps, bs_intercept_reps


```


####
 Pairs bootstrap of literacy/fertility data




```python

# Generate replicates of slope and intercept using pairs bootstrap
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(illiteracy, fertility, size=1000)

# Compute and print 95% CI for slope
print(np.percentile(bs_slope_reps, [2.5, 97.5]))

# Plot the histogram
_ = plt.hist(bs_slope_reps, bins=50, normed=True)
_ = plt.xlabel('slope')
_ = plt.ylabel('PDF')
plt.show()

# [0.04378061 0.0551616 ]

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture13-4.png)

####
 Plotting bootstrap regressions




```python

# Generate array of x-values for bootstrap lines: x
x = np.array([0, 100])

# Plot the bootstrap lines
for i in range(100):
    _ = plt.plot(x,
                 bs_slope_reps[i]*x + bs_intercept_reps[i],
                 linewidth=0.5, alpha=0.2, color='red')

# Plot the data
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')

# Label axes, set the margins, and show the plot
_ = plt.xlabel('illiteracy')
_ = plt.ylabel('fertility')
plt.margins(0.02)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture14-3.png)


 Introduction to hypothesis testing
------------------------------------


###
 Formulating and simulating a hypothesis



**Null hypothesis**
 : another name for the hypothesis you are testing




**Permutation**
 : random reordering of entries in an array



####
 Generating a permutation sample

 np.random.permutation()



 Permutation sampling is a great way to simulate the hypothesis that two variables have identical probability distributions. This is often a hypothesis you want to test, so in this exercise, you will write a function to generate a permutation sample from two data sets.





```

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

```


####
 Visualizing permutation sampling




```

for _ in range(50):
    # Generate permutation samples
    perm_sample_1, perm_sample_2 = permutation_sample(rain_june, rain_november)


    # Compute ECDFs
    x_1, y_1 = ecdf(perm_sample_1)
    x_2, y_2 = ecdf(perm_sample_2)

    # Plot ECDFs of permutation sample
    _ = plt.plot(x_1, y_1, marker='.', linestyle='none',
                 color='red', alpha=0.02)
    _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                 color='blue', alpha=0.02)

# Create and plot ECDFs from original data
x_1, y_1 = ecdf(rain_june)
x_2, y_2 = ecdf(rain_november)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('monthly rainfall (mm)')
_ = plt.ylabel('ECDF')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture1-16.png)


 Notice that the permutation samples ECDFs overlap and give a purple haze. None of the ECDFs from the permutation samples overlap with the observed data, suggesting that the hypothesis is not commensurate with the data. June and November rainfall are not identically distributed.



###
 Test statistics and p-values


####
 Test statistics



 When performing hypothesis tests, your choice of test statistic should be pertinent to the question you are seeking to answer in your hypothesis test.


 The most important thing to consider is:
 **What are you asking?**



####
 What is a p-value?



 The p-value is generally a measure of the probability of observing a test statistic equally or more extreme than the one you observed, given that the null hypothesis is true.



####
 Generating permutation replicates




```python

# In most circumstances, func will be a function you write yourself.
def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates

```


####
 Look before you leap: EDA before hypothesis testing



 Kleinteich and Gorb (
 *Sci. Rep.*
 ,
 **4**
 , 5225, 2014) performed an interesting experiment with South American horned frogs. They held a plate connected to a force transducer, along with a bait fly, in front of them. They then measured the impact force and adhesive force of the frog’s tongue when it struck the target.




 Frog A is an adult and Frog B is a juvenile. The researchers measured the impact force of 20 strikes for each frog. In the next exercise, we will test the hypothesis that the two frogs have the same distribution of impact forces. But, remember, it is important to do EDA first! Let’s make a bee swarm plot for the data. They are stored in a Pandas data frame,
 `df`
 , where column
 `ID`
 is the identity of the frog and column
 `impact_force`
 is the impact force in Newtons (N).





```

df.head()
   ID  impact_force
20  A         1.612
21  A         0.605
22  A         0.327
23  A         0.946
24  A         0.541

```




```python

# Make bee swarm plot
_ = sns.swarmplot(x='ID', y='impact_force', data=df)

# Label axes
_ = plt.xlabel('frog')
_ = plt.ylabel('impact force (N)')

# Show the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture2-15.png)


 Eyeballing it, it does not look like they come from the same distribution. Frog A, the adult, has three or four very hard strikes, and Frog B, the juvenile, has a couple weak ones. However, it is possible that with only 20 samples it might be too difficult to tell if they have difference distributions, so we should proceed with the hypothesis test.



####
 Permutation test on frog data



 The average strike force of Frog A was 0.71 Newtons (N), and that of Frog B was 0.42 N for a difference of 0.29 N. It is possible the frogs strike with the same force and this observed difference was by chance. You will compute the probability of getting at least a 0.29 N difference in mean strike force under the hypothesis that the distributions of strike forces for the two frogs are identical. We use a permutation test with a test statistic of the difference of means to test this hypothesis.





```

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff

# Compute difference of mean impact force from experiment: empirical_diff_means
empirical_diff_means = diff_of_means(force_a, force_b)

# Draw 10,000 permutation replicates: perm_replicates
perm_replicates = draw_perm_reps(force_a, force_b,
                                 diff_of_means, size=10000)

# Compute p-value: p
p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

# Print the result
print('p-value =', p)

# p-value = 0.0063
# p-value = 0.63%

```



 The p-value tells you that there is about a 0.6% chance that you would get the difference of means observed in the experiment if frogs were exactly the same.




 A p-value below 0.01 is typically said to be “statistically significant,” but: warning! warning! warning! You have computed a p-value; it is a number. I encourage you not to distill it to a yes-or-no phrase. p = 0.006 and p = 0.000000006 are both said to be “statistically significant,” but they are definitely not the same!



###
 Bootstrap hypothesis tests



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture3-14.png)

####
 A one-sample bootstrap hypothesis test



 Another juvenile frog was studied, Frog C, and you want to see if Frog B and Frog C have similar impact forces. Unfortunately, you do not have Frog C’s impact forces available, but you know they have a mean of 0.55 N. Because you don’t have the original data, you cannot do a permutation test, and you cannot assess the hypothesis that the forces from Frog B and Frog C come from the same distribution. You will therefore test another, less restrictive hypothesis: The mean strike force of Frog B is equal to that of Frog C.




 To set up the bootstrap hypothesis test, you will take the mean as our test statistic. Remember, your goal is to calculate the probability of getting a mean impact force less than or equal to what was observed for Frog B
 *if the hypothesis that the true mean of Frog B’s impact forces is equal to that of Frog C is true*
 . You first translate all of the data of Frog B such that the mean is 0.55 N. This involves adding the mean force of Frog C and subtracting the mean force of Frog B from each measurement of Frog B. This leaves other properties of Frog B’s distribution, such as the variance, unchanged.





```python

# Make an array of translated impact forces: translated_force_b
translated_force_b = force_b - np.mean(force_b) + 0.55

# Take bootstrap replicates of Frog B's translated impact forces: bs_replicates
bs_replicates = draw_bs_reps(translated_force_b, np.mean, 10000)

# Compute fraction of replicates that are less than the observed Frog B force: p
p = np.sum(bs_replicates <= np.mean(force_b)) / 10000

# Print the p-value
print('p = ', p)

# p =  0.0046
# p = 0.46%

```



 The low p-value suggests that the null hypothesis that Frog B and Frog C have the same mean impact force is false.



####
 A two-sample bootstrap hypothesis test for difference of means



 We now want to test the hypothesis that Frog A and Frog B have the same mean impact force, but not necessarily the same distribution, which is also impossible with a permutation test.




 To do the two-sample bootstrap test, we shift
 *both*
 arrays to have the same mean, since we are simulating the hypothesis that their means are, in fact, equal. We then draw bootstrap samples out of the shifted arrays and compute the difference in means. This constitutes a bootstrap replicate, and we generate many of them. The p-value is the fraction of replicates with a difference in means greater than or equal to what was observed.





```python

# Compute mean of all forces: mean_force
mean_force = np.mean(forces_concat)

# Generate shifted arrays
force_a_shifted = force_a - np.mean(force_a) + mean_force
force_b_shifted = force_b - np.mean(force_b) + mean_force

# Compute 10,000 bootstrap replicates from shifted arrays
bs_replicates_a = draw_bs_reps(force_a_shifted, np.mean, size=10000)
bs_replicates_b = draw_bs_reps(force_b_shifted, np.mean, size=10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_replicates_a - bs_replicates_b

# Compute and print p-value: p
p = np.sum(bs_replicates >= empirical_diff_means) / 10000
print('p-value =', p)

# p-value = 0.0043
# p-value = 0.43%

```



 You got a similar result as when you did the permutation test. Nonetheless, remember that it is important to carefully think about what question you want to ask. Are you only interested in the mean impact force, or in the distribution of impact forces?




 Hypothesis test examples
--------------------------


###
 A/B testing


####
 The vote for the Civil Rights Act in 1964



 The Civil Rights Act of 1964 was one of the most important pieces of legislation ever passed in the USA. Excluding “present” and “abstain” votes, 153 House Democrats and 136 Republicans voted yea. However, 91 Democrats and 35 Republicans voted nay. Did party affiliation make a difference in the vote?




 To answer this question, you will evaluate the hypothesis that the party of a House member has no bearing on his or her vote. You will use the fraction of Democrats voting in favor as your test statistic and evaluate the probability of observing a fraction of Democrats voting in favor at least as small as the observed fraction of 153/244. (That’s right, at least as
 *small*
 as. In 1964, it was the
 *Democrats*
 who were less progressive on civil rights issues.) To do this, permute the party labels of the House voters and then arbitrarily divide them into “Democrats” and “Republicans” and compute the fraction of Democrats voting yea.





```python

# Construct arrays of data: dems, reps
dems = np.array([True] * 153 + [False] * 91)
reps = np.array([True] * 136 + [False] * 35)

def frac_yea_dems(dems, reps):
    """Compute fraction of Democrat yea votes."""
    frac = np.sum(dems) / len(dems)
    return frac

# Acquire permutation samples: perm_replicates
perm_replicates = draw_perm_reps(dems, reps, frac_yea_dems, size=10000)

# Compute and print p-value: p
p = np.sum(perm_replicates <= 153/244) / len(perm_replicates)
print('p-value =', p)

# p-value = 0.0002
# p-value = 0.02%

```



 This small p-value suggests that party identity had a lot to do with the voting. Importantly, the South had a higher fraction of Democrat representatives, and consequently also a more racist bias.



####
 What is equivalent?



 You have experience matching a stories to probability distributions. Similarly, you use the same procedure for two different A/B tests if their stories match. In the Civil Rights Act example you just did, you performed an A/B test on voting data, which has a Yes/No type of outcome for each subject (in that case, a voter). Which of the following situations involving testing by a web-based company has an equivalent set up for an A/B test as the one you just did with the Civil Rights Act of 1964?




 You measure the number of people who click on an ad on your company’s website before and after changing its color.




 The “Democrats” are those who view the ad before the color change, and the “Republicans” are those who view it after.



####
 A time-on-website analog



 It turns out that you already did a hypothesis test analogous to an A/B test where you are interested in how much time is spent on the website before and after an ad campaign. The frog tongue force (a continuous quantity like time on the website) is an analog. “Before” = Frog A and “after” = Frog B. Let’s practice this again with something that actually is a before/after scenario.




 We return to the no-hitter data set. In 1920, Major League Baseball implemented important rule changes that ended the so-called dead ball era. Importantly, the pitcher was no longer allowed to spit on or scuff the ball, an activity that greatly favors pitchers. In this problem you will perform an A/B test to determine if these rule changes resulted in a slower rate of no-hitters (i.e., longer average time between no-hitters) using the difference in mean inter-no-hitter time as your test statistic. The inter-no-hitter times for the respective eras are stored in the arrays
 `nht_dead`
 and
 `nht_live`
 , where “nht” is meant to stand for “no-hitter time.”





```

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates



# Compute the observed difference in mean inter-no-hitter times: nht_diff_obs
nht_diff_obs = diff_of_means(nht_dead, nht_live)

# Acquire 10,000 permutation replicates of difference in mean no-hitter time: perm_replicates
perm_replicates = draw_perm_reps(nht_dead, nht_live, diff_of_means, size=10000)

# Compute and print the p-value: p
p = np.sum(perm_replicates <= nht_diff_obs) /len(perm_replicates)
print('p-val =', p)

p-val = 0.0001

```



 Your p-value is 0.0001, which means that only one out of your 10,000 replicates had a result as extreme as the actual difference between the dead ball and live ball eras. This suggests strong statistical significance. Watch out, though, you could very well have gotten zero replicates that were as extreme as the observed value. This just means that the p-value is quite small, almost certainly smaller than 0.001.



####
 What should you have done first?



 That was a nice hypothesis test you just did to check out whether the rule changes in 1920 changed the rate of no-hitters. But what
 *should*
 you have done with the data first?




 Performed EDA, perhaps plotting the ECDFs of inter-no-hitter times in the dead ball and live ball eras.




 Always a good idea to do first! I encourage you to go ahead and plot the ECDFs right now. You will see by eye that the null hypothesis that the distributions are the same is almost certainly not true.





```python

# Create and plot ECDFs
x_1, y_1 = ecdf(nht_dead)
x_2, y_2 = ecdf(nht_live)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('Games between no-hitters')
_ = plt.ylabel('ECDF')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture1-17.png)

###
 Test of correlation



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture2-16.png)

####
 Simulating a null hypothesis concerning correlation



 The observed correlation between female illiteracy and fertility in the data set of 162 countries may just be by chance; the fertility of a given country may actually be totally independent of its illiteracy. You will test this null hypothesis in the next exercise.




 To do the test, you need to simulate the data assuming the null hypothesis is true. Of the following choices, which is the best way to to do it?




 Answer: Do a permutation test: Permute the illiteracy values but leave the fertility values fixed to generate a new set of (illiteracy, fertility) data.




 This exactly simulates the null hypothesis and does so more efficiently than the last option. It is exact because it uses all data and eliminates any correlation because which illiteracy value pairs to which fertility value is shuffled.




 Last option: Do a permutation test: Permute both the illiteracy and fertility values to generate a new set of (illiteracy, fertility data). This exactly simulates the null hypothesis and does so more efficiently than the last option. It is exact because it uses all data and eliminates any correlation because which illiteracy value pairs to which fertility value is shuffled.



####
 Hypothesis test on Pearson correlation



 The observed correlation between female illiteracy and fertility may just be by chance; the fertility of a given country may actually be totally independent of its illiteracy. You will test this hypothesis. To do so, permute the illiteracy values but leave the fertility values fixed. This simulates the hypothesis that they are totally independent of each other. For each permutation, compute the Pearson correlation coefficient and assess how many of your permutation replicates have a Pearson correlation coefficient greater than the observed one.





```

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0,1]

# Compute observed correlation: r_obs
r_obs = pearson_r(illiteracy, fertility)

# r_obs = 0.8041324026815344

# Initialize permutation replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute illiteracy measurments: illiteracy_permuted
    illiteracy_permuted = np.random.permutation(illiteracy)

    # Compute Pearson correlation
    perm_replicates[i] = pearson_r(illiteracy_permuted, fertility)

# Compute p-value: p
p = np.sum(perm_replicates >= r_obs) /len(perm_replicates)
print('p-val =', p)

# p-val = 0.0

```



 You got a p-value of zero. In hacker statistics, this means that your p-value is very low, since you never got a single replicate in the 10,000 you took that had a Pearson correlation greater than the observed one. You could try increasing the number of replicates you take to continue to move the upper bound on your p-value lower and lower.



####
 Do neonicotinoid insecticides have unintended consequences?



 As a final exercise in hypothesis testing before we put everything together in our case study in the next chapter, you will investigate the effects of neonicotinoid insecticides on bee reproduction. These insecticides are very widely used in the United States to combat aphids and other pests that damage plants.




 In a recent study, Straub, et al. (
 [*Proc. Roy. Soc. B*
 , 2016](http://dx.doi.org/10.1098/rspb.2016.0506)
 ) investigated the effects of neonicotinoids on the sperm of pollinating bees. In this and the next exercise, you will study how the pesticide treatment affected the count of live sperm per half milliliter of semen.




 First, we will do EDA, as usual. Plot ECDFs of the alive sperm count for untreated bees (stored in the Numpy array
 `control`
 ) and bees treated with pesticide (stored in the Numpy array
 `treated`
 ).





```python

# Compute x,y values for ECDFs
x_control, y_control = ecdf(control)
x_treated, y_treated = ecdf(treated)

# Plot the ECDFs
plt.plot(x_control, y_control, marker='.', linestyle='none')
plt.plot(x_treated, y_treated, marker='.', linestyle='none')

# Set the margins
plt.margins(0.02)

# Add a legend
plt.legend(('control', 'treated'), loc='lower right')

# Label axes and show plot
plt.xlabel('millions of alive sperm per mL')
plt.ylabel('ECDF')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture3-15.png)


 The ECDFs show a pretty clear difference between the treatment and control; treated bees have fewer alive sperm. Let’s now do a hypothesis test in the next exercise.



####
 Bootstrap hypothesis test on bee sperm counts



 Now, you will test the following hypothesis:




**On average, male bees treated with neonicotinoid insecticide have the same number of active sperm per milliliter of semen than do untreated male bees.**




 You will use the difference of means as your test statistic.





```

def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates


# Compute the difference in mean sperm count: diff_means
diff_means = np.mean(control) - np.mean(treated)

# Compute mean of pooled data: mean_count
mean_count = np.mean(np.concatenate((control, treated)))

# Generate shifted data sets
control_shifted = control - np.mean(control) + mean_count
treated_shifted = treated - np.mean(treated) + mean_count

# Generate bootstrap replicates
bs_reps_control = draw_bs_reps(control_shifted,
                       np.mean, size=10000)
bs_reps_treated = draw_bs_reps(treated_shifted,
                       np.mean, size=10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_reps_control - bs_reps_treated

# Compute and print p-value: p
p = np.sum(bs_replicates >= np.mean(control) - np.mean(treated)) \
            / len(bs_replicates)
print('p-value =', p)

# p-value = 0.0

```



 The p-value is small, most likely less than 0.0001, since you never saw a bootstrap replicated with a difference of means at least as extreme as what was observed. In fact, when I did the calculation with 10 million replicates, I got a p-value of
 `2e-05`




 Putting it all together: a case study
---------------------------------------


###
 Finch beaks and the need for statistics



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture4-14.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture5-12.png)

####
 EDA of beak depths of Darwin’s finches



 For your first foray into the Darwin finch data, you will study how the beak depth (the distance, top to bottom, of a closed beak) of the finch species
 *Geospiza scandens*
 has changed over time. The Grants have noticed some changes of beak geometry depending on the types of seeds available on the island, and they also noticed that there was some interbreeding with another major species on Daphne Major,
 *Geospiza fortis*
 . These effects can lead to changes in the species over time.




 In the next few problems, you will look at the beak depth of
 *G. scandens*
 on Daphne Major in 1975 and in 2012. To start with, let’s plot all of the beak depth measurements in 1975 and 2012 in a bee swarm plot.




 The data are stored in a pandas DataFrame called
 `df`
 with columns
 `'year'`
 and
 `'beak_depth'`
 . The units of beak depth are millimeters (mm).





```

df.head()
   beak_depth  year
0         8.4  1975
1         8.8  1975
2         8.4  1975
3         8.0  1975
4         7.9  1975

```




```python

# Create bee swarm plot
_ = sns.swarmplot('year', 'beak_depth', data=df)

# Label the axes
_ = plt.xlabel('year')
_ = plt.ylabel('beak depth (mm)')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture6-9.png)


 It is kind of hard to see if there is a clear difference between the 1975 and 2012 data set. Eyeballing it, it appears as though the mean of the 2012 data set might be slightly higher, and it might have a bigger variance.



####
 ECDFs of beak depths



 While bee swarm plots are useful, we found that ECDFs are often even better when doing EDA. Plot the ECDFs for the 1975 and 2012 beak depth measurements on the same plot.





```python

# Compute ECDFs
x_1975, y_1975 = ecdf(bd_1975)
x_2012, y_2012 = ecdf(bd_2012)

# Plot the ECDFs
_ = plt.plot(x_1975, y_1975, marker='.', linestyle='none')
_ = plt.plot(x_2012, y_2012, marker='.', linestyle='none')

# Set margins
plt.margins(0.02)

# Add axis labels and legend
_ = plt.xlabel('beak depth (mm)')
_ = plt.ylabel('ECDF')
_ = plt.legend(('1975', '2012'), loc='lower right')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture7-10.png)


 The differences are much clearer in the ECDF. The mean is larger in the 2012 data, and the variance does appear larger as well.



####
 Parameter estimates of beak depths



 Estimate the
 *difference*
 of the mean beak depth of the
 *G. scandens*
 samples from 1975 and 2012 and report a 95% confidence interval.





```

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates


# Compute the difference of the sample means: mean_diff
mean_diff = np.mean(bd_2012) - np.mean(bd_1975)

# Get bootstrap replicates of means
bs_replicates_1975 = draw_bs_reps(bd_1975, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(bd_2012, np.mean, size=10000)

# Compute samples of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute 95% confidence interval: conf_int
conf_int = np.percentile(bs_diff_replicates, [2.5, 97.5])

# Print the results
print('difference of means =', mean_diff, 'mm')
print('95% confidence interval =', conf_int, 'mm')

# difference of means = 0.22622047244094645 mm
# 95% confidence interval = [0.05633521 0.39190544] mm

```


####
 Hypothesis test: Are beaks deeper in 2012?



 Your plot of the ECDF and determination of the confidence interval make it pretty clear that the beaks of
 *G. scandens*
 on Daphne Major have gotten deeper. But is it possible that this effect is just due to random chance? In other words, what is the probability that we would get the observed difference in mean beak depth if the means were the same?




 Be careful! The hypothesis we are testing is
 *not*
 that the beak depths come from the same distribution. For that we could use a permutation test.
 **The hypothesis is that the means are equal.**
 To perform this hypothesis test, we need to shift the two data sets so that they have the same mean and then use bootstrap sampling to compute the difference of means.





```python

# Compute mean of combined data set: combined_mean
combined_mean = np.mean(np.concatenate((bd_1975, bd_2012)))

# Shift the samples
# why shift the mean?
# to make np.mean(bd_1975_shifted) - np.mean(bd_2012_shifted) = 0 #1
# why make #1 = 0?
# because our hypothesis is "beak depth are the same in 1975 and 2012"
bd_1975_shifted = bd_1975 - np.mean(bd_1975) + combined_mean
bd_2012_shifted = bd_2012 - np.mean(bd_2012) + combined_mean

# Get bootstrap replicates of shifted data sets
bs_replicates_1975 = draw_bs_reps(bd_1975_shifted, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(bd_2012_shifted, np.mean, size=10000)

# Compute replicates of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute the p-value
p = np.sum(bs_diff_replicates >= mean_diff) / len(bs_diff_replicates)

# Print p-value
print('p =', p)

# p = 0.0034
# p = 0.34%

```



 We get a p-value of 0.0034, which suggests that there is a statistically significant difference. But remember: it is very important to know how different they are! In the previous exercise, you got a difference of 0.2 mm between the means. You should combine this with the statistical significance. Changing by 0.2 mm in 37 years is substantial by evolutionary standards. If it kept changing at that rate, the beak depth would double in only 400 years.



###
 Variation of beak shapes


####
 EDA of beak length and depth



 The beak length data are stored as
 `bl_1975`
 and
 `bl_2012`
 , again with units of millimeters (mm). You still have the beak depth data stored in
 `bd_1975`
 and
 `bd_2012`
 . Make scatter plots of beak depth (y-axis) versus beak length (x-axis) for the 1975 and 2012 specimens.





```python

# Make scatter plot of 1975 data
_ = plt.plot(bl_1975, bd_1975, marker='.',
             linestyle='None', color='blue', alpha=0.5)

# Make scatter plot of 2012 data
_ = plt.plot(bl_2012, bd_2012, marker='.',
            linestyle='None', color='red', alpha=0.5)

# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture8-7.png)


 In looking at the plot, we see that beaks got deeper (the red points are higher up in the y-direction), but not really longer. If anything, they got a bit shorter, since the red dots are to the left of the blue dots. So, it does not look like the beaks kept the same shape; they became shorter and deeper.



####
 Linear regressions



 Perform a linear regression for both the 1975 and 2012 data. Then, perform pairs bootstrap estimates for the regression parameters. Report 95% confidence intervals on the slope and intercept of the regression line.





```

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, deg=1)

    return bs_slope_reps, bs_intercept_reps


# Compute the linear regressions
slope_1975, intercept_1975 = np.polyfit(bl_1975, bd_1975, deg=1)
slope_2012, intercept_2012 = np.polyfit(bl_2012, bd_2012, deg=1)

# Perform pairs bootstrap for the linear regressions
bs_slope_reps_1975, bs_intercept_reps_1975 = draw_bs_pairs_linreg(bl_1975, bd_1975, size=1000)
bs_slope_reps_2012, bs_intercept_reps_2012 = draw_bs_pairs_linreg(bl_2012, bd_2012, size=1000)

# Compute confidence intervals of slopes
slope_conf_int_1975 = np.percentile(bs_slope_reps_1975, [2.5, 97.5])
slope_conf_int_2012 = np.percentile(bs_slope_reps_2012, [2.5, 97.5])
intercept_conf_int_1975 = np.percentile(bs_intercept_reps_1975, [2.5, 97.5])
intercept_conf_int_2012 = np.percentile(bs_intercept_reps_2012, [2.5, 97.5])


# Print the results
print('1975: slope =', slope_1975,
      'conf int =', slope_conf_int_1975)
print('1975: intercept =', intercept_1975,
      'conf int =', intercept_conf_int_1975)
print('2012: slope =', slope_2012,
      'conf int =', slope_conf_int_2012)
print('2012: intercept =', intercept_2012,
      'conf int =', intercept_conf_int_2012)

#   1975: slope = 0.4652051691605937 conf int = [0.33851226 0.59306491]
#   1975: intercept = 2.3908752365842263 conf int = [0.64892945 4.18037063]
#   2012: slope = 0.462630358835313 conf int = [0.33137479 0.60695527]
#   2012: intercept = 2.977247498236019 conf int = [1.06792753 4.70599387]

```



 It looks like they have the same slope, but different intercepts.



####
 Displaying the linear regression results




```python

# Make scatter plot of 1975 data
_ = plt.plot(bl_1975, bd_1975, marker='.',
             linestyle='none', color='blue', alpha=0.5)

# Make scatter plot of 2012 data
_ = plt.plot(bl_2012, bd_2012, marker='.',
             linestyle='none', color='red', alpha=0.5)

# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')

# Generate x-values for bootstrap lines: x
x = np.array([10, 17])

# Plot the bootstrap lines
for i in range(100):
    plt.plot(x, bs_slope_reps_1975[i] * x + bs_intercept_reps_1975[i],
             linewidth=0.5, alpha=0.2, color='blue')
    plt.plot(x, bs_slope_reps_2012[i] * x + bs_intercept_reps_2012[i],
             linewidth=0.5, alpha=0.2, color='red')

# Draw the plot again
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture9-6.png)

####
 Beak length to depth ratio



 The linear regressions showed interesting information about the beak geometry. The slope was the same in 1975 and 2012, suggesting that for every millimeter gained in beak length, the birds gained about half a millimeter in depth in both years. However, if we are interested in the shape of the beak, we want to compare the
 *ratio*
 of beak length to beak depth. Let’s make that comparison.





```python

# Compute length-to-depth ratios
ratio_1975 = bl_1975 / bd_1975
ratio_2012 = bl_2012 / bd_2012

# Compute means
mean_ratio_1975 = np.mean(ratio_1975)
mean_ratio_2012 = np.mean(ratio_2012)

# Generate bootstrap replicates of the means
bs_replicates_1975 = draw_bs_reps(ratio_1975, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(ratio_2012, np.mean, size=10000)

# Compute the 99% confidence intervals
conf_int_1975 = np.percentile(bs_replicates_1975, [0.5, 99.5])
conf_int_2012 = np.percentile(bs_replicates_2012, [0.5, 99.5])

# Print the results
print('1975: mean ratio =', mean_ratio_1975,
      'conf int =', conf_int_1975)
print('2012: mean ratio =', mean_ratio_2012,
      'conf int =', conf_int_2012)

# 1975: mean ratio = 1.5788823771858533 conf int = [1.55668803 1.60073509]
# 2012: mean ratio = 1.4658342276847767 conf int = [1.44363932 1.48729149]

```


####
 How different is the ratio?



 In the previous exercise, you computed the mean beak length to depth ratio with 99% confidence intervals for 1975 and for 2012. The results of that calculation are shown graphically in the plot accompanying this problem. In addition to these results, what would you say about the ratio of beak length to depth?




![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture10-6.png)


 The mean beak length-to-depth ratio decreased by about 0.1, or 7%, from 1975 to 2012. The 99% confidence intervals are not even close to overlapping, so this is a real change. The beak shape changed.



###
 Calculation of heritability


####
 EDA of heritability



 The array
 `bd_parent_scandens`
 contains the average beak depth (in mm) of two parents of the species
 `G. scandens`
 . The array
 `bd_offspring_scandens`
 contains the average beak depth of the offspring of the respective parents. The arrays
 `bd_parent_fortis`
 and
 `bd_offspring_fortis`
 contain the same information about measurements from
 *G. fortis*
 birds.




 Make a scatter plot of the average offspring beak depth (y-axis) versus average parental beak depth (x-axis) for both species. Use the
 `alpha=0.5`
 keyword argument to help you see overlapping points.





```python

# Make scatter plots
_ = plt.plot(bd_parent_fortis, bd_offspring_fortis,
             marker='.', linestyle='none', color='blue', alpha=0.5)
_ = plt.plot(bd_parent_scandens, bd_offspring_scandens,
             marker='.', linestyle='none', color='red', alpha=0.5)

# Label axes
_ = plt.xlabel('parental beak depth (mm)')
_ = plt.ylabel('offspring beak depth (mm)')

# Add legend
_ = plt.legend(('G. fortis', 'G. scandens'), loc='lower right')

# Show plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture11-6.png)


 It appears as though there is a stronger correlation in
 *G. fortis*
 than than in
 *G. scandens*
 . This suggests that beak depth is more strongly inherited in
 *G. fortis*
 . We’ll quantify this correlation next.



####
 Correlation of offspring and parental data



 In an effort to quantify the correlation between offspring and parent beak depths, we would like to compute statistics, such as the Pearson correlation coefficient, between parents and offspring. To get confidence intervals on this, we need to do a pairs bootstrap.




 You have
 [already written](https://campus.datacamp.com/courses/statistical-thinking-in-python-part-2/bootstrap-confidence-intervals?ex=12)
 a function to do pairs bootstrap to get estimates for parameters derived from linear regression. Your task in this exercise is to make a new function with call signature
 `draw_bs_pairs(x, y, func, size=1)`
 that performs pairs bootstrap and computes a single statistic on pairs samples defined. The statistic of interest is computed by calling
 `func(bs_x, bs_y)`
 . In the next exercise, you will use
 `pearson_r`
 for
 `func`
 .





```

def draw_bs_pairs(x, y, func, size=1):
    """Perform pairs bootstrap for a single statistic."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_replicates[i] = func(bs_x, bs_y)

    return bs_replicates


```


####
 Pearson correlation of offspring and parental data



 The Pearson correlation coefficient seems like a useful measure of how strongly the beak depth of parents are inherited by their offspring. Compute the Pearson correlation coefficient between parental and offspring beak depths for
 *G. scandens*
 . Do the same for
 *G. fortis*
 . Then, use the function you wrote in the last exercise to compute a 95% confidence interval using pairs bootstrap.





```

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0,1]


# Compute the Pearson correlation coefficients
r_scandens = pearson_r(bd_parent_scandens, bd_offspring_scandens)
r_fortis = pearson_r(bd_parent_fortis, bd_offspring_fortis)

# Acquire 1000 bootstrap replicates of Pearson r
bs_replicates_scandens = draw_bs_pairs(bd_parent_scandens, bd_offspring_scandens, pearson_r, size=1000)

bs_replicates_fortis = draw_bs_pairs(bd_parent_fortis, bd_offspring_fortis, pearson_r, size=1000)


# Compute 95% confidence intervals
conf_int_scandens = np.percentile(bs_replicates_scandens, [2.5, 97.5])
conf_int_fortis = np.percentile(bs_replicates_fortis, [2.5, 97.5])

# Print results
print('G. scandens:', r_scandens, conf_int_scandens)
print('G. fortis:', r_fortis, conf_int_fortis)

#    G. scandens: 0.4117063629401258 [0.26564228 0.54388972]
#    G. fortis: 0.7283412395518487 [0.6694112  0.77840616]

```



 It is clear from the confidence intervals that beak depth of the offspring of
 *G. fortis*
 parents is more strongly correlated with their offspring than their
 *G. scandens*
 counterparts.



####
 Measuring heritability



 Remember that the Pearson correlation coefficient is the ratio of the covariance to the geometric mean of the variances of the two data sets. This is a measure of the correlation between parents and offspring, but might not be the best estimate of heritability. If we stop and think, it makes more sense to define heritability as the ratio of the covariance between parent and offspring to the
 *variance of the parents alone*
 . In this exercise, you will estimate the heritability and perform a pairs bootstrap calculation to get the 95% confidence interval.




 This exercise highlights a very important point. Statistical inference (and data analysis in general) is not a plug-n-chug enterprise. You need to think carefully about the questions you are seeking to answer with your data and analyze them appropriately. If you are interested in how heritable traits are, the quantity we defined as the heritability is more apt than the off-the-shelf statistic, the Pearson correlation coefficient.





```

def heritability(parents, offspring):
    """Compute the heritability from parent and offspring samples."""
    covariance_matrix = np.cov(parents, offspring)
    return covariance_matrix[0,1] / covariance_matrix[0,0]

# Compute the heritability
heritability_scandens = heritability(bd_parent_scandens, bd_offspring_scandens)
heritability_fortis = heritability(bd_parent_fortis, bd_offspring_fortis)

# Acquire 1000 bootstrap replicates of heritability
replicates_scandens = draw_bs_pairs(
        bd_parent_scandens, bd_offspring_scandens, heritability, size=1000)

replicates_fortis = draw_bs_pairs(
        bd_parent_fortis, bd_offspring_fortis, heritability, size=1000)


# Compute 95% confidence intervals
conf_int_scandens = np.percentile(replicates_scandens, [2.5, 97.5])
conf_int_fortis = np.percentile(replicates_fortis, [2.5, 97.5])

# Print results
print('G. scandens:', heritability_scandens, conf_int_scandens)
print('G. fortis:', heritability_fortis, conf_int_fortis)


#   G. scandens: 0.5485340868685982 [0.34395487 0.75638267]
#   G. fortis: 0.7229051911438159 [0.64655013 0.79688342]

```



 Here again, we see that
 *G. fortis*
 has stronger heritability than
 *G. scandens*
 . This suggests that the traits of
 *G. fortis*
 may be strongly incorporated into
 *G. scandens*
 by introgressive hybridization.



####
 Is beak depth heritable at all in G. scandens?



 The heritability of beak depth in
 *G. scandens*
 seems low. It could be that this observed heritability was just achieved by chance and
 **beak depth is actually not really heritable in the species**
 . You will test that hypothesis here. To do this, you will do a pairs permutation test.





```python

# Initialize array of replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute parent beak depths
    bd_parent_permuted = np.random.permutation(bd_parent_scandens)
    perm_replicates[i] = heritability(bd_parent_permuted,
                                      bd_offspring_scandens)

# Compute p-value: p
p = np.sum(perm_replicates >= heritability_scandens) / len(perm_replicates)

# Print the p-value
print('p-val =', p)

# p-val = 0.0

```



 You get a p-value of zero, which means that none of the 10,000 permutation pairs replicates you drew had a heritability high enough to match that which was observed. This strongly suggests that beak depth is heritable in
 *G. scandens*
 , just not as much as in
 *G. fortis*
 . If you like, you can plot a histogram of the heritability replicates to get a feel for how extreme of a value of heritability you might expect by chance.





```

plt.hist(perm_replicates)
plt.axvline(x=heritability_scandens, color = 'red')
plt.text(heritability_scandens, 1500, 'heritability_scandens', ha='center', va='center',rotation='vertical', backgroundcolor='white')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-thinking-in-python-(part-2)/capture12-5.png)

