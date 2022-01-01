---
title: Case Studies in Statistical Thinking
date: 2021-12-07 11:22:08 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Case Studies in Statistical Thinking
=======================================







 This is the memo of the 5th course (5 courses in all) of ‘Statistics Fundamentals with Python’ skill track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/case-studies-in-statistical-thinking)**
 .



###
**Table of contents**


1. Fish sleep and bacteria growth: A review of Statistical Thinking I and II
2. [Analysis of results of the 2015 FINA World Swimming Championships](https://datascience103579984.wordpress.com/2019/09/29/case-studies-in-statistical-thinking-from-datacamp/2/)
3. [The “Current Controversy” of the 2013 World Championships](https://datascience103579984.wordpress.com/2019/09/29/case-studies-in-statistical-thinking-from-datacamp/3/)
4. [Statistical seismology and the Parkfield region](https://datascience103579984.wordpress.com/2019/09/29/case-studies-in-statistical-thinking-from-datacamp/4/)
5. [Earthquakes and oil mining in Oklahoma](https://datascience103579984.wordpress.com/2019/09/29/case-studies-in-statistical-thinking-from-datacamp/5/)






---



# **1. Fish sleep and bacteria growth: A review of Statistical Thinking I and II**
---------------------------------------------------------------------------------


## **1.1 Activity of zebrafish and melatonin**


####
**EDA: Plot ECDFs of active bout length**



 An active bout is a stretch of time where a fish is constantly moving. Plot an ECDF of active bout length for the mutant and wild type fish for the seventh night of their lives. The data sets are in the
 `numpy`
 arrays
 `bout_lengths_wt`
 and
 `bout_lengths_mut`
 . The bout lengths are in units of minutes.





```python

# Import the dc_stat_think module as dcst
import dc_stat_think as dcst

# Generate x and y values for plotting ECDFs
x_wt, y_wt = dcst.ecdf(bout_lengths_wt)
x_mut, y_mut = dcst.ecdf(bout_lengths_mut)

# Plot the ECDFs
_ = plt.plot(x_wt, y_wt, marker='.', linestyle='none')
_ = plt.plot(x_mut, y_mut, marker='.', linestyle='none')

# Make a legend, label axes, and show plot
_ = plt.legend(('wt', 'mut'))
_ = plt.xlabel('active bout length (min)')
_ = plt.ylabel('ECDF')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/1-8.png?w=1024)


 There is an outlier of one active bout for a mutant fish, and the ECDF exposes this clearly. It is important to know about, but we will not focus on it going forward, though.



####
**Interpreting ECDFs and the story**



 While a more detailed analysis of distributions is often warranted for careful analyses, you can already get a feel for the distributions and the story behind the data by eyeballing the ECDFs. Which of the following would be the most reasonable statement to make about how the active bout lengths are distributed and what kind of process might be behind exiting the active bout to rest?




 If you need a refresher, here are videos from Statistical Thinking I about stories behind probability distributions.



* [Discrete Uniform and Binomial](https://campus.datacamp.com/courses/statistical-thinking-in-python-part-1/thinking-probabilistically-discrete-variables?ex=9)
* [Poisson processes and Poisson distribution](https://campus.datacamp.com/courses/statistical-thinking-in-python-part-1/thinking-probabilistically-discrete-variables?ex=12)
* [Normal distribution](https://campus.datacamp.com/courses/statistical-thinking-in-python-part-1/thinking-probabilistically-continuous-variables?ex=4)
* [Exponential Distribution](https://campus.datacamp.com/courses/statistical-thinking-in-python-part-1/thinking-probabilistically-continuous-variables?ex=11)



 The bout lengths appear Exponentially distributed, which implies that exiting an active bout to rest is a Poisson process; the fish have no apparent memory about when they became active.




 While not
 *exactly*
 Exponentially distributed, the ECDF has no left tail, and no discernible inflection point, which is very much like the Exponential CDF.





---


## **1.2 Bootstrap confidence intervals**


####
**Parameter estimation: active bout length**



 Compute the mean active bout length for wild type and mutant, with 95% bootstrap confidence interval. The data sets are again available in the
 `numpy`
 arrays
 `bout_lengths_wt`
 and
 `bout_lengths_mut`
 . The
 `dc_stat_think`
 module has been imported as
 `dcst`
 .





```python

# Compute mean active bout length
mean_wt = np.mean(bout_lengths_wt)
mean_mut = np.mean(bout_lengths_mut)

# Draw bootstrap replicates
bs_reps_wt = dcst.draw_bs_reps(bout_lengths_wt, np.mean, size=10000)
bs_reps_mut = dcst.draw_bs_reps(bout_lengths_mut, np.mean, size=10000)

# Compute 95% confidence intervals
conf_int_wt = np.percentile(bs_reps_wt, [2.5, 97.5])
conf_int_mut = np.percentile(bs_reps_mut, [2.5, 97.5])

# Print the results
print("""
wt:  mean = {0:.3f} min., conf. int. = [{1:.1f}, {2:.1f}] min.
mut: mean = {3:.3f} min., conf. int. = [{4:.1f}, {5:.1f}] min.
""".format(mean_wt, *conf_int_wt, mean_mut, *conf_int_mut))

# wt:  mean = 3.874 min., conf. int. = [3.6, 4.1] min.
# mut: mean = 6.543 min., conf. int. = [6.1, 7.0] min.

```



 The confidence intervals are quite separated. Nonetheless, we will proceed to perform hypothesis tests.





---


## **1.3 Permutation and bootstrap hypothesis tests**



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/3-7.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/4-6.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/5-6.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/6-6.png?w=1024)



####
**Permutation test: wild type versus heterozygote**



 Test the hypothesis that the heterozygote and wild type bout lengths are identically distributed using a permutation test.





```python

# Compute the difference of means: diff_means_exp
diff_means_exp = np.mean(bout_lengths_het) - np.mean(bout_lengths_wt)

# Draw permutation replicates: perm_reps
perm_reps = dcst.draw_perm_reps(bout_lengths_het, bout_lengths_wt,
                               dcst.diff_of_means, size=10000)

# Compute the p-value: p-val
p_val = np.sum(perm_reps >= diff_means_exp) / len(perm_reps)

# Print the result
print('p =', p_val)

# p = 0.001

```



 A p-value of 0.001 suggests that the observed difference in means is unlikely to occur if heterozygotic and wild type fish have active bout lengths that are identically distributed.



####
**Bootstrap hypothesis test**



 The permutation test has a pretty restrictive hypothesis, that the heterozygotic and wild type bout lengths are identically distributed. Now, use a bootstrap hypothesis test to test the hypothesis that the means are equal, making no assumptions about the distributions.





```python

# Concatenate arrays: bout_lengths_concat
bout_lengths_concat = np.concatenate((bout_lengths_wt, bout_lengths_het))

# Compute mean of all bout_lengths: mean_bout_length
mean_bout_length = np.mean(bout_lengths_concat)

# Generate shifted arrays
wt_shifted = bout_lengths_wt - np.mean(bout_lengths_wt) + mean_bout_length
het_shifted = bout_lengths_het - np.mean(bout_lengths_het) + mean_bout_length

# Compute 10,000 bootstrap replicates from shifted arrays
bs_reps_wt = dcst.draw_bs_reps(wt_shifted, np.mean, size=10000)
bs_reps_het = dcst.draw_bs_reps(het_shifted, np.mean, size=10000)

# Get replicates of difference of means: bs_replicates
bs_reps = bs_reps_het - bs_reps_wt

# Compute and print p-value: p
p = np.sum(bs_reps >= diff_means_exp) / len(bs_reps)
print('p-value =', p)

# p-value = 0.0004

```



 We get a result of similar magnitude as the permutation test, though slightly smaller, probably because the heterozygote bout length distribution has a heavier tail to the right.





---


## **1.4 Linear regressions and pairs bootstrap**


####
**Assessing the growth rate**



 To compute the growth rate, you can do a linear regression of the logarithm of the total bacterial area versus time. Compute the growth rate and get a 95% confidence interval using pairs bootstrap. The time points, in units of hours, are stored in the
 `numpy`
 array
 `t`
 and the bacterial area, in units of square micrometers, is stored in
 `bac_area`
 .





```python

# Compute logarithm of the bacterial area: log_bac_area
log_bac_area = np.log(bac_area)

# Compute the slope and intercept: growth_rate, log_a0
growth_rate, log_a0 = np.polyfit(t, log_bac_area, 1)

# Draw 10,000 pairs bootstrap replicates: growth_rate_bs_reps, log_a0_bs_reps
growth_rate_bs_reps, log_a0_bs_reps = \
            dcst.draw_bs_pairs_linreg(t, log_bac_area, size=10000)

# Compute confidence intervals: growth_rate_conf_int
growth_rate_conf_int = np.percentile(growth_rate_bs_reps, [2.5, 97.5])

# Print the result to the screen
print("""
Growth rate: {0:.4f} sq. µm/hour
95% conf int: [{1:.4f}, {2:.4f}] sq. µm/hour
""".format(growth_rate, *growth_rate_conf_int))

# Growth rate: 0.2301 sq. µm/hour
# 95% conf int: [0.2266, 0.2336] sq. µm/hour

```



 Under these conditions, the bacteria add about 0.23 square micrometers worth of mass each hour. The error bar is very tight, which we will see graphically in the next exercise.



####
**Plotting the growth curve**



 You saw in the previous exercise that the confidence interval on the growth curve is very tight. You will explore this graphically here by plotting several bootstrap lines along with the growth curve. You will use the
 `plt.semilogy()`
 function to make the plot with the y-axis on a log scale. This means that you will need to transform your theoretical linear regression curve for plotting by exponentiating it.





```python

# Plot data points in a semilog-y plot with axis labeles
_ = plt.semilogy(t, bac_area, marker='.', linestyle='none')

# Generate x-values for the bootstrap lines: t_bs
t_bs = np.array([0, 14])

# Plot the first 100 bootstrap lines
for i in range(100):
    y = np.exp(growth_rate_bs_reps[i] * t_bs + log_a0_bs_reps[i])
    _ = plt.semilogy(t_bs, y, linewidth=0.5, alpha=0.05, color='red')

# Label axes and show plot
_ = plt.xlabel('time (hr)')
_ = plt.ylabel('area (sq. µm)')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/7-7.png?w=1024)


 You can see that the bootstrap replicates do not stray much. This is due to the exquisitly exponential nature of the bacterial growth under these experimental conditions.


# **2. Analysis of results of the 2015 FINA World Swimming Championships**
-------------------------------------------------------------------------


## **2.1 Introduction to swimming data**



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/1.png?w=905)
![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/2.png?w=828)



####
**Graphical EDA of men’s 200 free heats**



 In the heats, all contestants swim, the very fast and the very slow. To explore how the swim times are distributed, plot an ECDF of the men’s 200 freestyle.





```python

# Generate x and y values for ECDF: x, y
x, y = dcst.ecdf(mens_200_free_heats)

# Plot the ECDF as dots
plt.plot(x, y, marker='.', linestyle='none')

# Label axes and show plot
plt.xlabel('time (s)')
plt.ylabel('ECDF')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/3.png?w=1024)


 Graphical EDA is always a great start. We see that fast swimmers are below 115 seconds, with a smattering of slow swimmers past that, including one very slow swimmer.



####
**200 m free time with confidence interval**



 Now, you will practice parameter estimation and computation of confidence intervals by computing the mean and median swim time for the men’s 200 freestyle heats. The median is useful because it is immune to heavy tails in the distribution of swim times, such as the slow swimmers in the heats.
 `mens_200_free_heats`
 is still in your namespace.





```python

# Compute mean and median swim times
mean_time = np.mean(mens_200_free_heats)
median_time = np.median(mens_200_free_heats)

# Draw 10,000 bootstrap replicates of the mean and median
bs_reps_mean = dcst.draw_bs_reps(mens_200_free_heats, np.mean, size=10000)
bs_reps_median = dcst.draw_bs_reps(mens_200_free_heats, np.median, size=10000)


# Compute the 95% confidence intervals
conf_int_mean = np.percentile(bs_reps_mean, [2.5, 97.5])
conf_int_median = np.percentile(bs_reps_median, [2.5, 97.5])

# Print the result to the screen
print("""
mean time: {0:.2f} sec.
95% conf int of mean: [{1:.2f}, {2:.2f}] sec.

median time: {3:.2f} sec.
95% conf int of median: [{4:.2f}, {5:.2f}] sec.
""".format(mean_time, *conf_int_mean, median_time, *conf_int_median))

```



 Indeed, the mean swim time is longer than the median because of the effect of the very slow swimmers.





---


## **2.2 Do swimmers go faster in the finals?**



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/6.png?w=830)
![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/7.png?w=1024)



####
**EDA: finals versus semifinals**



 First, you will get an understanding of how athletes’ performance changes from the semifinals to the finals by computing the fractional improvement from the semifinals to finals and plotting an ECDF of all of these values.




 The arrays
 `final_times`
 and
 `semi_times`
 contain the swim times of the respective rounds. The arrays are aligned such that
 `final_times[i]`
 and
 `semi_times[i]`
 are for the same swimmer/event. If you are interested in the strokes/events, you can check out the data frame
 `df`
 in your namespace, which has more detailed information, but is not used in the analysis.





```

df.head()
   athleteid stroke  distance  final_swimtime  lastname  semi_swimtime
0     100537   FREE       100           52.52  CAMPBELL          53.00
1     100537   FREE        50           24.12  CAMPBELL          24.32
2     100631   FREE       100           52.82  CAMPBELL          52.84
3     100631   FREE        50           24.36  CAMPBELL          24.22
4     100650    FLY       100           57.67    MCKEON          57.59

```




```python

# Compute fractional difference in time between finals and semis
f = (semi_times - final_times) / semi_times

# Generate x and y values for the ECDF: x, y
x, y = dcst.ecdf(f)

# Make a plot of the ECDF
plt.plot(x, y, marker='.', linestyle='none')

# Label axes and show plot
_ = plt.xlabel('f')
_ = plt.ylabel('ECDF')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/8.png?w=642)


 The median of the ECDF is just above zero. But at first glance, it does not look like there is much of any difference between semifinals and finals. We’ll check this carefully in the next exercises.



####
**Parameter estimates of difference between finals and semifinals**



 Compute the mean fractional improvement from the semifinals to finals, along with a 95% confidence interval of the mean. The Numpy array
 `f`
 that you computed in the last exercise is in your namespace.





```python

# Mean fractional time difference: f_mean
f_mean = np.mean(f)

# Get bootstrap reps of mean: bs_reps
bs_reps = dcst.draw_bs_reps(f, func=np.mean, size=10000)

# Compute confidence intervals: conf_int
conf_int = np.percentile(bs_reps, [2.5, 97.5])

# Report
print("""
mean frac. diff.: {0:.5f}
95% conf int of mean frac. diff.: [{1:.5f}, {2:.5f}]""".format(f_mean, *conf_int))

# mean frac. diff.: 0.00040
# 95% conf int of mean frac. diff.: [-0.00092, 0.00176]

```



 It looks like the mean finals time is just faster than the mean semifinal time, and they very well may be the same. We’ll test this hypothesis next.



####
**How to do the permutation test**



 Based on our EDA and parameter estimates, it is tough to discern improvement from the semifinals to finals. In the next exercise, you will test the hypothesis that there is no difference in performance between the semifinals and finals. A permutation test is fitting for this. We will use the mean value of
 *f*
 as the test statistic.




 Step of the permutation test:



* Take an array of semifinal times and an array of final times for each swimmer for each stroke/distance pair.
* Go through each array, and for each index, swap the entry in the respective final and semifinal array with a 50% probability.
* Use the resulting final and semifinal arrays to compute
 `f`
 and then the mean of
 `f`
 .


####
**Generating permutation samples**



 As you worked out in the last exercise, we need to generate a permutation sample by randomly swapping corresponding entries in the
 `semi_times`
 and
 `final_times`
 array. Write a function with signature
 `swap_random(a, b)`
 that returns arrays where random indices have the entries in
 `a`
 and
 `b`
 swapped.





```

def swap_random(a, b):
    """Randomly swap entries in two arrays."""
    # Indices to swap
    swap_inds = np.random.random(size=len(a)) < 0.5

    # Make copies of arrays a and b for output
    a_out = np.copy(a)
    b_out = np.copy(b)

    # Swap values
    a_out[swap_inds] = b[swap_inds]
    b_out[swap_inds] = a[swap_inds]

    return a_out, b_out

```



 Now you have this function in hand to do the permutation test.



####
**Hypothesis test: Do women swim the same way in semis and finals?**



 Test the hypothesis that performance in the finals and semifinals are identical using the mean of the fractional improvement as your test statistic. The test statistic under the null hypothesis is considered to be at least as extreme as what was observed if it is greater than or equal to
 `f_mean`
 , which is already in your namespace.




 The semifinal and final times are contained in the
 `numpy`
 arrays
 `semi_times`
 and
 `final_times`
 .





```python

# Set up array of permutation replicates
perm_reps = np.empty(1000)

for i in range(1000):
    # Generate a permutation sample
    semi_perm, final_perm = swap_random(semi_times, final_times)

    # Compute f from the permutation sample
    f = (semi_perm - final_perm) / semi_perm

    # Compute and store permutation replicate
    perm_reps[i] = np.mean(f)

# Compute and print p-value
print('p =', np.sum(perm_reps >= f_mean) / 1000)

```



 That was a little tricky… Nice work! The p-value is large, about 0.27, which suggests that the results of the 2015 World Championships are consistent with there being no difference in performance between the finals and semifinals.





---


## **2.3 How does the performance of swimmers decline over long events?**



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/2-1.png?w=1024)

####
**EDA: Plot all your data**



 To get a graphical overview of a data set, it is often useful to plot all of your data. In this exercise, plot all of the splits for all female swimmers in the 800 meter heats.





```python

# Plot the splits for each swimmer
for splitset in splits:
    _ = plt.plot(split_number, splitset, linewidth=1, color='lightgray')

# Compute the mean split times
mean_splits = np.mean(splits, axis=0)

# Plot the mean split times
_ = plt.plot(split_number, mean_splits, linewidth=3, markersize=12)

# Label axes and show plot
_ = plt.xlabel('split number')
_ = plt.ylabel('split time (s)')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/3-1.png?w=1024)


 You can see that there is wide variability in the splits among the swimmers, and what appears to be a slight trend toward slower split times.



####
**Linear regression of average split time**



 We will assume that the swimmers slow down in a linear fashion over the course of the 800 m event. The slowdown per split is then the slope of the mean split time versus split number plot. Perform a linear regression to estimate the slowdown per split and compute a pairs bootstrap 95% confidence interval on the slowdown. Also show a plot of the best fit line.





```python

# Perform regression
slowdown, split_3 = np.polyfit(split_number, mean_splits, deg=1)

# Compute pairs bootstrap
bs_reps, _ = dcst.draw_bs_pairs_linreg(split_number, mean_splits, size=10000)

# Compute confidence interval
conf_int = np.percentile(bs_reps, [2.5, 97.5])

# Plot the data with regressions line
_ = plt.plot(split_number, mean_splits, marker='.', linestyle='none')
_ = plt.plot(split_number, slowdown * split_number + split_3, '-')

# Label axes and show plot
_ = plt.xlabel('split number')
_ = plt.ylabel('split time (s)')
plt.show()

# Print the slowdown per split
print("""
mean slowdown: {0:.3f} sec./split
95% conf int of mean slowdown: [{1:.3f}, {2:.3f}] sec./split""".format(
    slowdown, *conf_int))

#    mean slowdown: 0.065 sec./split
#    95% conf int of mean slowdown: [0.051, 0.078] sec./split

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/4-1.png?w=1024)


 There is a small (about 6 hundreths of a second), but discernible, slowdown per split. We’ll do a hypothesis test next.



####
**Hypothesis test: are they slowing down?**



 Now we will test the null hypothesis that the swimmer’s split time is not at all correlated with the distance they are at in the swim. We will use the Pearson correlation coefficient (computed using
 `dcst.pearson_r()`
 ) as the test statistic.





```python

# Observed correlation
rho = dcst.pearson_r(split_number, mean_splits)

# Initialize permutation reps
perm_reps_rho = np.empty(10000)

# Make permutation reps
for i in range(10000):
    # Scramble the split number array
    scrambled_split_number = np.random.permutation(split_number)

    # Compute the Pearson correlation coefficient
    perm_reps_rho[i] = dcst.pearson_r(scrambled_split_number, mean_splits)

# Compute and print p-value
p_val = np.sum(perm_reps_rho >= rho) / len(perm_reps_rho)
print('p =', p_val)


```



 The tiny effect is very real! With 10,000 replicates, we never got a correlation as big as observed under the hypothesis that the swimmers do not change speed as the race progresses.


# **3. The “Current Controversy” of the 2013 World Championships**
-----------------------------------------------------------------


## **3.1 Introduction to the current controversy**



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/5-1.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/6-1.png?w=1024)



####
**A metric for improvement**



 In your first analysis, you will investigate how times of swimmers in 50 m events change as they move between low numbered lanes (1-3) to high numbered lanes (6-8) in the semifinals and finals. We showed in the previous chapter that there is little difference between semifinal and final performance, so you will neglect any differences due to it being the final versus the semifinal.




 You want to use as much data as you can, so use all four strokes for both the men’s and women’s competitions. As such, what would be a good metric for improvement from one round to the next for an individual swimmer, where
 *t
 a*
 is the swim time in a low numbered lane and
 *t
 b*
 is the swim time in a high numbered lane?




 The fractional improvement of swim time, (
 *t
 a*
 –
 *t
 b*
 ) /
 *t
 a*
 .




 This is a good metric; it is the fractional improvement, and therefore independent of the basal speed (which is itself dependent on stroke and gender).



####
**ECDF of improvement from low to high lanes**



 Now that you have a metric for improvement going from low- to high-numbered lanes, plot an ECDF of this metric.





```python

# Compute the fractional improvement of being in high lane: f
f = (swimtime_low_lanes - swimtime_high_lanes) / swimtime_low_lanes

# Make x and y values for ECDF: x, y
x, y = dcst.ecdf(f)

# Plot the ECDFs as dots
_ = plt.plot(x, y, marker='.', linestyle='none')

# Label the axes and show the plot
_ = plt.xlabel('f')
_ = plt.ylabel('ECDF')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/1-1.png?w=1024)


 This is starting to paint a picture of lane bias. The ECDF demonstrates that all but three of the 26 swimmers swam faster in the high numbered lanes.



####
**Estimation of mean improvement**



 You will now estimate how big this current effect is. Compute the mean fractional improvement for being in a high-numbered lane versus a low-numbered lane, along with a 95% confidence interval of the mean.





```python

# Compute the mean difference: f_mean
f_mean = np.mean(f)

# Draw 10,000 bootstrap replicates: bs_reps
bs_reps = dcst.draw_bs_reps(f, np.mean, size=10000)

# Compute 95% confidence interval: conf_int
conf_int = np.percentile(bs_reps, [2.5, 97.5])

# Print the result
print("""
mean frac. diff.: {0:.5f}
95% conf int of mean frac. diff.: [{1:.5f}, {2:.5f}]""".format(f_mean, *conf_int))

#    mean frac. diff.: 0.01051
#    95% conf int of mean frac. diff.: [0.00612, 0.01591]

```



 It sure looks like swimmers are faster in lanes 6-8.



####
**How should we test the hypothesis?**



 You are interested in the presence of lane bias toward higher lanes, presumably due to a slight current in the pool.
 **A natural null hypothesis to test, then, is that the mean fractional improvement going from low to high lane numbers is zero.**
 Which of the following is a good way to simulate this null hypothesis?




 As a reminder, the arrays
 `swimtime_low_lanes`
 and
 `swimtime_high_lanes`
 contain the swim times for lanes 1-3 and 6-8, respectively, and we define the fractional improvement as
 `f = (swimtime_low_lanes - swimtime_high_lanes) / swimtime_low_lanes`
 .




**Subtract the mean of
 `f`
 from
 `f`
 to generate
 `f_shift`
 . Then, take bootstrap replicate of the mean from this
 `f_shift`
 .**



####
**Hypothesis test: Does lane assignment affect performance?**



 Perform a bootstrap hypothesis test of the null hypothesis that the mean fractional improvement going from low-numbered lanes to high-numbered lanes is zero. Take the fractional improvement as your test statistic, and “at least as extreme as” to mean that the test statistic under the null hypothesis is greater than or equal to what was observed.





```python

# Shift f: f_shift
f_shift = f - f_mean

# Draw 100,000 bootstrap replicates of the mean: bs_reps
bs_reps = dcst.draw_bs_reps(f_shift, np.mean, size=100000)

# Compute and report the p-value
p_val = np.sum(bs_reps >= f_mean) / 100000
print('p =', p_val)

```



 A p-value of 0.0003 is quite small and suggests that the mean fractional improvment is greater than zero.



####
**Did the 2015 event have this problem?**



 You would like to know if this is a typical problem with pools in competitive swimming. To address this question, perform a similar analysis for the results of the 2015 FINA World Championships. That is, compute the mean fractional improvement for going from lanes 1-3 to lanes 6-8 for the 2015 competition, along with a 95% confidence interval on the mean. Also test the hypothesis that the mean fractional improvement is zero.




 The arrays
 `swimtime_low_lanes_15`
 and
 `swimtime_high_lanes_15`
 have the pertinent data.





```python

# Compute f and its mean
f = (swimtime_low_lanes_15 - swimtime_high_lanes_15) / swimtime_low_lanes_15
f_mean = np.mean(f)

# Draw 10,000 bootstrap replicates
bs_reps = dcst.draw_bs_reps(f, np.mean, size=10000)

# Compute 95% confidence interval
conf_int = np.percentile(bs_reps, [2.5, 97.5])

# Shift f
f_shift = f - f_mean

# Draw 100,000 bootstrap replicates of the mean
bs_reps = dcst.draw_bs_reps(f_shift, np.mean, size=100000)

# Compute the p-value
p_val = np.sum(bs_reps >= f_mean) / 100000

# Print the results
print("""
mean frac. diff.: {0:.5f}
95% conf int of mean frac. diff.: [{1:.5f}, {2:.5f}]
p-value: {3:.5f}""".format(f_mean, *conf_int, p_val))

#    mean frac. diff.: 0.00079
#    95% conf int of mean frac. diff.: [-0.00198, 0.00341]
#    p-value: 0.28179

```



 Both the confidence interval an the p-value suggest that there was no lane bias in 2015.





---


## **3.2 The zigzag effect**



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/2-2.png?w=976)

####
**Which splits should we consider?**



 As you proceed to quantitatively analyze the zigzag effect in the 1500 m, which splits should you include in our analysis? For reference, the plot of the zigzag effect from the video is shown to the right.




**You should include all splits except the first two and the last two.**
 You should neglect the last two because swimmers stop pacing themselves and “kick” for the final stretch. The first two are different because they involve jumping off the starting blocks and more underwater swimming than others.




 You want to use splits where the swimmers are swimming as consistently as they can.



####
**EDA: mean differences between odd and even splits**



 To investigate the differences between odd and even splits, you first need to define a difference metric. In previous exercises, you investigated the
 *improvement*
 of moving from a low-numbered lane to a high-numbered lane, defining
 *f*
 = (
 *t
 a*
 –
 *t
 b*
 ) /
 *t
 a*
 . There, the
 *t
 a*
 in the denominator served as our reference time for improvement. Here, you are considering both improvement and decline in performance depending on the direction of swimming, so you want the reference to be an average. So, we will define the
 **fractional difference**
 as
 *f*
 = 2(
 *t
 a*
 –
 *t
 b*
 ) / (
 *t
 a*
 +
 *t
 b*
 ).




 Your task here is to plot the mean fractional difference between odd and even splits versus lane number. I have already calculated the mean fractional differences for the 2013 and 2015 Worlds for you, and they are stored in
 `f_13`
 and
 `f_15`
 . The corresponding lane numbers are in the array
 `lanes`
 .





```python

# Plot the the fractional difference for 2013 and 2015
plt.plot(lanes, f_13, marker='.', markersize=12, linestyle='none')
plt.plot(lanes, f_15, marker='.', markersize=12, linestyle='none')

# Add a legend
_ = plt.legend((2013, 2015))

# Label axes and show plot
plt.xlabel('lane')
plt.ylabel('frac. diff. (odd - even)')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/3-2.png?w=1024)


 EDA has exposed a strong slope in 2013 compared to 2015!



####
**How does the current effect depend on lane position?**



 To quantify the effect of lane number on performance, perform a linear regression on the
 `f_13`
 versus
 `lanes`
 data. Do a pairs bootstrap calculation to get a 95% confidence interval. Finally, make a plot of the regression. The arrays
 `lanes`
 and
 `f_13`
 are in your namespace.




 Note that we could compute error bars on the mean fractional differences and use them in the regression, but that is beyond the scope of this course.





```python

# Compute the slope and intercept of the frac diff/lane curve
slope, intercept  = np.polyfit(lanes, f_13, 1)

# Compute bootstrap replicates
bs_reps_slope, bs_reps_int = dcst.draw_bs_pairs_linreg(lanes, f_13, size=10000)

# Compute 95% confidence interval of slope
conf_int = np.percentile(bs_reps_slope, [2.5, 97.5])

# Print slope and confidence interval
print("""
slope: {0:.5f} per lane
95% conf int: [{1:.5f}, {2:.5f}] per lane""".format(slope, *conf_int))

# x-values for plotting regression lines
x = np.array([1, 8])

# Plot 100 bootstrap replicate lines
for i in range(100):
    _ = plt.plot(x, bs_reps_slope[i] * x + bs_reps_int[i],
                 color='red', alpha=0.2, linewidth=0.5)

# Update the plot
plt.draw()
plt.show()

#    slope: 0.00447 per lane
#    95% conf int: [0.00394, 0.00501] per lane

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/4-2.png?w=1024)


 The slope is a fractional difference of about 0.4% per lane. This is quite a substantial difference at this elite level of swimming where races can be decided by tiny differences.



####
**Hypothesis test: can this be by chance?**



 The EDA and linear regression analysis is pretty conclusive. Nonetheless, you will top off the analysis of the zigzag effect by testing the hypothesis that lane assignment has nothing to do with the mean fractional difference between even and odd lanes using a permutation test. You will use the Pearson correlation coefficient, which you can compute with
 `dcst.pearson_r()`
 as the test statistic. The variables
 `lanes`
 and
 `f_13`
 are already in your namespace.





```python

# Compute observed correlation: rho
rho = dcst.pearson_r(lanes, f_13)

# Initialize permutation reps: perm_reps_rho
perm_reps_rho = np.empty(10000)

# Make permutation reps
for i in range(10000):
    # Scramble the lanes array: scrambled_lanes
    scrambled_lanes = np.random.permutation(lanes)

    # Compute the Pearson correlation coefficient
    perm_reps_rho[i] = dcst.pearson_r(scrambled_lanes, f_13)

# Compute and print p-value
p_val = np.sum(perm_reps_rho >= rho) / 10000
print('p =', p_val)

#    p = 0.0

```



 The p-value is very small, as you would expect from the confidence interval of the last exercise.





---


## **3.3 Recap of swimming analysis**



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/5-2.png?w=931)
# **4. Statistical seismology and the Parkfield region**
-------------------------------------------------------


## **4.1 Introduction to statistical seismology and the Parkfield experiment**



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/1-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/2-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/3-3.png?w=953)
![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/4-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/5-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/6-2.png?w=1024)




[Gutenberg–Richter law](https://en.wikipedia.org/wiki/Gutenberg%E2%80%93Richter_law)



### **4.1.1 Parkfield earthquake magnitudes**



 As usual, you will start with EDA and plot the ECDF of the magnitudes of earthquakes detected in the Parkfield region from 1950 to 2016. The magnitudes of all earthquakes in the region from the ANSS ComCat are stored in the Numpy array
 `mags`
 .




 When you do it this time, though, take a shortcut in generating the ECDF. You may recall that putting an asterisk before an argument in a function splits what follows into separate arguments. Since
 `dcst.ecdf()`
 returns two values, we can pass them as the
 `x`
 ,
 `y`
 positional arguments to
 `plt.plot()`
 as
 `plt.plot(*dcst.ecdf(data_you_want_to_plot))`
 .




 You will use this shortcut in this exercise and going forward.





```python

# Make the plot
plt.plot(*dcst.ecdf(mags), marker='.', linestyle='none')

# Label axes and show plot
plt.xlabel('magnitude')
plt.ylabel('ECDF')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/7-1.png?w=1024)


 Note the distinctive roll-off at magnitudes below 1.0.



### **4.1.2 Computing the b-value**



 The
 *b*
 -value is a common metric for the seismicity of a region. You can imagine you would like to calculate it often when working with earthquake data. For tasks like this that you will do often, it is best to write a function! So, write a function with signature
 `b_value(mags, mt, perc=[2.5, 97.5], n_reps=None)`
 that returns the
 *b*
 -value and (optionally, if
 `n_reps`
 is not
 `None`
 ) its confidence interval for a set of magnitudes,
 `mags`
 . The completeness threshold is given by
 `mt`
 . The
 `perc`
 keyword argument gives the percentiles for the lower and upper bounds of the confidence interval, and
 `n_reps`
 is the number of bootstrap replicates to use in computing the confidence interval.





```

def b_value(mags, mt, perc=[2.5, 97.5], n_reps=None):
    """Compute the b-value and optionally its confidence interval."""
    # Extract magnitudes above completeness threshold: m
    m = mags[mags >= mt]

    # Compute b-value: b
    b = (np.mean(m) - mt) * np.log(10)

    # Draw bootstrap replicates
    if n_reps is None:
        return b
    else:
        m_bs_reps = dcst.draw_bs_reps(m, np.mean, size=n_reps)

        # Compute b-value from replicates: b_bs_reps
        b_bs_reps = (m_bs_reps - mt) * np.log(10)

        # Compute confidence interval: conf_int
        conf_int = np.percentile(b_bs_reps, perc)

        return b, conf_int

```



 You now have a very handy function for computing b-values. You’ll use it in this and the next chapter.



### **4.1.3 The b-value for Parkfield**



 The ECDF is effective at exposing roll-off, as you could see below magnitude 1. Because there are plenty of earthquakes above magnitude 3, you can use
 *m
 t
 = 3*
 as your completeness threshold. With this completeness threshold, compute the
 *b*
 -value for the Parkfield region from 1950 to 2016, along with the 95% confidence interval. Print the results to the screen. The variable
 `mags`
 with all the magnitudes is in your namespace.




 Overlay the theoretical Exponential CDF to verify that the Parkfield region follows the Gutenberg-Richter Law.





```python

# Compute b-value and confidence interval
b, conf_int = b_value(mags, mt, perc=[2.5, 97.5], n_reps=10000)

# Generate samples to for theoretical ECDF
m_theor = np.random.exponential(b/np.log(10), size=100000) + mt

# Plot the theoretical CDF
_ = plt.plot(*dcst.ecdf(m_theor))

# Plot the ECDF (slicing mags >= mt)
_ = plt.plot(*dcst.ecdf(mags[mags >= mt]), marker='.', linestyle='none')

# Pretty up and show the plot
_ = plt.xlabel('magnitude')
_ = plt.ylabel('ECDF')
_ = plt.xlim(2.8, 6.2)
plt.show()

# Report the results
print("""
b-value: {0:.2f}
95% conf int: [{1:.2f}, {2:.2f}]""".format(b, *conf_int))

#    b-value: 1.08
#    95% conf int: [0.94, 1.24]

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/8-1.png?w=1024)


 Parkfield seems to follow the Gutenberg-Richter law very well. The b-value of about 1 is typical for regions along fault zones.





---


## **4.2 Timing of major earthquakes and the Parkfield sequence**



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/1-4.png?w=961)
![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/2-4.png?w=942)



### **4.2.1 Interearthquake time estimates for Parkfield**



 In this exercise, you will first compute the best estimates for the parameters for the Exponential and Gaussian models for interearthquake times. You will then plot the theoretical CDFs for the respective models along with the formal ECDF of the actual Parkfield interearthquake times.





```python

# Compute the mean time gap: mean_time_gap
mean_time_gap = np.mean(time_gap)

# Standard deviation of the time gap: std_time_gap
std_time_gap = np.std(time_gap)

# Generate theoretical Exponential distribution of timings: time_gap_exp
time_gap_exp = np.random.exponential(scale=mean_time_gap, size=10000)

# Generate theoretical Normal distribution of timings: time_gap_norm
time_gap_norm = np.random.normal(loc=mean_time_gap, scale=std_time_gap, size=10000)

# Plot theoretical CDFs
_ = plt.plot(*dcst.ecdf(time_gap_exp))
_ = plt.plot(*dcst.ecdf(time_gap_norm))

# Plot Parkfield ECDF
_ = plt.plot(*dcst.ecdf(time_gap, formal=True, min_x=-10, max_x=50))

# Add legend
_ = plt.legend(('Exp.', 'Norm.'), loc='upper left')

# Label axes, set limits and show plot
_ = plt.xlabel('time gap (years)')
_ = plt.ylabel('ECDF')
_ = plt.xlim(-10, 50)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/3-4.png?w=873)


 By eye, the Gaussian model seems to describe the observed data best. We will investigate the consequences of this in the next exercise, and see if we can reject the Exponential model in coming exercises.



### **4.2.2 When will the next big Parkfield quake be?**



 The last big earthquake in the Parkfield region was on the evening of September 27, 2004 local time. Your task is to get an estimate as to when the next Parkfield quake will be, assuming the Exponential model and also the Gaussian model. In both cases, the best estimate is given by the mean time gap, which you computed in the last exercise to be 24.62 years, meaning that the next earthquake would be in 2029. Compute 95% confidence intervals on when the next earthquake will be assuming an Exponential distribution parametrized by
 `mean_time_gap`
 you computed in the last exercise. Do the same assuming a Normal distribution parametrized by
 `mean_time_gap`
 and
 `std_time_gap`
 .





```python

# Draw samples from the Exponential distribution: exp_samples
exp_samples = np.random.exponential(scale=mean_time_gap, size=100000)

# Draw samples from the Normal distribution: norm_samples
norm_samples = np.random.normal(loc=mean_time_gap, scale=std_time_gap, size=100000)

# No earthquake as of today, so only keep samples that are long enough
exp_samples = exp_samples[exp_samples > today - last_quake]
norm_samples = norm_samples[norm_samples > today - last_quake]

# Compute the confidence intervals with medians
conf_int_exp = np.percentile(exp_samples, [2.5, 50, 97.5]) + last_quake
conf_int_norm = np.percentile(norm_samples, [2.5, 50, 97.5]) + last_quake

# Print the results
print('Exponential:', conf_int_exp)
print('     Normal:', conf_int_norm)

#    Exponential: [2020.43020248 2036.77538201 2110.14809932]
#           Normal: [2020.64362947 2030.72447973 2046.46834012]

```



 Great work! The models given decidedly different predictions. The Gaussian model says the next earthquake is almost sure to be in the next few decades, but the Exponential model says we may very well have to wait longer.





---


## **4.3 How are the Parkfield interearthquake times distributed?**



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/4-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/5-4.png?w=1024)



### **4.3.1 Computing the value of a formal ECDF**



 To be able to do the Kolmogorov-Smirnov test, we need to compute the value of a formal ECDF at arbitrary points. In other words, we need a function,
 `ecdf_formal(x, data)`
 that returns the value of the formal ECDF derived from the data set
 `data`
 for each value in the array
 `x`
 . Two of the functions accomplish this. One will not. Of the two that do the calculation correctly, one is faster. Label each.




 As a reminder, the ECDF is formally defined as ECDF(
 *x*
 ) = (number of samples ≤
 *x*
 ) / (total number of samples). You also might want to check out the doc string of
 `np.searchsorted()`
 .





```python

# a)
def ecdf_formal(x, data):
    return np.searchsorted(np.sort(data), x) / len(data)

# b)
def ecdf_formal(x, data):
    return np.searchsorted(np.sort(data), x, side='right') / len(data)

# c)
def ecdf_formal(x, data):
    output = np.empty(len(x))

    data = np.sort(data)

    for i, x_val in x:
        j = 0
        while j < len(data) and x_val >= data[j]:
            j += 1

        output[i] = j

    return output / len(data)

```



 (a) Incorrect; (b) Correct, fast; (c) Correct, slow.



### **4.3.2 Computing the K-S statistic**



 Write a function to compute the Kolmogorov-Smirnov statistic from two datasets,
 `data1`
 and
 `data2`
 , in which
 `data2`
 consists of samples from the theoretical distribution you are comparing your data to. Note that this means we are using hacker stats to compute the K-S statistic for a dataset and a theoretical distribution,
 *not*
 the K-S statistic for two empirical datasets. Conveniently, the function you just selected for computing values of the formal ECDF is given as
 `dcst.ecdf_formal()`
 .





```

def ks_stat(data1, data2):
    # Compute ECDF from data: x, y
    x, y = dcst.ecdf(data1)

    # Compute corresponding values of the target CDF
    cdf = dcst.ecdf_formal(x, data2)

    # Compute distances between concave corners and CDF
    D_top = y - cdf

    # Compute distance between convex corners and CDF
    D_bottom = cdf - y + 1/len(data1)

    return np.max((D_top, D_bottom))

```


### **4.3.3 Drawing K-S replicates**



 Now, you need a function to draw Kolmogorov-Smirnov replicates out of a target distribution,
 `f`
 . Construct a function with signature
 `draw_ks_reps(n, f, args=(), size=10000, n_reps=10000)`
 to do so. Here,
 `n`
 is the number of data points, and
 `f`
 is the function you will use to generate samples from the target CDF. For example, to test against an Exponential distribution, you would pass
 `np.random.exponential`
 as
 `f`
 . This function usually takes arguments, which must be passed as a tuple. So, if you wanted to take samples from an Exponential distribution with mean
 `x_mean`
 , you would use the
 `args=(x_mean,)`
 keyword. The keyword arguments
 `size`
 and
 `n_reps`
 respectively represent the number of samples to take from the target distribution and the number of replicates to draw.





```

def draw_ks_reps(n, f, args=(), size=10000, n_reps=10000):
    # Generate samples from target distribution
    x_f = f(*args, size=size)

    # Initialize K-S replicates
    reps = np.empty(n_reps)

    # Draw replicates
    for i in range(n_reps):
        # Draw samples for comparison
        x_samp = f(*args, size=n)

        # Compute K-S statistic
        reps[i] = dcst.ks_stat(x_samp, x_f)

    return reps

```


### **4.3.4 The K-S test for Exponentiality**



 Test the null hypothesis that the interearthquake times of the Parkfield sequence are Exponentially distributed. That is, earthquakes happen at random with no memory of when the last one was.
 *Note*
 : This calculation is computationally intensive (you will draw more than 10
 8
 random numbers), so it will take about 10 seconds to complete.





```python

# Draw target distribution: x_f
x_f = np.random.exponential(scale=mean_time_gap, size=10000)

# Compute K-S stat: d
d = dcst.ks_stat(x_f, time_gap)

# Draw K-S replicates: reps
reps = dcst.draw_ks_reps(len(time_gap), np.random.exponential,
                         args=(mean_time_gap,), size=10000, n_reps=10000)

# Compute and print p-value
p_val = np.sum(reps >= d) / 10000
print('p =', p_val)

# p = 0.2199

```



 That’s a p-value above 0.2. This means that the Parkfield sequence is not outside the realm of possibility if earthquakes there are a Poisson process. This does
 *not*
 mean that they are generated by a Poisson process, but that the observed sequence is not incongruous with that model. The upshot is that it is really hard to say when the next Parkfield quake will be.


# **5. Earthquakes and oil mining in Oklahoma**
----------------------------------------------


## **5.1 Variations in earthquake frequency and seismicity**


### **5.1.1 EDA: Plotting earthquakes over time**



 Make a plot where the
 *y*
 -axis is the magnitude and the
 *x*
 -axis is the time of all earthquakes in Oklahoma between 1980 and the first half of 2017. Each dot in the plot represents a single earthquake. The time of the earthquakes, as decimal years, is stored in the Numpy array
 `time`
 , and the magnitudes in the Numpy array
 `mags`
 .





```python

# Plot time vs. magnitude
plt.plot(time, mags, marker='.', linestyle='none', alpha=0.1)

# Label axes and show the plot
plt.xlabel('time (year)')
plt.ylabel('magnitude')

plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/6-3.png?w=1024)

### **5.1.2 Estimates of the mean interearthquake times**



 The graphical EDA in the last exercise shows an obvious change in earthquake frequency around 2010. To compare, compute the mean time between earthquakes of magnitude 3 and larger from 1980 through 2009 and also from 2010 through mid-2017. Also include 95% confidence intervals of the mean. The variables
 `dt_pre`
 and
 `dt_post`
 respectively contain the time gap between all earthquakes of magnitude at least 3 from pre-2010 and post-2010 in units of days.





```python

# Compute mean interearthquake time
mean_dt_pre = np.mean(dt_pre)
mean_dt_post = np.mean(dt_post)

# Draw 10,000 bootstrap replicates of the mean
bs_reps_pre = dcst.draw_bs_reps(dt_pre, np.mean, size=10000)
bs_reps_post = dcst.draw_bs_reps(dt_post, np.mean, size=10000)

# Compute the confidence interval
conf_int_pre = np.percentile(bs_reps_pre, [2.5, 97.5])
conf_int_post = np.percentile(bs_reps_post, [2.5, 97.5])

# Print the results
print("""1980 through 2009
mean time gap: {0:.2f} days
95% conf int: [{1:.2f}, {2:.2f}] days""".format(mean_dt_pre, *conf_int_pre))

print("""
2010 through mid-2017
mean time gap: {0:.2f} days
95% conf int: [{1:.2f}, {2:.2f}] days""".format(mean_dt_post, *conf_int_post))

```




```

    1980 through 2009
    mean time gap: 204.61 days
    95% conf int: [140.30, 276.13] days

    2010 through mid-2017
    mean time gap: 1.12 days
    95% conf int: [0.97, 1.29] days

```



 There is almost a 200-fold increase in earthquake frequency after 2010.



### **5.1.3 Hypothesis test: did earthquake frequency change?**



 Obviously, there was a massive increase in earthquake frequency once wastewater injection began. Nonetheless, you will still do a hypothesis test for practice. You will not test the hypothesis that the interearthquake times have the same distribution before and after 2010, since wastewater injection may affect the distribution. Instead, you will assume that they have the same mean. So, compute the p-value associated with the hypothesis that the pre- and post-2010 interearthquake times have the same mean, using the mean of pre-2010 time gaps minus the mean of post-2010 time gaps as your test statistic.





```python

# Compute the observed test statistic
mean_dt_diff = mean_dt_pre - mean_dt_post

# Shift the post-2010 data to have the same mean as the pre-2010 data
dt_post_shift = dt_post - mean_dt_post + mean_dt_pre

# Compute 10,000 bootstrap replicates from arrays
bs_reps_pre = dcst.draw_bs_reps(dt_pre, np.mean, size=10000)
bs_reps_post = dcst.draw_bs_reps(dt_post_shift, np.mean, size=10000)

# Get replicates of difference of means
bs_reps = bs_reps_pre - bs_reps_post

# Compute and print the p-value
p_val = np.sum(bs_reps >= mean_dt_diff) / 10000
print('p =', p_val)

# p = 0.0

```



 In 10,000 samples, not one had a test statistic greater than was was observed. The p-value is, predictably based on what we have done so far, is tiny!



### **5.1.4 How to display your analysis**



 In the last three exercises, you generated a plot, computed means/confidence intervals, and did a hypothesis test. If you were to present your results to others, which of the following is the most effective order of emphasis, from greatest-to-least, you should put on the respective results?




 plot, mean/confidence interval, hypothesis test




 The plot graphically shows all data, and the scale of the effect is evident. The mean and confidence interval quantify how big the effect is. The hypothesis test, by this point, is so obvious it is useless.





---


## **5.2 Earthquake magnitudes in Oklahoma**



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/7-2.png?w=1024)

### **5.2.1 EDA: Comparing magnitudes before and after 2010**



 Make an ECDF of earthquake magnitudes from 1980 through 2009. On the same plot, show an ECDF of magnitudes of earthquakes from 2010 through mid-2017. The time of the earthquakes, as decimal years, are stored in the Numpy array
 `time`
 and the magnitudes in the Numpy array
 `mags`
 .





```python

# Get magnitudes before and after 2010
mags_pre = mags[time < 2010]
mags_post = mags[time >= 2010]

# Generate ECDFs
plt.plot(*dcst.ecdf(mags_pre), marker='.', linestyle='none')
plt.plot(*dcst.ecdf(mags_post), marker='.', linestyle='none')


# Label axes and show plot
_ = plt.xlabel('magnitude')
_ = plt.ylabel('ECDF')
plt.legend(('1980 though 2009', '2010 through mid-2017'), loc='upper left')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/case-studies-in-statistical-thinking/8-2.png?w=1024)


 Both curves seem to follow the Gutenberg-Richter Law, but with different completeness thresholds, probably due to improvements in sensing capabilities in more recent years.



### **5.2.2 Quantification of the b-values**



 Based on the plot you generated in the previous exercise, you can safely use a completeness threshold of
 `mt = 3`
 . Using this threshold, compute
 *b*
 -values for the period between 1980 and 2009 and for 2010 through mid-2017. The function
 `b_value()`
 you wrote last chapter, which computes the
 *b*
 -value and confidence interval from a set of magnitudes and completeness threshold, is available in your namespace, as are the
 `numpy`
 arrays
 `mags_pre`
 and
 `mags_post`
 from the last exercise, and
 `mt`
 .





```python

# Compute b-value and confidence interval for pre-2010
b_pre, conf_int_pre = b_value(mags_pre, mt, perc=[2.5, 97.5], n_reps=10000)

# Compute b-value and confidence interval for post-2010
b_post, conf_int_post = b_value(mags_post, mt, perc=[2.5, 97.5], n_reps=10000)

# Report the results
print("""
1980 through 2009
b-value: {0:.2f}
95% conf int: [{1:.2f}, {2:.2f}]

2010 through mid-2017
b-value: {3:.2f}
95% conf int: [{4:.2f}, {5:.2f}]
""".format(b_pre, *conf_int_pre, b_post, *conf_int_post))


```




```

    1980 through 2009
    b-value: 0.74
    95% conf int: [0.54, 0.96]

    2010 through mid-2017
    b-value: 0.62
    95% conf int: [0.60, 0.65]

```



 The confidence interval for the
 *b*
 -value for recent earthquakes is tighter than for earlier ones because there are many more recent ones. Still, the confidence intervals overlap, and we can perform a hypothesis test to see if we might get these results if the
 *b*
 -values are actually the same.



### **5.2.3 How should we do a hypothesis test on differences of the b-value?**



 We wish to test the hypothesis that the
 *b*
 -value in Oklahoma from 1980 through 2009 is the same as that from 2010 through mid-2017. Which of the first five statements is false? If none of them are false, select the last choice.



* You should only include earthquakes that have magnitudes above the completeness threshold. A value of 3 is reasonable.
* You should perform a permutation test because asserting a null hypothesis that the
 *b*
 -values are the same implicitly assumes that the magnitudes are identically distributed, specifically Exponentially, by the Gutenberg-Richter Law.
* A reasonable test statistic is the difference between the mean post-2010 magnitude and the mean pre-2010 magnitude.
* You do not need to worry about the fact that there were far fewer earthquakes before 2010 than there were after. That is to say, there are fewer earthquakes before 2010, but sufficiently many to do a permutation test.
* You do not need to worry about the fact that the two time intervals are of different length.
* **None of the above statements are false.**



 For instructional purposes, here are reasons why each is true: Option 1 is true because below the completeness threshold, we are not comparing earthquakes before and after 2010, but
 *observed*
 earthquakes before and after 2010. We do not have a complete data set below the completeness threshold.




 Option 2 is true because we really are assuming the Gutenberg-Richter law holds, in part because we are only considering earthquakes above the completeness threshold. We are using a model (the G-R law) to deal with missing data. So, since both sets of quakes follow the same statistical model, and that model has a single parameter, a permutation test is appropriate.




 Option 3 is true, even though you may be thinking that the mean values are not the
 *b*
 -values, and that you should be using the difference in
 *b*
 -value as your test statistic. However, the difference in mean magnitude is directly proportional to the difference in
 *b*
 -value, so the result of the hypothesis test will be identical if we use
 *b*
 -values of mean magnitudes.




 Option 4 is true because even though they have different numbers of earthquakes, you are only interested in summary statistics about their magnitude. There were 53 earthquakes between 1980 and 2009 with magnitude 3 or greater, so we have enough to compute a reliable mean.




 Option 5 is true because, provided the time interval is long enough, the
 *b*
 -value is independent of the time interval, just like the mean of Exponentially distributed values is independent of how many there are, provided there are not too few.



### **5.2.4 Hypothesis test: are the b-values different?**



 Perform the hypothesis test sketched out on the previous exercise. The variables
 `mags_pre`
 and
 `mags_post`
 are already loaded into your namespace, as is
 `mt = 3`
 .





```python

# Only magnitudes above completeness threshold
mags_pre = mags_pre[mags_pre >= mt]
mags_post = mags_post[mags_post >= mt]

# Observed difference in mean magnitudes: diff_obs
diff_obs = np.mean(mags_post) - np.mean(mags_pre)

# Generate permutation replicates: perm_reps
perm_reps = dcst.draw_perm_reps(mags_post, mags_pre, dcst.diff_of_means, size=10000)

# Compute and print p-value
p_val = np.sum(perm_reps < diff_obs) / 10000
print('p =', p_val)

#     p = 0.0993

```



 A p-value around 0.1 suggests that the observed magnitudes are commensurate with there being no change in
 *b*
 -value after wastewater injection began.



### **5.2.5 What can you conclude from this analysis?**



 All but one of the following constitute reasonable conclusions from our analysis of earthquakes. Which one does not?



* The seismicity, as measured by the
 *b*
 -value, is comparable before and after wastewater injection.
* Earthquakes are over 100 times more frequent in Oklahoma after widespread wastewater injection began.
* **Oklahoma has a smaller
 *b*
 -value than the Parkfield region, so the Parkfield region has more earthquakes.**
* Oklahoma has a
 *b*
 -value smaller than the Parkfield region, so a randomly selected earthquake above magnitude 3 in Oklahoma more likely than not has a smaller magnitude than one above magnitude 3 randomly selected from the Parkfield region.



 One cannot conclude information about frequency of earthquakes from the
 *b*
 -value alone. It is also true that from 2010-mid 2017, Oklahoma had twice as many earthquakes of magnitude 3 and higher than the entire state of California!





---



 Thank you for reading and hope you’ve learned a lot.



