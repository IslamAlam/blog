---
title: Manipulating Time Series Data in Python
date: 2021-12-07 11:22:11 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Manipulating Time Series Data in Python
==========================================







 This is a memo. This course does not have a track yet.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/manipulating-time-series-data-in-python)**
 .



#####
 PREREQUISITES


* [Introduction to Python](https://www.datacamp.com/courses/intro-to-python-for-data-science)
* [Intermediate Python for Data Science](https://www.datacamp.com/courses/intermediate-python-for-data-science)



# **1. Working with Time Series in Pandas**
------------------------------------------


## **1.1 How to use dates & times with pandas**




![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture-10.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture1-12.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture2-16.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture3-16.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture4-16.png?w=1024)


####
**Your first time series**



 You have learned in the video how to create a sequence of dates using
 `pd.date_range()`
 . You have also seen that each date in the resulting
 `pd.DatetimeIndex`
 is a
 `pd.Timestamp`
 with various attributes that you can access to obtain information about the date.




 Now, you’ll create a week of data, iterate over the result, and obtain the
 `dayofweek`
 and
 `weekday_name`
 for each date.





```python

# import libraries
import pandas as pd
import matplotlib.pyplot as plt

```




```python

# Create the range of dates here
seven_days = pd.date_range(start='2017-1-1', periods=7)

# Iterate over the dates and print the number and name of the weekday
for day in seven_days:
    print(day.dayofweek, day.weekday_name)

6 Sunday
0 Monday
1 Tuesday
2 Wednesday
3 Thursday
4 Friday
5 Saturday


seven_days
DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07'], dtype='datetime64[ns]', freq='D')

```




---


## **1.2 Indexing & resampling time series**


![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture5-19.png?w=960)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture6-16.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture7-13.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture8-11.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture9-10.png?w=1024)


####
**Create a time series of air quality data**



 You have seen in the video how to deal with dates that are not in the correct format, but instead are provided as
 `string`
 types, represented as
 `dtype`
`object`
 in
 `pandas`
 .




 We have prepared a data set with air quality data (ozone, pm25, and carbon monoxide for NYC, 2000-2017) for you to practice the use of
 `pd.to_datetime()`
 .





```python

# original data
data.head()
         date     ozone       pm25        co
0  1999-07-01  0.012024  20.000000  1.300686
1  1999-07-02  0.027699  23.900000  0.958194
2  1999-07-03  0.043969  36.700000  1.194444
3  1999-07-04  0.035162  39.000000  1.081548
4  1999-07-05  0.038359  28.171429  0.939583

print(data.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6317 entries, 0 to 6316
Data columns (total 4 columns):
date     6317 non-null object
ozone    6317 non-null float64
pm25     6317 non-null float64
co       6317 non-null float64
dtypes: float64(3), object(1)
memory usage: 197.5+ KB
None

```




```python

# Convert the date column to datetime64
data.date = pd.to_datetime(data.date)

# Set date column as index
data.set_index('date', inplace=True)

# Plot data
data.plot(subplots=True)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture10-11.png?w=1024)



```python

# After transform date to datetime type and set it to index
data.head()
               ozone       pm25        co
date
1999-07-01  0.012024  20.000000  1.300686
1999-07-02  0.027699  23.900000  0.958194
1999-07-03  0.043969  36.700000  1.194444
1999-07-04  0.035162  39.000000  1.081548
1999-07-05  0.038359  28.171429  0.939583

print(data.info())
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 6317 entries, 1999-07-01 to 2017-03-31
Data columns (total 3 columns):
ozone    6317 non-null float64
pm25     6317 non-null float64
co       6317 non-null float64
dtypes: float64(3)
memory usage: 197.4 KB
None

```


####
**Compare annual stock price trends**



 You’ll compare the performance for three years of Yahoo stock prices.





```

yahoo.head()
            price
date
2013-01-02  20.08
2013-01-03  19.78
2013-01-04  19.86
2013-01-07  19.40
2013-01-08  19.66

yahoo.info()
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 782 entries, 2013-01-02 to 2015-12-31
Data columns (total 1 columns):
price    756 non-null float64
dtypes: float64(1)
memory usage: 32.2 KB

```




```python

# Create dataframe prices here
prices = pd.DataFrame()

# Select data for each year and concatenate with prices here
for year in ['2013', '2014', '2015']:
    price_per_year = yahoo.loc[year, ['price']].reset_index(drop=True)
    price_per_year.rename(columns={'price': year}, inplace=True)
    prices = pd.concat([prices, price_per_year], axis=1)

# Plot prices
prices.plot()
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture11-8.png?w=1024)



```

prices.head()
    2013   2014   2015
0  20.08    NaN    NaN
1  19.78  39.59  50.17
2  19.86  40.12  49.13
3  19.40  39.93  49.21
4  19.66  40.92  48.59

In [4]: prices.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 261 entries, 0 to 260
Data columns (total 3 columns):
2013    252 non-null float64
2014    252 non-null float64
2015    252 non-null float64
dtypes: float64(3)
memory usage: 6.2 KB

```


####
**Set and change time series frequency**



 You have seen how to assign a frequency to a
 `DateTimeIndex`
 , and then change this frequency.




 Now, you’ll use data on the daily carbon monoxide concentration in NYC, LA and Chicago from 2005-17.




 You’ll set the frequency to calendar daily and then resample to monthly frequency, and visualize both series to see how the different frequencies affect the data.





```

co.info()
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 1898 entries, 2005-01-01 to 2010-12-31
Data columns (total 3 columns):
Chicago        1898 non-null float64
Los Angeles    1898 non-null float64
New York       1898 non-null float64
dtypes: float64(3)
memory usage: 139.3 KB


co.head()
             Chicago  Los Angeles  New York
date
2005-01-01  0.317763     0.777657  0.639830
2005-01-03  0.520833     0.349547  0.969572
2005-01-04  0.477083     0.626630  0.905208
2005-01-05  0.348822     0.613814  0.769176
2005-01-06  0.572917     0.792596  0.815761

```




```python

# Set the frequency to calendar daily
co = co.asfreq('D')

# Plot the data
co.plot(subplots=True)
plt.show()


# Set frequency to monthly
co = co.asfreq('M')

# Plot the data
co.plot(subplots=True)
plt.show()


```


![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture12-8.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture13-7.png?w=1024)




---


## **1.3 Lags, changes, and returns for stock price series**


![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture14-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture15-6.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture16-6.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture17-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture18-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture19-3.png?w=1024)


####
**Shifting stock prices across time**



 The first method to manipulate time series is
 `.shift()`
 , which allows you shift all values in a
 `Series`
 or
 `DataFrame`
 by a number of periods to a different time along the
 `DateTimeIndex`
 .




 Let’s use this to visually compare a stock price series for Google shifted 90 business days into both past and future.





```python

# original data
google.head()
             Close
Date
2014-01-02  556.00
2014-01-03  551.95
2014-01-06  558.10
2014-01-07  568.86
2014-01-08  570.04

```




```python

# Import data here
google = pd.read_csv('google.csv', parse_dates=['Date'], index_col='Date')

# Set data frequency to business daily
google = google.asfreq('B')

# Create 'lagged' and 'shifted'
google['lagged'] = google.Close.shift(-90)
google['shifted'] = google.Close.shift(90)

# Plot the google price series
google.plot()
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture1-13.png?w=1024)



```python

# shifted data
google.head()
             Close  lagged  shifted
Date
2014-01-02  556.00  511.00      NaN
2014-01-03  551.95  518.73      NaN
2014-01-06  558.10  529.92      NaN
2014-01-07  568.86  533.09      NaN
2014-01-08  570.04  526.65      NaN

```


####
**Calculating stock price changes**



 Now you’ll practice a similar calculation to calculate absolute changes from current and shifted prices, and compare the result to the function
 `.diff()`
 .





```python

# Created shifted_30 here
yahoo['shifted_30'] = yahoo.price.shift(periods=30)

# Subtract shifted_30 from price
yahoo['change_30'] = yahoo['price'] - yahoo['shifted_30']

# Get the 30-day price difference
yahoo['diff_30'] = yahoo.price.diff(periods=30)

# Inspect the last five rows of price
print(yahoo.tail())

# Show the value_counts of the difference between change_30 and diff_30
print(yahoo.change_30.sub(yahoo.diff_30).value_counts())


```




```

            price  shifted_30  change_30  diff_30
date
2015-12-25    NaN       32.19        NaN      NaN
2015-12-28  33.60       32.94       0.66     0.66
2015-12-29  34.04       32.86       1.18     1.18
2015-12-30  33.37       32.98       0.39     0.39
2015-12-31  33.26       32.62       0.64     0.64

0.0    703
dtype: int64

```



 There’s usually more than one way to get to the same result when working with data.



####
**Plotting multi-period returns**



 The last time series method you have learned about was
 `.pct_change()`
 . Let’s use this function to calculate returns for various calendar day periods, and plot the result to compare the different patterns.




 We’ll be using Google stock prices from 2014-2016.





```python

# Create daily_return
google['daily_return'] = google.Close.pct_change(1).mul(100)

# Create monthly_return
google['monthly_return'] = google.Close.pct_change(30).mul(100)

# Create annual_return
google['annual_return'] = google.Close.pct_change(360).mul(100)

# Plot the result
google.plot(subplots=True)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture2-17.png?w=1024)




# **2. Basic Time Series Metrics & Resampling**
----------------------------------------------


## **2.1 Compare time series growth rates**


![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture-11.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture1-14.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture2-18.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture3-17.png?w=1024)


####
**Compare the performance of several asset classes**



 You have seen how you can easily compare several time series by normalizing their starting points to 100, and plot the result.




 To broaden your perspective on financial markets, let’s compare four key assets: stocks, bonds, gold, and oil.





```

print(prices.info())
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 2469 entries, 2007-06-29 to 2017-06-26
Data columns (total 4 columns):
SP500    2469 non-null float64
Bonds    2469 non-null float64
Gold     2469 non-null float64
Oil      2469 non-null float64
dtypes: float64(4)
memory usage: 96.4 KB
None

prices.head()
              SP500   Bonds    Gold    Oil
DATE
2007-06-29  1503.35  402.15  648.50  70.47
2007-07-02  1519.43  402.96  650.50  71.11
2007-07-03  1524.87  402.02  657.25  71.41
2007-07-05  1525.40  400.15  655.90  71.81
2007-07-06  1530.44  399.31  647.75  72.80

```




```python

# Import data here
prices = pd.read_csv('asset_classes.csv', parse_dates=['DATE'], index_col='DATE')

# Inspect prices here
print(prices.info())

# Select first prices
first_prices = prices.iloc[0]

# Create normalized
normalized = prices.div(first_prices).mul(100)

# Plot normalized
normalized.plot()
plt.show()


```




```

first_prices
SP500    1503.35
Bonds     402.15
Gold      648.50
Oil        70.47
Name: 2007-06-29 00:00:00, dtype: float64

normalized.head()
                 SP500       Bonds        Gold         Oil
DATE
2007-06-29  100.000000  100.000000  100.000000  100.000000
2007-07-02  101.069611  100.201417  100.308404  100.908188
2007-07-03  101.431470   99.967674  101.349268  101.333901
2007-07-05  101.466724   99.502673  101.141095  101.901518
2007-07-06  101.801976   99.293796   99.884348  103.306372

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture4-17.png?w=1024)


 Normalizing series is a common step in time series analysis.



####
**Comparing stock prices with a benchmark**



 You also learned in the video how to compare the performance of various stocks against a benchmark. Now you’ll learn more about the stock market by comparing the three largest stocks on the NYSE to the Dow Jones Industrial Average, which contains the 30 largest US companies.




 The three largest companies on the NYSE are:






|
 Company
  |
 Stock Ticker
  |
| --- | --- |
|
 Johnson & Johnson
  |
 JNJ
  |
|
 Exxon Mobil
  |
 XOM
  |
|
 JP Morgan Chase
  |
 JPM
  |





```python

# Import stock prices and index here
stocks = pd.read_csv('nyse.csv', parse_dates=['date'], index_col='date')
dow_jones = pd.read_csv('dow_jones.csv', parse_dates=['date'], index_col='date')

# Concatenate data and inspect result here
data = pd.concat([stocks, dow_jones], axis=1)
print(data.info())

# Normalize and plot your data here
data.div(data.iloc[0]).mul(100).plot()
plt.show()

```




```

<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 1762 entries, 2010-01-04 to 2016-12-30
Data columns (total 4 columns):
JNJ     1762 non-null float64
JPM     1762 non-null float64
XOM     1762 non-null float64
DJIA    1762 non-null float64
dtypes: float64(4)
memory usage: 68.8 KB
None

data.head()
              JNJ    JPM    XOM      DJIA
date
2010-01-04  64.68  42.85  69.15  10583.96
2010-01-05  63.93  43.68  69.42  10572.02
2010-01-06  64.45  43.92  70.02  10573.68
2010-01-07  63.99  44.79  69.80  10606.86
2010-01-08  64.21  44.68  69.52  10618.19

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture5-20.png?w=1024)

####
**Plot performance difference vs benchmark index**



 You learned how to calculate and plot the performance difference of a stock in percentage points relative to a benchmark index.




 Let’s compare the performance of Microsoft (
 `MSFT`
 ) and Apple (
 `AAPL`
 ) to the S&P 500 over the last 10 years.





```python

# Create tickers
tickers = ['MSFT', 'AAPL']

# Import stock data here
stocks = pd.read_csv('msft_aapl.csv', parse_dates=['date'], index_col='date')

# Import index here
sp500 = pd.read_csv('sp500.csv', parse_dates=['date'], index_col='date')

# Concatenate stocks and index here
data = pd.concat([stocks, sp500], axis=1).dropna()

# Normalize data
normalized = data.div(data.iloc[0]).mul(100)

# Subtract the normalized index from the normalized stock prices, and plot the result
normalized[tickers].sub(normalized['SP500'], axis=0).plot()
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture6-17.png?w=1024)


 Now you can compare these stocks to the overall market so you can more easily spot trends and outliers.





---


## **2.2 Changing the time series frequency: resampling**


![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture1-17.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture2-20.png?w=1024)


####
**Convert monthly to weekly data**



 You have learned how to use
 `.reindex()`
 to conform an existing time series to a
 `DateTimeIndex`
 at a different frequency.




 Let’s practice this method by creating monthly data and then converting this data to weekly frequency while applying various fill logic options.





```python

# Set start and end dates
start = '2016-1-1'
end = '2016-2-29'

# Create monthly_dates here
monthly_dates = pd.date_range(start=start, end=end, freq='M')

# Create and print monthly here
monthly = pd.Series(data=[1,2], index=monthly_dates)
print(monthly)


2016-01-31    1
2016-02-29    2
Freq: M, dtype: int64

```




```python

# Create weekly_dates here
weekly_dates = pd.date_range(start=start, end=end, freq='W')

weekly_dates
# DatetimeIndex(['2016-01-03', '2016-01-10', '2016-01-17', '2016-01-24', '2016-01-31', '2016-02-07', '2016-02-14', '2016-02-21', '2016-02-28'], dtype='datetime64[ns]', freq='W-SUN')


# Print monthly, reindexed using weekly_dates
print(monthly.reindex(weekly_dates))
2016-01-03    NaN
2016-01-10    NaN
2016-01-17    NaN
2016-01-24    NaN
2016-01-31    1.0
2016-02-07    NaN
2016-02-14    NaN
2016-02-21    NaN
2016-02-28    NaN
Freq: W-SUN, dtype: float64


print(monthly.reindex(weekly_dates, method='bfill'))
2016-01-03    1
2016-01-10    1
2016-01-17    1
2016-01-24    1
2016-01-31    1
2016-02-07    2
2016-02-14    2
2016-02-21    2
2016-02-28    2
Freq: W-SUN, dtype: int64


print(monthly.reindex(weekly_dates, method='ffill'))
2016-01-03    NaN
2016-01-10    NaN
2016-01-17    NaN
2016-01-24    NaN
2016-01-31    1.0
2016-02-07    1.0
2016-02-14    1.0
2016-02-21    1.0
2016-02-28    1.0
Freq: W-SUN, dtype: float64

```


####
**Create weekly from monthly unemployment data**



 The civilian US unemployment rate is reported monthly. You may need more frequent data, but that’s no problem because you just learned how to upsample a time series.




 You’ll work with the time series data for the last 20 years, and apply a few options to fill in missing values before plotting the weekly series.





```python

# Import data here
data = pd.read_csv('unemployment.csv', parse_dates=['date'], index_col='date')

# Show first five rows of weekly series
print(data.asfreq('W').head())

# Show first five rows of weekly series with bfill option
print(data.asfreq('W', method='bfill').head())

# Create weekly series with ffill option and show first five rows
weekly_ffill = data.asfreq('W', method='ffill')
print(weekly_ffill.head())

# Plot weekly_fill starting 2015 here
weekly_ffill['2015':].plot()
plt.show()


```




```

data.head()
            UNRATE
date
2000-01-01     4.0
2000-02-01     4.1
2000-03-01     4.0
2000-04-01     3.8
2000-05-01     4.0

print(data.asfreq('W').head())
            UNRATE
date
2000-01-02     NaN
2000-01-09     NaN
2000-01-16     NaN
2000-01-23     NaN
2000-01-30     NaN

print(data.asfreq('W', method='bfill').head())
            UNRATE
date
2000-01-02     4.1
2000-01-09     4.1
2000-01-16     4.1
2000-01-23     4.1
2000-01-30     4.1

print(data.asfreq('W', method='ffill').head())
            UNRATE
date
2000-01-02     4.0
2000-01-09     4.0
2000-01-16     4.0
2000-01-23     4.0
2000-01-30     4.0

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture3-18.png?w=1024)



---


## **2.3 Upsampling & interpolation with `.resample()`**


![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture4-18.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture5-21.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture6-18.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture7-14.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture8-12.png?w=1024)


####
**Use interpolation to create weekly employment data**



 You have recently used the civilian US unemployment rate, and converted it from monthly to weekly frequency using simple
 `forward`
 or
 `backfill`
 methods.




 Compare your previous approach to the new
 `.interpolate()`
 method.





```

monthly.head()
            UNRATE
DATE
2010-01-01     9.8
2010-02-01     9.8
2010-03-01     9.9
2010-04-01     9.9
2010-05-01     9.6

```




```python

# Create weekly dates
weekly_dates = pd.date_range(monthly.index.min(), monthly.index.max(), freq='W')

# Reindex monthly to weekly data
weekly = monthly.reindex(weekly_dates)

# Create ffill and interpolated columns
weekly['ffill'] = weekly.UNRATE.ffill()
weekly['interpolated'] = weekly.UNRATE.interpolate()

# Plot weekly
weekly.plot()
plt.show()

```




```

weekly.tail()
            UNRATE  ffill  interpolated
2016-12-04     NaN    4.7      4.788571
2016-12-11     NaN    4.7      4.791429
2016-12-18     NaN    4.7      4.794286
2016-12-25     NaN    4.7      4.797143
2017-01-01     4.8    4.8      4.800000

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture9-11.png?w=1024)


 Interpolating is a useful way to create smoother time series when resampling.



####
**Interpolate debt/GDP and compare to unemployment**



 Since you have learned how to interpolate time series, you can now apply this new skill to the quarterly debt/GDP series, and compare the result to the monthly unemployment rate.





```

data.head()
            Debt/GDP  Unemployment
date
2010-01-01  87.00386           9.8
2010-02-01       NaN           9.8
2010-03-01       NaN           9.9
2010-04-01  88.67047           9.9
2010-05-01       NaN           9.6

```




```python

# Import & inspect data here
data = pd.read_csv('debt_unemployment.csv', parse_dates=['date'], index_col='date')

# Interpolate and inspect here
interpolated = data.interpolate()

# Plot interpolated data here
interpolated.plot(secondary_y='Unemployment')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture10-12.png?w=1024)



```

interpolated.head()
             Debt/GDP  Unemployment
date
2010-01-01  87.003860           9.8
2010-02-01  87.559397           9.8
2010-03-01  88.114933           9.9
2010-04-01  88.670470           9.9
2010-05-01  89.135103           9.6

```




---


## **2.4 Downsampling & aggregation**


![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture1-18.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture2-21.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture3-19.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture4-19.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture5-22.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture6-19.png?w=1024)


####
**Compare weekly, monthly and annual ozone trends for NYC & LA**



 You’ll apply this new skill to ozone data for both NYC and LA since 2000 to compare the air quality trend at weekly, monthly and annual frequencies and explore how different resampling periods impact the visualization.





```python

# Import and inspect data here
ozone = pd.read_csv('ozone.csv', parse_dates=['date'], index_col='date')
print(ozone.info())

# Calculate and plot the weekly average ozone trend
ozone.resample('W').mean().plot()
plt.title('Ozone Weekly')
plt.show()

# Calculate and plot the monthly average ozone trend
ozone.resample('M').mean().plot()
plt.title('Ozone Monthly')
plt.show()

# Calculate and plot the annual average ozone trend
ozone.resample('A').mean().plot()
plt.title('Ozone Annualy')
plt.show()


```


![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture7-15.png?w=654)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture8-13.png?w=644)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture9-12.png?w=652)


####
**Compare monthly average stock prices for Facebook and Google**




```python

# Import and inspect data here
stocks = pd.read_csv('stocks.csv', parse_dates=['date'], index_col='date')
print(stocks.info())

# Calculate and plot the monthly averages
monthly_average = stocks.resample('M').mean()
monthly_average.plot(subplots=True)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture10-13.png?w=657)


 How do the two stock price series compare?



####
**Compare quarterly GDP growth rate and stock returns**



 With your new skill to downsample and aggregate time series, you can compare higher-frequency stock price series to lower-frequency economic time series.




 As a first example, let’s compare the quarterly GDP growth rate to the quarterly rate of return on the (resampled) Dow Jones Industrial index of 30 large US stocks.




 GDP growth is reported at the beginning of each quarter for the previous quarter. To calculate matching stock returns, you’ll resample the stock index to quarter start frequency using the alias
 `'QS'`
 , and aggregating using the
 `.first()`
 observations.





```

gdp_growth.head()
            gdp_growth
date
2007-01-01         0.2
2007-04-01         3.1
2007-07-01         2.7
2007-10-01         1.4
2008-01-01        -2.7


djia.head()
                djia
date
2007-06-29  13408.62
2007-07-02  13535.43
2007-07-03  13577.30
2007-07-04       NaN
2007-07-05  13565.84

```




```python

# Import and inspect gdp_growth here
gdp_growth = pd.read_csv('gdp_growth.csv', parse_dates=['date'], index_col='date')
print(gdp_growth.info())

# Import and inspect djia here
djia = pd.read_csv('djia.csv', parse_dates=['date'], index_col='date')
print(djia.info())


# Calculate djia quarterly returns here
djia_quarterly = djia.resample('QS').first()
djia_quarterly_return = djia_quarterly.pct_change().mul(100)

# Concatenate, rename and plot djia_quarterly_return and gdp_growth here
data = pd.concat([gdp_growth, djia_quarterly_return], axis=1)
data.columns = ['gdp', 'djia']

data.plot()
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture11-9.png?w=1024)

####
**Visualize monthly mean, median and standard deviation of S&P500 returns**



 You have also learned how to calculate several aggregate statistics from upsampled data.




 Let’s use this to explore how the monthly mean, median and standard deviation of daily S&P500 returns have trended over the last 10 years.





```python

# Import data here
sp500 = pd.read_csv('sp500.csv', parse_dates=['date'], index_col='date')

# Calculate daily returns here
daily_returns = sp500.squeeze().pct_change()

# Resample and calculate statistics
stats = daily_returns.resample('M').agg(['mean', 'median', 'std'])

# Plot stats here
stats.plot()
plt.show()


```




```

sp500.head()
              SP500
date
2007-06-29  1503.35
2007-07-02  1519.43
2007-07-03  1524.87
2007-07-05  1525.40
2007-07-06  1530.44


daily_returns.head()
date
2007-06-29         NaN
2007-07-02    0.010696
2007-07-03    0.003580
2007-07-05    0.000348
2007-07-06    0.003304
Name: SP500, dtype: float64


stats.head()
                mean    median       std
date
2007-06-30       NaN       NaN       NaN
2007-07-31 -0.001490  0.000921  0.010908
2007-08-31  0.000668  0.001086  0.015261
2007-09-30  0.001900  0.000202  0.010000
2007-10-31  0.000676 -0.000265  0.008719

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture12-9.png?w=1024)




# **3. Window Functions: Rolling & Expanding Metrics**
-----------------------------------------------------


## **3.1 Rolling window functions with pandas**


![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture13-8.png?w=1002)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture14-6.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture15-7.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture16-7.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture17-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture18-6.png?w=1024)


####
**Rolling average air quality since 2010 for new york city**



 The last video was about rolling window functions. To practice this new tool, you’ll start with air quality trends for New York City since 2010. In particular, you’ll be using the daily Ozone concentration levels provided by the Environmental Protection Agency to calculate & plot the 90 and 360 day rolling average.





```python

# Import and inspect ozone data here
data = pd.read_csv('ozone.csv', parse_dates=['date'], index_col='date')
print(data.info())

# Calculate 90d and 360d rolling mean for the last price
data['90D'] = data.Ozone.rolling('90D').mean()
data['360D'] = data.Ozone.rolling('360D').mean()

# Plot data
data['2010':].plot()
plt.title('New York City')
plt.show()


```




```

data.head()
               Ozone       90D      360D
date
2000-01-01  0.004032  0.004032  0.004032
2000-01-02  0.009486  0.006759  0.006759
2000-01-03  0.005580  0.006366  0.006366
2000-01-04  0.008717  0.006954  0.006954
2000-01-05  0.013754  0.008314  0.008314


data.tail()
               Ozone       90D      360D
date
2017-03-27  0.005640  0.021992  0.026629
2017-03-28  0.013870  0.021999  0.026583
2017-03-29  0.034341  0.022235  0.026584
2017-03-30  0.026059  0.022334  0.026599
2017-03-31  0.035983  0.022467  0.026607

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture19-4.png?w=1024)


**Do the different rolling windows help you see any long term trends that are hard to spot in the original data.**



####
**Rolling 360-day median & std. deviation for nyc ozone data since 2000**



 The last video also showed you how to calculate several rolling statistics using the
 `.agg()`
 method, similar to
 `.groupby()`
 .




 Let’s take a closer look at the air quality history of NYC using the Ozone data you have seen before. The daily data are very volatile, so using a longer term rolling average can help reveal a longer term trend.




 You’ll be using a 360 day rolling window, and
 `.agg()`
 to calculate the rolling median and standard deviation for the daily average ozone values since 2000.





```python

# Import and inspect ozone data here
data = pd.read_csv('ozone.csv', parse_dates=['date'], index_col='date').dropna()

# Calculate the rolling mean and std here
rolling_stats = data.Ozone.rolling(360).agg(['mean', 'std'])

# Join rolling_stats with ozone data
stats = data.join(rolling_stats)

# Plot stats
stats.plot(subplots=True)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture20-5.png?w=1024)

####
**Rolling quantiles for daily air quality in nyc**



 You learned in the last video how to calculate rolling quantiles to describe changes in the dispersion of a time series over time in a way that is less sensitive to outliers than using the mean and standard deviation.




 Let’s calculate rolling quantiles – at 10%, 50% (median) and 90% – of the distribution of daily average ozone concentration in NYC using a 360-day rolling window.





```python

# before interpolate
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 6291 entries, 2000-01-01 to 2017-03-31
Data columns (total 1 columns):
Ozone    6167 non-null float64
dtypes: float64(1)
memory usage: 98.3 KB
None

# after interpolate
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 6300 entries, 2000-01-01 to 2017-03-31
Freq: D
Data columns (total 1 columns):
Ozone    6300 non-null float64
dtypes: float64(1)
memory usage: 98.4 KB
None

```




```python

# Resample, interpolate and inspect ozone data here
print(data.info())
data = data.resample('D').interpolate()
print(data.info())

# Create the rolling window
rolling = data.Ozone.rolling(360)

# Insert the rolling quantiles to the monthly returns
data['q10'] = rolling.quantile(0.1)
data['q50'] = rolling.quantile(0.5)
data['q90'] = rolling.quantile(0.9)

# Plot monthly returns
data.plot()
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture21-4.png?w=1024)


 The rolling quantiles help show the volatility of the series.





---


## **3.2 Expanding window functions with pandas**


![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture1-19.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture2-22.png?w=885)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture3-20.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture4-20.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture5-23.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture6-20.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture7-16.png?w=1015)


####
**Cumulative sum vs .diff()**



 You have learned about expanding windows that allow you to run cumulative calculations.




 The cumulative sum method has in fact the opposite effect of the
 `.diff()`
 method that you came across in chapter 1.




 To illustrate this, let’s use the Google stock price time series, create the differences between prices, and reconstruct the series using the cumulative sum.





```

data.head()
             Close
Date
2014-01-02  556.00
2014-01-03  551.95
2014-01-06  558.10
2014-01-07  568.86
2014-01-08  570.04

```




```python

# Calculate differences
differences = data.diff().dropna()

# Select start price
start_price = data.first('D')

# Calculate cumulative sum
cumulative_sum = start_price.append(differences).cumsum()

# Validate cumulative sum equals data
print(data.equals(cumulative_sum))
# True

```




```

differences.head()
            Close
Date
2014-01-03  -4.05
2014-01-06   6.15
2014-01-07  10.76
2014-01-08   1.18
2014-01-09  -5.49

start_price
            Close
Date
2014-01-02  556.0


```



 The
 `.cumsum()`
 method allows you to reconstruct the original data from the differences.



####
**Cumulative return on $1,000 invested in google vs apple I**



 To put your new ability to do cumulative return calculations to practical use, let’s compare how much $1,000 would be worth if invested in Google (
 `'GOOG'`
 ) or Apple (
 `'AAPL'`
 ) in 2010.





```

data.tail()
              AAPL    GOOG
Date
2017-05-24  153.34  954.96
2017-05-25  153.87  969.54
2017-05-26  153.61  971.47
2017-05-30  153.67  975.88
2017-05-31  152.76  964.86

returns.tail()
                AAPL      GOOG
Date
2017-05-24 -0.002991  0.006471
2017-05-25  0.003456  0.015268
2017-05-26 -0.001690  0.001991
2017-05-30  0.000391  0.004540
2017-05-31 -0.005922 -0.011292

```




```python

# Define your investment
investment = 1000

# Calculate the daily returns here
returns = data.pct_change()

# Calculate the cumulative returns here
returns_plus_one = returns + 1
cumulative_return = returns_plus_one.cumprod()

# Calculate and plot the investment return here
cumulative_return.mul(investment).plot()
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture8-14.png?w=648)

####
**Cumulative return on $1,000 invested in google vs apple II**



 Apple outperformed Google over the entire period, but this may have been different over various 1-year sub periods, so that switching between the two stocks might have yielded an even better result.




 To analyze this, calculate that cumulative return for rolling 1-year periods, and then plot the returns to see when each stock was superior.





```python

# Import numpy
import numpy as np

# Define a multi_period_return function
def multi_period_return(period_returns):
    return np.prod(period_returns + 1) - 1

# Calculate daily returns
daily_returns = data.pct_change()

# Calculate rolling_annual_returns
rolling_annual_returns = daily_returns.rolling('360D').apply(multi_period_return)

# Plot rolling_annual_returns
rolling_annual_returns.mul(100).plot()
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture9-13.png?w=1024)



---


## **3.3 Case study: S&P500 price simulation**


![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture10-14.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture11-10.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture12-10.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture13-9.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture14-7.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture15-8.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture16-8.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture17-5.png?w=1024)


####
**Random walk I**



 In the last video, you have seen how to generate a random walk of returns, and how to convert this random return series into a random stock price path.




 In this exercise, you’ll build your own random walk by drawing random numbers from the normal distribution with the help of
 `numpy`
 .





```

import pandas as pd
from numpy.random import normal, random
import matplotlib.pyplot as plt
# Set seed here
seed = 42

# Create random_walk
random_walk = normal(loc=0.001, scale=0.01, size=2500)

# Convert random_walk to pd.series
random_walk = pd.Series(random_walk)

# Create random_prices
random_prices = random_walk.add(1).cumprod()

# Plot random_prices here
random_prices.mul(1000).plot()
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture1-20.png?w=1024)

####
**Random walk II**



 In the last video, you have also seen how to create a random walk of returns by sampling from actual returns, and how to use this random sample to create a random stock price path.




 In this exercise, you’ll build a random walk using historical returns from Facebook’s stock price since IPO through the end of May 31, 2017. Then you’ll simulate an alternative random price path in the next exercise.





```

fb.head()
date
2012-05-17    38.00
2012-05-18    38.23
2012-05-21    34.03
2012-05-22    31.00
2012-05-23    32.00
Name: 1, dtype: float64

```




```

import pandas as pd
from numpy.random import choice, random
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed here
seed = 42

# Calculate daily_returns here
daily_returns = fb.pct_change().dropna()

# Get n_obs
n_obs = daily_returns.count()

# Create random_walk
random_walk = choice(daily_returns, size=n_obs)

# Convert random_walk to pd.series
random_walk = pd.Series(random_walk)

# Plot random_walk distribution
sns.distplot(random_walk)
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture2-23.png?w=655)

####
**Random walk III**



 In this exercise, you’ll complete your random walk simulation using Facebook stock returns over the last five years. You’ll start off with a random sample of returns like the one you’ve generated during the last exercise and use it to create a random stock price path.





```python

# Select fb start price here
start = fb.price.first('D')

# Add 1 to random walk and append to start
random_walk = random_walk.add(1)
random_price = start.append(random_walk)

# Calculate cumulative product here
random_price = random_price.cumprod()

# Insert into fb and plot
fb['random'] = random_price

fb.plot()
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture3-21.png?w=1024)



---


## **3.4 Relationships between time series: correlation**


####
**Annual return correlations among several stocks**



 You have seen in the video how to calculate correlations, and visualize the result.




 In this exercise, we have provided you with the historical stock prices for Apple (
 `AAPL`
 ), Amazon (
 `AMZN`
 ), IBM (
 `IBM`
 ), WalMart (
 `WMT`
 ), and Exxon Mobile (
 `XOM`
 ) for the last 4,000 trading days from July 2001 until the end of May 2017.




 You’ll calculate the year-end returns, the pairwise correlations among all stocks, and visualize the result as an annotated heatmap.





```python

# Inspect data here
print(data.info())

# Calculate year-end prices here
annual_prices = data.resample('A').last()

# Calculate annual returns here
annual_returns = annual_prices.pct_change()

# Calculate and print the correlation matrix here
correlations = annual_returns.corr()
print(correlations)

# Visualize the correlations as heatmap here
sns.heatmap(correlations, annot=True)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture6-21.png?w=1024)




# **4. Putting it all together: Building a value-weighted index**
----------------------------------------------------------------


## **4.1 Select index components & import data**


![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture7-17.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture8-15.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture9-14.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture10-15.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture11-11.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture12-11.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture13-10.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture14-8.png?w=1024)


####
**Explore and clean company listing information**



 To get started with the construction of a market-value based index, you’ll work with the combined listing info for the three largest US stock exchanges, the NYSE, the NASDAQ and the AMEX.




 In this and the next exercise, you will calculate market-cap weights for these stocks.




 We have already imported
 `pandas`
 as
 `pd`
 , and loaded the
 `listings`
 data set with listings information from the NYSE, NASDAQ, and AMEX. The column
 `'Market Capitalization'`
 is already measured in USD mn.





```

listings.head(3)
  Exchange Stock Symbol                           Company Name  Last Sale  Market Capitalization  IPO Year                 Sector               Industry
0     amex         XXII                22nd Century Group, Inc       1.33             120.628490       NaN  Consumer Non-Durables  Farming/Seeds/Milling
1     amex          FAX  Aberdeen Asia-Pacific Income Fund Inc       5.00            1266.332595    1986.0                    NaN                    NaN
2     amex          IAF     Aberdeen Australia Equity Fund Inc       6.15             139.865305       NaN                    NaN                    NaN


listings.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6674 entries, 0 to 6673
Data columns (total 8 columns):
Exchange                 6674 non-null object
Stock Symbol             6674 non-null object
Company Name             6674 non-null object
Last Sale                6590 non-null float64
Market Capitalization    6674 non-null float64
IPO Year                 2852 non-null float64
Sector                   5182 non-null object
Industry                 5182 non-null object
dtypes: float64(3), object(5)
memory usage: 417.2+ KB

```




```python

# Move 'stock symbol' into the index
listings.set_index('Stock Symbol', inplace=True)

# Drop rows with missing 'sector' data
listings.dropna(subset=['Sector'], inplace=True)

# Select companies with IPO Year before 2019
listings = listings.loc[listings['IPO Year'] < 2019]

# Inspect the new listings data
print(listings.info())

# Show the number of companies per sector
print(listings.groupby('Sector').size().sort_values(ascending=False))

```




```python

# after cleaning
<class 'pandas.core.frame.DataFrame'>
Index: 2349 entries, ACU to ZTO
Data columns (total 7 columns):
Exchange                 2349 non-null object
Company Name             2349 non-null object
Last Sale                2349 non-null float64
Market Capitalization    2349 non-null float64
IPO Year                 2349 non-null float64
Sector                   2349 non-null object
Industry                 2349 non-null object
dtypes: float64(3), object(4)
memory usage: 146.8+ KB
None


Sector
Health Care              445
Consumer Services        402
Technology               386
Finance                  351
Energy                   144
Capital Goods            143
Public Utilities         104
Basic Industries         104
Consumer Non-Durables     89
Miscellaneous             68
Transportation            58
Consumer Durables         55
dtype: int64

```



 Your data is squeaky clean now!



####
**Select and inspect index components**



 Now that you have imported and cleaned the
 `listings`
 data, you can proceed to select the
 `index`
 components as the largest company for each sector by market capitalization.




 You’ll also have the opportunity to take a closer look at the components, their last market value, and last price.





```python

# Select largest company for each sector
components = listings.groupby('Sector')['Market Capitalization'].nlargest(1)

# Print components, sorted by market cap
print(components.sort_values(ascending=False))

```




```

Sector                 Stock Symbol
Technology             AAPL           740,024.47
Consumer Services      AMZN           422,138.53
Miscellaneous          MA             123,330.09
Health Care            AMGN           118,927.21
Transportation         UPS             90,180.89
Finance                GS              88,840.59
Basic Industries       RIO             70,431.48
Public Utilities       TEF             54,609.81
Consumer Non-Durables  EL              31,122.51
Capital Goods          ILMN            25,409.38
Energy                 PAA             22,223.00
Consumer Durables      CPRT            13,620.92
Name: Market Capitalization, dtype: float64


```




```python

# Select stock symbols and print the result
tickers = components.index.get_level_values('Stock Symbol')
print(tickers)
# Index(['RIO', 'ILMN', 'CPRT', 'EL', 'AMZN', 'PAA', 'GS', 'AMGN', 'MA', 'TEF', 'AAPL', 'UPS'], dtype='object', name='Stock Symbol')

# Print company name, market cap, and last price for each component
info_cols = ['Company Name', 'Market Capitalization', 'Last Sale']
info_cols = listings.loc[tickers, info_cols].sort_values('Market Capitalization', ascending=False)
print(info_cols)

```




```

                                    Company Name  Market Capitalization  Last Sale
Stock Symbol
AAPL                                  Apple Inc.             740,024.47     141.05
AMZN                            Amazon.com, Inc.             422,138.53     884.67
MA                       Mastercard Incorporated             123,330.09     111.22
AMGN                                  Amgen Inc.             118,927.21     161.61
UPS                  United Parcel Service, Inc.              90,180.89     103.74
GS               Goldman Sachs Group, Inc. (The)              88,840.59     223.32
RIO                                Rio Tinto Plc              70,431.48      38.94
TEF                                Telefonica SA              54,609.81      10.84
EL            Estee Lauder Companies, Inc. (The)              31,122.51      84.94
ILMN                              Illumina, Inc.              25,409.38     173.68
PAA           Plains All American Pipeline, L.P.              22,223.00      30.72
CPRT                                Copart, Inc.              13,620.92      29.65

```


####
**Import index component price information**



 Now you’ll use the stock symbols for the companies you selected in the last exercise to calculate returns for each company.





```python

# Print tickers
print(tickers)
['RIO', 'ILMN', 'CPRT', 'EL', 'AMZN', 'PAA', 'GS', 'AMGN', 'MA', 'TEF', 'AAPL', 'UPS']

# Import prices and inspect result
stock_prices = pd.read_csv('stock_prices.csv', parse_dates=['Date'], index_col='Date')
print(stock_prices.info())

# Calculate the returns
price_return = stock_prices.iloc[-1].div(stock_prices.iloc[0]).sub(1).mul(100)

# Plot horizontal bar chart of sorted price_return
price_return.sort_values().plot(kind='barh', title='Stock Price Returns')
plt.show()


```




```

<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 1762 entries, 2010-01-04 to 2016-12-30
Data columns (total 12 columns):
AAPL    1761 non-null float64
AMGN    1761 non-null float64
AMZN    1761 non-null float64
CPRT    1761 non-null float64
EL      1762 non-null float64
GS      1762 non-null float64
ILMN    1761 non-null float64
MA      1762 non-null float64
PAA     1762 non-null float64
RIO     1762 non-null float64
TEF     1762 non-null float64
UPS     1762 non-null float64
dtypes: float64(12)
memory usage: 179.0 KB
None

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture15-9.png?w=1024)



---


## **4.2 Build a market-cap weighted index**


![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture1-21.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture2-24.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture3-22.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture4-21.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture5-24.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture6-22.png?w=1024)


####
**Calculate number of shares outstanding**



 The next step towards building a value-weighted index is to calculate the number of shares for each index component.




 The number of shares will allow you to calculate the total market capitalization for each component given the historical price series in the next exercise.





```python

# Inspect listings and print tickers
print(listings.info())
<class 'pandas.core.frame.DataFrame'>
Index: 1015 entries, ACU to YPF
Data columns (total 7 columns):
Exchange                 1015 non-null object
Company Name             1015 non-null object
Last Sale                1015 non-null float64
Market Capitalization    1015 non-null float64
IPO Year                 1015 non-null float64
Sector                   1015 non-null object
Industry                 1015 non-null object
dtypes: float64(3), object(4)
memory usage: 103.4+ KB


listings.head(2)
             Exchange              Company Name  Last Sale  Market Capitalization  IPO Year                 Sector                             Industry
Stock Symbol
ACU              amex  Acme United Corporation.      27.39              91.138992    1988.0          Capital Goods      Industrial Machinery/Components
ROX              amex       Castle Brands, Inc.       1.46             237.644444    2006.0  Consumer Non-Durables  Beverages (Production/Distribution)


print(tickers)
['RIO', 'ILMN', 'CPRT', 'EL', 'AMZN', 'PAA', 'GS', 'AMGN', 'MA', 'TEF', 'AAPL', 'UPS']

```




```python

# Select components and relevant columns from listings
components = listings.loc[tickers, ['Market Capitalization', 'Last Sale']]

# Print the first rows of components
print(components.head())

              Market Capitalization  Last Sale
Stock Symbol
RIO                    70431.476895      38.94
ILMN                   25409.384000     173.68
CPRT                   13620.922869      29.65
EL                     31122.510011      84.94
AMZN                  422138.530626     884.67


```




```python

# Calculate the number of shares here
no_shares = components['Market Capitalization'].div(components['Last Sale'])

# Print the sorted no_shares
print(no_shares.sort_values(ascending=False))

Stock Symbol
AAPL    5246.540000
TEF     5037.804990
RIO     1808.717948
MA      1108.884100
UPS      869.297154
AMGN     735.890171
PAA      723.404994
AMZN     477.170618
CPRT     459.390316
GS       397.817439
EL       366.405816
ILMN     146.300000
dtype: float64

```


####
**Create time series of market value**



 You can now use the number of shares to calculate the total market capitalization for each component and trading date from the historical price series.




 The result will be the key input to construct the value-weighted stock index, which you will complete in the next exercise.





```

components.head()
              Market Capitalization  Last Sale  Number of Shares
Stock Symbol
RIO                    70431.476895      38.94       1808.717948
ILMN                   25409.384000     173.68        146.300000
CPRT                   13620.922869      29.65        459.390316
EL                     31122.510011      84.94        366.405816
AMZN                  422138.530626     884.67        477.170618

```




```python

# Select the number of shares
no_shares = components['Number of Shares']
print(no_shares.sort_values())

Stock Symbol
ILMN     146.300000
EL       366.405816
GS       397.817439
CPRT     459.390316
AMZN     477.170618
PAA      723.404994
AMGN     735.890171
UPS      869.297154
MA      1108.884100
RIO     1808.717948
TEF     5037.804990
AAPL    5246.540000
Name: Number of Shares, dtype: float64

```




```

stock_prices.head()
             AAPL   AMGN    AMZN  CPRT     EL  ...     MA    PAA    RIO    TEF    UPS
Date                                           ...
2010-01-04  30.57  57.72  133.90  4.55  24.27  ...  25.68  27.00  56.03  28.55  58.18
2010-01-05  30.63  57.22  134.69  4.55  24.18  ...  25.61  27.30  56.90  28.53  58.28
2010-01-06  30.14  56.79  132.25  4.53  24.25  ...  25.56  27.29  58.64  28.23  57.85
2010-01-07  30.08  56.27  130.00  4.50  24.56  ...  25.39  26.96  58.65  27.75  57.41
2010-01-08  30.28  56.77  133.52  4.52  24.66  ...  25.40  27.05  59.30  27.57  60.17

[5 rows x 12 columns]


# Create the series of market cap per ticker
market_cap = stock_prices.mul(no_shares)


market_cap.head()
                   AAPL          AMGN          AMZN         CPRT           EL  ...            MA           PAA            RIO            TEF           UPS
Date                                                                           ...
2010-01-04  160386.7278  42475.580670  63893.145750  2090.225938  8892.669154  ...  28476.143688  19531.934838  101342.466626  143829.332465  50575.708420
2010-01-05  160701.5202  42107.635585  64270.110538  2090.225938  8859.692631  ...  28398.521801  19748.956336  102916.051241  143728.576365  50662.638135
2010-01-06  158130.7156  41791.202811  63105.814231  2081.038131  8885.341038  ...  28343.077596  19741.722286  106063.220471  142217.234868  50288.840359
2010-01-07  157815.9232  41408.539922  62032.180340  2067.256422  8998.926841  ...  28154.567299  19502.998638  106081.307650  139799.088473  49906.349611
2010-01-08  158865.2312  41776.485008  63711.820915  2076.444228  9035.567423  ...  28165.656140  19568.105088  107256.974316  138892.283574  52305.609756

[5 rows x 12 columns]

```




```python

# Select first and last market cap here
first_value = market_cap.iloc[0]
last_value = market_cap.iloc[-1]

# Concatenate and plot first and last market cap here
pd.concat([first_value, last_value], axis=1).plot(kind='barh')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture7-18.png?w=648)


 You’ve made one of the essential ingredients of the index.



###
**Calculate & plot the composite index**



 By now you have all ingredients that you need to calculate the aggregate stock performance for your group of companies.




 Use the time series of market capitalization that you created in the last exercise to aggregate the market value for each period, and then normalize this series to convert it to an index.





```

market_cap_series.head(2)
                   AAPL          AMGN          AMZN         CPRT           EL  ...            MA           PAA            RIO            TEF           UPS
Date                                                                           ...
2010-01-04  160386.7278  42475.580670  63893.145750  2090.225938  8892.669154  ...  28476.143688  19531.934838  101342.466626  143829.332465  50575.708420
2010-01-05  160701.5202  42107.635585  64270.110538  2090.225938  8859.692631  ...  28398.521801  19748.956336  102916.051241  143728.576365  50662.638135

[2 rows x 12 columns]

raw_index.head(2)
Date
2010-01-04    694817.642691
2010-01-05    697995.697475
dtype: float64

```




```python

# Aggregate and print the market cap per trading day
raw_index = market_cap_series.sum(axis=1)
print(raw_index)

Date
2010-01-04    6.948176e+05
2010-01-05    6.979957e+05
                  ...
2016-12-29    1.589422e+06
2016-12-30    1.574862e+06
Length: 1761, dtype: float64


# Normalize the aggregate market cap here
index = raw_index.div(raw_index.iloc[0]).mul(100)


# Plot the index here
index.plot(title='Market-Cap Weighted Index')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture8-16.png?w=1024)



---


## **4.3 Evaluate index performance**


![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture1-22.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture2-25.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture3-23.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture4-22.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture5-25.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture6-23.png?w=1024)


####
**Calculate the contribution of each stock to the index**



 You have successfully built the value-weighted index. Let’s now explore how it performed over the 2010-2016 period.




 Let’s also determine how much each stock has contributed to the index return.





```

index.head()
Date
1/4/10    100.000000
1/5/10    100.457394
1/6/10     99.981005
1/7/10     99.485328
1/8/10    100.148231
Name: Unnamed: 1, dtype: float64

components.head(1)
              Market Capitalization  Last Sale  Number of Shares
Stock Symbol
RIO                    70431.476895      38.94       1808.717948

```




```python

# Calculate and print the index return here
index_return = (index[-1] / index[0] - 1) * 100
print(index_return)

# 126.65826659999996

# Select the market capitalization
market_cap = components['Market Capitalization']

# Calculate the total market cap
total_market_cap = market_cap.sum()
# 1800858.8762796503

# Calculate the component weights, and print the result
weights = market_cap / total_market_cap
print(weights.sort_values())

Stock Symbol
CPRT    0.007564
PAA     0.012340
...
AMZN    0.234410
AAPL    0.410929
Name: Market Capitalization, dtype: float64

# Calculate and plot the contribution by component
weights.mul(index_return).sort_values().plot(kind='barh')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture7-19.png?w=1024)

 return contribution by component



 The next step is to take a look at how your index stacks up against a benchmark!



####
**Compare index performance against benchmark I**



 The next step in analyzing the performance of your index is to compare it against a benchmark.




 In the video, we used the S&P 500 as benchmark. You can also use the Dow Jones Industrial Average, which contains the 30 largest stocks, and would also be a reasonable benchmark for the largest stocks from all sectors across the three exchanges.





```

djia.head()
DATE
2010-01-04    100.000000
2010-01-05     99.887188
2010-01-06     99.902872
2010-01-07    100.216365
2010-01-08    100.323414
Name: DJIA, dtype: float64

index.head()
Date
2010-01-04    100.000000
2010-01-05    100.457394
2010-01-06     99.981005
2010-01-07     99.485328
2010-01-08    100.148231
Name: Unnamed: 1, dtype: float64

```




```python

# Convert index series to dataframe here
data = index.to_frame('Index')

# Normalize djia series and add as new column to data
djia = djia.div(djia.iloc[0]).mul(100)
data['DJIA'] = djia

# Show total return for both index and djia
print((data.iloc[-1] / data.iloc[0] -1 ) * 100)

Index    126.658267
DJIA      86.722172
dtype: float64


# Plot both series
data.plot()
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture8-17.png?w=649)

####
**Compare index performance against benchmark II**



 The next step in analyzing the performance of your index is to compare it against a benchmark.




 In the video, we have use the S&P 500 as benchmark. You can also use the Dow Jones Industrial Average, which contains the 30 largest stocks, and would also be a reasonable benchmark for the largest stocks from all sectors across the three exchanges.





```python

# Inspect data
print(data.info())
print(data.head())

<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 1761 entries, 2010-01-04 to 2016-12-30
Data columns (total 2 columns):
Index    1761 non-null float64
DJIA     1761 non-null float64
dtypes: float64(2)
memory usage: 41.3 KB
None

                 Index        DJIA
Date
2010-01-04  100.000000  100.000000
2010-01-05  100.457394   99.887188
2010-01-06   99.981005   99.902872
2010-01-07   99.485328  100.216365
2010-01-08  100.148231  100.323414

```




```python

# Create multi_period_return function here
def multi_period_return(r):
    return (np.prod(r + 1) - 1) * 100

# Calculate rolling_return_360
rolling_return_360 = data.pct_change().rolling('360D').apply(multi_period_return)

# Plot rolling_return_360 here
rolling_return_360.plot(title='Rolling 360D Return')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture9-15.png?w=1024)



---


###
 4.4
 **Index correlation & exporting to excel**


####
**Visualize your index constituent correlations**



 To better understand the characteristics of your index constituents, you can calculate the return correlations.




 Use the daily stock prices or your index companies, and show a heatmap of the daily return correlations!





```

stock_prices.head(3)
            AAPL  AMGN   AMZN  CPRT    EL  ...    MA   PAA   RIO   TEF   UPS
Date                                       ...
2010-01-04 30.57 57.72 133.90  4.55 24.27  ... 25.68 27.00 56.03 28.55 58.18
2010-01-05 30.63 57.22 134.69  4.55 24.18  ... 25.61 27.30 56.90 28.53 58.28
2010-01-06 30.14 56.79 132.25  4.53 24.25  ... 25.56 27.29 58.64 28.23 57.85

[3 rows x 12 columns]

```




```python

# Calculate the daily returns
returns = stock_prices.pct_change()

# Calculate and print the pairwise correlations
correlations = returns.corr()
print(correlations)

# Plot a heatmap of daily return correlations
sns.heatmap(correlations, annot=True)
plt.title('Daily Return Correlations')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/manipulating-time-series-data-in-python/capture10-16.png?w=1024)

####
**Save your analysis to multiple excel worksheets**



 Now that you have completed your analysis, you may want to save all results into a single Excel workbook.




 Let’s practice exporting various
 `DataFrame`
 to multiple Excel worksheets.





```python

# Inspect index and stock_prices
print(index.info())
print(stock_prices.info())

# Join index to stock_prices, and inspect the result
data = stock_prices.join(index)
print(data.info())

# Create index & stock price returns
returns = data.pct_change()

# Export data and data as returns to excel
with pd.ExcelWriter('data.xls') as writer:
    data.to_excel(writer, sheet_name='data')
    returns.to_excel(writer, sheet_name='returns')


```



 The End.


 Thank you for reading and hope you’ve learned a lot.



