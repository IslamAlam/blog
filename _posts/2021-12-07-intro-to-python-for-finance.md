---
title: Intro to Python for Finance
date: 2021-12-07 11:22:08 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Intro to Python for Finance
=============================







 This is a memo. This course does not have a track yet.


**A introduction course for those who have no experience in Python.


 You can find the original course
 [HERE](https://www.datacamp.com/courses/intro-to-python-for-finance)**
 .





---



# **1. Basic of Python**
-----------------------





 Nothing special here.





---



# **2. Lists**
-------------


####
**sort() & max()**




```python

# list sort
prices = [159.54, 37.13, 71.17]
prices.sort()
print(prices)
# [37.13, 71.17, 159.54]

# max value of a list
prices = [159.54, 37.13, 71.17]
price_max = max(prices)
print(price_max)
# 159.54

```


####
**append() & extend()**




```python

# Append a name to the list names
names.append('Amazon.com')
print(names)

# Extend list names
more_elements = ['DowDuPont', 'Alphabet Inc']
names.append(more_elements)
print(names)

# ['Apple Inc', 'Coca-Cola', 'Walmart', 'Amazon.com']
# append a list to the list will produce a nested list.
# ['Apple Inc', 'Coca-Cola', 'Walmart', 'Amazon.com', ['DowDuPont', 'Alphabet Inc']]
# to get a flat list, use extend
['Apple Inc', 'Coca-Cola', 'Walmart', 'Amazon.com', 'DowDuPont', 'Alphabet Inc']

```


####
**index()**




```python

# get max price
max_price = max(prices)

# Identify index of max price
max_index = prices.index(max_price)

# Identify the name of the company with max price
max_stock_name = names[max_index]

print('The largest stock price is associated with ' + max_stock_name + ' and is $' + str(max_price) + '.')

# The largest stock price is associated with Amazon.com and is $1705.54.

```




---



# **3. Numpy arrays**
--------------------


####
**step slicing**




```python

# Subset every third element
print(prices_array)
# [170.12  93.29  55.28 145.3  171.81  59.5  100.5 ]

prices_subset_3 = prices_array[0:7:3]
print(prices_subset_3)
# [170.12 145.3  100.5 ]

```


####
**2D array**




```python

# Create a 2D array of prices and earnings
stock_array = np.array([prices, earnings])
print(stock_array)
# [[170.12  93.29  55.28 145.3  171.81  59.5  100.5 ]
#  [  9.2    5.31   2.41   5.91  15.42   2.51   6.79]]

# Print the shape of stock_array
print(stock_array.shape)
# (2, 7)

# Print the size of stock_array
print(stock_array.size)
# 14

```


####
**np.transpose()**




```python

# Transpose stock_array
stock_array_transposed = np.transpose(stock_array)
print(stock_array_transposed)
# [[170.12   9.2 ]
 [ 93.29   5.31]
 [ 55.28   2.41]
 [145.3    5.91]
 [171.81  15.42]
 [ 59.5    2.51]
 [100.5    6.79]]

# Print the shape of stock_array
print(stock_array_transposed.shape)
# (7, 2)

# Print the size of stock_array
print(stock_array_transposed.size)
# 14

```


####
**Subsetting 2D arrays**




```python

# original array
stock_array_transposed
[[170.12   9.2 ]
 [ 93.29   5.31]
 [ 55.28   2.41]
 [145.3    5.91]
 [171.81  15.42]
 [ 59.5    2.51]
 [100.5    6.79]]

```




```python

# Subset all rows and column
array[:, column_index]

# Subset the first (0th) columns
prices = stock_array_transposed[:, 0]
print(prices)

# [170.12  93.29  55.28 145.3  171.81  59.5  100.5 ]

##########################################
# Subset all columns and row
array[row_index, :]

# Subset the first 3 rows (0,1,2)
stock_array_transposed[0:3]
array([[170.12,   9.2 ],
       [ 93.29,   5.31],
       [ 55.28,   2.41]])

##########################################
# Subset single value
array[row_index][column_index]

# Subset the 3rd row, 1st column
stock_array_transposed[2][0]
55.28


```


####
 np.mean & np.std()




```python

# Calculate mean and standard deviation
np.mean(array)
np.std(array)

```


####
**np.arange()**




```python

# Create and print company IDs
company_ids = np.arange(1, 8, 1)
print(company_ids)
# [1 2 3 4 5 6 7]

# Use array slicing to select specific company IDs
company_ids_odd = np.arange(1, 8, 2)
print(company_ids_odd)
# [1 3 5 7]

```


####
**boolean slice**
 I: numeric




```python

# Find the mean
price_mean = np.mean(prices)

# Create boolean array
boolean_array = (prices > price_mean)
print(boolean_array)
[ True False False  True  True False False]

# Select prices that are greater than average
above_avg = prices[boolean_array]
print(above_avg)
[170.12 145.3  171.81]


```


####
**boolean slice**
 II: string




```

sectors
array(['Information Technology', 'Health Care', 'Health Care',
       'Information Technologies', 'Health Care'], dtype='<U24')

names
array(['Apple Inc', 'Abbvie Inc', 'Abbott Laboratories',
       'Accenture Technologies', 'Allergan Plc'], dtype='<U22')

```




```python

# Create boolean array
boolean_array = (sectors == 'Health Care')
print(boolean_array)
# [False  True  True False  True]

# Print only health care companies
health_care = names[boolean_array]
print(health_care)
# ['Abbvie Inc' 'Abbott Laboratories' 'Allergan Plc']

```




---



 4. Visualization
------------------


####
**First plot (line plot)**




```python

# Import matplotlib.pyplot with the alias plt
import matplotlib.pyplot as plt

# Plot the price of stock over time
plt.plot(days, prices, color="red", linestyle="--")

# Display the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-python-for-finance/capture3-13.png?w=637)

####
**Add label and title**




```

import matplotlib.pyplot as plt

# Plot price as a function of time
plt.plot(days, prices, color="red", linestyle="--")

# Add x and y labels
plt.xlabel('Days')
plt.ylabel('Prices, $')

# Add plot title
plt.title('Company Stock Prices Over Time')

# Show plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-python-for-finance/capture4-13.png?w=641)

####
**Multi graphs**




```python

# Plot two lines of varying colors
plt.plot(days, prices1, color='red')
plt.plot(days, prices2, color='green')

# Add labels
plt.xlabel('Days')
plt.ylabel('Prices, $')
plt.title('Stock Prices Over Time')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-python-for-finance/capture5-16.png?w=646)

####
**Scatter plot**




```python

# Import pyplot as plt
import matplotlib.pyplot as plt

# Plot price as a function of time
plt.scatter(days, prices, color='green', s=0.1)

# Show plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-python-for-finance/capture6-13.png?w=635)

####
**histogram**




```python

# Plot histogram
plt.hist(prices, bins=100)

# Display plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-python-for-finance/capture7-10.png?w=635)


 You can see that these prices are not normally distributed, they are skewed to the left!



####
**Comparing two histograms**




```python

# Plot histogram of stocks_A
plt.hist(stock_A, bins=100, alpha=0.4)

# Plot histogram of stocks_B
plt.hist(stock_B, bins=100, alpha=0.4)

# Display plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-python-for-finance/capture8-8.png?w=644)

####
**Add legend**




```python

# Plot stock_A and stock_B histograms
plt.hist(stock_A, bins=100, alpha=0.4, label='Stock A')
plt.hist(stock_B, bins=100, alpha=0.4, label='Stock B')

# Add the legend
plt.legend()

# Display the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-python-for-finance/capture9-7.png?w=638)



---



# **5. S&P 100 Case Study**
--------------------------



 Nothing special here.




 The End.


 Thank you for reading.



