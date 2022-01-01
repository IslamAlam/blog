---
title: Working with Geospatial Data in Python
date: 2021-12-07 11:22:12 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Working with Geospatial Data in Python
=========================================







 This is the memo of
 **Working with Geospatial Data in Python from DataCamp**
 .


**You can find the original course
 [HERE](https://www.datacamp.com/courses/working-with-geospatial-data-in-python)**
 .



###
 Course Description



 A good proportion of the data out there in the real world is inherently spatial. From the population recorded in the national census, to every shop in your neighborhood, the majority of datasets have a location aspect that you can exploit to make the most of what they have to offer. This course will show you how to integrate spatial data into your Python Data Science workflow. You will learn how to interact with, manipulate and augment real-world data using their geographic dimension. You will learn to read tabular spatial data in the most common formats (e.g. GeoJSON, shapefile, geopackage) and visualize them in maps. You will then combine different sources using their location as the bridge that puts them in relation to each other. And, by the end of the course, you will be able to understand what makes geographic data unique, allowing you to transform and repurpose them in different contexts.



###
**Table of contents**


1. Introduction to geospatial vector data
2. [Spatial relationships](https://datascience103579984.wordpress.com/2019/11/25/working-with-geospatial-data-in-python-from-datacamp/2/)
3. [Projecting and transforming geometries](https://datascience103579984.wordpress.com/2019/11/25/working-with-geospatial-data-in-python-from-datacamp/3/)
4. [Putting it all together – Artisanal mining sites case study](https://datascience103579984.wordpress.com/2019/11/25/working-with-geospatial-data-in-python-from-datacamp/4/)





# **1. Introduction to geospatial vector data**
----------------------------------------------



 In this chapter, you will be introduced to the concepts of geospatial data, and more specifically of vector data. You will then learn how to represent such data in Python using the GeoPandas library, and the basics to read, explore and visualize such data. And you will exercise all this with some datasets about the city of Paris.



## **1.1 Geospatial data**



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/1-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/2-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/3-5.png?w=1024)



### **1.1.1 Restaurants in Paris**



 Throughout the exercises in this course, we will work with several datasets about the city of Paris.




 In this exercise, we will start with exploring a dataset about the restaurants in the center of Paris (compiled from a
 [Paris Data open dataset](https://opendata.paris.fr/explore/dataset/commercesparis/)
 ). The data contains the coordinates of the point locations of the restaurants and a description of the type of restaurant.




 We expect that you are familiar with the basics of the pandas library to work with tabular data (
 `DataFrame`
 objects) in Python. Here, we will use pandas to read the provided csv file, and then use matplotlib to make a visualization of the points. With matplotlib, we first create a figure and axes object with
 `fig, ax = plt.subplots()`
 , and then use this axes object
 `ax`
 to create the plot.





```python

# Import pandas and matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# Read the restaurants csv file
restaurants = pd.read_csv("paris_restaurants.csv")

# Inspect the first rows of restaurants
print(restaurants.head())

# Make a plot of all points
fig, ax = plt.subplots()
ax.plot(restaurants.x, restaurants.y, 'o')
plt.show()

```




```

                                             type              x             y
0                             Restaurant européen  259641.691646  6.251867e+06
1                Restaurant traditionnel français  259572.339603  6.252030e+06
2                Restaurant traditionnel français  259657.276374  6.252143e+06
3  Restaurant indien, pakistanais et Moyen Orient  259684.438330  6.252203e+06
4                Restaurant traditionnel français  259597.943086  6.252230e+06

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/4-4.png?w=1024)

### **1.1.2 Adding a background map**



 A plot with just some points can be hard to interpret without any spatial context. Therefore, in this exercise we will learn how to add a background map.




 We are going to make use of the
 `contextily`
 package. The
 `add_basemap()`
 function of this package makes it easy to add a background web map to our plot. We begin by plotting our data first, and then pass the matplotlib axes object to the
 `add_basemap()`
 function.
 `contextily`
 will then download the web tiles needed for the geographical extent of your plot.




 To set the size of the plotted points, we can use the
 `markersize`
 keyword of the
 `plot()`
 method.




 Pandas has been imported as
 `pd`
 and matplotlib’s pyplot functionality as
 `plt`
 .





```python

# Read the restaurants csv file
restaurants = pd.read_csv("paris_restaurants.csv")

# Import contextily
import contextily

# A figure of all restaurants with background
fig, ax = plt.subplots()
ax.plot(restaurants.x, restaurants.y, 'o', markersize=1)
contextily.add_basemap(ax)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/5-4.png?w=1024)



---


## **1.2 Introduction to GeoPandas**


### **1.2.1 Explore the Paris districts (I)**



 In this exercise, we introduce a next dataset about Paris: the administrative districts of Paris (compiled from a
 [Paris Data open dataset](https://opendata.paris.fr/explore/dataset/quartier_paris/)
 ).




 The dataset is available as a GeoPackage file, a specialised format to store geospatial vector data, and such a file can be read by GeoPandas using the
 `geopandas.read_file()`
 function.




 To get a first idea of the dataset, we can inspect the first rows with
 `head()`
 and do a quick visualization with `plot(). The attribute information about the districts included in the dataset is the district name and the population (total number of inhabitants of each district).





```python

# Import GeoPandas
import geopandas

# Read the Paris districts dataset
districts = geopandas.read_file('paris_districts.gpkg')

# Inspect the first rows
print(districts.head())

# Make a quick visualization of the districts
districts.plot()
plt.show()

```




```

   id           district_name  population                                           geometry
0   1  St-Germain-l'Auxerrois        1672  POLYGON ((451922.1333912524 5411438.484355546,...
1   2                  Halles        8984  POLYGON ((452278.4194036503 5412160.89282334, ...
2   3            Palais-Royal        3195  POLYGON ((451553.8057660239 5412340.522224233,...
3   4           Place-Vendôme        3044  POLYGON ((451004.907944323 5412654.094913081, ...
4   5                 Gaillon        1345  POLYGON ((451328.7522686935 5412991.278156867,...

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/6-2.png?w=1024)

### **1.2.2 Explore the Paris districts (II)**



 In the previous exercise, we used the customized
 `plot()`
 method of the GeoDataFrame, which produces a simple visualization of the geometries in the dataset. The GeoDataFrame and GeoSeries objects can be seen as “spatial-aware” DataFrame and Series objects, and compared to their pandas counterparts, they expose additional spatial-specific methods and attributes.




 The
 `.geometry`
 attribute of a GeoDataFrame always returns the column with the geometry objects as a
 `GeoSeries`
 , whichever the actual name of the column (in the default case it will also be called ‘geometry’).




 Another example of extra spatial functionality is the
 `area`
 attribute, giving the area of the polygons.




 GeoPandas has been imported as
 `geopandas`
 and the districts dataset is available as the
 `districts`
 variable.





```python

# Check what kind of object districts is
print(type(districts))

# Check the type of the geometry attribute
print(type(districts.geometry))

# Inspect the first rows of the geometry
print(districts.geometry.head())

# Inspect the area of the districts
print(districts.geometry.area)

```




```

<class 'geopandas.geodataframe.GeoDataFrame'>
<class 'geopandas.geoseries.GeoSeries'>

0    POLYGON ((451922.1333912524 5411438.484355546,...
1    POLYGON ((452278.4194036503 5412160.89282334, ...
2    POLYGON ((451553.8057660239 5412340.522224233,...
3    POLYGON ((451004.907944323 5412654.094913081, ...
4    POLYGON ((451328.7522686935 5412991.278156867,...
Name: geometry, dtype: object


0     8.685379e+05
1     4.122371e+05
2     2.735494e+05
          ...
78    1.598127e+06
79    2.089783e+06
Length: 80, dtype: float64

```


### **1.2.3 The Paris restaurants as a GeoDataFrame**



 In the first coding exercise of this chapter, we imported the locations of the restaurants in Paris from a csv file. To enable the geospatial functionality of GeoPandas, we want to convert the pandas DataFrame to a GeoDataFrame. This can be done with the
 `GeoDataFrame()`
 constructor and the
 `geopandas.points_from_xy()`
 function, and is done for you.




 Now we have a GeoDataFrame, all spatial functionality becomes available, such as plotting the geometries. In this exercise we will make the same figure as in the first exercise with the restaurants dataset, but now using the GeoDataFrame’s
 `plot()`
 method.




 Pandas has been imported as
 `pd`
 , GeoPandas as
 `geopandas`
 and matplotlib’s pyplot functionality as
 `plt`
 .





```python

# Read the restaurants csv file into a DataFrame
df = pd.read_csv("paris_restaurants.csv")

# Convert it to a GeoDataFrame
restaurants = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.x, df.y))

# Inspect the first rows of the restaurants GeoDataFrame
print(restaurants.head())

# Make a plot of the restaurants
ax = restaurants.plot(markersize=1)
import contextily
contextily.add_basemap(ax)
plt.show()

```




```

                                 type              x             y  \
0                 European restuarant  259641.691646  6.251867e+06
1       Traditional French restaurant  259572.339603  6.252030e+06
2       Traditional French restaurant  259657.276374  6.252143e+06
3  Indian / Middle Eastern restaurant  259684.438330  6.252203e+06
4       Traditional French restaurant  259597.943086  6.252230e+06

                                      geometry
0  POINT (259641.6916457232 6251867.062617987)
1  POINT (259572.3396029567 6252029.683163137)
2  POINT (259657.2763744336 6252143.400946028)
3  POINT (259684.4383301869 6252203.137238394)
4  POINT (259597.9430858413 6252230.044091299)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/7-2.png?w=1024)



---


## **1.3 Exploring and visualizing spatial data**


### **1.3.1 Visualizing the population density**



 Let’s get back to the districts dataset. In a previous exercise we visualized the districts with a uniform column. But often we want to show the spatial variation of a variable, and color the polygons accordingly.




 In this exercise we will visualize the spatial variation of the population density within the center of Paris. For this, we will first calculate the population density by dividing the population number with the area, and add it as a new column to the dataframe.




 The districts dataset is already loaded as
 `districts`
 , GeoPandas has been imported as
 `geopandas`
 and matplotlib.pyplot as
 `plt`
 .





```python

# Inspect the first rows of the districts dataset
print(districts.head())

# Inspect the area of the districts
print(districts.area)

# Add a population density column
districts['population_density'] = districts.population / districts.area * 10**6

# Make a plot of the districts colored by the population density
districts.plot(column='population_density', legend=True)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/1-6.png?w=1024)



```

   id           district_name  population                                           geometry
0   1  St-Germain-l'Auxerrois        1672  POLYGON ((451922.1333912524 5411438.484355546,...
1   2                  Halles        8984  POLYGON ((452278.4194036503 5412160.89282334, ...
2   3            Palais-Royal        3195  POLYGON ((451553.8057660239 5412340.522224233,...
3   4           Place-Vendôme        3044  POLYGON ((451004.907944323 5412654.094913081, ...
4   5                 Gaillon        1345  POLYGON ((451328.7522686935 5412991.278156867,...


0     8.685379e+05
1     4.122371e+05
2     2.735494e+05
3     2.693111e+05
          ...
78    1.598127e+06
79    2.089783e+06
Length: 80, dtype: float64

```


### **1.3.2 Using pandas functionality: groupby**



 This course will focus on the spatial functionality of GeoPandas, but don’t forget that we still have a dataframe, and all functionality you know from Pandas is still applicable.




 In this exercise, we will recap a common functionality: the groupby operation. You may want to use this operation when you have a column containing groups, and you want to calculate a statistic for each group. In the
 `groupby()`
 method, you pass the column that contains the groups. On the resulting object, you can then call the method you want to calculate for each group. In this exercise, we want to know the size of each group of type of restaurants.




 We refer to the course on Manipulating DataFrames with pandas for more information and exercises on this groupby operation.





```python

# Load the restaurants data
restaurants = geopandas.read_file("paris_restaurants.geosjon")

# Calculate the number of restaurants of each type
type_counts = restaurants.groupby('type').size()

# Print the result
print(type_counts)

```




```

type
African restaurant                        138
Asian restaurant                         1642
Caribbean restaurant                       27
Central and South American restuarant      97
European restuarant                      1178
Indian / Middle Eastern restaurant        394
Maghrebian restaurant                     207
Other world restaurant                    107
Traditional French restaurant            1945
dtype: int64

```


### **1.3.3 Plotting multiple layers**



 Another typical pandas functionality is filtering a dataframe: taking a subset of the rows based on a condition (which generates a boolean mask).




 In this exercise, we will take the subset of all African restaurants, and then make a multi-layered plot. In such a plot, we combine the visualization of several GeoDataFrames on a single figure. To add one layer, we can use the
 `ax`
 keyword of the
 `plot()`
 method of a GeoDataFrame to pass it a matplotlib axes object.




 The restaurants data is already loaded as the
 `restaurants`
 GeoDataFrame. GeoPandas is imported as
 `geopandas`
 and matplotlib.pyplot as
 `plt`
 .





```python

# Load the restaurants dataset
restaurants = geopandas.read_file("paris_restaurants.geosjon")

# Take a subset of the African restaurants
african_restaurants = restaurants[restaurants['type']=='African restaurant']

# Make a multi-layered plot
fig, ax = plt.subplots(figsize=(10, 10))
restaurants.plot(ax=ax, color='grey')
african_restaurants.plot(ax=ax, color='red')
# Remove the box, ticks and labels
ax.set_axis_off()
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/2-6.png?w=1024)


# **2. Spatial relationships**
-----------------------------


## **2.1 Shapely geometries and spatial relationships**


### **2.1.1 Creating a Point geometry**



 The Eiffel Tower is an iron lattice tower built in the 19th century, and is probably the most iconic view of Paris.




![Eiffel Tower](https://assets.datacamp.com/production/repositories/2561/datasets/699329ee5cda7c70a745a4fb150b3e7e136e6e3c/Eiffel_Tower.png)


 (By couscouschocolat [
 [CC BY 2.0](https://creativecommons.org/licenses/by/2.0)
 ], via Wikimedia Commons)




 The location of the Eiffel Tower is: x of 255422.6 and y of 6250868.9.





```python

# Import the Point geometry
from shapely.geometry import Point

# Construct a point object for the Eiffel Tower
eiffel_tower = Point(255422.6, 6250868.9)

# Print the result
print(eiffel_tower)
# POINT (255422.6 6250868.9)

```


### **2.1.2 Shapely’s spatial methods**



 Now we have a shapely
 `Point`
 object for the Eiffel Tower, we can use the different methods available on such a geometry object to perform spatial operations, such as calculating a distance or checking a spatial relationship.




 We repeated the construction of
 `eiffel_tower`
 , and also provide the code that extracts one of the neighbourhoods (the Montparnasse district), as well as one of the restaurants located within Paris.





```python

# Construct a point object for the Eiffel Tower
eiffel_tower = Point(255422.6, 6250868.9)

# Accessing the Montparnasse geometry (Polygon) and restaurant
district_montparnasse = districts.loc[52, 'geometry']
resto = restaurants.loc[956, 'geometry']

# Is the Eiffel Tower located within the Montparnasse district?
print(eiffel_tower.within(district_montparnasse))
# False

# Does the Montparnasse district contains the restaurant?
print(district_montparnasse.contains(resto))
# True

# The distance between the Eiffel Tower and the restaurant?
print(eiffel_tower.distance(resto))
# 4431.459825587039

```



 Note that the
 `contains()`
 and
 `within()`
 methods are the opposite of each other: if
 `geom1.contains(geom2)`
 is
 `True`
 , then also
 `geom2.within(geom1)`
 will be
 `True`
 .





---


## **2.2 Spatial relationships with GeoPandas**


### **2.2.1 In which district in the Eiffel Tower located?**



 Let’s return to the Eiffel Tower example. In previous exercises, we constructed a
 `Point`
 geometry for its location, and we checked that it was not located in the Montparnasse district. Let’s now determine in which of the districts of Paris it
 *is*
 located.




 The
 `districts`
 GeoDataFrame has been loaded, and the Shapely and GeoPandas libraries are imported.





```python

# Construct a point object for the Eiffel Tower
eiffel_tower = Point(255422.6, 6250868.9)

# Create a boolean Series
mask = districts.contains(eiffel_tower)

# Print the boolean Series
print(mask.head())

# Filter the districts with the boolean mask
print(districts[mask])

```




```

0    False
1    False
2    False
3    False
4    False
dtype: bool

    id district_name  population                                           geometry
27  28  Gros-Caillou       25156  POLYGON ((257097.2898896902 6250116.967139574,...


```


### **2.2.2 How far is the closest restaurant?**



 Now, we might be interested in the restaurants nearby the Eiffel Tower. To explore them, let’s visualize the Eiffel Tower itself as well as the restaurants within 1km.




 To do this, we can calculate the distance to the Eiffel Tower for each of the restaurants. Based on this result, we can then create a mask that takes
 `True`
 if the restaurant is within 1km, and
 `False`
 otherwise, and use it to filter the restaurants GeoDataFrame. Finally, we make a visualization of this subset.




 The
 `restaurants`
 GeoDataFrame has been loaded, and the
 `eiffel_tower`
 object created. Further, matplotlib, GeoPandas and contextily have been imported.





```python

# The distance from each restaurant to the Eiffel Tower
dist_eiffel = restaurants.distance(eiffel_tower)

# The distance to the closest restaurant
print(dist_eiffel.min())
# 460.6976028277898

# Filter the restaurants for closer than 1 km
restaurants_eiffel = restaurants[dist_eiffel<1000]

# Make a plot of the close-by restaurants
ax = restaurants_eiffel.plot()
geopandas.GeoSeries([eiffel_tower]).plot(ax=ax, color='red')
contextily.add_basemap(ax)
ax.set_axis_off()
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/1-7.png?w=1022)



---


## **2.3 The spatial join operation**



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/2-7.png?w=1024)

### **2.3.1 Paris: spatial join of districts and bike stations**



 Let’s return to the Paris data on districts and bike stations. We will now use the spatial join operation to identify the district in which each station is located.




 The districts and bike sharing stations datasets are already pre-loaded for you as the
 `districts`
 and
 `stations`
 GeoDataFrames, and GeoPandas has been imported as
 `geopandas`





```python

# Join the districts and stations datasets
joined = geopandas.sjoin(stations, districts, op='within')

# Inspect the first five rows of the result
print(joined.head())

```




```

                                       name  bike_stands  available_bikes  \
0                    14002 - RASPAIL QUINET           44                4
143  14112 - FAUBOURG SAINT JACQUES CASSINI           16                0
293               14033 - DAGUERRE GASSENDI           38                1
346     14006 - SAINT JACQUES TOMBE ISSOIRE           22                0
429       14111 - DENFERT-ROCHEREAU CASSINI           24                8

                                        geometry  index_right  id district_name
0     POINT (450804.448740735 5409797.268203795)           52  53  Montparnasse
143   POINT (451419.446715647 5409421.528587255)           52  53  Montparnasse
293  POINT (450708.2275807534 5409406.941172979)           52  53  Montparnasse
346  POINT (451340.0264470892 5409124.574548723)           52  53  Montparnasse
429  POINT (451274.5111513372 5409609.730783217)           52  53  Montparnasse

```


### **2.3.2 Map of tree density by district (1)**



 Using a dataset of all trees in public spaces in Paris, the goal is to make a map of the tree density by district. For this, we first need to find out how many trees each district contains, which we will do in this exercise. In the following exercise, we will then use this result to calculate the density and create a map.




 To obtain the tree count by district, we first need to know in which district each tree is located, which we can do with a spatial join. Then, using the result of the spatial join, we will calculate the number of trees located in each district using the pandas ‘group-by’ functionality.




 GeoPandas has been imported as
 `geopandas`
 .





```python

# trees
      species location_type                                     geometry
0  Marronnier    Alignement  POINT (455834.1224756146 5410780.605718749)
1  Marronnier    Alignement  POINT (446546.2841757428 5412574.696813397)
2  Marronnier    Alignement    POINT (449768.283096671 5409876.55691999)
3  Marronnier    Alignement   POINT (451779.7079508423 5409292.07146508)
4     Sophora    Alignement  POINT (447041.3613609616 5409756.711514045)

```




```python

# Read the trees and districts data
trees = geopandas.read_file("paris_trees.gpkg")
districts = geopandas.read_file("paris_districts_utm.geojson")

# Spatial join of the trees and districts datasets
joined = geopandas.sjoin(trees, districts, op='within')

# Calculate the number of trees in each district
trees_by_district = joined.groupby('district_name').size()

# Convert the series to a DataFrame and specify column name
trees_by_district = trees_by_district.to_frame(name='n_trees')

# Inspect the result
print(trees_by_district.head())

```




```

                 n_trees
district_name
Amérique             183
Archives               8
Arsenal               60
Arts-et-Metiers       20
Auteuil              392

```


### **2.3.3 Map of tree density by district (2)**



 Now we have obtained the number of trees by district, we can make the map of the districts colored by the tree density.




 For this, we first need to merge the number of trees in each district we calculated in the previous step (
 `trees_by_district`
 ) back to the districts dataset. We will use the
 [`pd.merge()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html)
 function to join two dataframes based on a common column.




 Since not all districts have the same size, it is a fairer comparison to visualize the tree density: the number of trees relative to the area.




 The district dataset has been pre-loaded as
 `districts`
 , and the final result of the previous exercise (a DataFrame with the number of trees for each district) is available as
 `trees_by_district`
 . GeoPandas has been imported as
 `geopandas`
 and Pandas as
 `pd`
 .





```python

# Print the first rows of the result of the previous exercise
print(trees_by_district.head())

# Merge the 'districts' and 'trees_by_district' dataframes
districts_trees = pd.merge(districts, trees_by_district, on='district_name')

# Inspect the result
print(districts_trees.head())

```




```

     district_name  n_trees
0         Amérique      728
1         Archives       34
2          Arsenal      213
3  Arts-et-Metiers       79
4          Auteuil     1474


   id           district_name                                           geometry  n_trees
0   1  St-Germain-l'Auxerrois  POLYGON ((451922.1333912524 5411438.484355546,...      152
1   2                  Halles  POLYGON ((452278.4194036503 5412160.89282334, ...      149
2   3            Palais-Royal  POLYGON ((451553.8057660239 5412340.522224233,...        6
3   4           Place-Vendôme  POLYGON ((451004.907944323 5412654.094913081, ...       17
4   5                 Gaillon  POLYGON ((451328.7522686935 5412991.278156867,...       18

```




```python

# Merge the 'districts' and 'trees_by_district' dataframes
districts_trees = pd.merge(districts, trees_by_district, on='district_name')

# Add a column with the tree density
districts_trees['n_trees_per_area'] = districts_trees['n_trees'] / districts_trees.geometry.area

# Make of map of the districts colored by 'n_trees_per_area'
districts_trees.plot(column='n_trees_per_area')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/3-6.png?w=1024)



---


## **2.4 Choropleths**



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/4-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/5-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/6-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/7-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/8-2.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/9-2.png?w=1024)



### **2.4.1 Equal interval choropleth**



 In the last exercise, we created a map of the tree density. Now we know more about choropleths, we will explore this visualisation in more detail.




 First, let’s visualize the effect of just using the number of trees versus the number of trees normalized by the area of the district (the tree density). Second, we will create an equal interval version of this map instead of using a continuous color scale. This classification algorithm will split the value space in equal bins and assign a color to each.




 The
 `district_trees`
 GeoDataFrame, the final result of the previous exercise is already loaded. It includes the variable
 `n_trees_per_area`
 , measuring tree density by district (note the variable has been multiplied by 10,000).





```python

# Print the first rows of the tree density dataset
print(districts_trees.head())

# Make a choropleth of the number of trees
districts_trees.plot(column='n_trees', legend=True)
plt.show()

# Make a choropleth of the number of trees per area
districts_trees.plot(column='n_trees_per_area', legend=True)
plt.show()

# Make a choropleth of the number of trees
districts_trees.plot(column='n_trees_per_area', scheme='equal_interval', legend=True)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/10-2.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/11-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/12-2.png?w=1024)



### **2.4.2 Quantiles choropleth**



 In this exercise we will create a quantile version of the tree density map. Remember that the quantile algorithm will rank and split the values into groups with the same number of elements to assign a color to each. This time, we will create seven groups that allocate the colors of the
 `YlGn`
 colormap across the entire set of values.




 The
 `district_trees`
 GeoDataFrame is again already loaded. It includes the variable
 `n_trees_per_area`
 , measuring tree density by district (note the variable has been multiplied by 10,000).





```python

# Generate the choropleth and store the axis
ax = districts_trees.plot(column='n_trees_per_area', scheme='quantiles',
                          k=7, cmap='YlGn', legend=True)

# Remove frames, ticks and tick labels from the axis
ax.set_axis_off()
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/13-2.png?w=1024)

### **2.4.3 Compare classification algorithms**



 In this final exercise, you will build a multi map figure that will allow you to compare the two approaches to map variables we have seen.




 You will rely on standard
 `matplotlib`
 patterns to build a figure with two subplots (Axes
 `axes[0]`
 and
 `axes[1]`
 ) and display in each of them, respectively, an equal interval and quantile based choropleth. Once created, compare them visually to explore the differences that the classification algorithm can have on the final result.




 This exercise comes with a GeoDataFrame object loaded under the name
 `district_trees`
 that includes the variable
 `n_trees_per_area`
 , measuring tree density by district.





```python

# Set up figure and subplots
fig, axes = plt.subplots(nrows=2)

# Plot equal interval map
districts_trees.plot('n_trees_per_area', scheme='equal_interval', k=5, legend=True, ax=axes[0])
axes[0].set_title('Equal Interval')
axes[0].set_axis_off()

# Plot quantiles map
districts_trees.plot('n_trees_per_area', scheme='quantiles', k=5, legend=True, ax=axes[1])
axes[1].set_title('Quantiles')
axes[1].set_axis_off()

# Display maps
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/1-8.png?w=811)


# **3. Projecting and transforming geometries**
----------------------------------------------


## **3.1 Coordinate Reference Systems**



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/2-8.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/3-7.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/4-6.png?w=965)



## **3.1.1 Geographic vs projected coordinates**



 The CRS attribute stores the information about the Coordinate Reference System in which the data is represented. In this exercises, we will explore the CRS and the coordinates of the
 `districts`
 dataset about the districts of Paris.





```python

# Import the districts dataset
districts = geopandas.read_file("paris_districts.geojson")

# Print the CRS information
print(districts.crs)
# {'init': 'epsg:4326'}

# Print the first rows of the GeoDataFrame
print(districts.head())

```




```

   id           district_name  population                                           geometry
0   1  St-Germain-l'Auxerrois        1672  POLYGON ((2.344593389828428 48.85404991486192,...
1   2                  Halles        8984  POLYGON ((2.349365804803003 48.86057567227663,...
2   3            Palais-Royal        3195  POLYGON ((2.339465868602756 48.86213531210705,...
3   4           Place-Vendôme        3044  POLYGON ((2.331944969393234 48.86491285292422,...
4   5                 Gaillon        1345  POLYGON ((2.336320212305949 48.8679713890312, ...


```



 Indeed, this dataset is using geographic coordinates: longitude and latitude in degrees. We could see that the
 `crs`
 attribute referenced the EPSG:4326 (the code for WGS84, the most common used geographic coordinate system). A further rule of thumb is that the coordinates were in a small range (<180) and cannot be expressing meters.





---


## **3.2 Working with coordinate systems in GeoPandas**



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/5-6.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/6-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/7-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/8-3.png?w=755)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/9-3.png?w=978)



### **3.2.1 Projecting a GeoDataFrame**



 The Paris districts dataset is provided in geographical coordinates (longitude/latitude in WGS84). To see the result of naively using the data as is for plotting or doing calculations, we will first plot the data as is, and then plot a projected version.




 The standard projected CRS for France is the RGF93 / Lambert-93 reference system (referenced by the
 `EPSG:2154`
 number).




 GeoPandas and matplotlib have already been imported, and the districts dataset is read and assigned to the
 `districts`
 variable.





```python

# Print the CRS information
print(districts.crs)
# {'init': 'epsg:4326'}

# Plot the districts dataset
districts.plot()
plt.show()

# Convert the districts to the RGF93 reference system
districts_RGF93 = districts.to_crs(epsg=2154)

# Plot the districts dataset again
districts_RGF93.plot()
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/10-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/11-4.png?w=1024)




 The plot using longitude/latitude degrees distorted the shape of Paris quite a bit.



### **3.2.2 Projecting a Point**



 In the previous chapter, we worked with the Eiffel Tower location. Again, we provided you the coordinates in a projected coordinate system, so you could, for example, calculate distances. Let’s return to this iconic landmark, and express its location in geographical coordinates: 48°51′29.6″N, 2°17′40.2″E. Or, in decimals: latitude of 48.8584 and longitude of 2.2945.




 Shapely geometry objects have no notion of a CRS, and thus cannot be directly converted to another CRS. Therefore, we are going to use the GeoPandas to transform the Eiffel Tower point location to an alternative CRS. We will put the single point in a GeoSeries, use the
 `to_crs()`
 method, and extract the point again.





```python

# Construct a Point object for the Eiffel Tower
from shapely.geometry import Point
eiffel_tower = Point(2.2945, 48.8584)

# Put the point in a GeoSeries with the correct CRS
s_eiffel_tower = geopandas.GeoSeries([eiffel_tower], crs={'init': 'EPSG:4326'})

# Convert to other CRS
s_eiffel_tower_projected = s_eiffel_tower.to_crs(epsg=2154)

# Print the projected point
print(s_eiffel_tower_projected)

0    POINT (648237.3015492002 6862271.681553576)
dtype: object

```


####
 3.2.3
 **Calculating distance in a projected CRS**



 Now we have the Eiffel Tower location in a projected coordinate system, we can calculate the distance to other points.




 The final
 `s_eiffel_tower_projected`
 of the previous exercise containing the projected Point is already provided, and we extract the single point into the
 `eiffel_tower`
 variable. Further, the
 `restaurants`
 dataframe (using WGS84 coordinates) is also loaded.





```python

# Extract the single Point
eiffel_tower = s_eiffel_tower_projected[0]

# Ensure the restaurants use the same CRS
restaurants = restaurants.to_crs(s_eiffel_tower_projected.crs)

# The distance from each restaurant to the Eiffel Tower
dist_eiffel = restaurants.distance(eiffel_tower)

# The distance to the closest restaurant
print(min(dist_eiffel))
# 303.56255387894674

```



 Because our data was now in a projected coordinate reference system that used meters as unit, we know that the result of 303 is actually 303 meter.



### **3.2.4 Projecting to Web Mercator for using web tiles**



 In the first chapter, we did an exercise on plotting the restaurant locations in Paris and adding a background map to it using the
 `contextily`
 package.




 Currently,
 `contextily`
 assumes that your data is in the Web Mercator projection, the system used by most web tile services. And in that first exercise, we provided the data in the appropriate CRS so you didn’t need to care about this aspect.




 However, typically, your data will not come in Web Mercator (
 `EPSG:3857`
 ) and you will have to align them with web tiles on your own.




 GeoPandas, matplotlib and contextily are already imported.





```python

# Convert to the Web Mercator projection
restaurants_webmercator = restaurants.to_crs(epsg=3857)

# Plot the restaurants with a background map
ax = restaurants_webmercator.plot(markersize=1)
contextily.add_basemap(ax)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/1-9.png?w=1024)



---


## **3.3 Spatial operations: creating new geometries**



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/1-10.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/2-9.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/3-8.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/4-7.png?w=1024)



### **3.3.1 Exploring a Land Use dataset**



 For the following exercises, we first introduce a new dataset: a dataset about the land use of Paris (a simplified version based on the open European
 [Urban Atlas](https://land.copernicus.eu/local/urban-atlas)
 ). The land use indicates for what kind of activity a certain area is used, such as residential area or for recreation. It is a polygon dataset, with a label representing the land use class for different areas in Paris.




 In this exercise, we will read the data, explore it visually, and calculate the total area of the different classes of land use in the area of Paris.





```python

# Import the land use dataset
land_use = geopandas.read_file('paris_land_use.shp')
print(land_use.head())

# Make a plot of the land use with 'class' as the color
land_use.plot(column='class', legend=True, figsize=(15, 10))
plt.show()

# Add the area as a new column
land_use['area'] = land_use.area

# Calculate the total area for each land use class
total_area = land_use.groupby('class')['area'].sum() / 1000**2
print(total_area)

```




```

                       class                                           geometry
0               Water bodies  POLYGON ((3751386.280643055 2890064.32259039, ...
1  Roads and associated land  POLYGON ((3751390.345445618 2886000, 3751390.3...
2  Roads and associated land  POLYGON ((3751390.345445618 2886898.191588611,...
3  Roads and associated land  POLYGON ((3751390.345445618 2887500, 3751390.3...
4  Roads and associated land  POLYGON ((3751390.345445618 2888647.356784857,...

class
Continuous Urban Fabric             45.943090
Discontinuous Dense Urban Fabric     3.657343
Green urban areas                    9.858438
Industrial, commercial, public      13.295042
Railways and associated land         1.935793
Roads and associated land            7.401574
Sports and leisure facilities        3.578509
Water bodies                         3.189706
Name: area, dtype: float64

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/5-7.png?w=1024)

### **3.3.2 Intersection of two polygons**



 For this exercise, we are going to use 2 individual polygons: the district of Muette extracted from the
 `districts`
 dataset, and the green urban area of Boulogne, a large public park in the west of Paris, extracted from the
 `land_use`
 dataset. The two polygons have already been assigned to the
 `muette`
 and
 `park_boulogne`
 variables.




 We first visualize the two polygons. You will see that they overlap, but the park is not fully located in the district of Muette. Let’s determine the overlapping part.





```python

# Plot the two polygons
geopandas.GeoSeries([park_boulogne, muette]).plot(alpha=0.5, color=['green', 'blue'])
plt.show()

# Calculate the intersection of both polygons
intersection = park_boulogne.intersection(muette)

# Plot the intersection
geopandas.GeoSeries([intersection]).plot()
plt.show()

# Print proportion of district area that occupied park
print(intersection.area / muette.area)
# 0.4352082235641065

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/6-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/7-5.png?w=1024)



### **3.3.3 Intersecting a GeoDataFrame with a Polygon**



 Combining the land use dataset and the districts dataset, we can now investigate what the land use is in a certain district.




 For that, we first need to determine the intersection of the land use dataset with a given district. Let’s take again the
 *Muette*
 district as example case.




 The land use and districts datasets have already been imported as
 `land_use`
 and
 `districts`
 , and the Muette district has been extracted into the
 `muette`
 shapely polygon. Further, GeoPandas and matplotlib are imported.





```python

# Print the land use datset and Notre-Dame district polygon
print(land_use.head())
print(type(muette))

# Calculate the intersection of the land use polygons with Notre Dame
land_use_muette = land_use.geometry.intersection(muette)

# Plot the intersection
land_use_muette.plot(edgecolor='black')
plt.show()

# Print the first five rows of the intersection
print(land_use_muette.head())

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/8-4.png?w=1024)



```

                            class                                           geometry
0  Industrial, commercial, public  POLYGON ((3751385.614444552 2895114.54542058, ...
1                    Water bodies  POLYGON ((3751386.280643055 2890064.32259039, ...
2       Roads and associated land  POLYGON ((3751390.345445618 2886000, 3751390.3...
3       Roads and associated land  POLYGON ((3751390.345445618 2886898.191588611,...
4       Roads and associated land  POLYGON ((3751390.345445618 2887500, 3751390.3...

<class 'shapely.geometry.polygon.Polygon'>

0    ()
1    ()
2    ()
3    ()
4    ()
dtype: object

```




---


## **3.4 Overlaying spatial datasets**



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/9-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/10-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/11-5.png?w=1024)



### **3.4.1 Overlay of two polygon layers**



 Going back to the land use and districts datasets, we will now combine both datasets in an overlay operation. Create a new
 `GeoDataFrame`
 consisting of the intersection of the land use polygons wich each of the districts, but make sure to bring the attribute data from both source layers.





```python

# Print the first five rows of both datasets
print(land_use.head())
print(districts.head())

# Overlay both datasets based on the intersection
combined = geopandas.overlay(land_use, districts, how='intersection')

# Print the first five rows of the result
print(combined.head())

```




```

                            class                                           geometry
0  Industrial, commercial, public  POLYGON ((3751385.614444552 2895114.54542058, ...
1                    Water bodies  POLYGON ((3751386.280643055 2890064.32259039, ...
2       Roads and associated land  POLYGON ((3751390.345445618 2886000, 3751390.3...
3       Roads and associated land  POLYGON ((3751390.345445618 2886898.191588611,...
4       Roads and associated land  POLYGON ((3751390.345445618 2887500, 3751390.3...


   id           district_name  population                                           geometry
0   1  St-Germain-l'Auxerrois        1672  POLYGON ((3760188.134760949 2889260.456597198,...
1   2                  Halles        8984  POLYGON ((3760610.022313007 2889946.421907361,...
2   3            Palais-Royal        3195  POLYGON ((3759905.524344832 2890194.453753149,...
3   4           Place-Vendôme        3044  POLYGON ((3759388.396359455 2890559.229067303,...
4   5                 Gaillon        1345  POLYGON ((3759742.125111854 2890864.393745991,...



                       class  id district_name  population  \
0               Water bodies  61       Auteuil       67967
1    Continuous Urban Fabric  61       Auteuil       67967
2  Roads and associated land  61       Auteuil       67967
3          Green urban areas  61       Auteuil       67967
4  Roads and associated land  61       Auteuil       67967

                                            geometry
0  POLYGON ((3751395.345451574 2890118.001377039,...
1  (POLYGON ((3753253.104067317 2888254.529208081...
2  POLYGON ((3751519.830145844 2890061.508628568,...
3  (POLYGON ((3754314.716711559 2890283.121013219...
4  POLYGON ((3751619.112743544 2890500, 3751626.6...


```


### **3.4.2 Inspecting the overlay result**



 Now that we created the overlay of the land use and districts datasets, we can more easily inspect the land use for the different districts. Let’s get back to the example district of Muette, and inspect the land use of that district.




 GeoPandas and Matplotlib are already imported. The result of the
 `overlay()`
 function from the previous exercises is available as
 `combined`
 .





```python

# Print the first rows of the overlay result
print(combined.head())

# Add the area as a column
combined['area'] = combined.area

# Take a subset for the Muette district
land_use_muette = combined[combined.district_name=='Muette']

# Visualize the land use of the Muette district
land_use_muette.plot(column='class')
plt.show()

# Calculate the total area for each land use class
print(land_use_muette.groupby('class')['area'].sum() / 1000**2)

```




```

                       class  id district_name  population  \
0               Water bodies  61       Auteuil       67967
1    Continuous Urban Fabric  61       Auteuil       67967
2  Roads and associated land  61       Auteuil       67967
3          Green urban areas  61       Auteuil       67967
4  Roads and associated land  61       Auteuil       67967

                                            geometry
0  POLYGON ((3751395.345451574 2890118.001377039,...
1  (POLYGON ((3753253.104067317 2888254.529208081...
2  POLYGON ((3751519.830145844 2890061.508628568,...
3  (POLYGON ((3754314.716711559 2890283.121013219...
4  POLYGON ((3751619.112743544 2890500, 3751626.6...


class
Continuous Urban Fabric             1.275297
Discontinuous Dense Urban Fabric    0.088289
Green urban areas                   2.624229
Industrial, commercial, public      0.362990
Railways and associated land        0.005424
Roads and associated land           0.226271
Sports and leisure facilities       0.603989
Water bodies                        0.292194
Name: area, dtype: float64

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/12-3.png?w=1024)


# **4. Putting it all together – Artisanal mining sites case study**
-------------------------------------------------------------------


## **4.1 Introduction to the dataset**



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/13-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/14-1.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/15-1.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/16-1.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/17-1.png?w=1024)



### **4.1.1 Import and explore the data**



 In this exercise, we will start with reading and exploring two new datasets:



* First, a dataset on artisanal mining sites in Eastern Congo (adapted from
 [IPIS open data](http://ipisresearch.be/home/conflict-mapping/maps/open-data/)
 ).
* Second, a dataset on the national parks in Congo (adapted from the
 [World Resources Institute](https://www.wri.org/)
 ).



 For each of those datasets, the exercise consists of importing the necessary packages, reading the data with
 `geopandas.read_file()`
 , inspecting the first 5 rows and the Coordinate Reference System (CRS) of the data, and making a quick visualization.





```python

# Import GeoPandas and Matplotlib
import geopandas
import matplotlib.pyplot as plt

# Read the mining site data
mining_sites = geopandas.read_file('ipis_cod_mines.geojson')

# Print the first rows and the CRS information
print(mining_sites.head())
print(mining_sites.crs)

# Make a quick visualisation
mining_sites.plot()
plt.show()

```




```

   visit_date                                name  n_workers mineral                  geometry
0  2013-03-27                           Mayi-Tatu      150.0    Gold  POINT (29.66033 1.01089)
1  2013-03-27                             Mabanga      115.0    Gold  POINT (29.65862 1.00308)
2  2013-03-27                             Molende      130.0    Gold  POINT (29.65629 0.98563)
3  2013-03-27                          Embouchure      135.0    Gold  POINT (29.64494 0.99976)
4  2013-03-27  Apumu-Atandele-Jerusalem-Luka Yayo      270.0    Gold       POINT (29.66 0.956)
{'init': 'epsg:4326'}

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/18-1.png?w=444)



```python

# Import GeoPandas and Matplotlib
import geopandas
import matplotlib.pyplot as plt

# Read the mining site data
national_parks = geopandas.read_file("cod_conservation.shp")

# Print the first rows and the CRS information
print(national_parks.head())
print(national_parks.crs)

# Make a quick visualisation
national_parks.plot()
plt.show()

```




```

             Type                    Name                                           geometry
0  Nature Reserve  Luki Biosphere Reserve  POLYGON ((1469015.469222862 -605537.8418950802...
1  Nature Reserve  Itombwe Nature Reserve  POLYGON ((3132067.8539 -408115.0111999996, 313...
2  Nature Reserve    Okapi Faunal Reserve  POLYGON ((3197982.926399998 148235.506099999, ...
3   National park   Salonga National park  POLYGON ((2384337.1864 -280729.9739000015, 238...
4   National park   Salonga National park  POLYGON ((2399938.984200001 -152211.4943000004...
{'no_defs': True, 'y_0': 0, 'x_0': 0, 'proj': 'merc', 'units': 'm', 'datum': 'WGS84', 'lon_0': 0, 'lat_ts': 5}


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/1-11.png?w=1024)


 For the mining sites, it indicated EPSG:4326, so the dataset is expressed in geographical longitude/latitude. The last dataset, the national parks, is in projected coordinates. So we will need to make sure both datasets are in the same CRS to be able to use them together.



### **4.1.2 Convert to common CRS and save to a file**



 As we have seen in the previous exercises, both datasets are using a different Coordinate Reference System (CRS). This is also illustrated by the first plot in this exercise (for which the code is already provided in the script): both datasets are about the same region, so they should normally overlap in their coordinates; but they don’t.




 For further analyses in the rest of this chapter, we will convert both datasets to the same CRS, and save both to a new file. To ensure we can do distance-based calculations, we will convert them to a projected CRS: the local UTM zone 35, which is identified by EPSG:32735 (
 <https://epsg.io/32735>
 ).




 The mining sites (
 `mining_sites`
 ) and national parks (
 `national_parks`
 ) datasets are already loaded, and GeoPandas and matplotlib are imported.





```python

# Plot the natural parks and mining site data
ax = national_parks.plot()
mining_sites.plot(ax=ax, color='red')
plt.show()

# Convert both datasets to UTM projection
mining_sites_utm = mining_sites.to_crs(epsg=32735)
national_parks_utm = national_parks.to_crs(epsg=32735)

# Plot the converted data again
ax = national_parks_utm.plot()
mining_sites_utm.plot(ax=ax, color='red')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/2-10.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/1-12.png?w=1024)





```python

# Read the mining site data
mining_sites = geopandas.read_file("ipis_cod_mines.geojson")
national_parks = geopandas.read_file("cod_conservation.shp")

# Convert both datasets to UTM projection
mining_sites_utm = mining_sites.to_crs(epsg=32735)
national_parks_utm = national_parks.to_crs(epsg=32735)

# Write converted data to a file
mining_sites_utm.to_file('ipis_cod_mines_utm.gpkg', driver='GPKG')
national_parks_utm.to_file("cod_conservation_utm.shp", driver='ESRI Shapefile')

```


### **4.1.3 Styling a multi-layered plot**



 Now we have converted both datasets to the same Coordinate Reference System, let’s make a nicer plot combining the two.




 The datasets in UTM coordinates as we saved them to files in the last exercise are read back in and made available as the
 `mining_sites`
 and
 `national_parks`
 variables. GeoPandas and matplotlib are already imported.





```python

# Plot of the parks and mining sites
ax = national_parks.plot(color='green')
mining_sites.plot(ax=ax, markersize=5)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/3-9.png?w=1024)



```python

# Plot of the parks and mining sites
ax = national_parks.plot(color='green')
mining_sites.plot(ax=ax, markersize=5, alpha=0.5)
ax.set_axis_off()
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/4-8.png?w=1024)



```python

# Plot of the parks and mining sites
ax = national_parks.plot(color='green')
mining_sites.plot(ax=ax, column = 'mineral', markersize=5, legend=True)
ax.set_axis_off()
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/5-8.png?w=1024)



---


## **4.2 Additional spatial operations**



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/6-6.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/7-6.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/8-5.png?w=979)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/9-5.png?w=1024)



### **4.2.1 Buffer around a point**



 Consider the city of Goma, the capital of the North Kivu province of Congo, close to the border with Rwanda. Its coordinates are 1.66°S 29.22°E (the
 `Point`
 is already provided in UTM coordinates as the
 `goma`
 variable).




*How many mining sites are located within 50 km of Goma? And how much area of national park?*
 Let’s determine that using the buffer operation. Remember that distances should be expressed in the unit of the CRS (i.e. in meter in this case).




 Note: if you have a boolean Series (for example as result of a spatial relationship method), then you can calculate how many
 `True`
 values (ie. how many geometries passed the check) by taking the sum of those booleans because in that case the
 `True`
 and
 `False`
 values will be seen as ones and zeros.





```python

# goma is a Point
print(type(goma))

# Create a buffer of 50km around Goma
goma_buffer = goma.buffer(50000)

# The buffer is a polygon
print(type(goma_buffer))

# Check how many sites are located within the buffer
mask = mining_sites.within(goma_buffer)
print(mask.sum())

# Calculate the area of national park within the buffer
print(national_parks.intersection(goma_buffer).area.sum() / (1000**2))

```


### **4.2.2 Mining sites within national parks**



 For this exercise, let’s start with one of the national parks, the Kahuzi-Biega National park (which was extracted from the
 `national_parks`
 dataset and is provided as the
 `kahuzi`
 variable).




*Which of the mining sites are located within this national park?*




 And as a second step:
 *can we determine all mining sites that are located within one of the national parks and in which park?*




 The mining sites (
 `mining_sites`
 ) and national parks (
 `national_parks`
 ) datasets are already loaded, and GeoPandas is already imported.





```python

# Extract the single polygon for the Kahuzi-Biega National park
kahuzi = national_parks[national_parks['Name'] == "Kahuzi-Biega National park"].geometry.squeeze()

# Take a subset of the mining sites located within Kahuzi
sites_kahuzi = mining_sites[mining_sites.within(kahuzi)]
print(sites_kahuzi)

# Determine in which national park a mining site is located
sites_within_park = geopandas.sjoin(mining_sites, national_parks, op='within', how='inner')
print(sites_within_park.head())

# The number of mining sites in each national park
print(sites_within_park['name'].value_counts())

```




```

       visit_date                   name  n_workers      mineral  \
661   2013-08-28Z          Ibozia/Kalumé       80.0  Cassiterite
662   2013-08-26Z                Matamba      150.0  Cassiterite
663   2013-08-27Z          Mutete/Mukina      170.0  Cassiterite
664   2013-08-28Z                 Mutete      100.0  Cassiterite
760   2014-02-25Z              Mazankala      120.0  Cassiterite
813   2015-07-28Z             Kitendebwa       50.0         Gold
869   2013-09-28Z           Sebwa-Lukoma      130.0  Cassiterite
870   2013-10-30Z              Rwamakaza      160.0  Cassiterite
1481  2009-01-01Z               Mugaba I       50.0         Gold
1482  2009-01-01Z           Mugaba Ouest       46.0         Gold
1676  2015-08-02Z  Nguba(Nkuba) kamisoke      122.0  Cassiterite

                                         geometry
661   POINT (567832.7086093378 9759143.339360647)
662   POINT (598323.5389475008 9758688.142411157)
663   POINT (570733.4369126211 9761871.114227083)
664   POINT (569881.0930415759 9762219.110778008)
760   POINT (613075.5326777868 9722956.979837928)
813   POINT (693078.9282059025 9770107.517721133)
869   POINT (660406.3452248175 9715261.717041001)
870    POINT (661266.834456568 9716072.198784607)
1481  POINT (685167.3714990132 9744069.967416598)
1482  POINT (683156.6865782175 9746324.416321497)
1676  POINT (622151.3489110788 9808363.111073116)


      visit_date          name  n_workers mineral                                     geometry  \
253  2013-09-05Z  Kiviri/Tayna      244.0    Gold   POINT (709734.912568812 9961013.720415946)
578  2015-09-02Z   Lubondozi 3       30.0    Gold  POINT (578464.3150203574 9555456.293453641)
579  2015-09-02Z        Katamu      180.0    Gold  POINT (576249.9033853477 9554313.725408439)
580  2015-09-02Z     Kimabwe 1      120.0    Gold  POINT (576425.7766608761 9556329.633628448)
581  2015-09-02Z   Lubondozi 1      300.0    Gold   POINT (579164.711161439 9554722.924142597)

     index_right            Type                       Name
253           23  Nature Reserve       Tayna Nature Reserve
578           15  Hunting Domain  Luama-Kivu Hunting Domain
579           15  Hunting Domain  Luama-Kivu Hunting Domain
580           15  Hunting Domain  Luama-Kivu Hunting Domain
581           15  Hunting Domain  Luama-Kivu Hunting Domain

Colline 7           1
Mutete              1
                   ..
Kimabwe 1           1
Muchacha            1
Name: name, Length: 64, dtype: int64

```




---


## **4.3 Applying custom spatial operations**


####
 4.3.1
 **Finding the name of the closest National Park**



 Let’s start with a custom query for a single mining site. Here, we will determine the name of the national park that is the closest to the specific mining site.




 The datasets on the mining sites (
 `mining_sites`
 ) and national parks (
 `national_parks`
 ) are already loaded.





```

mining_sites.head(1)
    visit_date       name  n_workers mineral                                     geometry
0  2013-03-27Z  Mayi-Tatu      150.0    Gold  POINT (796089.4159891906 10111855.17426374)


national_parks.head(1)
             Type                    Name                                           geometry
0  Nature Reserve  Luki Biosphere Reserve  POLYGON ((-1038121.47250213 9375412.18990065, ...

```




```python

# Get the geometry of the first row
single_mine = mining_sites.geometry[0]

# Calculate the distance from each national park to this mine
dist = national_parks.distance(single_mine)

# The index of the minimal distance
idx = dist.idxmin()

# Access the name of the corresponding national park
closest_park = national_parks.loc[idx, 'Name']
print(closest_park)

```


### **4.3.2 Applying a custom operation to each geometry**



 Now we know how to get the closest national park for a single point, let’s do this for all points. For this, we are first going to write a function, taking a single point as argument and returning the desired result. Then we can use this function to apply it to all points.




 The datasets on the mining sites (
 `mining_sites`
 ) and national parks (
 `national_parks`
 ) are already loaded. The single mining site from the previous exercises is already defined as
 `single_mine`
 .





```python

# Define a function that returns the closest national park
def closest_national_park(geom, national_parks):
    dist = national_parks.distance(geom)
    idx = dist.idxmin()
    closest_park = national_parks.loc[idx, 'Name']
    return closest_park

# Call the function on single_mine
print(closest_national_park(single_mine, national_parks))

# Apply the function to all mining sites
mining_sites['closest_park'] = mining_sites.geometry.apply(closest_national_park, national_parks=national_parks)
print(mining_sites.head())

```




```

Virunga National park


    visit_date                                name  n_workers mineral  \
0  2013-03-27Z                           Mayi-Tatu      150.0    Gold
1  2013-03-27Z                             Mabanga      115.0    Gold
2  2013-03-27Z                             Molende      130.0    Gold
3  2013-03-27Z                          Embouchure      135.0    Gold
4  2013-03-27Z  Apumu-Atandele-Jerusalem-Luka Yayo      270.0    Gold

                                      geometry           closest_park
0  POINT (796089.4159891906 10111855.17426374)  Virunga National park
1  POINT (795899.6640655082 10110990.83998195)  Virunga National park
2  POINT (795641.7066578076 10109059.78659637)  Virunga National park
3  POINT (794376.3093052682 10110622.24995522)  Virunga National park
4  POINT (796057.5042468573 10105781.54751797)  Virunga National park

```




---


## **4.4 Working with raster data**



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/1.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/2.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/4.png?w=1024)



### **4.4.1 Import and plot raster data**



 In this exercise, we are going to use a raster dataset of the vegetation types map (available from
 [http://www.wri.org](http://www.wri.org/)
 ). The raster values take a set of discrete values indicating the type of vegetation. Let’s start with reading the data and plotting it together with the mining site data.




 The mining sites dataset (
 `mining_sites`
 ) is already loaded, and GeoPandas and matplotlib are already imported.





```python

# Import the rasterio package
import rasterio

# Open the raster dataset
src = rasterio.open("central_africa_vegetation_map_foraf.tif")

# Import the plotting functionality of rasterio
import rasterio.plot

# Plot the raster layer with the mining sites
ax = rasterio.plot.show(src)
mining_sites.plot(ax=ax, color='red', markersize=1)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/5.png?w=877)

### **4.4.2 Extract information from raster layer**



 Let’s now extract information from the raster layer, based on a vector file. This functionality is provided by the
 [`rasterstats`](https://github.com/perrygeo/python-rasterstats)
 package. Specifically for this exercise, we will determine the vegetation type at all mining sites, by getting the nearest raster pixel value at each point of the mining site dataset.




 A subset of the mining sites dataset (
 `mining_sites`
 ) is already loaded, and GeoPandas and matplotlib are already imported.





```python

# Import the rasterstats package
import rasterstats

# Extract the nearest value in the raster for all mining sites
vegetation_raster = "central_africa_vegetation_map_foraf.tif"
mining_sites['vegetation'] = rasterstats.point_query(mining_sites.geometry, vegetation_raster, interpolate='nearest')
print(mining_sites.head())

# Replace numeric vegation types codes with description
mining_sites['vegetation'] = mining_sites['vegetation'].replace(vegetation_types)

# Make a plot indicating the vegetation type
mining_sites.plot(column='vegetation', legend=True)
plt.show()

```




```

          visit_date         name  n_workers      mineral                     geometry  vegetation
    350   2013-05-30       Kunguo      154.0         Gold      POINT (28.274 -1.00941)           1
    2056  2017-08-10      Masange        4.0   Wolframite   POINT (27.329875 -1.08929)           1
    686   2013-09-09  Kabusangala      120.0         Gold  POINT (27.136147 -3.425685)           1
    602   2013-12-23     Simunofu      120.0  Cassiterite  POINT (26.541903 -1.540585)           1
    571   2013-09-07       Kigali       26.0  Cassiterite  POINT (26.601226 -1.398051)           7

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/6.png?w=1024)

### **4.4.3 Further reference**



![Desktop View]({{ site.baseurl }}/assets/datacamp/working-with-geospatial-data-in-python/7.png?w=1024)



---



 Thank you for reading and hope you’ve learned a lot.



