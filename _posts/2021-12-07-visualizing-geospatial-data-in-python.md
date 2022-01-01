---
title: Visualizing Geospatial Data in Python
date: 2021-12-07 11:22:12 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Visualizing Geospatial Data in Python
========================================







 This is the memo of the 5th course (5 courses in all) of ‘Data Visualization with Python’ skill track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/visualizing-geospatial-data-in-python)**
 .



###
**Course Description**



 One of the most important tasks of a data scientist is to understand the relationships between their data’s physical location and their geographical context. In this course you’ll be learning to make attractive visualizations of geospatial data with the GeoPandas package. You will learn to spatially join datasets, linking data to context. Finally you will learn to overlay geospatial data to maps to add even more spatial cues to your work. You will use several datasets from the City of Nashville’s open data portal to find out where the chickens are in Nashville, which neighborhood has the most public art, and more!



###
**Table of contents**


1. Building 2-layer maps : combining polygons and scatterplots
2. [Creating and joining GeoDataFrames](https://datascience103579984.wordpress.com/2019/11/08/visualizing-geospatial-data-in-python-from-datacamp/2/)
3. [GeoSeries and folium](https://datascience103579984.wordpress.com/2019/11/08/visualizing-geospatial-data-in-python-from-datacamp/3/)
4. [Creating a choropleth building permit density in Nashville](https://datascience103579984.wordpress.com/2019/11/08/visualizing-geospatial-data-in-python-from-datacamp/4/)





# **1. Building 2-layer maps : combining polygons and scatterplots**
-------------------------------------------------------------------


## **1.1 Introduction**



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/6.png?w=838)

### **1.1.1 Plotting a scatterplot from longitude and latitude**



 When using latitude and longitude to create a scatterplot, which value is plotted along the horizontal axis (as x)?




 Longitude is plotted as x on the horizontal axis.



### **1.1.2 Styling a scatterplot**



 In this exercise, you’ll be using
 `plt.scatter()`
 to plot the father and son height data from the video. The
 `father_son`
 DataFrame is available in your workspace. In each scatterplot, plot
 `father_son.fheight`
 as x-axis and
 `father_son.sheight`
 as y-axis.





```python

# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Scatterplot 3
plt.scatter(father_son.fheight, father_son.sheight,  c = 'yellow', edgecolor = 'darkblue')
plt.show()
plt.xlabel('father height (inches)')
plt.ylabel('son height (inches)')
plt.title('Son Height as a Function of Father Height')

plt.grid()

# Show your plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/7.png?w=1024)

### **1.1.2 Extracting longitude and latitude**



 A DataFrame named
 `df`
 has been loaded to your workspace. Complete the code to extract longitude and latitude to new,
 *separate*
 columns.





```

df.head()
   StopID                             StopName                 Location
0    4431      MUSIC CITY CENTRAL 5TH - BAY 11   (36.16659, -86.781996)
1     588         CHARLOTTE AVE & 7TH AVE N WB      (36.165, -86.78406)
2     590         CHARLOTTE AVE & 8TH AVE N WB  (36.164393, -86.785451)
3     541  11TH AVE / N GULCH STATION OUTBOUND  (36.162249, -86.790464)

type(df.Location[0])
tuple

```




```python

# extract latitude to a new column: lat
df['lat'] = [loc[0] for loc in df.Location]

# extract longitude to a new column: lng
df['lng'] = [loc[1] for loc in df.Location]

# print the first few rows of df again
print(df.head())

```




```

   StopID                             StopName                 Location        lat        lng
0    4431      MUSIC CITY CENTRAL 5TH - BAY 11   (36.16659, -86.781996)  36.166590 -86.781996
1     588         CHARLOTTE AVE & 7TH AVE N WB      (36.165, -86.78406)  36.165000 -86.784060
2     590         CHARLOTTE AVE & 8TH AVE N WB  (36.164393, -86.785451)  36.164393 -86.785451
3     541  11TH AVE / N GULCH STATION OUTBOUND  (36.162249, -86.790464)  36.162249 -86.790464

```


####
**Plotting chicken locations**



 Now you will create a scatterplot that shows where the Nashville chickens are!





```python

# Import pandas and matplotlib.pyplot using their customary aliases
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
chickens = pd.read_csv(chickens_path)

# Look at the first few rows of the chickens DataFrame
print(chickens.head())

# Plot the locations of all Nashville chicken permits
plt.scatter(x = chickens.lng, y = chickens.lat)

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/8.png?w=1024)



---


## **1.2 Geometries and shapefiles**



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/9.png?w=919)
![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/10.png?w=886)



### **1.2.1 Creating a GeoDataFrame & examining the geometry**



 Let’s see where service districts are in Nashville. The path to the service district shapefile has been stored in the variable
 `shapefile_path`
 .





```python

# Import geopandas
import geopandas as gpd

# Read in the services district shapefile and look at the first few rows.
service_district = gpd.read_file(shapefile_path)
print(service_district.head())

# Print the contents of the service districts geometry in the first row
print(service_district.loc[0, 'geometry'])

```




```

       area_sq_mi  objectid                       name                                           geometry
    0       198.0       0.0    Urban Services District  POLYGON ((-86.68680500011935 36.28670500013504...
    1       327.0       4.0  General Services District  (POLYGON ((-86.56776164301485 36.0342383159728...
    POLYGON ((-86.68680500011935 36.28670500013504, -86.68706099969657 36.28550299967364, -86.68709498823965 36.28511683351293, -86.68712691935902 36.28475404474551, -86.6871549990252 36.28443499969863, -86.68715025108719 36.28438104319917, -86.68708600011215 36.2836510002216, ...

```


### **1.2.2 Plotting shapefile polygons**



 The next step is to show the map of polygons. We have imported
 `matplotlib.pyplot`
 as
 `plt`
 and
 `geopandas`
 as
 `gpd`
 , A GeoDataFrame of the service districts called
 `service_district`
 is in your workspace.





```python

# Import packages
import geopandas as gpd
import matplotlib.pyplot as plt

# Plot the Service Districts without any additional arguments
service_district.head()
service_district.plot()
plt.show()


# Plot the Service Districts, color them according to name, and show a legend
service_district.plot(column = 'name', legend = True)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/11.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/12.png?w=1024)





---


## **1.3 Scatterplots over polygons**


### **1.3.1 Geometry**



 Geometry is a special type of data structure. What types of geometries might be stored in a geometry field?




**lines, points, and polygons**



### **1.3.2 Plotting points over polygons – part 1**



 Make a basic plot of the service districts with the chicken locations. The packages needed have already been imported for you. The
 `chickens`
 DataFrame and
 `service_district`
 GeoDataFrame are in your workspace.





```python

# Plot the service district shapefile
service_district.plot(column='name')

# Add the chicken locations
plt.scatter(x=chickens.lng, y=chickens.lat, c = 'black')

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/13.png?w=1024)

####
 1.3.3 Plotting points over polygons – part 2



 We have loaded the usual libraries as
 `pd`
 ,
 `plt`
 , and
 `gpd`
 , the chickens dataset as
 `chickens`
 , and the service districts as
 `service_district`
 . Plot the service districts and chicken permits together to see what story your visualization tells.





```python

# Plot the service district shapefile
service_district.plot(column='name', legend=True)

# Add the chicken locations
plt.scatter(x=chickens.lng, y=chickens.lat, c='black', edgecolor = 'white')


# Add labels and title
plt.title('Nashville Chicken Permits')
plt.xlabel('longitude')
plt.ylabel('latitude')

# Add grid lines and show the plot
plt.grid()
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/14.png?w=1024)




# **2. Creating and joining GeoDataFrames**
------------------------------------------


## **2.1 GeoJSON and plotting with geopandas**



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/15.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/16.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/17.png?w=859)
![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/18.png?w=1024)



### **2.1.1 Working with GeoJSON**



 The advantage of GeoJSON over shapefiles is:



* The file is human readable, so you can open it in a text editor and understand the contents.
* The file stands alone and doesn’t rely on other files.
* GeoJSON supports multi-part geometries.


### **2.1.2 Colormaps**



 When you want to differentiate regions, but not imply any type of relationship between the regions, a qualitative colormap is the best choice. In this exercise you’ll compare a qualitative colormap to a sequential (quantitative) colormap using the school districts GeoDataFrame. It is available in your workspace as
 `school_districts`
 .





```

school_districts.head()
   first_name       city    zip                           email state  ...    position term_expir district         phone                                           geometry
0  Dr. Sharon  Nashville  37218  gentryfordistrict1@comcast.net    TN  ...      Member       2016        1  615-268-5269  (POLYGON ((-86.77136400034288 36.3835669997190...
1        Jill    Madison  37115          jill.speering@mnps.org    TN  ...  Vice-Chair       2016        3  615-562-5234  (POLYGON ((-86.75364713283636 36.4042760799855...
2  Dr. Jo Ann  Nashville  37220          joann.brannon@mnps.org    TN  ...      Member       2018        2  615-833-5976  (POLYGON ((-86.76696199971282 36.0833250002130...
3        Anna  Hermitage  37076          anna.shepherd@mnps.org    TN  ...       Chair       2018        4  615-210-3768  (POLYGON ((-86.5809831462547 36.20934685360503...
4         Amy  Nashville  37221             amy.frogge@mnps.org    TN  ...      Member       2016        9  615-521-5650  (POLYGON ((-86.97287099971373 36.2082789997189...

[5 rows x 12 columns]


```




```python

# Set legend style
lgnd_kwds = {'title': 'School Districts',
               'loc': 'upper left', 'bbox_to_anchor': (1, 1.03), 'ncol': 1}

# Plot the school districts using the tab20 colormap (qualitative)
school_districts.plot(column = 'district', cmap = 'tab20', legend = True, legend_kwds  = lgnd_kwds)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Nashville School Districts')
plt.show();

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/19.png?w=1024)



```python

# Set legend style
lgnd_kwds = {'title': 'School Districts',
               'loc': 'upper left', 'bbox_to_anchor': (1, 1.03), 'ncol': 1}

# Plot the school districts using the summer colormap (sequential)
school_districts.plot(column = 'district', cmap = 'summer', legend = True, legend_kwds = lgnd_kwds)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Nashville School Districts')
plt.show();

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/20.png?w=1024)



```python

# Set legend style
lgnd_kwds = {'title': 'School Districts',
               'loc': 'upper left', 'bbox_to_anchor': (1, 1.03), 'ncol': 1}

# Plot the school districts using Set3 colormap without the column argument
school_districts.plot(cmap = 'Set3', legend = True, legend_kwds = lgnd_kwds)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Nashville School Districts')
plt.show();

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/21.png?w=1024)

 There is no legend when the column argument is not supplied even if you set legend to
 `True`
 !


### **2.1.3 Map Nashville neighborhoods**



 This time you’ll read a GeoJSON file in to a GeoDataFrame to take a quick peek at where Nashville neighborhoods are.





```

                   name                                           geometry
0  Historic Buena Vista  (POLYGON ((-86.79511056795417 36.1757596496334...
1        Charlotte Park  (POLYGON ((-86.87459668651866 36.1575770268129...
2              Hillwood  (POLYGON ((-86.87613708067906 36.1355409894979...
3            West Meade  (POLYGON ((-86.9038380396094 36.1255414807897,...
4          White Bridge  (POLYGON ((-86.86321427797685 36.1288622289404...


```




```

import geopandas as gpd
import matplotlib.pyplot as plt

# Read in the neighborhoods geojson file
neighborhoods = gpd.read_file(neighborhoods_path)

# Print the first few rows of neighborhoods
print(neighborhoods.head())

# Plot the neighborhoods, color according to name and use the Dark2 colormap
neighborhoods.plot(column = 'name', cmap = 'Dark2')

# Show the plot.
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/22.png?w=1024)



---


###
 2.2 Projections and coordinate reference systems



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/27.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/23.png?w=902)
![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/24.png?w=909)
![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/25.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/26.png?w=962)



### **2.2.1 Changing coordinate reference systems**



 In this exercise you will learn how to find a GeoDataFrame’s coordinate reference system and how to change it. The school districts GeoDataFrame is available in your workspace as
 `school_districts`
 .





```python

# Print the first row of school districts GeoDataFrame and the crs
print(school_districts.head(1))
print(school_districts.crs)

# Convert the crs to epsg:3857
school_districts.geometry = school_districts.geometry.to_crs(epsg = 3857)

# Print the first row of school districts GeoDataFrame and the crs again
print(school_districts.head(1))
print(school_districts.crs)

```




```

       first_name       city    zip                           email state  ... position term_expir district         phone                                           geometry
    0  Dr. Sharon  Nashville  37218  gentryfordistrict1@comcast.net    TN  ...   Member       2016        1  615-268-5269  (POLYGON ((-86.77136400034288 36.3835669997190...

    [1 rows x 12 columns]
    {'init': 'epsg:4326'}

       first_name       city    zip                           email state  ... position term_expir district         phone                                           geometry
    0  Dr. Sharon  Nashville  37218  gentryfordistrict1@comcast.net    TN  ...   Member       2016        1  615-268-5269  (POLYGON ((-9659344.055955959 4353528.76657080...

    [1 rows x 12 columns]
    {'init': 'epsg:3857', 'no_defs': True}

```



 You can change the coordinate reference system of a GeoDataFrame by changing the crs property of the GeoDataFrame. Notice that the units for geometry change when you change the CRS. You always need to ensure two GeoDataFrames share the same crs before you spatially join them.



### **2.2.2 Construct a GeoDataFrame from a DataFrame**



 In this exercise, you will construct a
 `geopandas`
 GeoDataFrame from the Nashville Public Art DataFrame. You will need to import the
 `Point`
 constructor from the
 `shapely.geometry`
 module to create a geometry column in
 `art`
 before you can create a GeoDataFrame from
 `art`
 . This will get you ready to spatially join the art data and the neighborhoods data in order to discover which neighborhood has the most art.




 The Nashville Public Art data has been loaded for you as
 `art`
 .





```

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Print the first few rows of the art DataFrame
print(art.head())

# Create a geometry column from lng & lat
art['geometry'] = art.apply(lambda x: Point(float(x.lng), float(x.lat)), axis=1)

# Create a GeoDataFrame from art and verify the type
art_geo = gpd.GeoDataFrame(art, crs = neighborhoods.crs, geometry = art.geometry)
print(type(art_geo))


```




```

                                 title                                 last_name                 first_name                            address                                             medium  \
    0          [Cross Country Runners]                                     Frost                      Miley     4001 Harding Rd., Nashville TN                                             Bronze
    1  [Fourth and Commerce Sculpture]                                    Walker                        Lin  333 Commerce Street, Nashville TN                                                NaN
    2              12th & Porter Mural                                   Kennedy                        Kim                  114 12th Avenue N                   Porter all-weather outdoor paint
    3                A Splash of Color  Stevenson and Stanley and ROFF (Harroff)  Doug and Ronnica and Lynn                   616 17th Ave. N.  Steel, brick, wood, and fabric on frostproof c...
    4             A Story of Nashville                                    Ridley                       Greg    615 Church Street, Nashville TN                           Hammered copper repousse

            type                                               desc       lat       lng                    loc
    0  Sculpture                                                NaN  36.12856 -86.83660   (36.12856, -86.8366)
    1  Sculpture                                                NaN  36.16234 -86.77774  (36.16234, -86.77774)
    2      Mural  Kim Kennedy is a musician and visual artist wh...  36.15790 -86.78817   (36.1579, -86.78817)
    3      Mural  Painted wooden hoop dancer on a twenty foot po...  36.16202 -86.79975  (36.16202, -86.79975)
    4     Frieze  Inside the Grand Reading Room, this is a serie...  36.16215 -86.78205  (36.16215, -86.78205)
    <class 'geopandas.geodataframe.GeoDataFrame'>

```



 Now that the public art data is in a GeoDataFrame we can join it to the neighborhoods with a special kind of join called a spatial join.





---


## **2.3 Spatial joins**



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/1-1.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/2-1.png?w=783)
![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/3-1.png?w=859)
![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/4-1.png?w=928)
![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/5.png?w=942)
![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/6-1.png?w=922)
![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/7-1.png?w=948)



### **2.3.1 Spatial join practice**



 Is there a difference between art (point data) that
 ***intersects***
 with neighborhoods (polygon data) and art (point data)
 ***within***
 neighborhoods (polygon data)? Explore different spatial joins with the
 `art_geo`
 and
 `neighborhoods`
 GeoDataFrames, which are available in your workspace.





```python

# Spatially join art_geo and neighborhoods
art_intersect_neighborhoods = gpd.sjoin(art_geo, neighborhoods, op = 'intersects')

# Print the shape property of art_intersect_neighborhoods
print(art_intersect_neighborhoods.shape)
# (40, 13)


art_geo.head(1)
                     title last_name first_name                         address  medium  ... desc       lat      lng                   loc                   geometry
0  [Cross Country Runners]     Frost      Miley  4001 Harding Rd., Nashville TN  Bronze  ...  NaN  36.12856 -86.8366  (36.12856, -86.8366)  POINT (-86.8366 36.12856)

[1 rows x 11 columns]


neighborhoods.head(1)
                   name                                           geometry
0  Historic Buena Vista  (POLYGON ((-86.79511056795417 36.1757596496334...


art_intersect_neighborhoods.head(1)
                             title last_name first_name                            address medium  ...       lng                    loc                             geometry  index_right  \
1  [Fourth and Commerce Sculpture]    Walker        Lin  333 Commerce Street, Nashville TN    NaN  ... -86.77774  (36.16234, -86.77774)  POINT (-86.77774000000001 36.16234)           41

              name
1  Urban Residents

[1 rows x 13 columns]

```




```python

# Create art_within_neighborhoods by spatially joining art_geo and neighborhoods
art_within_neighborhoods = gpd.sjoin(art_geo, neighborhoods, op = 'within')

# Print the shape property of art_within_neighborhoods
print(art_within_neighborhoods.shape)
# (40, 13)

```




```python

# Spatially join art_geo and neighborhoods and using the contains op
art_containing_neighborhoods = gpd.sjoin(art_geo, neighborhoods, op = 'contains')

# Print the shape property of art_containing_neighborhoods
print(art_containing_neighborhoods.shape)
# (0, 13)

```


### **2.3.2 Finding the neighborhood with the most public art**



 Now that you have created
 `art_geo`
 , a GeoDataFrame, from the
 `art`
 DataFrame, you can join it
 *spatially*
 to the
 `neighborhoods`
 data to see what art is in each neighborhood.





```python

# import packages
import geopandas as gpd
import pandas as pd

# Spatially join neighborhoods with art_geo
neighborhood_art = gpd.sjoin(art_geo, neighborhoods, op = "within")

# Print the first few rows
print(neighborhood_art.head())


neighborhood_art.head(1)
                             title last_name first_name                            address medium  ...       lng                    loc                             geometry  index_right  \
1  [Fourth and Commerce Sculpture]    Walker        Lin  333 Commerce Street, Nashville TN    NaN  ... -86.77774  (36.16234, -86.77774)  POINT (-86.77774000000001 36.16234)           41

              name
1  Urban Residents

[1 rows x 13 columns]

```


### **2.3.3 Aggregating points within polygons**



 Now that you have spatially joined
 `art`
 and
 `neighborhoods`
 , you can group, aggregate, and sort the data to find which neighborhood has the most public art. You can count artwork titles to see how many artworks are in each neighborhood.





```python

# Get name and title from neighborhood_art and group by name
neighborhood_art_grouped = neighborhood_art[['name', 'title']].groupby('name')

# Aggregate the grouped data and count the artworks within each polygon
print(neighborhood_art_grouped.agg('count').sort_values(by = 'title', ascending = False))

```




```

                          title
name
Urban Residents              22
Lockeland Springs             3
Edgehill (ONE)                2
Germantown                    2
Hillsboro-West End            2
Inglewood                     2
Sunnyside                     2
Chestnut Hill (TAG)           1
Historic Edgefield            1
McFerrin Park                 1
Renraw                        1
Wedgewood Houston (SNAP)      1

```


### **2.3.4 Plotting the Urban Residents neighborhood and art**



 Now you know that most art is in the
 **Urban Residents**
 neighborhood. In this exercise, you’ll create a plot of art in that neighborhood. First you will subset just the
 `urban_art`
 from
 `neighborhood_art`
 and you’ll subset the
 `urban_polygon`
 from
 `neighborhoods`
 . Then you will create a plot of the polygon as
 `ax`
 before adding a plot of the art.





```python

# Create urban_art from neighborhood_art where the neighborhood name is Urban Residents
urban_art = neighborhood_art.loc[neighborhood_art.name == 'Urban Residents']

# Get just the Urban Residents neighborhood polygon and save it as urban_polygon
urban_polygon = neighborhoods.loc[neighborhoods.name == "Urban Residents"]

# Plot the urban_polygon as ax
ax = urban_polygon.plot(color = 'lightgreen')

# Add a plot of the urban_art and show it
urban_art.plot( ax = ax, column = 'type', legend = True);
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/11-1.png?w=1024)




# 3. **GeoSeries and folium**



## 3.1 GeoSeries attributes and methods I



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/1-2.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/2-2.png?w=860)



### **3.1.1 Find the area of the Urban Residents neighborhood**



 How big is the Urban Residents neighborhood?





```python

# Print the head of the urban polygon
print(urban_polygon.head())

# Create a copy of the urban_polygon using EPSG:3857 and print the head
urban_poly_3857 = urban_polygon.to_crs(epsg = 3857)
print(urban_poly_3857.head())

# Print the area of urban_poly_3857 in kilometers squared
area = urban_poly_3857.geometry.area / 10**6
print('The area of the Urban Residents neighborhood is ', area[0], ' km squared')

'''
   index             name                                           geometry
0     41  Urban Residents  (POLYGON ((-86.78122053774267 36.1645653773768...
   index             name                                           geometry
0     41  Urban Residents  (POLYGON ((-9660441.280680289 4323289.00479539...
The area of the Urban Residents neighborhood is  1.1289896057984288  km squared
'''

```




---


## **3.2 GeoSeries attributes and methods II**



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/3-2.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/4-2.png?w=1024)



### **3.2.1 The center of the Urban Residents neighborhood**



 Now you’ll find the center point of the
 `urban_poly_3857`
 and plot it over the polygon.





```python

# Create downtown_center from urban_poly_3857
downtown_center = urban_poly_3857.geometry.centroid

# Print the type of downtown_center
print(type(downtown_center))

# Plot the urban_poly_3857 as ax and add the center point
ax = urban_poly_3857.plot(color = 'lightgreen')
downtown_center.plot(ax = ax, color = 'black')
plt.xticks(rotation = 45)

# Show the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/5-1.png?w=1024)

### **3.2.2 Prepare to calculate distances**



 In this exercise you will prepare a GeoDataFrame called
 `art_dist_meters`
 with the locations of downtown art converted to meters using EPSG:3857. You will use
 `art_dist_meters`
 in the next exercise to calculate the distance of each artwork from the center of the Urban Residents neighborhood in meters.




 The
 `art`
 data is in your workspace, along with
 `urban_poly_3857`
 and
 `center_point`
 , the center point of the Urban Residents neighborhood. A geometry column called
 `geometry`
 that uses degrees has already been created in the
 `art`
 DataFrame.





```

art.columns
Index(['title', 'last_name', 'first_name', 'address', 'medium', 'type', 'desc', 'lat', 'lng', 'loc', 'geometry', 'center'], dtype='object')

type(art)
pandas.core.frame.DataFrame

art.head(1)
                     title last_name first_name                         address  medium  ...       lat      lng                   loc                                     geometry  \
0  [Cross Country Runners]     Frost      Miley  4001 Harding Rd., Nashville TN  Bronze  ...  36.12856 -86.8366  (36.12856, -86.8366)  POINT (-9666606.09421918 4318325.479300267)

                                         center
0  POINT (-9660034.312198792 4322835.782813124)

[1 rows x 12 columns]

```




```python

# Import packages
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd

# Create art_dist_meters using art and the geometry from art
art_dist_meters = gpd.GeoDataFrame(art, geometry = art.geometry, crs = {'init': 'epsg:4326'})
print(art_dist_meters.head(2))

# Set the crs of art_dist_meters to use EPSG:3857
art_dist_meters.geometry = art_dist_meters.geometry.to_crs(epsg = 3857)
print(art_dist_meters.head(2))

# Add a column to art_meters, center
art_dist_meters['center'] = center_point

```




```

                             title last_name first_name                            address  medium  ... desc       lat       lng                    loc                             geometry
0          [Cross Country Runners]     Frost      Miley     4001 Harding Rd., Nashville TN  Bronze  ...  NaN  36.12856 -86.83660   (36.12856, -86.8366)            POINT (-86.8366 36.12856)
1  [Fourth and Commerce Sculpture]    Walker        Lin  333 Commerce Street, Nashville TN     NaN  ...  NaN  36.16234 -86.77774  (36.16234, -86.77774)  POINT (-86.77774000000001 36.16234)

[2 rows x 11 columns]


                             title last_name first_name                            address  medium  ... desc       lat       lng                    loc                                      geometry
0          [Cross Country Runners]     Frost      Miley     4001 Harding Rd., Nashville TN  Bronze  ...  NaN  36.12856 -86.83660   (36.12856, -86.8366)   POINT (-9666606.09421918 4318325.479300267)
1  [Fourth and Commerce Sculpture]    Walker        Lin  333 Commerce Street, Nashville TN     NaN  ...  NaN  36.16234 -86.77774  (36.16234, -86.77774)  POINT (-9660053.828991087 4322982.159062029)

[2 rows x 11 columns]

```


### **3.2.3 Art distances from neighborhood center**



 Now that you have the center point and the art locations in the units we need to calculate distances in meters, it’s time to perform that step.





```python

# Import package for pretty printing
import pprint

# Build a dictionary of titles and distances for Urban Residents art
art_distances = {}
for row in art_dist_meters.iterrows():
    vals = row[1]
    key = vals['title']
    ctr = vals['center']
    art_distances[key] = vals['geometry'].distance(other=ctr)

# Pretty print the art_distances
pprint.pprint(art_distances)

```




```

{'12th & Porter Mural': 1269.1502879119878,
 'A Splash of Color': 2471.774738455904,
 'A Story of Nashville': 513.5632030470281,
 'Aerial Innovations Mural': 4516.755210408422,
 'Airport Sun Project': 12797.594229783645,
 'Andrew Jackson': 948.9812821640502,
 'Angel': 10202.565989739454,
 'Anticipation': 688.8349105273556,
......

```




---


## **3.3 Street maps with folium**


### **3.3.1 Create a folium location from the urban centroid**



 In order to construct a folium map of the Urban Residents neighborhood, you need to build a coordinate pair location that is formatted for folium.





```python

# Print the head of the urban_polygon
print(urban_polygon.head())

# Create urban_center from the urban_polygon center
urban_center = urban_polygon.center[0]

# Print urban_center
print(urban_center)

# Create array for folium called urban_location
urban_location = [urban_center.y, urban_center.x]

# Print urban_location
print(urban_location)

```




```

       index             name                                           geometry                                        center
    0     41  Urban Residents  (POLYGON ((-86.78122053774267 36.1645653773768...  POINT (-86.77756457127047 36.16127820928791)
    POINT (-86.77756457127047 36.16127820928791)
    [36.161278209287914, -86.77756457127047]

```


### **3.3.2 Create a folium map of downtown Nashville**



 In this exercise you will create a street map of downtown Nashville using folium.





```python

# Construct a folium map with urban_location
downtown_map = folium.Map(location = urban_location, zoom_start = 15)

# Display the map
display(downtown_map)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/1-3.png?w=1024)

### **3.3.3 Folium street map of the downtown neighborhood**



 This time you will create the folium map of downtown and add the Urban Residents neighborhood area from
 `urban_polygon`
 . The
 `urban_polygon`
 has been printed to your console.





```python

# Create array for called folium_loc from the urban_polygon center point
point = urban_polygon.centroid[0]
folium_loc = [point.y, point.x]

# Construct a map from folium_loc: downtown_map
downtown_map = folium.Map(location = folium_loc, zoom_start = 15)

# Draw our neighborhood: Urban Residents
folium.GeoJson(urban_polygon.geometry).add_to(downtown_map)

# Display the map
display(downtown_map)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/2-3.png?w=1024)



---


## **3.4 Creating markers and popups in folium**


### **3.4.1 Adding markers for the public art**



 Now that you have added the polygon for the Urban Residents neighborhood to your folium street map, it’s time to add the locations of the art within the neighborhood. You can do that by creating folium markers. Each marker needs a location assigned. Use
 `iterrows()`
 to loop through the data to grab the values you need.





```python

# Iterate through the urban_art and print each part of tuple returned
for row in urban_art.iterrows():
  print('first part: ', row[0])
  print('second part: ', row[1])

# Create a location and marker with each iteration for the downtown_map
for row in urban_art.iterrows():
    row_values = row[1]
    location = [row_values['lat'], row_values['lng']]
    marker = folium.Marker(location = location)
    marker.add_to(downtown_map)

# Display the map
display(downtown_map)

```




```

first part:  1
second part:  title              [Fourth and Commerce Sculpture]
last_name                                   Walker
first_name                                     Lin
address          333 Commerce Street, Nashville TN
medium                                         NaN
type                                     Sculpture
desc                                           NaN
lat                                        36.1623
lng                                       -86.7777
loc                          (36.16234, -86.77774)
geometry       POINT (-86.77774000000001 36.16234)
index_right                                     41
name                               Urban Residents
Name: 1, dtype: object

...
...

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/3-3.png?w=1024)

### **3.4.2 Troubleshooting data issues**



 You will be building popups for the downtown art using the
 `title`
 and
 `desc`
 columns from the
 `urban_art`
 DataFrame. Here, you will inspect those columns to identify and clean up any problematic values.





```python

# Print the urban_art titles
print(urban_art.title)

#Print the urban_art descriptions
print(urban_art.desc)

# Replace Nan and ' values in description
urban_art.desc.fillna('', inplace = True)
urban_art.desc = urban_art.desc.str.replace("'", "`")

#Print the urban_art descriptions again
print(urban_art.desc)

```




```

    1                        [Fourth and Commerce Sculpture]
    4                                   A Story of Nashville
    21                                           Chet Atkins
...
    Name: title, dtype: object

    1                                                    NaN
    4      Inside the Grand Reading Room, this is a serie...
    21     A sculpture of a young Chet Atkins seated on a...
...
    Name: desc, dtype: object

    1
    4      Inside the Grand Reading Room, this is a serie...
    21     A sculpture of a young Chet Atkins seated on a...
...
    Name: desc, dtype: object

```


### **3.4.3 A map of downtown art**



 Now you will assign a
 `popup`
 to each marker to give information about the artwork at each location. In particular you will assign the art title and description to the
 `popup`
 for each marker. You will do so by creating the map object
 `downtown_map`
 , then add the popups, and finally use the
 `display`
 function to show your map.




 One warning before you start: you’ll need to ensure that all instances of single quotes (
 `'`
 ) are removed from the pop-up message, otherwise your plot will not render!





```python

# Construct downtown map
downtown_map = folium.Map(location = nashville, zoom_start = 15)
folium.GeoJson(urban_polygon).add_to(downtown_map)

# Create popups inside the loop you built to create the markers
for row in urban_art.iterrows():
    row_values = row[1]
    location = [row_values['lat'], row_values['lng']]
    popup = (str(row_values['title']) + ': ' +
             str(row_values['desc'])).replace("'", "`")
    marker = folium.Marker(location = location, popup = popup)
    marker.add_to(downtown_map)

# Display the map.
display(downtown_map)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/4-3.png?w=1024)




# **4. Creating a choropleth building permit density in Nashville**
------------------------------------------------------------------


## **4.1 What is a choropleth?**



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/5-2.png?w=1024)

### **4.1.1 Finding counts from a spatial join**



 You will be using a dataset of the building permits issued in Nashville during 2017. This DataFrame called
 `permits`
 is in your workspace along with the
 `council_districts`
 GeoDataFrame.





```

permits.head(3)
    permit_id      issued      cost        lat        lng
0  2017032777  2017-05-24  226201.0  36.198241 -86.742235
1  2017061156  2017-10-23   15000.0  36.151554 -86.830222
2  2017074521  2017-11-20   13389.0  36.034239 -86.708892


council_districts.head(3)
  district                                           geometry
0        1  (POLYGON ((-86.90738248774342 36.3905151283193...
1        2  (POLYGON ((-86.75902399986667 36.2309080000732...
2        8  (POLYGON ((-86.72850199989709 36.2832840002146...


council_districts.crs
{'init': 'epsg:4326'}

```




```

from shapely.geometry import Point

# Create a shapely Point from lat and lng
permits['geometry'] = permits.apply(lambda x: Point((x.lng , x.lat)), axis = 1)

# Build a GeoDataFrame: permits_geo
permits_geo = gpd.GeoDataFrame(permits, crs = council_districts.crs, geometry = permits.geometry)

# Spatial join of permits_geo and council_districts
permits_by_district = gpd.sjoin(permits_geo, council_districts, op = 'within')
print(permits_by_district.head(2))

# Create permit_counts
permit_counts = permits_by_district.groupby(['district']).size()
print(permit_counts)

```




```

     permit_id      issued      cost        lat        lng                              geometry  index_right district
0   2017032777  2017-05-24  226201.0  36.198241 -86.742235  POINT (-86.74223499999999 36.198241)            5        5
68  2017053890  2017-09-05       0.0  36.185442 -86.768239  POINT (-86.76823900000001 36.185442)            5        5


district
1     146
10    119
11    239
...
dtype: int64

```



 Now you have a count of building permits issued for each council district. Next you’ll get the area of each council_district.



### **4.1.2 Council district areas and permit counts**



 In order to create a normalized value for the building permits issued in each council district, you will need to find the area of each council district. Remember that you can leverage the
 `area`
 attribute of a GeoSeries to do this. You will need to convert
 `permit_counts`
 to a DataFrame so you can merge it with the
 `council_districts`
 data. Both
 `permit_counts`
 and
 `council_districts`
 are in your workspace.





```python

# Create an area column in council_districts
council_districts['area'] = council_districts.geometry.area
print(council_districts.head(2))

'''
  district                                           geometry      area
0        1  (POLYGON ((-86.90738248774342 36.3905151283193...  0.022786
1        2  (POLYGON ((-86.75902399986667 36.2309080000732...  0.002927
'''

# Convert permit_counts to a DataFrame
permits_df = permit_counts.to_frame()
print(permits_df.head(2))

'''
            0
district
1         146
10        119
'''

# Reset index and column names
permits_df.reset_index(inplace=True)
permits_df.columns = ['district', 'bldg_permits']
print(permits_df.head(2))

'''
  district  bldg_permits
0        1           146
1       10           119
'''

# Merge council_districts and permits_df:
districts_and_permits = pd.merge(council_districts, permits_df, on = 'district')
print(districts_and_permits.head(2))

'''
  district                                           geometry    bldg_permits
0        1  (POLYGON ((-86.90738248774342 36.3905151283193...  146
1        2  (POLYGON ((-86.75902399986667 36.2309080000732...  399
'''

```



 You have created a column with the
 `area`
 in the
 `council_districts`
 Geo DataFrame and built a DataFrame from the
 `permit_counts`
 . You have merged all the information into a single GeoDataFrame. Next you will calculate the permits by area for each council district.



### **4.1.3 Calculating a normalized metric**



 Now you are ready to divide the number of building permits issued for projects in each council district by the area of that district to get a normalized value for the permits issued. First you will verify that the
 `districts_and_permits`
 is still a GeoDataFrame.





```python

# Print the type of districts_and_permits
print(type(districts_and_permits))

# Create permit_density column in districts_and_permits
districts_and_permits['permit_density'] = districts_and_permits.apply(lambda row: row.bldg_permits / row.area, axis = 1)

# Print the head of districts_and_permits
print(districts_and_permits.head())

```




```

<class 'geopandas.geodataframe.GeoDataFrame'>

  district                                           geometry        area  bldg_permits  permit_density                                        center
0        1  (POLYGON ((-86.9073824877434 36.39051512831934...  350.194851           146        0.416911  POINT (-86.89459869514988 36.26266635824652)
1        2  (POLYGON ((-86.75902399986667 36.2309080000731...   44.956987           399        8.875150  POINT (-86.80270842421444 36.20859420830921)
2        8  (POLYGON ((-86.72850199989709 36.2832840002146...   38.667932           209        5.404995   POINT (-86.7377559683446 36.24515598511006)
3        9  (POLYGON ((-86.68680500011934 36.2867050001350...   44.295293           186        4.199092  POINT (-86.67436394441576 36.23852818463936)
4        4  (POLYGON ((-86.74488864807593 36.0531632050230...   31.441618           139        4.420892  POINT (-86.73914087216721 36.02939641896401)


```




---


## **4.2 Choropleths with geopandas**


### **4.2.1 Geopandas choropleths**



 First you will plot a choropleth of the building permit density for each council district using the default colormap. Then you will polish it by changing the colormap and adding labels and a title.





```python

# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

# Simple plot of building permit_density
districts_and_permits.plot(column = 'permit_density', legend = True);
plt.show();


# Polished choropleth of building permit_density
districts_and_permits.plot(column = 'permit_density', cmap = 'BuGn', edgecolor = 'black', legend = True)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.xticks(rotation = 'vertical')
plt.title('2017 Building Project Density by Council District')
plt.show();

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/8-1.png?w=982)
![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/9-1.png?w=995)



### **4.2.2 Area in km squared, geometry in decimal degrees**



 In this exercise, you’ll start again with the
 `council_districts`
 GeoDataFrame and the
 `permits`
 DataFrame. You will change the
 `council_districts`
 to use the EPSG 3857 coordinate reference system before creating a column for
 `area`
 . Once the
 `area`
 column has been created, you will change the CRS back to EPSG 4326 so that the geometry is in decimal degrees.





```python

# Change council_districts crs to epsg 3857
council_districts = council_districts.to_crs(epsg = 3857)
print(council_districts.crs)
print(council_districts.head())

# Create area in square km
sqm_to_sqkm = 10**6
council_districts['area'] = council_districts.geometry.area / sqm_to_sqkm

# Change council_districts crs back to epsg 4326
council_districts = council_districts.to_crs(epsg = 4326)
print(council_districts.crs)
print(council_districts.head())

```




```

    {'init': 'epsg:3857', 'no_defs': True}
      district                                           geometry
    0        1  (POLYGON ((-9674485.564711858 4354489.55569189...
    1        2  (POLYGON ((-9657970.37338656 4332440.649821124...
    2        8  (POLYGON ((-9654572.679891953 4339671.15221535...
    3        9  (POLYGON ((-9649930.991109086 4340143.58970314...
    4        4  (POLYGON ((-9656396.83322303 4307939.01495162,...

    {'init': 'epsg:4326', 'no_defs': True}
      district                                           geometry        area
    0        1  (POLYGON ((-86.9073824877434 36.39051512831934...  350.194851
    1        2  (POLYGON ((-86.75902399986667 36.2309080000731...   44.956987
    2        8  (POLYGON ((-86.72850199989709 36.2832840002146...   38.667932
    3        9  (POLYGON ((-86.68680500011934 36.2867050001350...   44.295293
    4        4  (POLYGON ((-86.74488864807593 36.0531632050230...   31.441618

```



 The
 `council_districts`
 have
 `area`
 in kilometers squared and
 `geometry`
 measures in decimal degrees.



### **4.2.3 Spatially joining and getting counts**



 You will continue preparing your dataset for plotting a
 `geopandas`
 choropleth by creating a GeoDataFrame of the building permits spatially joined to the council districts. After that, you will be able to get counts of the building permits issued in each council district.





```python

# Create permits_geo
permits_geo = gpd.GeoDataFrame(permits, crs = council_districts.crs, geometry = permits.geometry)

# Spatially join permits_geo and council_districts
permits_by_district = gpd.sjoin(permits_geo, council_districts, op = 'within')
print(permits_by_district.head(2))

# Count permits in each district
permit_counts = permits_by_district.groupby('district').size()

# Convert permit_counts to a df with 2 columns: district and bldg_permits
counts_df = permit_counts.to_frame()
counts_df.reset_index(inplace=True)
counts_df.columns = ['district', 'bldg_permits']
print(counts_df.head(2))

```




```

     permit_id      issued      cost        lat        lng                              geometry  index_right district       area
0   2017032777  2017-05-24  226201.0  36.198241 -86.742235  POINT (-86.74223499999999 36.198241)            5        5  19.030612
68  2017053890  2017-09-05       0.0  36.185442 -86.768239  POINT (-86.76823900000001 36.185442)            5        5  19.030612


  district  bldg_permits
0        1           146
1       10           119

```


### **4.2.4 Building a polished Geopandas choropleth**



 After merging the
 `counts_df`
 with
 `permits_by_district`
 , you will create a column with normalized
 `permit_density`
 by dividing the count of permits in each council district by the area of that council district. Then you will plot your final
 `geopandas`
 choropleth of the building projects in each council district.





```python

# Merge permits_by_district and counts_df
districts_and_permits = pd.merge(permits_by_district, counts_df, on = 'district')

# Create permit_density column
districts_and_permits['permit_density'] = districts_and_permits.apply(lambda row: row.bldg_permits / row.area, axis = 1)
print(districts_and_permits.head(2))

# Create choropleth plot
districts_and_permits.plot(column = 'permit_density', cmap = 'OrRd', edgecolor = 'black', legend = True)

# Add axis labels and title
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('2017 Building Project Density by Council District')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/10-1.png?w=1024)



---


## **4.3 Choropleths with folium**



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/11-2.png?w=1024)

### **4.3.1 Folium choropleth**



 In this exercise, you will construct a folium choropleth to show the density of permitted construction projects in different Nashville council districts. You will be using a single data source, the
 `districts_and_permits`
 GeoDataFrame, which is in your workspace.





```

districts_and_permits.head(1)
  district                                           geometry        area  bldg_permits  permit_density
0        1  (POLYGON ((-86.9073824877434 36.39051512831934...  350.194851           146        0.416911

```




```python

# Center point for Nashville
nashville = [36.1636,-86.7823]

# Create map
m = folium.Map(location=nashville, zoom_start=10)

# Build choropleth
m.choropleth(
    geo_data=districts_and_permits,
    name='geometry',
    data=districts_and_permits,
    columns=['district', 'permit_density'],
    key_on='feature.properties.district',
    fill_color='Reds',
    fill_opacity=0.5,
    line_opacity=1.0,
    legend_name='2017 Permitted Building Projects per km squared'
)

# Create LayerControl and add it to the map
folium.LayerControl().add_to(m)

# Display the map
display(m)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/12-1.png?w=1024)

### **4.3.2 Folium choropleth with markers and popups**



 Now you will add a marker to the center of each council district that shows the district number along with the count of building permits issued in 2017 for that district. The map you created in the last exercise is in your workspace as
 `m`
 .





```python

# Create center column for the centroid of each district
districts_and_permits['center'] = districts_and_permits.geometry.centroid

# Build markers and popups
for row in districts_and_permits.iterrows():
    row_values = row[1]
    center_point = row_values['center']
    location = [center_point.y, center_point.x]
    popup = ('Council District: ' + str(row_values['district']) +
             ';  ' + 'permits issued: ' + str(row_values['bldg_permits']))
    marker = folium.Marker(location = location, popup = popup)
    marker.add_to(m)

# Display the map
display(m)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/visualizing-geospatial-data-in-python/13-1.png?w=1024)



---



 Thank you for reading and hope you’ve learned a lot.



