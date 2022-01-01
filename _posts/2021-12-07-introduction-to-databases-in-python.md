---
title: Introduction to Databases in Python
date: 2021-12-07 11:22:08 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Introduction to Databases in Python
======================================







 This is the memo of the 3rd course (4 courses in all) of ‘Importing & Cleaning Data with Python’ skill track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/introduction-to-relational-databases-in-python)**
 .





---



# **1. Basics of Relational Databases**
--------------------------------------




## **1.1 Introduction to Databases**


####
**Relational model**



 Tables, Columns, Rows, and Relationships are part of the relational model.





---


## **1.2 Connecting to your database**



 Database types:



* SQLite
* PostgreSQL
* MySQL
* MS SQL
* Oracle
* etc.



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-databases-in-python/capture3-9.png?w=1011)

####
**Engines and connection strings**



 Alright, it’s time to create your first engine! An engine is just a common interface to a database, and the information it requires to connect to one is contained in a connection string, for example
 `sqlite:///example.sqlite`
 . Here,
 `sqlite`
 in
 `sqlite:///`
 is the database driver, while
 `example.sqlite`
 is a SQLite file contained in the local directory.




 You can learn a lot more about connection strings in the
 [SQLAlchemy documentation](http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls)
 .




 Your job in this exercise is to create an engine that connects to a local SQLite file named
 `census.sqlite`
 . Then, print the names of the tables the engine contains using the
 `.table_names()`
 method.





```python

# Import create_engine
from sqlalchemy import create_engine

# Create an engine that connects to the census.sqlite file: engine
engine = create_engine('sqlite:///census.sqlite')

# Print table names
print(engine.table_names())

# ['census', 'state_fact']

```



 This database has two tables, as you can see:
 `'census'`
 and
 `'state_fact'`
 . You’ll be exploring both of these and more throughout this course!



####
**Autoloading Tables from a database**



 SQLAlchemy can be used to automatically load tables from a database using something called reflection. Reflection is the process of reading the database and building the metadata based on that information. It’s the opposite of creating a Table by hand and is very useful for working with existing databases.




 To perform reflection, you will first need to import and initialize a
 `MetaData`
 object.
 `MetaData`
 objects contain information about tables stored in a database. During reflection, the
 `MetaData`
 object will be populated with information about the reflected table automatically, so we only need to initialize it before reflecting by calling
 `MetaData()`
 .




 You will also need to import the
 `Table`
 object from the SQLAlchemy package. Then, you use this
 `Table`
 object to read your table from the engine, autoload the columns, and populate the metadata. This can be done with a single call to
 `Table()`
 : using the
 `Table`
 object in this manner is a lot like passing arguments to a function. For example, to autoload the columns with the engine, you have to specify the keyword arguments
 `autoload=True`
 and
 `autoload_with=engine`
 to
 `Table()`
 .




 Finally, to view information about the object you just created, you will use the
 `repr()`
 function. For any Python object,
 `repr()`
 returns a text representation of that object. For SQLAlchemy
 `Table`
 objects, it will return the information about that table contained in the metadata.




 In this exercise, your job is to reflect the
 `"census"`
 table available on your
 `engine`
 into a variable called
 `census`
 .





```python

# Import create_engine, MetaData, and Table
from sqlalchemy import create_engine, MetaData, Table

# Create engine: engine
engine = create_engine('sqlite:///census.sqlite')

# Create a metadata object: metadata
metadata = MetaData()

# Reflect census table from the engine: census
census = Table('census', metadata, autoload=True, autoload_with=engine)

# Print census table metadata
print(repr(census))


```




```

Table('census', MetaData(bind=None), Column('state', VARCHAR(length=30), table=<census>), Column('sex', VARCHAR(length=1), table=<census>), Column('age', INTEGER(), table=<census>), Column('pop2000', INTEGER(), table=<census>), Column('pop2008', INTEGER(), table=<census>), schema=None)

```



 Reflecting a table allows you to work with it in Python.



####
**Viewing Table details**



 Now you can begin to learn more about the columns and structure of your table. It is important to get an understanding of your database by examining the column names. This can be done by using the
 `.columns`
 attribute and accessing the
 `.keys()`
 method. For example,
 `census.columns.keys()`
 would return a list of column names of the
 `census`
 table.




 Following this, we can use the metadata container to find out more details about the reflected table such as the columns and their types. For example, information about the table objects are stored in the
 `metadata.tables`
 dictionary, so you can get the metadata of your
 `census`
 table with
 `metadata.tables['census']`
 . This is similar to your use of the
 `repr()`
 function on the
 `census`
 table from the previous exercise.





```

from sqlalchemy import create_engine, MetaData, Table

engine = create_engine('sqlite:///census.sqlite')

metadata = MetaData()

# Reflect the census table from the engine: census
census = Table('census', metadata, autoload=True, autoload_with=engine)

# Print the column names
print(census.columns.keys())

# Print full metadata of census
print(repr(metadata.tables['census']))


```




```

['state', 'sex', 'age', 'pop2000', 'pop2008']

Table('census', MetaData(bind=None), Column('state', VARCHAR(length=30), table=<census>), Column('sex', VARCHAR(length=1), table=<census>), Column('age', INTEGER(), table=<census>), Column('pop2000', INTEGER(), table=<census>), Column('pop2008', INTEGER(), table=<census>), schema=None)

```



 The
 `census`
 table, as you can see, has five columns. Knowing the names of these columns and their data types will make it easier for you to structure your queries.





---


## **1.3 Introduction to SQL**



**raw SQL vs SQLAlchemy**




![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-databases-in-python/capture4-8.png?w=1017)

####
**Selecting data from a Table: raw SQL**



 As you have seen in the video, to access and manipulate the data in the database, we will first need to establish a connection to it by using the
 `.connect()`
 method on the engine. This is because the
 `create_engine()`
 function that you have used before returns an instance of an engine, but it does not actually open a connection until an action is called that would require a connection, such as a query.




 Using what we just learned about SQL and applying the
 `.execute()`
 method on our connection, we can leverage a raw SQL query to query all the records in our
 `census`
 table. The object returned by the
 `.execute()`
 method is a
 **ResultProxy**
 . On this ResultProxy, we can then use the
 `.fetchall()`
 method to get our results – that is, the
 **ResultSet**
 .




 In this exercise, you’ll use a traditional SQL query. Notice that when you execute a query using raw SQL, you will query the table
 *in the database directly*
 . In particular, no reflection step is needed.





```

from sqlalchemy import create_engine
engine = create_engine('sqlite:///census.sqlite')

# Create a connection on engine
connection = engine.connect()

# Build select statement for census table: stmt
stmt = 'SELECT * FROM census'

# Execute the statement and fetch the results: results
results = connection.execute(stmt).fetchall()

connection.execute(stmt)
# <sqlalchemy.engine.result.ResultProxy at 0x7f19c77ec0b8>

# Print results
print(results)

```




```

[('Illinois', 'M', 0, 89600, 95012), ('Illinois', 'M', 1, 88445, 91829), ('Illinois', 'M', 2, 88729, 89547), ('Illinois', 'M', 3, 88868, 90037), ('Illinois', 'M', 4, 91947, 91111), ('Illinois', 'M', 5, 93894, 89802), ('Illinois', 'M', 6, 93676, 88931), ('Illinois', 'M', 7, 94818, 90940), ('Illinois', 'M', 8, 95035, 86943), ('Illinois', 'M', 9, 96436, 86055), ('Illinois', 'M', 10, 97280, 86565), ('Illinois', 'M', 11, 94029, 86606), ('Illinois', 'M', 12, 92402, 89596), ('Illinois', 'M', 13, 89926, 91661), ('Illinois', 'M', 14, 90717, 91256), ('Illinois', 'M', 15, 92178, 92729), ('Illinois', 'M', 16, 90587, 93083),
...

```



 Notice that the stmt converts into a SQL statement listing all the records for all the columns in the table.This output is quite unwieldy though, and fetching all the records in the table might take a long time, so in the next exercises, you will learn how to fetch only the first few records of a ResultProxy.



####
**Selecting data from a Table with SQLAlchemy**



 It’s now time to build your first select statement using SQLAlchemy. SQLAlchemy provides a nice “Pythonic” way of interacting with databases. When you used raw SQL in the last exercise, you queried the database directly. When using SQLAlchemy, you will go through a
 `Table`
 object instead, and SQLAlchemy will take case of translating your query to an appropriate SQL statement for you. So rather than dealing with the differences between specific dialects of traditional SQL such as MySQL or PostgreSQL, you can leverage the Pythonic framework of SQLAlchemy to streamline your workflow and more efficiently query your data. For this reason, it is worth learning even if you may already be familiar with traditional SQL.




 In this exercise, you’ll once again build a statement to query all records from the
 `census`
 table. This time, however, you’ll make use of the
 `select()`
 function of the
 `sqlalchemy`
 module. This function requires a
 **list**
 of tables or columns as the only required argument: for example,
 `select([my_table])`
 .




 You will also fetch only a few records of the ResultProxy by using
 `.fetchmany()`
 with a
 `size`
 argument specifying the number of records to fetch.





```python

# Import select
from sqlalchemy import select

# Reflect census table via engine: census
census = Table('census', metadata, autoload=True, autoload_with=engine)

# Build select statement for census table: stmt
stmt = select([census])

# Print the emitted statement to see the SQL string
print(stmt)
# SELECT census.state, census.sex, census.age, census.pop2000, census.pop2008
FROM census

# Execute the statement on connection and fetch 10 records: result
results = connection.execute(stmt).fetchmany(size=10)

# Execute the statement and print the results
print(results)

```




```

[('Illinois', 'M', 0, 89600, 95012), ('Illinois', 'M', 1, 88445, 91829), ('Illinois', 'M', 2, 88729, 89547), ('Illinois', 'M', 3, 88868, 90037), ('Illinois', 'M', 4, 91947, 91111), ('Illinois', 'M', 5, 93894, 89802), ('Illinois', 'M', 6, 93676, 88931), ('Illinois', 'M', 7, 94818, 90940), ('Illinois', 'M', 8, 95035, 86943), ('Illinois', 'M', 9, 96436, 86055)]

```


####
**Handling a ResultSet**



 Recall the differences between a ResultProxy and a ResultSet:



* ResultProxy: The object returned by the
 `.execute()`
 method. It can be used in a variety of ways to get the data returned by the query.
* ResultSet: The actual data asked for in the query when using a fetch method such as
 `.fetchall()`
 on a ResultProxy.



 This separation between the ResultSet and ResultProxy allows us to fetch as much or as little data as we desire.




 Once we have a ResultSet, we can use Python to access all the data within it by column name and by list style indexes. For example, you can get the first row of the results by using
 `results[0]`
 . With that first row then assigned to a variable
 `first_row`
 , you can get data from the first column by either using
 `first_row[0]`
 or by column name such as
 `first_row['column_name']`
 . You’ll now practice exactly this using the ResultSet you obtained from the
 `census`
 table in the previous exercise.





```python

# Get the first row of the results by using an index: first_row
first_row = results[0]

# Print the first row of the results
print(first_row)

# Print the first column of the first row by accessing it by its index
print(first_row[0])
print(results[0][0])

# Print the 'state' column of the first row by using its name
print(first_row['state'])
print(results[0]['state'])


```




```

results[:3]
[('Illinois', 'M', 0, 89600, 95012),
 ('Illinois', 'M', 1, 88445, 91829),
 ('Illinois', 'M', 2, 88729, 89547)]

results[:3][0]
('Illinois', 'M', 0, 89600, 95012)

results[:3][0][3]
89600

```




---



# **2. Applying Filtering, Ordering and Grouping to Queries**
------------------------------------------------------------


## **2.1 Filtering and targeting data**


####
**Connecting to a PostgreSQL database**



 In these exercises, you will be working with real databases hosted on the cloud via Amazon Web Services (AWS)!




 Let’s begin by connecting to a PostgreSQL database. When connecting to a PostgreSQL database, many prefer to use the psycopg2 database driver as it supports practically all of PostgreSQL’s features efficiently and is the standard dialect for PostgreSQL in SQLAlchemy.




 You might recall from Chapter 1 that we use the
 `create_engine()`
 function and a connection string to connect to a database. In general, connection strings have the form
 `"dialect+driver://username:password@host:port/database"`




 There are three components to the connection string in this exercise: the dialect and driver (
 `'postgresql+psycopg2://'`
 ), followed by the username and password (
 `'`
 username:password
 `'`
 ), followed by the host and port (
 `'@postgresql.csrrinzqubik.us-east-1.rds.amazonaws.com:1234/'`
 ), and finally, the database name (
 `'census'`
 ). You will have to pass this string as an argument to
 `create_engine()`
 in order to connect to the database.





```python

# Import create_engine function
from sqlalchemy import create_engine

# Create an engine to the census database
engine = create_engine('postgresql+psycopg2://username:password@postgresql.csrrinzqubik.us-east-1.rds.amazonaws.com:1234/census')

# Use the .table_names() method on the engine to print the table names
print(engine.table_names())

# ['census', 'state_fact', 'vrska', 'census1', 'data', 'data1', 'employees3', 'users', 'employees', 'employees_2']

```


####
**Filter data selected from a Table – Simple**



 Having connected to the database, it’s now time to practice filtering your queries!




 As mentioned in the video, a
 `where()`
 clause is used to filter the data that a statement returns. For example, to select all the records from the
 `census`
 table where the sex is Female (or
 `'F'`
 ) we would do the following:




`select([census]).where(census.columns.sex == 'F')`




 In addition to
 `==`
 we can use basically any python comparison operator (such as
 `<=`
 ,
 `!=`
 , etc) in the
 `where()`
 clause.





```python

# Create a select query: stmt
stmt = select([census])

# Add a where clause to filter the results to only those for New York : stmt_filtered
stmt = stmt.where(census.columns.state == 'New York')

# Execute the query to retrieve all the data returned: results
results = connection.execute(stmt).fetchall()

# Loop over the results and print the age, sex, and pop2000
for result in results:
    print(result.age, result.sex, result.pop2000)


```




```

0 M 126237
1 M 124008
2 M 124725
...
83 M 21687
84 M 18873
85 M 88366
0 F 120355
1 F 118219
2 F 119577
...
84 F 37436
85 F 226378

```


####
**Filter data selected from a Table – Expressions**



 In addition to standard Python comparators, we can also use methods such as
 `in_()`
 to create more powerful
 `where()`
 clauses. You can see a full list of expressions in the
 [SQLAlchemy Documentation](http://docs.sqlalchemy.org/en/latest/core/sqlelement.html#module-sqlalchemy.sql.expression)
 .




 Method
 `in_()`
 , when used on a column, allows us to include records where the value of a column is among a list of possible values. For example,
 `where(census.columns.age.in_([20, 30, 40]))`
 will return only records for people who are exactly 20, 30, or 40 years old.




 In this exercise, you will continue working with the
 `census`
 table, and select the records for people from the three most densely populated states.





```python

# Define a list of states for which we want results
states = ['New York', 'California', 'Texas']

# Create a query for the census table: stmt
stmt = select([census])

# Append a where clause to match all the states in_ the list states
stmt = stmt.where(census.columns.state.in_(states))

# Loop over the ResultProxy and print the state and its population in 2000
for result in connection.execute(stmt):
    print(result.state, result.pop2000)


```




```

California 252494
California 247978
..
New York 122770
New York 123978
...
Texas 27961
Texas 171538


```



 Along with
 `in_`
 , you can also use methods like
 `and_`
 ,
 `any_`
 to create more powerful
 `where()`
 clauses. You might have noticed that we did not use any of the fetch methods to retrieve a ResultSet like in the previous exercises. Indeed, if you are only interested in manipulating one record at a time, you can iterate over the ResultProxy directly!



####
**Filter data selected from a Table – Advanced**



 SQLAlchemy also allows users to use conjunctions such as
 `and_()`
 ,
 `or_()`
 , and
 `not_()`
 to build more complex filtering. For example, we can get a set of records for people in New York who are 21 or 37 years old with the following code:





```

select([census]).where(
  and_(census.columns.state == 'New York',
       or_(census.columns.age == 21,
          census.columns.age == 37
         )
      )
  )

```



 An equivalent SQL statement would be,for example,





```

SELECT * FROM census WHERE state = 'New York' AND (age = 21 OR age = 37)

```




```python

# Import and_
from sqlalchemy import and_

# Build a query for the census table: stmt
stmt = select([census])

# Append a where clause to select only non-male records from California using and_
stmt = stmt.where(
    # The state of California with a non-male sex
    and_(census.columns.state == 'California',
         census.columns.sex != 'M'
         )
)

# Loop over the ResultProxy printing the age and sex
for result in connection.execute(stmt):
    print(result.age, result.sex)


```




```

0 F
1 F
2 F
...
84 F
85 F

```




---


## **2.2 Overview of ordering**


####
**Ordering by a single column**



 To sort the result output by a field, we use the
 `.order_by()`
 method. By default, the
 `.order_by()`
 method sorts from lowest to highest on the supplied column.





```python

# Build a query to select the state column: stmt
stmt = select([census.columns.state])

# Order stmt by the state column
stmt = stmt.order_by(census.columns.state)

# Execute the query and store the results: results
results = connection.execute(stmt).fetchall()

# Print the first 10 results
print(results[:10])

# [('Alabama',), ('Alabama',), ('Alabama',), ('Alabama',), ('Alabama',), ('Alabama',), ('Alabama',), ('Alabama',), ('Alabama',), ('Alabama',)]

```


####
**Ordering in descending order by a single column**



 You can also use
 `.order_by()`
 to sort from highest to lowest by wrapping a column in the
 `desc()`
 function. Although you haven’t seen this function in action, it generalizes what you have already learned.




 Pass
 `desc()`
 (for “descending”) inside an
 `.order_by()`
 with the name of the column you want to sort by. For instance,
 `stmt.order_by(desc(table.columns.column_name))`
 sorts
 `column_name`
 in descending order.





```python

# Import desc
from sqlalchemy import desc

# Build a query to select the state column: stmt
stmt = select([census.columns.state])

# Order stmt by state in descending order: rev_stmt
rev_stmt = stmt.order_by(desc(census.columns.state))

# Execute the query and store the results: rev_results
rev_results = connection.execute(rev_stmt).fetchall()

# Print the first 10 rev_results
print(rev_results[:10])

# [('Wyoming',), ('Wyoming',), ('Wyoming',), ('Wyoming',), ('Wyoming',), ('Wyoming',), ('Wyoming',), ('Wyoming',), ('Wyoming',), ('Wyoming',)]

```


####
**Ordering by multiple columns**



 We can pass multiple arguments to the
 `.order_by()`
 method to order by multiple columns. In fact, we can also sort in ascending or descending order for each individual column.





```python

# Build a query to select state and age: stmt
stmt = select([census.columns.state, census.columns.age])

# Append order by to ascend by state and descend by age
stmt = stmt.order_by(census.columns.state, desc(census.columns.age))

# Execute the statement and store all the records: results
results = connection.execute(stmt).fetchall()

# Print the first 20 results
print(results[:20])


```




```

[('Alabama', 85), ('Alabama', 85), ('Alabama', 84), ('Alabama', 84), ('Alabama', 83), ('Alabama', 83), ('Alabama', 82), ('Alabama', 82), ('Alabama', 81), ('Alabama', 81), ('Alabama', 80), ('Alabama', 80), ('Alabama', 79), ('Alabama', 79), ('Alabama', 78), ('Alabama', 78), ('Alabama', 77), ('Alabama', 77), ('Alabama', 76), ('Alabama', 76)]

```




---


## **2.3 Counting, summing and grouping data**


####
**Counting distinct data**



 SQLAlchemy’s
 `func`
 module provides access to built-in SQL functions that can make operations like counting and summing faster and more efficient.




 We can use
 `func.sum()`
 to get a
 **sum**
 of the
 `pop2008`
 column of
 `census`
 as shown below:





```

select([func.sum(census.columns.pop2008)])

```



 If instead you want to
 **count**
 the number of values in
 `pop2008`
 , you could use
 `func.count()`
 like this:





```

select([func.count(census.columns.pop2008)])

```



 Furthermore, if you only want to count the
 **distinct**
 values of
 `pop2008`
 , you can use the
 `.distinct()`
 method:





```

select([func.count(census.columns.pop2008.distinct())])

```



 In this exercise, you will practice using
 `func.count()`
 and
 `.distinct()`
 to get a count of the distinct number of states in
 `census`
 .




 So far, you’ve seen
 `.fetchall()`
 ,
 `.fetchmany()`
 , and
 `.first()`
 used on a ResultProxy to get the results. The ResultProxy also has a method called
 `.scalar()`
 for getting just the value of a query that returns only one row and column.




 This can be very useful when you are querying for just a count or sum.





```python

# Build a query to count the distinct states values: stmt
stmt = select([func.count(census.columns.state.distinct())])

# Execute the query and store the scalar result: distinct_state_count
distinct_state_count = connection.execute(stmt).scalar()

# Print the distinct_state_count
print(distinct_state_count)

# 51

```




```

connection.execute(stmt).fetchall()
# [(51,)]

```



 Notice the use of the
 `.scalar()`
 method: This is useful when you want to get just the value of a query that returns only one row and column, like in this case.



####
**Count of records by state**



 Often, we want to get a count for each record with a particular value in another column. The
 `.group_by()`
 method helps answer this type of query. You can pass a column to the
 `.group_by()`
 method and use in an aggregate function like
 `sum()`
 or
 `count()`
 . Much like the
 `.order_by()`
 method,
 `.group_by()`
 can take multiple columns as arguments.





```python

# Import func
from sqlalchemy import func

# Build a query to select the state and count of ages by state: stmt
stmt = select([census.columns.state, func.count(census.columns.age)])

# Group stmt by state
stmt = stmt.group_by(census.columns.state)

# Execute the statement and store all the records: results
results = connection.execute(stmt).fetchall()

# Print results
print(results)

# Print the keys/column names of the results returned
print(results[0].keys())

```




```

[('Alabama', 172), ('Alaska', 172), ('Arizona', 172), ('Arkansas', 172), ('California', 172),
...
('Wisconsin', 172), ('Wyoming', 172)]

['state', 'count_1']

```



 Notice that the key for the count method just came out as
 `count_1`
 . This can make it hard in complex queries to tell what column is being referred to: In the next exercise, you’ll practice assigning more descriptive labels when performing such calculations.



####
**Determining the population sum by state**



 To avoid confusion with query result column names like
 `count_1`
 , we can use the
 `.label()`
 method to provide a name for the resulting column. This gets appended to the function method we are using, and its argument is the name we want to use.




 We can pair
 `func.sum()`
 with
 `.group_by()`
 to get a sum of the population by
 `State`
 and use the
 `label()`
 method to name the output.




 We can also create the
 `func.sum()`
 expression before using it in the select statement. We do it the same way we would inside the select statement and store it in a variable. Then we use that variable in the select statement where the
 `func.sum()`
 would normally be.





```python

# Import func
from sqlalchemy import func

# Build an expression to calculate the sum of pop2008 labeled as population
pop2008_sum = func.sum(census.columns.pop2008).label('population')

# Build a query to select the state and sum of pop2008: stmt
stmt = select([census.columns.state, pop2008_sum])

# Group stmt by state
stmt = stmt.group_by(census.columns.state)

# Execute the statement and store all the records: results
results = connection.execute(stmt).fetchall()

# Print results
print(results)

# Print the keys/column names of the results returned
print(results[0].keys())

```




```

[('Alabama', 4649367), ('Alaska', 664546), ('Arizona', 6480767), ('Arkansas', 2848432),
...
('Wisconsin', 5625013), ('Wyoming', 529490)]

['state', 'population']

```




---



**2.4 Use pandas and matplotlib to visualize our data**
--------------------------------------------------------


####
**ResultsSets and pandas dataframes**




```python

# import pandas
import pandas as pd

# Create a DataFrame from the results: df
df = pd.DataFrame(results)

# Set column names
df.columns = results[0].keys()

# Print the Dataframe
print(df)

```




```

        state  population
0  California    36609002
1       Texas    24214127
2    New York    19465159
3     Florida    18257662
4    Illinois    12867077

```



 If you enjoy using pandas for your data scientific needs, you’ll want to always feed ResultProxies into pandas DataFrames!



####
**From SQLAlchemy results to a plot**




```python

# Import pyplot as plt from matplotlib
from matplotlib import pyplot as plt

# Create a DataFrame from the results: df
df = pd.DataFrame(results)

# Set Column names
df.columns = results[0].keys()

# Print the DataFrame
print(df)

# Plot the DataFrame
df.plot.bar()

plt.show()

```




```

        state  population
0  California    36609002
1       Texas    24214127
2    New York    19465159
3     Florida    18257662
4    Illinois    12867077

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-databases-in-python/capture5-8.png?w=642)


 You’re ready to learn about more advanced SQLAlchemy Queries!





---



# **3. Advanced SQLAlchemy Queries**
-----------------------------------


## **3.1 Calculating values in a query**


####
**Connecting to a MySQL database**




```python

# Import create_engine function
from sqlalchemy import create_engine

# Create an engine to the census database
engine = create_engine('mysql+pymysql://username:password@courses.csrrinzqubik.us-east-1.rds.amazonaws.com:3306/census')

# Print the table names
print(engine.table_names())

#  ['census', 'state_fact']

```


####
**Calculating a difference between two columns**



 Often, you’ll need to perform math operations as part of a query, such as if you wanted to calculate the change in population from 2000 to 2008. For math operations on numbers, the operators in SQLAlchemy work the same way as they do in Python.




 You can use these operators to perform addition (
 `+`
 ), subtraction (
 `-`
 ), multiplication (
 `*`
 ), division (
 `/`
 ), and modulus (
 `%`
 ) operations. Note: They behave differently when used with non-numeric column types.




 Let’s now find the top 5 states by population growth between 2000 and 2008.





```

census.columns.keys()
['state', 'sex', 'age', 'pop2000', 'pop2008']

connection.execute(select([census])).fetchmany(3)
[('Illinois', 'M', 0, 89600, 95012),
 ('Illinois', 'M', 1, 88445, 91829),
 ('Illinois', 'M', 2, 88729, 89547)]

```




```python

# Build query to return state names by population difference from 2008 to 2000: stmt
stmt = select([census.columns.state, (census.columns.pop2008-census.columns.pop2000).label('pop_change')])

# Append group by for the state: stmt_grouped
stmt_grouped = stmt.group_by(census.columns.state)

# Append order by for pop_change descendingly: stmt_ordered
stmt_ordered = stmt_grouped.order_by(desc('pop_change'))

# Return only 5 results: stmt_top5
stmt_top5 = stmt_ordered.limit(5)

# Use connection to execute stmt_top5 and fetch all results
results = connection.execute(stmt_top5).fetchall()

# Print the state and population change for each record
for result in results:
    print('{}:{}'.format(result.state, result.pop_change))


```




```

California:105705
Florida:100984
Texas:51901
New York:47098
Pennsylvania:42387

```


####
**Determining the overall percentage of women**



 It’s possible to combine functions and operators in a single select statement as well. These combinations can be exceptionally handy when we want to calculate percentages or averages, and we can also use the
 `case()`
 expression to operate on data that meets specific criteria while not affecting the query as a whole. The
 `case()`
 expression accepts a list of conditions to match and the column to return if the condition matches, followed by an
 `else_`
 if none of the conditions match. We can wrap this entire expression in any function or math operation we like.




 Often when performing integer division, we want to get a float back. While some databases will do this automatically, you can use the
 `cast()`
 function to convert an expression to a particular type.





```python

# import case, cast and Float from sqlalchemy
from sqlalchemy import case, cast, Float

# Build an expression to calculate female population in 2000
female_pop2000 = func.sum(
    case([
        (census.columns.sex == 'F', census.columns.pop2000)
    ], else_=0))

# Cast an expression to calculate total population in 2000 to Float
total_pop2000 = cast(func.sum(census.columns.pop2000), Float)

# Build a query to calculate the percentage of women in 2000: stmt
stmt = select([female_pop2000 / total_pop2000 * 100])

# Execute the query and store the scalar result: percent_female
percent_female = connection.execute(stmt).scalar()

# Print the percentage
print(percent_female)

# 51.0946743229

```




---


## **3.2 SQL relationships**


####
**Automatic joins with an established relationship**



 If you have two tables that already have an established relationship, you can automatically use that relationship by just adding the columns we want from each table to the select statement. Recall that Jason constructed the following query:





```

stmt = select([census.columns.pop2008, state_fact.columns.abbreviation])

```



 in order to join the
 `census`
 and
 `state_fact`
 tables and select the
 `pop2008`
 column from the first and the
 `abbreviation`
 column from the second. In this case, the
 `census`
 and
 `state_fact`
 tables had a pre-defined relationship: the
 `state`
 column of the former corresponded to the
 `name`
 column of the latter.




 In this exercise, you’ll use the same predefined relationship to select the
 `pop2000`
 and
 `abbreviation`
 columns!





```python

# Build a statement to join census and state_fact tables: stmt
stmt = select([census.columns.pop2000, state_fact.columns.abbreviation])

# Execute the statement and get the first result: result
result = connection.execute(stmt).first()

# Loop over the keys in the result object and print the key and value
for key in result.keys():
    print(key, getattr(result, key))

pop2000 89600
abbreviation IL

```




```

result
# (89600, 'IL')

```


####
**Joins**



 If you aren’t selecting columns from both tables or the two tables don’t have a defined relationship, you can still use the
 `.join()`
 method on a table to join it with another table and get extra data related to our query.




 The
 `join()`
 takes the table object you want to join in as the first argument and a condition that indicates how the tables are related to the second argument.




 Finally, you use the
 `.select_from()`
 method on the select statement to wrap the join clause.




 For example, the following code joins the
 `census`
 table to the
 `state_fact`
 table such that the
 `state`
 column of the
 `census`
 table corresponded to the
 `name`
 column of the
 `state_fact`
 table.





```

stmt = stmt.select_from(
    census.join(
        state_fact, census.columns.state ==
        state_fact.columns.name)

```




```

connection.execute(select([state_fact])).keys()
['id',
 'name',
 'abbreviation',
 'country',
 'type',
 'sort',
 'status',
 'occupied',
 'notes',
 'fips_state',
 'assoc_press',
 'standard_federal_region',
 'census_region',
 'census_region_name',
 'census_division',
 'census_division_name',
 'circuit_court']


connection.execute(select([state_fact])).fetchmany(2)
[('13', 'Illinois', 'IL', 'USA', 'state', '10', 'current', 'occupied', '', '17', 'Ill.', 'V', '2', 'Midwest', '3', 'East North Central', '7'),
 ('30', 'New Jersey', 'NJ', 'USA', 'state', '10', 'current', 'occupied', '', '34', 'N.J.', 'II', '1', 'Northeast', '2', 'Mid-Atlantic', '3')]


```




```python

# Build a statement to select the census and state_fact tables: stmt
stmt = select([census, state_fact])

# Add a select_from clause that wraps a join for the census and state_fact
# tables where the census state column and state_fact name column match
stmt_join = stmt.select_from(
    census.join(state_fact, census.columns.state == state_fact.columns.name))

# Execute the statement and get the first result: result
result = connection.execute(stmt_join).first()

# Loop over the keys in the result object and print the key and value
for key in result.keys():
    print(key, getattr(result, key))


```




```

state Illinois
sex M
age 0
pop2000 89600
pop2008 95012
id 13
name Illinois
abbreviation IL
country USA
type state
sort 10
status current
occupied occupied
notes
fips_state 17
assoc_press Ill.
standard_federal_region V
census_region 2
census_region_name Midwest
census_division 3
census_division_name East North Central
circuit_court 7


```


####
**More practice with joins**



 You can use the same select statement you built in the last exercise, however, let’s add a twist and only return a few columns and use the other table in a
 `group_by()`
 clause.





```

stmt = select([
            census.columns.state,
            func.sum(census.columns.pop2008),
            state_fact.columns.census_division_name
        ])

connection.execute(stmt).fetchmany(3)
# [('Texas', 15446707263, 'South Atlantic')]

```




```

stmt_joined = stmt.select_from(
            census.join(state_fact, census.columns.state == state_fact.columns.name)
        )

connection.execute(stmt_joined).fetchmany(3)
# [('Texas', 302287703, 'West South Central')]

```




```

stmt_grouped = stmt_joined.group_by(state_fact.columns.name)

connection.execute(stmt_grouped).fetchmany(3)
# [('Alabama', 4649367, 'East South Central'),
 ('Alaska', 664546, 'Pacific'),
 ('Arizona', 6480767, 'Mountain')]

```




```python

# Build a statement to select the state, sum of 2008 population and census
# division name: stmt
stmt = select([
    census.columns.state,
    func.sum(census.columns.pop2008),
    state_fact.columns.census_division_name
])

# Append select_from to join the census and state_fact tables by the census state and state_fact name columns
stmt_joined = stmt.select_from(
    census.join(state_fact, census.columns.state == state_fact.columns.name)
)

# Append a group by for the state_fact name column
stmt_grouped = stmt_joined.group_by(state_fact.columns.name)

# Execute the statement and get the results: results
results = connection.execute(stmt_grouped).fetchall()

# Loop over the results object and print each record.
for record in results:
    print(record)


```




```

('Alabama', 4649367, 'East South Central')
('Alaska', 664546, 'Pacific')
...
('Wisconsin', 5625013, 'East North Central')
('Wyoming', 529490, 'Mountain')

```



 The ability to join tables like this is what makes relational databases so powerful.





---


## **3.3 Working with hierarchical tables**


####
**Using alias to handle same table joined queries**



 Often, you’ll have tables that contain hierarchical data, such as employees and managers who are also employees. For this reason, you may wish to join a table to itself on different columns. The
 `.alias()`
 method, which creates a copy of a table, helps accomplish this task. Because it’s the same table, you only need a where clause to specify the join condition.




 Here, you’ll use the
 `.alias()`
 method to build a query to join the
 `employees`
 table against itself to determine to whom everyone reports.





```

employees.columns.keys()
# ['id', 'name', 'job', 'mgr', 'hiredate', 'sal', 'comm', 'dept']

connection.execute(select([employees.columns.name, employees.columns.mgr])).fetchmany(5)
# [('JOHNSON', 6), ('HARDING', 9), ('TAFT', 2), ('HOOVER', 2), ('LINCOLN', 6)]

```




```python

# Make an alias of the employees table: managers
managers = employees.alias()

# Build a query to select names of managers and their employees: stmt
stmt = select(
    [managers.columns.name.label('manager'),
     employees.columns.name.label('employee')]
)

# Match managers id with employees mgr: stmt_matched
stmt_matched = stmt.where(managers.columns.id == employees.columns.mgr)

# Order the statement by the managers name: stmt_ordered
stmt_ordered = stmt_matched.order_by(managers.columns.name)

# Execute statement: results
results = connection.execute(stmt_ordered).fetchall()

# Print records
for record in results:
    print(record)


```




```

('FILLMORE', 'GRANT')
('FILLMORE', 'ADAMS')
('FILLMORE', 'MONROE')
('GARFIELD', 'JOHNSON')
('GARFIELD', 'LINCOLN')
('GARFIELD', 'POLK')
('GARFIELD', 'WASHINGTON')
('HARDING', 'TAFT')
('HARDING', 'HOOVER')
('JACKSON', 'HARDING')
('JACKSON', 'GARFIELD')
('JACKSON', 'FILLMORE')
('JACKSON', 'ROOSEVELT')


```


####
**Leveraging functions and group_bys with hierarchical data**



 It’s also common to want to roll up data which is in a hierarchical table. Rolling up data requires making sure you’re careful which alias you use to perform the group_bys and which table you use for the function.




 Here, your job is to get a count of employees for each manager.





```

connection.execute(select([func.count(employees.columns.id)])).fetchmany(3)
[(14,)]

```




```python

# Make an alias of the employees table: managers
managers = employees.alias()

# Build a query to select names of managers and counts of their employees: stmt
stmt = select([managers.columns.name, func.count(employees.columns.id)])

# Append a where clause that ensures the manager id and employee mgr are equal
stmt_matched = stmt.where(managers.columns.id == employees.columns.mgr)

# Group by Managers Name
stmt_grouped = stmt_matched.group_by(managers.columns.name)

# Execute statement: results
results = connection.execute(stmt_grouped).fetchall()

# print manager
for record in results:
    print(record)


```




```

('FILLMORE', 3)
('GARFIELD', 4)
('HARDING', 2)
('JACKSON', 4)

```




---


## **3.4 Dealing with large ResultSets**


####
**Working on blocks of records**



 Sometimes you may have the need to work on a large ResultProxy, and you may not have the memory to load all the results at once.


 To work around that issue, you can get blocks of rows from the ResultProxy by using the
 `.fetchmany()`
 method inside a loop. With
 `.fetchmany()`
 , give it an argument of the number of records you want. When you reach an empty list, there are no more rows left to fetch, and you have processed all the results of the query.


 Then you need to use the
 `.close()`
 method to close out the connection to the database.





```

results_proxy
 <sqlalchemy.engine.result.ResultProxy at 0x7f4b7dc70160>

results_proxy.fetchmany(3)
[('Illinois', 'M', 0, 89600, 95012),
 ('Illinois', 'M', 1, 88445, 91829),
 ('Illinois', 'M', 2, 88729, 89547)]

results_proxy.fetchmany(3)
[('Illinois', 'M', 3, 88868, 90037),
 ('Illinois', 'M', 4, 91947, 91111),
 ('Illinois', 'M', 5, 93894, 89802)]


results_proxy.fetchmany(1)[0]
('Illinois', 'M', 9, 96436, 86055)

results_proxy.fetchmany(1)[0]['state']
'Illinois'

state_count
{}

```




```python

# Start a while loop checking for more results
while more_results:
    # Fetch the first 50 results from the ResultProxy: partial_results
    partial_results = results_proxy.fetchmany(50)

    # if empty list, set more_results to False
    if partial_results == []:
        more_results = False

    # Loop over the fetched records and increment the count for the state
    for row in partial_results:
        if row.state in state_count:
            state_count[row.state] += 1
        else:
            state_count[row.state] = 1

# Close the ResultProxy, and thus the connection
results_proxy.close()

# Print the count by state
print(state_count)

```




```

{'Illinois': 172, 'New Jersey': 172, 'District of Columbia': 172, 'North Dakota': 75, 'Florida': 172, 'Maryland': 49, 'Idaho': 172, 'Massachusetts': 16}

```



 As a data scientist, you’ll inevitably come across huge databases, and being able to work on them in blocks is a vital skill.





---



# **4. Creating and Manipulating your own Databases**
----------------------------------------------------


## **4.1 Creating databases and tables**


####
**Creating tables with SQLAlchemy**



 Previously, you used the
 `Table`
 object to reflect a table from an
 *existing*
 database, but what if you wanted to create a
 *new*
 table? You’d still use the
 `Table`
 object; however, you’d need to replace the
 `autoload`
 and
 `autoload_with`
 parameters with Column objects.




 The
 `Column`
 object takes a name, a SQLAlchemy type with an optional format, and optional keyword arguments for different constraints.




 When defining the table, recall how in the video Jason passed in
 `255`
 as the maximum length of a String by using
 `Column('name', String(255))`
 . Checking out the slides from the video may help: you can download them by clicking on ‘Slides’ next to the IPython Shell.




 After defining the table, you can create the table in the database by using the
 `.create_all()`
 method on metadata and supplying the engine as the only parameter. Go for it!





```

metadata
MetaData(bind=None)

engine
Engine(sqlite:///:memory:)

```




```python

# Import Table, Column, String, Integer, Float, Boolean from sqlalchemy
from sqlalchemy import Table, Column, String, Integer, Float, Boolean

# Define a new table with a name, count, amount, and valid column: data
data = Table('data', metadata,
             Column('name', String(255)),
             Column('count', Integer()),
             Column('amount', Float()),
             Column('valid', Boolean())
)

# Use the metadata to create the table
metadata.create_all(engine)

# Print table details
print(repr(data))


```




```

Table('data', MetaData(bind=None), Column('name', String(length=255), table=<data>), Column('count', Integer(), table=<data>), Column('amount', Float(), table=<data>), Column('valid', Boolean(), table=<data>), schema=None)

```



 When creating a table, it’s important to carefully think about what data types each column should be.



####
**Constraints and data defaults**



 You’re now going to practice creating a table with some constraints! Often, you’ll need to make sure that a column is unique, nullable, a positive value, or related to a column in another table. This is where constraints come in.




 You can also set a default value for the column if no data is passed to it via the
 `default`
 keyword on the column.





```python

# Import Table, Column, String, Integer, Float, Boolean from sqlalchemy
from sqlalchemy import Table, Column, String, Integer, Float, Boolean

# Define a new table with a name, count, amount, and valid column: data
data = Table('data', metadata,
             Column('name', String(255), unique=True),
             Column('count', Integer(), default=1),
             Column('amount', Float()),
             Column('valid', Boolean(), default=False)
)

# Use the metadata to create the table
metadata.create_all(engine)

# Print the table details
print(repr(metadata.tables['data']))


```




```

Table('data', MetaData(bind=None), Column('name', String(length=255), table=<data>), Column('count', Integer(), table=<data>, default=ColumnDefault(1)), Column('amount', Float(), table=<data>), Column('valid', Boolean(), table=<data>, default=ColumnDefault(False)), schema=None)

```




---


## **4.2 Inserting data into a table**


####
**Inserting a single row**



 There are several ways to perform an insert with SQLAlchemy; however, we are going to focus on the one that follows the same pattern as the
 `select`
 statement.




 It uses an
 `insert`
 statement where you specify the table as an argument, and supply the data you wish to insert into the value via the
 `.values()`
 method as keyword arguments





```

data
Table('data', MetaData(bind=None), Column('name', String(length=255), table=<data>), Column('count', Integer(), table=<data>), Column('amount', Float(), table=<data>), Column('valid', Boolean(), table=<data>), schema=None)

type(data)
sqlalchemy.sql.schema.Table


repr(data)
"Table('data', MetaData(bind=None), Column('name', String(length=255), table=<data>), Column('count', Integer(), table=<data>), Column('amount', Float(), table=<data>), Column('valid', Boolean(), table=<data>), schema=None)"

type(repr(data))
str

```




```python

# Import insert and select from sqlalchemy
from sqlalchemy import insert, select

# Build an insert statement to insert a record into the data table: insert_stmt
insert_stmt = insert(data).values(name='Anna', count=1, amount=1000.00, valid=True)

# Execute the insert statement via the connection: results
results = connection.execute(insert_stmt)

# Print result rowcount
print(results.rowcount)
# 1

# Build a select statement to validate the insert: select_stmt
select_stmt = select([data]).where(data.columns.name == 'Anna')

# Print the result of executing the query.
print(connection.execute(select_stmt).first())
# ('Anna', 1, 1000.0, True)

```


####
**Inserting multiple records at once**



 When inserting multiple records at once, you do not use the
 `.values()`
 method. Instead, you’ll want to first build a
 **list of dictionaries**
 that represents the data you want to insert, with keys being the names of the columns.




 In the
 `.execute()`
 method, you can pair this list of dictionaries with an
 `insert`
 statement, which will insert all the records in your list of dictionaries.





```python

# Build a list of dictionaries: values_list
values_list = [
    {'name': 'Anna', 'count': 1, 'amount': 1000.00, 'valid': True},
    {'name': 'Taylor', 'count': 1, 'amount': 750.00, 'valid': False}]

# Build an insert statement for the data table: stmt
stmt = insert(data)

# Execute stmt with the values_list: results
results = connection.execute(stmt, values_list)

# Print rowcount
print(results.rowcount)
# 2

```




```

connection.execute(select([data])).fetchmany(3)
# [('Anna', 1, 1000.0, True), ('Taylor', 1, 750.0, False)]

```


####
**Loading a CSV into a table**



 You’re now going to learn how to load the contents of a CSV file into a table.




 One way to do that would be to read a CSV file line by line, create a dictionary from each line, and then use
 `insert()`
 , like you did in the previous exercise.




 But there is a faster way using
 `pandas`
 . You can read a CSV file into a DataFrame using the
 `read_csv()`
 function (this function should be familiar to you, but you can run
 `help(pd.read_csv)`
 in the console to refresh your memory!). Then, you can call the
 [`.to_sql()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html)
 method on the DataFrame to load it into a SQL table in a database. The columns of the DataFrame should match the columns of the SQL table.




`.to_sql()`
 has many parameters, but in this exercise we will use the following:



* `name`
 is the name of the SQL table (as a string).
* `con`
 is the connection to the database that you will use to upload the data.
* `if_exists`
 specifies how to behave if the table already exists in the database; possible values are
 `"fail"`
 ,
 `"replace"`
 , and
 `"append"`
 .
* `index`
 (
 `True`
 or
 `False`
 ) specifies whether to write the DataFrame’s index as a column.



 In this exercise, you will upload the data contained in the
 `census.csv`
 file into an existing table
 `"census"`
 . The
 `connection`
 to the database has already been created for you.





```python

# import pandas
import pandas as pd

# read census.csv into a dataframe : census_df
census_df = pd.read_csv("census.csv", header=None)

# rename the columns of the census dataframe
census_df.columns = ['state', 'sex', 'age', 'pop2000', 'pop2008']

census_df.head(3)

```




```

      state sex  age  pop2000  pop2008
0  Illinois   M    0    89600    95012
1  Illinois   M    1    88445    91829
2  Illinois   M    2    88729    89547

```




```

connection
<sqlalchemy.engine.base.Connection at 0x7feca8d64e10>

```




```python

# append the data from census_df to the "census" table via connection
census_df.to_sql(name='census', con=connection, if_exists='append', index=False)

```




```

connection.execute(select([func.count(census)])).fetchmany(3)
[(8772,)]

census_df.shape
(8772, 5)

```



 The
 `pandas`
 package provides us with an efficient way to load a DataFrames into a SQL table. If you would create and execute a statement to select all records
 `from`
 census, you would see that there are 8772 rows in the table.





---


## **4.3 Updating data in a database**


####
**Updating individual records**



 The
 `update`
 statement is very similar to an
 `insert`
 statement. For example, you can update all wages in the
 `employees`
 table as follows:





```

stmt = update(employees).values(wage=100.00)

```



 The
 `update`
 statement also typically uses a
 `where`
 clause to help us determine what data to update. For example, to only update the record for the employee with ID 15, you would append the previous statement as follows:





```

stmt = stmt.where(employees.columns.id == 15)

```




```

repr(state_fact)
#  "Table('state_fact', MetaData(bind=None), Column('id', VARCHAR(length=256), table=<state_fact>), Column('name', VARCHAR(length=256), table=<state_fact>), Column('abbreviation', VARCHAR(length=256), table=<state_fact>), Column('country', VARCHAR(length=256), table=<state_fact>), Column('type', VARCHAR(length=256), table=<state_fact>), Column('sort', VARCHAR(length=256), table=<state_fact>), Column('status', VARCHAR(length=256), table=<state_fact>), Column('occupied', VARCHAR(length=256), table=<state_fact>), Column('notes', VARCHAR(length=256), table=<state_fact>), Column('fips_state', VARCHAR(length=256), table=<state_fact>), Column('assoc_press', VARCHAR(length=256), table=<state_fact>), Column('standard_federal_region', VARCHAR(length=256), table=<state_fact>), Column('census_region', VARCHAR(length=256), table=<state_fact>), Column('census_region_name', VARCHAR(length=256), table=<state_fact>), Column('census_division', VARCHAR(length=256), table=<state_fact>), Column('census_division_name', VARCHAR(length=256), table=<state_fact>), Column('circuit_court', VARCHAR(length=256), table=<state_fact>), schema=None)"


```




```python

# Build a select statement: select_stmt
select_stmt = select([state_fact]).where(state_fact.columns.name == 'New York')

# Execute select_stmt and fetch the results
results = connection.execute(select_stmt).fetchall()

# Print the results of executing the select_stmt
print(results)
# [('32', 'New York', 'NY', 'USA', 'state', '10', 'current', 'occupied', '', '0', 'N.Y.', 'II', '1', 'Northeast', '2', 'Mid-Atlantic', '2')]

# Print the FIPS code for the first row of the result
print(results[0]['fips_state'])
# 0

```




```

select_stmt = select([state_fact]).where(state_fact.columns.name == 'New York')
results = connection.execute(select_stmt).fetchall()
print(results)
print(results[0]['fips_state'])

# Build a statement to update the fips_state to 36: update_stmt
update_stmt = update(state_fact).values(fips_state = 36)

# Append a where clause to limit it to records for New York state
update_stmt = update_stmt.where(state_fact.columns.name == 'New York')

# Execute the statement: update_results
update_results = connection.execute(update_stmt)

```




```python

# Execute select_stmt again and fetch the new results
new_results = connection.execute(select_stmt).fetchall()

# Print the new_results
print(new_results)
# [('32', 'New York', 'NY', 'USA', 'state', '10', 'current', 'occupied', '', '36', 'N.Y.', 'II', '1', 'Northeast', '2', 'Mid-Atlantic', '2')]

# Print the FIPS code for the first row of the new_results
print(new_results[0]['fips_state'])
# 36

```


####
**Updating multiple records**



 By using a
 `where`
 clause that selects more records, you can update multiple records at once.




 Unlike inserting, updating multiple records works exactly the same way as updating a single record (as long as you are updating them with the same value).





```python

# Build a statement to update the notes to 'The Wild West': stmt
stmt = update(state_fact).values(notes='The Wild West')

# Append a where clause to match the West census region records: stmt_west
stmt_west = stmt.where(state_fact.columns.census_region_name == 'West')

# Execute the statement: results
results = connection.execute(stmt_west)

# Print rowcount
print(results.rowcount)
# 13

```


####
**Correlated updates**



 You can also update records with data from a select statement. This is called a correlated update. It works by defining a
 `select`
 statement that returns the value you want to update the record with and assigning that select statement as the value in
 `update`
 .





```

state_fact.columns.keys()
['id',
 'name',
...
 'fips_state',
...]

connection.execute(select([state_fact])).fetchmany(3)
# [('13', 'Illinois', 'IL', 'USA', 'state', '10', 'current', 'occupied', '', '17', 'Ill.', 'V', '2', 'Midwest', '3', 'East North Central', '7'),
 ('30', 'New Jersey', 'NJ', 'USA', 'state', '10', 'current', 'occupied', '', '34', 'N.J.', 'II', '1', 'Northeast', '2', 'Mid-Atlantic', '3'),
 ('34', 'North Dakota', 'ND', 'USA', 'state', '10', 'current', 'occupied', '', '38', 'N.D.', 'VIII', '2', 'Midwest', '4', 'West North Central', '8')]


flat_census.columns.keys()
# ['state_name', 'fips_code']

connection.execute(select([flat_census])).fetchmany(3)
#  [(None, '17'), (None, '34'), (None, '38')]

```




```python

# Build a statement to select name from state_fact: fips_stmt
fips_stmt = select([state_fact.columns.name])

# Append a where clause to match the fips_state to flat_census fips_code: fips_stmt
fips_stmt = fips_stmt.where(
    state_fact.columns.fips_state == flat_census.columns.fips_code)

# Build an update statement to set the name to fips_stmt_where: update_stmt
update_stmt = update(flat_census).values(state_name=fips_stmt)

# Execute update_stmt: results
results = connection.execute(update_stmt)

# Print rowcount
print(results.rowcount)
# 51

```




```

connection.execute(select([flat_census])).fetchmany(3)
[('Illinois', '17'), ('New Jersey', '34'), ('North Dakota', '38')]

```




---


## **4.4 Removing data from a database**


####
**Deleting all the records from a table**



 Often, you’ll need to empty a table of all of its records so you can reload the data. You can do this with a
 `delete`
 statement with just the table as an argument. For example, delete the table
 `extra_employees`
 by executing as follows:





```

delete_stmt = delete(extra_employees)
result_proxy = connection.execute(delete_stmt)

```



**Do be careful, though, as deleting cannot be undone!**





```python

# Import delete, select
from sqlalchemy import delete, select

# Build a statement to empty the census table: stmt
delete_stmt = delete(census)

# Execute the statement: results
results = connection.execute(delete_stmt)

# Print affected rowcount
print(results.rowcount)
# 8772

# Build a statement to select all records from the census table : select_stmt
select_stmt = select([census])

# Print the results of executing the statement to verify there are no rows
print(connection.execute(select_stmt).fetchall())
# []

```



 As you can see, there are no records left in the
 `census`
 table after executing the delete statement!



####
**Deleting specific records**



 By using a
 `where()`
 clause, you can target the
 `delete`
 statement to remove only certain records. For example, delete all rows from the
 `employees`
 table that had
 `id`
 3 with the following delete statement:





```

delete(employees).where(employees.columns.id == 3)

```



 Here you’ll delete ALL rows which have
 `'M'`
 in the
 `sex`
 column and
 `36`
 in the
 `age`
 column.




 We have included code at the start which computes the total number of these rows. It is important to make sure that this is the number of rows that you actually delete.





```python

# Build a statement to count records using the sex column for Men ('M') age 36: count_stmt
count_stmt = select([func.count(census.columns.sex)]).where(
    and_(census.columns.sex == 'M',
         census.columns.age == 36)
)

# Execute the select statement and use the scalar() fetch method to save the record count
to_delete = connection.execute(count_stmt).scalar()

# Build a statement to delete records from the census table: delete_stmt
delete_stmt = delete(census)

# Append a where clause to target Men ('M') age 36: delete_stmt
delete_stmt = delete_stmt.where(
    and_(census.columns.sex == 'M',
         census.columns.age == 36))

# Execute the statement: results
results = connection.execute(delete_stmt)

# Print affected rowcount and to_delete record count, make sure they match
print(results.rowcount, to_delete)
# 51 51

```



 You may frequently be required to remove specific records from a table, like in this case.



####
**Deleting a table completely**



 You’re now going to practice dropping individual tables from a database with the
 `.drop()`
 method, as well as
 *all*
 tables in a database with the
 `.drop_all()`
 method!




 As Spider-Man’s Uncle Ben said:
 **With great power, comes great responsibility.**
 Do be careful when deleting tables, as it’s not simple or fast to restore large databases!




 Remember, you can check to see if a table exists on an
 `engine`
 with the
 `.exists(engine)`
 method.





```

engine
# Engine(sqlite:///census.sqlite)

repr(state_fact)
# "Table('state_fact', MetaData(bind=None), Column('id', VARCHAR(length=256), table=<state_fact>),
...
Column('circuit_court', VARCHAR(length=256), table=<state_fact>), schema=None)"

state_fact.exists(engine)
# True

repr(metadata.tables)
#  "immutabledict({'census': Table('census', MetaData(bind=None), Column('state', VARCHAR(length=30), table=<census>), ... , Column('pop2008', INTEGER(), table=<census>), schema=None), 'state_fact': Table('state_fact', MetaData(bind=None), Column('id', VARCHAR(length=256), table=<state_fact>), Column('circuit_court', VARCHAR(length=256), ... , table=<state_fact>), schema=None)})"

```




```python

# Drop the state_fact table
state_fact.drop(engine)

# Check to see if state_fact exists
print(state_fact.exists(engine))
# False

# Drop all tables
metadata.drop_all(engine)

# Check to see if census exists
print(census.exists(engine))
# False


```




---



# **5. Putting it all together**
-------------------------------


## **5.1 Census case study**


####
**Setup the engine and metadata**



 In this exercise, your job is to create an engine to the database that will be used in this chapter. Then, you need to initialize its metadata.




 Recall how you did this in Chapter 1 by leveraging
 `create_engine()`
 and
 `MetaData()`
 .





```python

# Import create_engine, MetaData
from sqlalchemy import create_engine, MetaData

# Define an engine to connect to chapter5.sqlite: engine
engine = create_engine('sqlite:///chapter5.sqlite')

# Initialize MetaData: metadata
metadata = MetaData()


```




```

engine
# Engine(sqlite:///chapter5.sqlite)

type(engine)
# sqlalchemy.engine.base.Engine

metadata
# MetaData(bind=None)

type(metadata)
# sqlalchemy.sql.schema.MetaData

```


####
**Create the table to the database**



 Having setup the engine and initialized the metadata, you will now define the
 `census`
 table object and then create it in the database using the
 `metadata`
 and
 `engine`
 from the previous exercise.




 To create it in the database, you will have to use the
 `.create_all()`
 method on the
 `metadata`
 with
 `engine`
 as the argument.





```python

# Import Table, Column, String, and Integer
from sqlalchemy import Table, Column, String, Integer

# Build a census table: census
census = Table('census', metadata,
               Column('state', String(30)),
               Column('sex', String(1)),
               Column('age', Integer()),
               Column('pop2000', Integer()),
               Column('pop2008', Integer()))

# Create the table in the database
metadata.create_all(engine)


```



 When creating columns of type
 `String()`
 , it’s important to spend some time thinking about what their maximum lengths should be.





---


## **5.2 Populating the database**


####
**Reading the data from the CSV**



 Leverage the Python CSV module from the standard library and load the data into a list of dictionaries.





```

csv_reader
<_csv.reader at 0x7f3e282604a8>

```




```python

# Create an empty list: values_list
values_list = []

# Iterate over the rows
for row in csv_reader:
    # Create a dictionary with the values
    data = {'state': row[0], 'sex': row[1], 'age':row[2], 'pop2000': row[3],
            'pop2008': row[4]}
    # Append the dictionary to the values list
    values_list.append(data)

values_list[:2]

```




```

[{'age': '0',
  'pop2000': '89600',
  'pop2008': '95012',
  'sex': 'M',
  'state': 'Illinois'},
 {'age': '1',
  'pop2000': '88445',
  'pop2008': '91829',
  'sex': 'M',
  'state': 'Illinois'}]

```


####
**Load data from a list into the Table**




```python

# Import insert
from sqlalchemy import insert

# Build insert statement: stmt
stmt = insert(census)

# Use values_list to insert data: results
results = connection.execute(stmt, values_list)

# Print rowcount
print(results.rowcount)
# 8772

from sqlalchemy import select
connection.execute(select([census])).fetchmany(3)

[('Illinois', 'M', 0, 89600, 95012),
 ('Illinois', 'M', 1, 88445, 91829),
 ('Illinois', 'M', 2, 88729, 89547)]

```




---


## **5.3 Example queries**


####
**Determine the average age by population**



 To calculate a weighted average, we first find the total sum of weights multiplied by the values we’re averaging, then divide by the sum of all the weights.




 In this exercise, however, you will make use of
 **`func.sum()`**
 together with
 `select`
 to select the weighted average of a column from a table. You will still work with the
 `census`
 data, and you will compute the average of age weighted by state population in the year 2000, and then group this weighted average by sex.





```python

# Import select and func
from sqlalchemy import select, func

# Select sex and average age weighted by 2000 population
stmt = select([(func.sum(census.columns.pop2000 * census.columns.age)
  					/ func.sum(census.columns.pop2000)).label('average_age'),
               census.columns.sex])

# Group by sex
stmt = stmt.group_by(census.columns.sex)

# Execute the query and fetch all the results
results = connection.execute(stmt).fetchall()

# Print the sex and average age column for each result
for result in results:
    print(result.sex, result.average_age)

# F 37
# M 34

results
# [(37, 'F'), (34, 'M')]

```


####
**Determine the percentage of population by gender and state**



 In this exercise, you will write a query to determine the percentage of the population in 2000 that comprised of women. You will group this query by state.





```python

# import case, cast and Float from sqlalchemy
from sqlalchemy import case, cast, Float

# Build a query to calculate the percentage of women in 2000: stmt
stmt = select([census.columns.state,
    (func.sum(
        case([
            (census.columns.sex == 'F', census.columns.pop2000)
        ], else_=0)) /
     cast(func.sum(census.columns.pop2000), Float) * 100).label('percent_female')
])

# Group By state
stmt = stmt.group_by(census.columns.state)

# Execute the query and store the results: results
results = connection.execute(stmt).fetchall()

# Print the percentage
for result in results:
    print(result.state, result.percent_female)


```




```

Alabama 51.8324077702
Alaska 49.3014978935
...
Wisconsin 50.6148645265
Wyoming 49.9459554265

```


####
**Determine the difference by state from the 2000 and 2008 censuses**



 In this final exercise, you will write a query to calculate the states that changed the most in population. You will limit your query to display only the top 10 states.





```python

# Build query to return state name and population difference from 2008 to 2000
stmt = select([census.columns.state,
     (census.columns.pop2008-census.columns.pop2000).label('pop_change')
])

# Group by State
stmt = stmt.group_by(census.columns.state)

# Order by Population Change
stmt = stmt.order_by(desc('pop_change'))

# Limit to top 10
stmt = stmt.limit(10)

# Use connection to execute the statement and fetch all results
results = connection.execute(stmt).fetchall()

# Print the state and population change for each record
for result in results:
    print('{}:{}'.format(result.state, result.pop_change))

```




```

California:105705
Florida:100984
Texas:51901
New York:47098
Pennsylvania:42387
Arizona:29509
Ohio:29392
Illinois:26221
Michigan:25126
North Carolina:24108

```



 The End.


 Thank you for reading.



