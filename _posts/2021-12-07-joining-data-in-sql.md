---
title: Joining Data in SQL
date: 2021-12-07 11:22:10 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Joining Data in SQL
====================







 This is the memo of the 18th course of ‘Data Scientist with Python’ track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/joining-data-in-postgresql)**
 .




 Further Reading:


[More dangerous subtleties of JOINs in SQL](https://alexpetralia.com/posts/2017/7/19/more-dangerous-subtleties-of-joins-in-sql)
 — Be careful when JOIN tables with duplications or NULLs





---



**Introduction to joins**
--------------------------


###
 Introduction to INNER JOIN


####
 Inner join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture-19.png)





```

SELECT table_name
FROM information_schema.tables
-- Specify the correct table_schema value
WHERE table_schema = 'public';


table_name
cities
countries
languages
economies
currencies
populations

```




```

SELECT *
FROM left_table
INNER JOIN right_table
ON left_table.id = right_table.id;

```




```

-- 1. Select name fields (with alias) and region
SELECT cities.name AS city, countries.name AS country, region
FROM cities
  INNER JOIN countries
    ON cities.country_code = countries.code;

city	country	region
Abidjan	Cote d'Ivoire	Western Africa
Abu Dhabi	United Arab Emirates	Middle East
Abuja	Nigeria	Western Africa

```


####
 Inner join (2)




```

SELECT c1.name AS city, c2.name AS country
FROM cities AS c1
INNER JOIN countries AS c2
ON c1.country_code = c2.code;

```




```

-- 3. Select fields with aliases
SELECT c.code AS country_code, name, year, inflation_rate
FROM countries AS c
  -- 1. Join to economies (alias e)
  INNER JOIN economies AS e
    -- 2. Match on code
    ON c.code = e.code;

```


####
 Inner join (3)




```

SELECT *
FROM left_table
  INNER JOIN right_table
    ON left_table.id = right_table.id
  INNER JOIN another_table
    ON left_table.id = another_table.id;

```




```

-- 6. Select fields
SELECT c.code, name, region, e.year, fertility_rate, unemployment_rate
  -- 1. From countries (alias as c)
  FROM countries AS c
  -- 2. Join to populations (as p)
  INNER JOIN populations AS p
    -- 3. Match on country code
    ON c.code = p.country_code
  -- 4. Join to economies (as e)
  INNER JOIN economies AS e
    -- 5. Match on country code
    ON c.code = e.code;


```




```

-- countries INNER JOIN populations table
code	name	fertility_rate
ABW	Aruba	1.704
ABW	Aruba	1.647
AFG	Afghanistan	5.746
AFG	Afghanistan	4.653

-- economies table
econ_id	code	year
1	AFG	2010
2	AFG	2015


code	name	region	year	fertility_rate	unemployment_rate
AFG	Afghanistan	Southern and Central Asia	2010	4.653	null
AFG	Afghanistan	Southern and Central Asia	2010	5.746	null
AFG	Afghanistan	Southern and Central Asia	2015	4.653	null
AFG	Afghanistan	Southern and Central Asia	2015	5.746	null
AGO	Angola	Central Africa	2010	5.996	null

```




```

-- 6. Select fields
SELECT c.code, name, region, e.year, fertility_rate, unemployment_rate
  -- 1. From countries (alias as c)
  FROM countries AS c
  -- 2. Join to populations (as p)
  INNER JOIN populations AS p
    -- 3. Match on country code
    ON c.code = p.country_code
  -- 4. Join to economies (as e)
  INNER JOIN economies AS e
    -- 5. Match on country code and year
    ON c.code = e.code AND p.year = e.year;

code	name	region	year	fertility_rate	unemployment_rate
AFG	Afghanistan	Southern and Central Asia	2010	5.746	null
AFG	Afghanistan	Southern and Central Asia	2015	4.653	null

```


###
 INNER JOIN via USING


####
 Inner join with using




```

SELECT *
FROM countries
  INNER JOIN economies
    ON countries.code = economies.code

-- is equal to

SELECT *
FROM countries
  INNER JOIN economies
    USING(code)

```




```

-- 4. Select fields
SELECT c.name AS country, continent, l.name AS language, official
  -- 1. From countries (alias as c)
  FROM countries as c
  -- 2. Join to languages (as l)
  INNER JOIN languages as l
    -- 3. Match using code
    USING (code)


country	continent	language	official
Afghanistan	Asia	Dari	true
Afghanistan	Asia	Pashto	true

```


###
 Self-ish joins, just in CASE


####
 Self-join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture1-19.png)



```

pop_id	country_code	year	fertility_rate	life_expectancy	size
20	ABW	2010	1.704	74.9535	101597
19	ABW	2015	1.647	75.5736	103889

```




```

-- 4. Select fields with aliases
SELECT p1.country_code, p1.size AS size2010, p2.size AS size2015
-- 1. From populations (alias as p1)
FROM populations AS p1
  -- 2. Join to itself (alias as p2)
  INNER JOIN populations AS p2
    -- 3. Match on country code
    ON p1.country_code = p2.country_code


country_code	size2010	size2015
ABW	101597	103889
ABW	101597	101597
ABW	103889	103889
ABW	103889	101597

```




```

-- 5. Select fields with aliases
SELECT p1.country_code,
       p1.size AS size2010,
       p2.size AS size2015
-- 1. From populations (alias as p1)
FROM populations as p1
  -- 2. Join to itself (alias as p2)
  INNER JOIN populations as p2
    -- 3. Match on country code
    ON p1.country_code = p2.country_code
        -- 4. and year (with calculation)
        AND p1.year = p2.year - 5


country_code	size2010	size2015
ABW	101597	103889
AFG	27962200	32526600
AGO	21220000	25022000
ALB	2913020	2889170

```




```

-- With two numeric fields A and B, the percentage growth from A to B can be calculated as (B−A)/A∗100.0.

SELECT p1.country_code,
       p1.size AS size2010,
       p2.size AS size2015,
       -- 1. calculate growth_perc
       ((p2.size - p1.size)/p1.size * 100.0) AS growth_perc
-- 2. From populations (alias as p1)
FROM populations AS p1
  -- 3. Join to itself (alias as p2)
  INNER JOIN populations AS p2
    -- 4. Match on country code
    ON p1.country_code = p2.country_code
        -- 5. and year (with calculation)
        AND p1.year = p2.year - 5;


country_code	size2010	size2015	growth_perc
ABW	101597	103889	2.25597210228443
AFG	27962200	32526600	16.32329672575
AGO	21220000	25022000	17.9171919822693
ALB	2913020	2889170	-0.818874966353178

```


####
 Case when and then




```

SELECT name, continent, code, surface_area,
    -- 1. First case
    CASE WHEN surface_area > 2000000 THEN 'large'
        -- 2. Second case
        WHEN surface_area > 350000 THEN 'medium'
        -- 3. Else clause + end
        ELSE 'small' END
        -- 4. Alias name
        AS geosize_group
-- 5. From table
FROM countries;


name	continent	code	surface_area	geosize_group
Afghanistan	Asia	AFG	652090	medium
Netherlands	Europe	NLD	41526	small
Albania	Europe	ALB	28748	small

```


####
 Inner challenge




```

SELECT name, continent, code, surface_area,
    CASE WHEN surface_area > 2000000
            THEN 'large'
       WHEN surface_area > 350000
            THEN 'medium'
       ELSE 'small' END
       AS geosize_group
INTO countries_plus
FROM countries;


name	continent	code	surface_area	geosize_group
Afghanistan	Asia	AFG	652090	medium
Netherlands	Europe	NLD	41526	small
Albania	Europe	ALB	28748	small
Algeria	Africa	DZA	2381740	large

```




```

SELECT country_code, size,
    -- 1. First case
    CASE WHEN size > 50000000 THEN 'large'
        -- 2. Second case
        WHEN size > 1000000 THEN 'medium'
        -- 3. Else clause + end
        ELSE 'small' END
        -- 4. Alias name
        AS popsize_group
-- 5. From table
FROM populations
-- 6. Focus on 2015
WHERE year = 2015;


country_code	size	popsize_group
ABW	103889	small
AFG	32526600	medium
AGO	25022000	medium
ALB	2889170	medium

```




```

SELECT country_code, size,
    CASE WHEN size > 50000000 THEN 'large'
        WHEN size > 1000000 THEN 'medium'
        ELSE 'small' END
        AS popsize_group
-- 1. Into table
INTO pop_plus
FROM populations
WHERE year = 2015;

-- 2. Select all columns of pop_plus
SELECT * FROM pop_plus;


country_code	size	popsize_group
ABW	103889	small
AFG	32526600	medium
AGO	25022000	medium
ALB	2889170	medium

```




```

SELECT country_code, size,
  CASE WHEN size > 50000000
            THEN 'large'
       WHEN size > 1000000
            THEN 'medium'
       ELSE 'small' END
       AS popsize_group
INTO pop_plus
FROM populations
WHERE year = 2015;

-- 5. Select fields
SELECT name, continent, geosize_group, popsize_group
-- 1. From countries_plus (alias as c)
FROM countries_plus AS c
  -- 2. Join to pop_plus (alias as p)
  INNER JOIN pop_plus AS p
    -- 3. Match on country code
    ON c.code = p.country_code
-- 4. Order the table
ORDER BY geosize_group;


name	continent	geosize_group	popsize_group
India	Asia	large	large
United States	North America	large	large
Saudi Arabia	Asia	large	medium
China	Asia	large	large

```




---



**Outer joins and cross joins**
--------------------------------


###
 LEFT and RIGHT JOINs



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture-18.png)

####
 Left Join




```

-- Select the city name (with alias), the country code,
-- the country name (with alias), the region,
-- and the city proper population
SELECT c1.name AS city, code, c2.name AS country,
       region, city_proper_pop
-- From left table (with alias)
FROM cities AS c1
  -- Join to right table (with alias)
  INNER JOIN countries AS c2
    -- Match on country code
    ON c1.country_code = c2.code
-- Order by descending country code
ORDER BY code DESC;


city	code	country	region	city_proper_pop
Harare	ZWE	Zimbabwe	Eastern Africa	1606000
Lusaka	ZMB	Zambia	Eastern Africa	1742980
Cape Town	ZAF	South Africa	Southern Africa	3740030

-- 230 rows

```




```

SELECT c1.name AS city, code, c2.name AS country,
       region, city_proper_pop
FROM cities AS c1
  -- 1. Join right table (with alias)
  LEFT JOIN countries AS c2
    -- 2. Match on country code
    ON c1.country_code = c2.code
-- 3. Order by descending country code
ORDER BY code DESC;


city	code	country	region	city_proper_pop
Taichung	null	null	null	2752410
Tainan	null	null	null	1885250
Kaohsiung	null	null	null	2778920
Bucharest	null	null	null	1883420

-- 236 rows

```


####
 Left join (2)




```

/*
5. Select country name AS country, the country's local name,
the language name AS language, and
the percent of the language spoken in the country
*/
SELECT c.name AS country, local_name, l.name AS language, percent
-- 1. From left table (alias as c)
FROM countries AS c
  -- 2. Join to right table (alias as l)
  INNER JOIN languages AS l
    -- 3. Match on fields
    ON c.code = l.code
-- 4. Order by descending country
ORDER BY country DESC;


country	local_name	language	percent
Zimbabwe	Zimbabwe	Shona	null
Zimbabwe	Zimbabwe	Tonga	null
Zimbabwe	Zimbabwe	Tswana	null

-- 914 rows

```




```

/*
5. Select country name AS country, the country's local name,
the language name AS language, and
the percent of the language spoken in the country
*/
SELECT c.name AS country, local_name, l.name AS language, percent
-- 1. From left table (alias as c)
FROM countries AS c
  -- 2. Join to right table (alias as l)
  LEFT JOIN languages AS l
    -- 3. Match on fields
    ON c.code = l.code
-- 4. Order by descending country
ORDER BY country DESC;


country	local_name	language	percent
Zimbabwe	Zimbabwe	Chibarwe	null
Zimbabwe	Zimbabwe	Shona	null
Zimbabwe	Zimbabwe	Ndebele	null
Zimbabwe	Zimbabwe	English	null

-- 921 rows

```


####
 Left join (3)




```

-- 5. Select name, region, and gdp_percapita
SELECT name, region, gdp_percapita
-- 1. From countries (alias as c)
FROM countries AS c
  -- 2. Left join with economies (alias as e)
  LEFT JOIN economies AS e
    -- 3. Match on code fields
    ON c.code = e.code
-- 4. Focus on 2010
WHERE year = 2010;


name	region	gdp_percapita
Afghanistan	Southern and Central Asia	539.667
Angola	Central Africa	3599.27
Albania	Southern Europe	4098.13
United Arab Emirates	Middle East	34628.6

```




```

-- Select fields
SELECT region, AVG(gdp_percapita) AS avg_gdp
-- From countries (alias as c)
FROM countries AS c
  -- Left join with economies (alias as e)
  LEFT JOIN economies AS e
    -- Match on code fields
    ON c.code = e.code
-- Focus on 2010
WHERE year = 2010
-- Group by region
GROUP BY region;


region	avg_gdp
Southern Africa	5051.59797363281
Australia and New Zealand	44792.384765625
Southeast Asia	10547.1541320801

```




```

-- Select fields
SELECT region, AVG(gdp_percapita) AS avg_gdp
-- From countries (alias as c)
FROM countries AS c
  -- Left join with economies (alias as e)
  LEFT JOIN economies AS e
    -- Match on code fields
    ON c.code = e.code
-- Focus on 2010
WHERE year = 2010
-- Group by region
GROUP BY region
-- Order by descending avg_gdp
ORDER BY avg_gdp DESC;


region	avg_gdp
Western Europe	58130.9614955357
Nordic Countries	57073.99765625
North America	47911.509765625
Australia and New Zealand	44792.384765625

```


####
 Right join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture3-17.png)



```

-- convert this code to use RIGHT JOINs instead of LEFT JOINs
/*
SELECT cities.name AS city, urbanarea_pop, countries.name AS country,
       indep_year, languages.name AS language, percent
FROM cities
  LEFT JOIN countries
    ON cities.country_code = countries.code
  LEFT JOIN languages
    ON countries.code = languages.code
ORDER BY city, language;
*/

SELECT cities.name AS city, urbanarea_pop, countries.name AS country,
       indep_year, languages.name AS language, percent
FROM languages
  RIGHT JOIN countries
    ON languages.code = countries.code
  RIGHT JOIN cities
    ON countries.code = cities.country_code
ORDER BY city, language;


city	urbanarea_pop	country	indep_year	language	percent
Abidjan	4765000	Cote d'Ivoire	1960	French	null
Abidjan	4765000	Cote d'Ivoire	1960	Other	null
Abu Dhabi	1145000	United Arab Emirates	1971	Arabic	null

```


###
 FULL JOINs


####
 Full join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture4-16.png)



```

SELECT name AS country, code, region, basic_unit
-- 3. From countries
FROM countries
  -- 4. Join to currencies
  FULL JOIN currencies
    -- 5. Match on code
    USING (code)
-- 1. Where region is North America or null
WHERE region = 'North America' OR region IS NULL
-- 2. Order by region
ORDER BY region;


country	code	region	basic_unit
Greenland	GRL	North America	null
null	TMP	null	United States dollar
null	FLK	null	Falkland Islands pound
null	AIA	null	East Caribbean dollar
null	NIU	null	New Zealand dollar
null	ROM	null	Romanian leu
null	SHN	null	Saint Helena pound
null	SGS	null	British pound
null	TWN	null	New Taiwan dollar
null	WLF	null	CFP franc
null	MSR	null	East Caribbean dollar
null	IOT	null	United States dollar
null	CCK	null	Australian dollar
null	COK	null	New Zealand dollar

```




```

SELECT name AS country, code, region, basic_unit
-- 1. From countries
FROM countries
  -- 2. Join to currencies
  LEFT JOIN currencies
    -- 3. Match on code
    USING (code)
-- 4. Where region is North America or null
WHERE region = 'North America' OR region IS NULL
-- 5. Order by region
ORDER BY region;


country	code	region	basic_unit
Bermuda	BMU	North America	Bermudian dollar
Canada	CAN	North America	Canadian dollar
United States	USA	North America	United States dollar
Greenland	GRL	North America	null

```




```

SELECT name AS country, code, region, basic_unit
FROM countries
  -- 1. Join to currencies
  INNER JOIN currencies
    USING (code)
-- 2. Where region is North America or null
WHERE region = 'North America' OR region IS NULL
-- 3. Order by region
ORDER BY region;


country	code	region	basic_unit
Bermuda	BMU	North America	Bermudian dollar
Canada	CAN	North America	Canadian dollar
United States	USA	North America	United States dollar

```


####
 Full join (2)




```

-- FULL JOIN

SELECT countries.name, code, languages.name AS language
-- 3. From languages
FROM languages
  -- 4. Join to countries
  FULL JOIN countries
    -- 5. Match on code
    USING (code)
-- 1. Where countries.name starts with V or is null
WHERE countries.name LIKE 'V%' OR countries.name IS NULL
-- 2. Order by ascending countries.name
ORDER BY countries.name;


name	code	language
Vanuatu	VUT	Tribal Languages
Vanuatu	VUT	English
Vanuatu	VUT	French
Vanuatu	VUT	Other

-- 53 rows

```




```

-- LEFT JOIN

SELECT countries.name, code, languages.name AS language
FROM languages
  -- 1. Join to countries
  LEFT JOIN countries
    -- 2. Match using code
    USING (code)
-- 3. Where countries.name starts with V or is null
WHERE countries.name LIKE 'V%' OR countries.name IS NULL
ORDER BY countries.name;


name	code	language
Vanuatu	VUT	English
Vanuatu	VUT	Other
Vanuatu	VUT	French

-- 51 rows

```




```

-- LEFT JOIN

SELECT countries.name, code, languages.name AS language
FROM languages
  -- 1. Join to countries
  INNER JOIN countries
    USING (code)
-- 2. Where countries.name starts with V or is null
WHERE countries.name LIKE 'V%' OR countries.name IS NULL
ORDER BY countries.name;

name	code	language
Vanuatu	VUT	Tribal Languages
Vanuatu	VUT	Bislama
Vanuatu	VUT	English

-- 10 rows

```


####
 Full join (3)




```

-- 7. Select fields (with aliases)
SELECT c1.name AS country, region, l.name AS language,
       basic_unit, frac_unit
-- 1. From countries (alias as c1)
FROM countries AS c1
  -- 2. Join with languages (alias as l)
  FULL JOIN languages AS l
    -- 3. Match on code
    USING (code)
  -- 4. Join with currencies (alias as c2)
  FULL JOIN currencies AS c2
    -- 5. Match on code
    USING (code)
-- 6. Where region like Melanesia and Micronesia
WHERE region LIKE 'M%esia';


country	region	language	basic_unit	frac_unit
Kiribati	Micronesia	English	Australian dollar	Cent
Kiribati	Micronesia	Kiribati	Australian dollar	Cent
Marshall Islands	Micronesia	Other	United States dollar	Cent
Marshall Islands	Micronesia	Marshallese	United States dollar	Cent

```


###
 CROSSing the rubicon



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture2-17.png)

####
 A table of two cities

 CROSS JOIN




```

-- 4. Select fields
SELECT c.name AS city, l.name AS language
-- 1. From cities (alias as c)
FROM cities AS c
  -- 2. Join to languages (alias as l)
  CROSS JOIN languages AS l
-- 3. Where c.name like Hyderabad
WHERE c.name LIKE 'Hyder%';


city	language
Hyderabad (India)	Dari
Hyderabad	Dari
Hyderabad (India)	Pashto
Hyderabad	Pashto

```




```

-- 5. Select fields
SELECT c.name AS city, l.name AS language
-- 1. From cities (alias as c)
FROM cities AS c
  -- 2. Join to languages (alias as l)
  INNER JOIN languages AS l
    -- 3. Match on country code
    ON c.country_code = l.code
-- 4. Where c.name like Hyderabad
WHERE c.name LIKE 'Hyder%';


city	language
Hyderabad (India)	Hindi
Hyderabad (India)	Bengali
Hyderabad (India)	Telugu
Hyderabad (India)	Marathi

```


####
 Outer challenge




```

-- Select fields
SELECT c.name AS country, region, life_expectancy AS life_exp
-- From countries (alias as c)
FROM countries as c
  -- Join to populations (alias as p)
  LEFT JOIN populations as p
    -- Match on country code
    ON c.code = p.country_code
-- Focus on 2010
WHERE year = 2010
-- Order by life_exp
ORDER BY life_exp
-- Limit to 5 records
LIMIT 5;

```




---



**Set theory clauses**
-----------------------


###
 State of the UNION



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture1-18.png)

####
 Union



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture5-13.png)



```

-- Select fields from 2010 table
SELECT *
  -- From 2010 table
  FROM economies2010
    -- Set theory clause
    UNION
-- Select fields from 2015 table
SELECT *
  -- From 2015 table
  FROM economies2015
-- Order by code and year
ORDER BY code, year;


code	year	income_group	gross_savings
AFG	2010	Low income	37.133
AFG	2015	Low income	21.466
AGO	2010	Upper middle income	23.534
AGO	2015	Upper middle income	-0.425

```


####
 Union (2)




```

-- Select field
SELECT country_code
  -- From cities
  FROM cities
	-- Set theory clause
	UNION
-- Select field
SELECT code AS country_code
  -- From currencies
  FROM currencies
-- Order by country_code
ORDER BY country_code;


country_code
ABW
AFG
AGO
AIA

```


####
 Union all



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture6-10.png)



```

-- Select fields
SELECT code, year
  -- From economies
  FROM economies
	-- Set theory clause
	UNION ALL
-- Select fields
SELECT country_code AS code, year
  -- From populations
  FROM populations
-- Order by code, year
ORDER BY code, year;


code	year
ABW	2010
ABW	2015
AFG	2010
AFG	2010

```


###
 INTERSECTional data science



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture2-18.png)

####
 Intersect




```

-- Select fields
SELECT code, year
  -- From economies
  FROM economies
	-- Set theory clause
	INTERSECT
-- Select fields
SELECT country_code AS code, year
  -- From populations
  FROM populations
-- Order by code and year
ORDER BY code, year;


code	year
AFG	2010
AFG	2015
AGO	2010

```


####
 Intersect (2)




```

-- Select fields
SELECT name
  -- From countries
  FROM countries
	-- Set theory clause
	INTERSECT
-- Select fields
SELECT name
  -- From cities
  FROM cities;


name
Singapore
Hong Kong

```



 Hong Kong is part of China, but it appears separately here because it has its own ISO country code. Depending upon your analysis, treating Hong Kong separately could be useful or a mistake. Always check your dataset closely before you perform an analysis!



###
 EXCEPTional



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture3-16.png)

####
 Except




```

-- Get the names of cities in cities which are not noted as capital cities in countries as a single field result.

-- Select field
SELECT name
  -- From cities
  FROM cities
	-- Set theory clause
	EXCEPT
-- Select field
SELECT capital
  -- From countries
  FROM countries
-- Order by result
ORDER BY name;


name
Abidjan
Ahmedabad
Alexandria

```


####
 Except (2)




```

-- Determine the names of capital cities that are not listed in the cities table.

-- Select field
SELECT capital
  -- From countries
  FROM countries
	-- Set theory clause
	EXCEPT
-- Select field
SELECT name
  -- From cities
  FROM cities
-- Order by ascending capital
ORDER BY capital;


capital
Agana
Amman
Amsterdam
...

```


###
 Semi-joins and Anti-joins



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture4-15.png)

####
 Semi-join




```

-- You are now going to use the concept of a semi-join to identify languages spoken in the Middle East.

-- Select distinct fields
SELECT DISTINCT name
  -- From languages
  FROM languages
-- Where in statement
WHERE code IN
  -- Subquery
  (SELECT code
    FROM countries
        WHERE region = 'Middle East')
-- Order by name
ORDER BY name;

```


####
 Relating semi-join to a tweaked inner join




```

SELECT DISTINCT name
FROM languages
WHERE code IN
  (SELECT code
   FROM countries
   WHERE region = 'Middle East')
ORDER BY name;

-- is equal to

SELECT DISTINCT languages.name AS language
FROM languages
INNER JOIN countries
ON languages.code = countries.code
WHERE region = 'Middle East'
ORDER BY language;

```


####
 Diagnosing problems using anti-join



 Your goal is to identify the currencies used in Oceanian countries!





```

-- Begin by determining the number of countries in countries that are listed in Oceania using SELECT, FROM, and WHERE.


-- Select statement
SELECT COUNT(*)
  -- From countries
  FROM countries
-- Where continent is Oceania
WHERE continent = 'Oceania';


count
19

```




```

-- 5. Select fields (with aliases)
SELECT c1.code, name, basic_unit AS currency
  -- 1. From countries (alias as c1)
  FROM countries AS c1
  	-- 2. Join with currencies (alias as c2)
  	INNER JOIN currencies c2
    -- 3. Match on code
    USING (code)
-- 4. Where continent is Oceania
WHERE continent = 'Oceania';


code	name	currency
AUS	Australia	Australian dollar
PYF	French Polynesia	CFP franc
KIR	Kiribati	Australian dollar

```




```

-- 3. Select fields
SELECT code, name
  -- 4. From Countries
  FROM countries
  -- 5. Where continent is Oceania
  WHERE continent = 'Oceania'
  	-- 1. And code not in
  	AND code NOT IN
  	-- 2. Subquery
  	(SELECT code
  	 FROM currencies);


code	name
ASM	American Samoa
FJI	Fiji Islands
GUM	Guam
FSM	Micronesia, Federated States of
MNP	Northern Mariana Islands

```


####
 Set theory challenge


* Identify the country codes that are included in either
 `economies`
 or
 `currencies`
 but not in
 `populations`
 .
* Use that result to determine the names of cities in the countries that match the specification in the previous instruction.




```

-- Select the city name
SELECT name
  -- Alias the table where city name resides
  FROM cities AS c1
  -- Choose only records matching the result of multiple set theory clauses
  WHERE country_code IN
(
    -- Select appropriate field from economies AS e
    SELECT e.code
    FROM economies AS e
    -- Get all additional (unique) values of the field from currencies AS c2
    UNION
    SELECT c2.code
    FROM currencies AS c2
    -- Exclude those appearing in populations AS p
    EXCEPT
    SELECT p.country_code
    FROM populations AS p
);

```




---



**Subqueries**
---------------


###
 Subqueries inside WHERE and SELECT clauses


####
 Subquery inside where



 You’ll now try to figure out which countries had high average life expectancies (at the country level) in 2015.





```

-- Select average life_expectancy
SELECT AVG(life_expectancy)
  -- From populations
  FROM populations
-- Where year is 2015
WHERE year = 2015


avg
71.6763415481105

```




```

-- Select fields
SELECT *
  -- From populations
  FROM populations
-- Where life_expectancy is greater than
WHERE life_expectancy >
  -- 1.15 * subquery
  1.15 * (SELECT AVG(life_expectancy)
   FROM populations
   WHERE year = 2015) AND
  year = 2015;


pop_id	country_code	year	fertility_rate	life_expectancy	size
21	AUS	2015	1.833	82.4512	23789800
376	CHE	2015	1.54	83.1976	8281430
356	ESP	2015	1.32	83.3805	46444000
134	FRA	2015	2.01	82.6707	66538400

```


####
 Subquery inside where (2)




```

-- 2. Select fields
SELECT name, country_code, urbanarea_pop
  -- 3. From cities
  FROM cities
-- 4. Where city name in the field of capital cities
WHERE name IN
  -- 1. Subquery
  (SELECT capital
   FROM countries)
ORDER BY urbanarea_pop DESC;


name	country_code	urbanarea_pop
Beijing	CHN	21516000
Dhaka	BGD	14543100
Tokyo	JPN	13513700

```


####
 Subquery inside select



 The code selects the top 9 countries in terms of number of cities appearing in the
 `cities`
 table.





```

SELECT countries.name AS country, COUNT(*) AS cities_num
  FROM cities
    INNER JOIN countries
    ON countries.code = cities.country_code
GROUP BY country
ORDER BY cities_num DESC, country
LIMIT 9;

-- is equal to

SELECT countries.name AS country,
  (SELECT COUNT(*)
   FROM cities
   WHERE countries.code = cities.country_code) AS cities_num
FROM countries
ORDER BY cities_num DESC, country
LIMIT 9;

country	cities_num
China	36
India	18
Japan	11

```


###
 Subquery inside FROM clause



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture7-11.png)

####
 Subquery inside from



 You will use this to determine the number of languages spoken for each country, identified by the country’s local name!





```

-- Select fields (with aliases)
SELECT code, COUNT(*) AS lang_num
  -- From languages
  From languages
-- Group by code
GROUP BY code;


code	lang_num
BLZ	9
BGD	2
ITA	4

```




```

-- Select fields
SELECT local_name, subquery.lang_num
  -- From countries
  FROM countries,
  	-- Subquery (alias as subquery)
  	(SELECT code, COUNT(*) AS lang_num
  	 From languages
  	 GROUP BY code) AS subquery
  -- Where codes match
  WHERE countries.code = subquery.code
-- Order by descending number of languages
ORDER BY lang_num DESC;


local_name	lang_num
Zambia	19
Zimbabwe	16
YeItyop´iya	16
Bharat/India	14

```


####
 Advanced subquery



 You can also nest multiple subqueries to answer even more specific questions.




 In this exercise, for each of the six continents listed in 2015, you’ll identify which country had the maximum inflation rate (and how high it was) using multiple subqueries. The table result of your query in
 **Task 3**
 should look something like the following, where anything between
 `<`
`>`
 will be filled in with appropriate values:





```

+------------+---------------+-------------------+
| name       | continent     | inflation_rate    |
|------------+---------------+-------------------|
| <country1> | North America | <max_inflation1>  |
| <country2> | Africa        | <max_inflation2>  |
| <country3> | Oceania       | <max_inflation3>  |
| <country4> | Europe        | <max_inflation4>  |
| <country5> | South America | <max_inflation5>  |
| <country6> | Asia          | <max_inflation6>  |
+------------+---------------+-------------------+

```



 Again, there are multiple ways to get to this solution using only joins, but the focus here is on showing you an introduction into advanced subqueries.





```

-- step 1

-- Select fields
SELECT name, continent, inflation_rate
  -- From countries
  FROM countries
  	-- Join to economies
  	INNER JOIN economies
    -- Match on code
    USING (code)
-- Where year is 2015
WHERE year = 2015;


name	continent	inflation_rate
Afghanistan	Asia	-1.549
Angola	Africa	10.287
Albania	Europe	1.896
United Arab Emirates	Asia	4.07

```




```

-- step 2

-- Select fields
SELECT MAX(inflation_rate) AS max_inf
  -- Subquery using FROM (alias as subquery)
  FROM (
      SELECT name, continent, inflation_rate
      FROM countries
        INNER JOIN economies
        USING (code)
      WHERE year = 2015) AS subquery
-- Group by continent
GROUP BY continent;


max_inf
48.684
9.784
39.403
21.858

```




```

-- step 3

-- Select fields
SELECT name, continent, inflation_rate
  -- From countries
  FROM countries
	-- Join to economies
	INNER JOIN economies
	-- Match on code
	ON countries.code = economies.code
  -- Where year is 2015
  WHERE year = 2015 AND inflation_rate
    -- And inflation rate in subquery (alias as subquery)
    IN (
        SELECT MAX(inflation_rate) AS max_inf
        FROM (
             SELECT name, continent, inflation_rate
             FROM countries
                INNER JOIN economies
                ON countries.code = economies.code
             WHERE year = 2015) AS subquery
        GROUP BY continent);


name	continent	inflation_rate
Haiti	North America	7.524
Malawi	Africa	21.858
Nauru	Oceania	9.784

```


####
 Subquery challenge



 Let’s test your understanding of the subqueries with a challenge problem! Use a subquery to get 2015 economic data for countries that do
 **not**
 have



* `gov_form`
 of
 `'Constitutional Monarchy'`
 or
* `'Republic'`
 in their
 `gov_form`
 .



 Here,
 `gov_form`
 stands for the form of the government for each country. Review the different entries for
 `gov_form`
 in the
 `countries`
 table.





```

-- Select fields
SELECT code, inflation_rate, unemployment_rate
  -- From economies
  FROM economies
  -- Where year is 2015 and code is not in
  WHERE year = 2015 AND code NOT IN
  	-- Subquery
  	(SELECT code
  	 FROM countries
  	 WHERE (gov_form = 'Constitutional Monarchy' OR gov_form LIKE '%Republic%'))
-- Order by inflation rate
ORDER BY inflation_rate;


code	inflation_rate	unemployment_rate
AFG	-1.549	null
CHE	-1.14	3.178
PRI	-0.751	12
ROU	-0.596	6.812

```


###
 Course review



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture8-8.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture9-7.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture10-7.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture11-7.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture12-6.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture13-5.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture14-4.png)

####
 Final challenge



 In this exercise, you’ll need to get the country names and other 2015 data in the
 `economies`
 table and the
 `countries`
 table for
 **Central American countries with an official language**
 .





```

-- Select fields
SELECT DISTINCT c.name, e.total_investment, e.imports
  -- From table (with alias)
  FROM countries AS c
    -- Join with table (with alias)
    LEFT JOIN economies AS e
      -- Match on code
      ON (c.code = e.code
      -- and code in Subquery
        AND c.code IN (
          SELECT l.code
          FROM languages AS l
          WHERE official = 'true'
        ) )
  -- Where region and year are correct
  WHERE region = 'Central America' AND year = 2015
-- Order by field
ORDER BY name;


name	total_investment	imports
Belize	22.014	6.743
Costa Rica	20.218	4.629
El Salvador	13.983	8.193

```


####
 Final challenge (2)



 Let’s ease up a bit and calculate the average fertility rate for each region in 2015.





```

-- Select fields
SELECT region, continent, AVG(fertility_rate) AS avg_fert_rate
  -- From left table
  FROM countries AS c
    -- Join to right table
    INNER JOIN populations AS p
      -- Match on join condition
      ON c.code = p.country_code
  -- Where specific records matching some condition
  WHERE year = 2015
-- Group appropriately
GROUP BY region, continent
-- Order appropriately
ORDER BY avg_fert_rate;


region	continent	avg_fert_rate
Southern Europe	Europe	1.42610000371933
Eastern Europe	Europe	1.49088890022702
Baltic Countries	Europe	1.60333331425985
Eastern Asia	Asia	1.62071430683136

```


####
 Final challenge (3)



 You are now tasked with determining the top 10 capital cities in Europe and the Americas in terms of a calculated percentage using
 `city_proper_pop`
 and
 `metroarea_pop`
 in
 `cities`
 .





```

-- Select fields
SELECT name, country_code, city_proper_pop, metroarea_pop,
      -- Calculate city_perc
      city_proper_pop / metroarea_pop * 100 AS city_perc
  -- From appropriate table
  FROM cities
  -- Where
  WHERE name IN
    -- Subquery
    (SELECT capital
     FROM countries
     WHERE (continent = 'Europe'
        OR continent LIKE '%America'))
       AND metroarea_pop IS NOT NULL
-- Order appropriately
ORDER BY city_perc DESC
-- Limit amount
LIMIT 10;


name	country_code	city_proper_pop	metroarea_pop	city_perc
Lima	PER	8852000	10750000	82.3441863059998
Bogota	COL	7878780	9800000	80.3957462310791
Moscow	RUS	12197600	16170000	75.4334926605225

```



 This is the memo of the 18th course of ‘Data Scientist with Python’ track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/joining-data-in-postgresql)**
 .




 Further Reading:


[More dangerous subtleties of JOINs in SQL](https://alexpetralia.com/posts/2017/7/19/more-dangerous-subtleties-of-joins-in-sql)
 — Be careful when JOIN tables with duplications or NULLs





---



**Introduction to joins**
--------------------------


###
 Introduction to INNER JOIN


####
 Inner join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture-19.png)





```

SELECT table_name
FROM information_schema.tables
-- Specify the correct table_schema value
WHERE table_schema = 'public';


table_name
cities
countries
languages
economies
currencies
populations

```




```

SELECT *
FROM left_table
INNER JOIN right_table
ON left_table.id = right_table.id;

```




```

-- 1. Select name fields (with alias) and region
SELECT cities.name AS city, countries.name AS country, region
FROM cities
  INNER JOIN countries
    ON cities.country_code = countries.code;

city	country	region
Abidjan	Cote d'Ivoire	Western Africa
Abu Dhabi	United Arab Emirates	Middle East
Abuja	Nigeria	Western Africa

```


####
 Inner join (2)




```

SELECT c1.name AS city, c2.name AS country
FROM cities AS c1
INNER JOIN countries AS c2
ON c1.country_code = c2.code;

```




```

-- 3. Select fields with aliases
SELECT c.code AS country_code, name, year, inflation_rate
FROM countries AS c
  -- 1. Join to economies (alias e)
  INNER JOIN economies AS e
    -- 2. Match on code
    ON c.code = e.code;

```


####
 Inner join (3)




```

SELECT *
FROM left_table
  INNER JOIN right_table
    ON left_table.id = right_table.id
  INNER JOIN another_table
    ON left_table.id = another_table.id;

```




```

-- 6. Select fields
SELECT c.code, name, region, e.year, fertility_rate, unemployment_rate
  -- 1. From countries (alias as c)
  FROM countries AS c
  -- 2. Join to populations (as p)
  INNER JOIN populations AS p
    -- 3. Match on country code
    ON c.code = p.country_code
  -- 4. Join to economies (as e)
  INNER JOIN economies AS e
    -- 5. Match on country code
    ON c.code = e.code;


```




```

-- countries INNER JOIN populations table
code	name	fertility_rate
ABW	Aruba	1.704
ABW	Aruba	1.647
AFG	Afghanistan	5.746
AFG	Afghanistan	4.653

-- economies table
econ_id	code	year
1	AFG	2010
2	AFG	2015


code	name	region	year	fertility_rate	unemployment_rate
AFG	Afghanistan	Southern and Central Asia	2010	4.653	null
AFG	Afghanistan	Southern and Central Asia	2010	5.746	null
AFG	Afghanistan	Southern and Central Asia	2015	4.653	null
AFG	Afghanistan	Southern and Central Asia	2015	5.746	null
AGO	Angola	Central Africa	2010	5.996	null

```




```

-- 6. Select fields
SELECT c.code, name, region, e.year, fertility_rate, unemployment_rate
  -- 1. From countries (alias as c)
  FROM countries AS c
  -- 2. Join to populations (as p)
  INNER JOIN populations AS p
    -- 3. Match on country code
    ON c.code = p.country_code
  -- 4. Join to economies (as e)
  INNER JOIN economies AS e
    -- 5. Match on country code and year
    ON c.code = e.code AND p.year = e.year;

code	name	region	year	fertility_rate	unemployment_rate
AFG	Afghanistan	Southern and Central Asia	2010	5.746	null
AFG	Afghanistan	Southern and Central Asia	2015	4.653	null

```


###
 INNER JOIN via USING


####
 Inner join with using




```

SELECT *
FROM countries
  INNER JOIN economies
    ON countries.code = economies.code

-- is equal to

SELECT *
FROM countries
  INNER JOIN economies
    USING(code)

```




```

-- 4. Select fields
SELECT c.name AS country, continent, l.name AS language, official
  -- 1. From countries (alias as c)
  FROM countries as c
  -- 2. Join to languages (as l)
  INNER JOIN languages as l
    -- 3. Match using code
    USING (code)


country	continent	language	official
Afghanistan	Asia	Dari	true
Afghanistan	Asia	Pashto	true

```


###
 Self-ish joins, just in CASE


####
 Self-join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture1-19.png)



```

pop_id	country_code	year	fertility_rate	life_expectancy	size
20	ABW	2010	1.704	74.9535	101597
19	ABW	2015	1.647	75.5736	103889

```




```

-- 4. Select fields with aliases
SELECT p1.country_code, p1.size AS size2010, p2.size AS size2015
-- 1. From populations (alias as p1)
FROM populations AS p1
  -- 2. Join to itself (alias as p2)
  INNER JOIN populations AS p2
    -- 3. Match on country code
    ON p1.country_code = p2.country_code


country_code	size2010	size2015
ABW	101597	103889
ABW	101597	101597
ABW	103889	103889
ABW	103889	101597

```




```

-- 5. Select fields with aliases
SELECT p1.country_code,
       p1.size AS size2010,
       p2.size AS size2015
-- 1. From populations (alias as p1)
FROM populations as p1
  -- 2. Join to itself (alias as p2)
  INNER JOIN populations as p2
    -- 3. Match on country code
    ON p1.country_code = p2.country_code
        -- 4. and year (with calculation)
        AND p1.year = p2.year - 5


country_code	size2010	size2015
ABW	101597	103889
AFG	27962200	32526600
AGO	21220000	25022000
ALB	2913020	2889170

```




```

-- With two numeric fields A and B, the percentage growth from A to B can be calculated as (B−A)/A∗100.0.

SELECT p1.country_code,
       p1.size AS size2010,
       p2.size AS size2015,
       -- 1. calculate growth_perc
       ((p2.size - p1.size)/p1.size * 100.0) AS growth_perc
-- 2. From populations (alias as p1)
FROM populations AS p1
  -- 3. Join to itself (alias as p2)
  INNER JOIN populations AS p2
    -- 4. Match on country code
    ON p1.country_code = p2.country_code
        -- 5. and year (with calculation)
        AND p1.year = p2.year - 5;


country_code	size2010	size2015	growth_perc
ABW	101597	103889	2.25597210228443
AFG	27962200	32526600	16.32329672575
AGO	21220000	25022000	17.9171919822693
ALB	2913020	2889170	-0.818874966353178

```


####
 Case when and then




```

SELECT name, continent, code, surface_area,
    -- 1. First case
    CASE WHEN surface_area > 2000000 THEN 'large'
        -- 2. Second case
        WHEN surface_area > 350000 THEN 'medium'
        -- 3. Else clause + end
        ELSE 'small' END
        -- 4. Alias name
        AS geosize_group
-- 5. From table
FROM countries;


name	continent	code	surface_area	geosize_group
Afghanistan	Asia	AFG	652090	medium
Netherlands	Europe	NLD	41526	small
Albania	Europe	ALB	28748	small

```


####
 Inner challenge




```

SELECT name, continent, code, surface_area,
    CASE WHEN surface_area > 2000000
            THEN 'large'
       WHEN surface_area > 350000
            THEN 'medium'
       ELSE 'small' END
       AS geosize_group
INTO countries_plus
FROM countries;


name	continent	code	surface_area	geosize_group
Afghanistan	Asia	AFG	652090	medium
Netherlands	Europe	NLD	41526	small
Albania	Europe	ALB	28748	small
Algeria	Africa	DZA	2381740	large

```




```

SELECT country_code, size,
    -- 1. First case
    CASE WHEN size > 50000000 THEN 'large'
        -- 2. Second case
        WHEN size > 1000000 THEN 'medium'
        -- 3. Else clause + end
        ELSE 'small' END
        -- 4. Alias name
        AS popsize_group
-- 5. From table
FROM populations
-- 6. Focus on 2015
WHERE year = 2015;


country_code	size	popsize_group
ABW	103889	small
AFG	32526600	medium
AGO	25022000	medium
ALB	2889170	medium

```




```

SELECT country_code, size,
    CASE WHEN size > 50000000 THEN 'large'
        WHEN size > 1000000 THEN 'medium'
        ELSE 'small' END
        AS popsize_group
-- 1. Into table
INTO pop_plus
FROM populations
WHERE year = 2015;

-- 2. Select all columns of pop_plus
SELECT * FROM pop_plus;


country_code	size	popsize_group
ABW	103889	small
AFG	32526600	medium
AGO	25022000	medium
ALB	2889170	medium

```




```

SELECT country_code, size,
  CASE WHEN size > 50000000
            THEN 'large'
       WHEN size > 1000000
            THEN 'medium'
       ELSE 'small' END
       AS popsize_group
INTO pop_plus
FROM populations
WHERE year = 2015;

-- 5. Select fields
SELECT name, continent, geosize_group, popsize_group
-- 1. From countries_plus (alias as c)
FROM countries_plus AS c
  -- 2. Join to pop_plus (alias as p)
  INNER JOIN pop_plus AS p
    -- 3. Match on country code
    ON c.code = p.country_code
-- 4. Order the table
ORDER BY geosize_group;


name	continent	geosize_group	popsize_group
India	Asia	large	large
United States	North America	large	large
Saudi Arabia	Asia	large	medium
China	Asia	large	large

```




---



**Outer joins and cross joins**
--------------------------------


###
 LEFT and RIGHT JOINs



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture-18.png)

####
 Left Join




```

-- Select the city name (with alias), the country code,
-- the country name (with alias), the region,
-- and the city proper population
SELECT c1.name AS city, code, c2.name AS country,
       region, city_proper_pop
-- From left table (with alias)
FROM cities AS c1
  -- Join to right table (with alias)
  INNER JOIN countries AS c2
    -- Match on country code
    ON c1.country_code = c2.code
-- Order by descending country code
ORDER BY code DESC;


city	code	country	region	city_proper_pop
Harare	ZWE	Zimbabwe	Eastern Africa	1606000
Lusaka	ZMB	Zambia	Eastern Africa	1742980
Cape Town	ZAF	South Africa	Southern Africa	3740030

-- 230 rows

```




```

SELECT c1.name AS city, code, c2.name AS country,
       region, city_proper_pop
FROM cities AS c1
  -- 1. Join right table (with alias)
  LEFT JOIN countries AS c2
    -- 2. Match on country code
    ON c1.country_code = c2.code
-- 3. Order by descending country code
ORDER BY code DESC;


city	code	country	region	city_proper_pop
Taichung	null	null	null	2752410
Tainan	null	null	null	1885250
Kaohsiung	null	null	null	2778920
Bucharest	null	null	null	1883420

-- 236 rows

```


####
 Left join (2)




```

/*
5. Select country name AS country, the country's local name,
the language name AS language, and
the percent of the language spoken in the country
*/
SELECT c.name AS country, local_name, l.name AS language, percent
-- 1. From left table (alias as c)
FROM countries AS c
  -- 2. Join to right table (alias as l)
  INNER JOIN languages AS l
    -- 3. Match on fields
    ON c.code = l.code
-- 4. Order by descending country
ORDER BY country DESC;


country	local_name	language	percent
Zimbabwe	Zimbabwe	Shona	null
Zimbabwe	Zimbabwe	Tonga	null
Zimbabwe	Zimbabwe	Tswana	null

-- 914 rows

```




```

/*
5. Select country name AS country, the country's local name,
the language name AS language, and
the percent of the language spoken in the country
*/
SELECT c.name AS country, local_name, l.name AS language, percent
-- 1. From left table (alias as c)
FROM countries AS c
  -- 2. Join to right table (alias as l)
  LEFT JOIN languages AS l
    -- 3. Match on fields
    ON c.code = l.code
-- 4. Order by descending country
ORDER BY country DESC;


country	local_name	language	percent
Zimbabwe	Zimbabwe	Chibarwe	null
Zimbabwe	Zimbabwe	Shona	null
Zimbabwe	Zimbabwe	Ndebele	null
Zimbabwe	Zimbabwe	English	null

-- 921 rows

```


####
 Left join (3)




```

-- 5. Select name, region, and gdp_percapita
SELECT name, region, gdp_percapita
-- 1. From countries (alias as c)
FROM countries AS c
  -- 2. Left join with economies (alias as e)
  LEFT JOIN economies AS e
    -- 3. Match on code fields
    ON c.code = e.code
-- 4. Focus on 2010
WHERE year = 2010;


name	region	gdp_percapita
Afghanistan	Southern and Central Asia	539.667
Angola	Central Africa	3599.27
Albania	Southern Europe	4098.13
United Arab Emirates	Middle East	34628.6

```




```

-- Select fields
SELECT region, AVG(gdp_percapita) AS avg_gdp
-- From countries (alias as c)
FROM countries AS c
  -- Left join with economies (alias as e)
  LEFT JOIN economies AS e
    -- Match on code fields
    ON c.code = e.code
-- Focus on 2010
WHERE year = 2010
-- Group by region
GROUP BY region;


region	avg_gdp
Southern Africa	5051.59797363281
Australia and New Zealand	44792.384765625
Southeast Asia	10547.1541320801

```




```

-- Select fields
SELECT region, AVG(gdp_percapita) AS avg_gdp
-- From countries (alias as c)
FROM countries AS c
  -- Left join with economies (alias as e)
  LEFT JOIN economies AS e
    -- Match on code fields
    ON c.code = e.code
-- Focus on 2010
WHERE year = 2010
-- Group by region
GROUP BY region
-- Order by descending avg_gdp
ORDER BY avg_gdp DESC;


region	avg_gdp
Western Europe	58130.9614955357
Nordic Countries	57073.99765625
North America	47911.509765625
Australia and New Zealand	44792.384765625

```


####
 Right join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture3-17.png)



```

-- convert this code to use RIGHT JOINs instead of LEFT JOINs
/*
SELECT cities.name AS city, urbanarea_pop, countries.name AS country,
       indep_year, languages.name AS language, percent
FROM cities
  LEFT JOIN countries
    ON cities.country_code = countries.code
  LEFT JOIN languages
    ON countries.code = languages.code
ORDER BY city, language;
*/

SELECT cities.name AS city, urbanarea_pop, countries.name AS country,
       indep_year, languages.name AS language, percent
FROM languages
  RIGHT JOIN countries
    ON languages.code = countries.code
  RIGHT JOIN cities
    ON countries.code = cities.country_code
ORDER BY city, language;


city	urbanarea_pop	country	indep_year	language	percent
Abidjan	4765000	Cote d'Ivoire	1960	French	null
Abidjan	4765000	Cote d'Ivoire	1960	Other	null
Abu Dhabi	1145000	United Arab Emirates	1971	Arabic	null

```


###
 FULL JOINs


####
 Full join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture4-16.png)



```

SELECT name AS country, code, region, basic_unit
-- 3. From countries
FROM countries
  -- 4. Join to currencies
  FULL JOIN currencies
    -- 5. Match on code
    USING (code)
-- 1. Where region is North America or null
WHERE region = 'North America' OR region IS NULL
-- 2. Order by region
ORDER BY region;


country	code	region	basic_unit
Greenland	GRL	North America	null
null	TMP	null	United States dollar
null	FLK	null	Falkland Islands pound
null	AIA	null	East Caribbean dollar
null	NIU	null	New Zealand dollar
null	ROM	null	Romanian leu
null	SHN	null	Saint Helena pound
null	SGS	null	British pound
null	TWN	null	New Taiwan dollar
null	WLF	null	CFP franc
null	MSR	null	East Caribbean dollar
null	IOT	null	United States dollar
null	CCK	null	Australian dollar
null	COK	null	New Zealand dollar

```




```

SELECT name AS country, code, region, basic_unit
-- 1. From countries
FROM countries
  -- 2. Join to currencies
  LEFT JOIN currencies
    -- 3. Match on code
    USING (code)
-- 4. Where region is North America or null
WHERE region = 'North America' OR region IS NULL
-- 5. Order by region
ORDER BY region;


country	code	region	basic_unit
Bermuda	BMU	North America	Bermudian dollar
Canada	CAN	North America	Canadian dollar
United States	USA	North America	United States dollar
Greenland	GRL	North America	null

```




```

SELECT name AS country, code, region, basic_unit
FROM countries
  -- 1. Join to currencies
  INNER JOIN currencies
    USING (code)
-- 2. Where region is North America or null
WHERE region = 'North America' OR region IS NULL
-- 3. Order by region
ORDER BY region;


country	code	region	basic_unit
Bermuda	BMU	North America	Bermudian dollar
Canada	CAN	North America	Canadian dollar
United States	USA	North America	United States dollar

```


####
 Full join (2)




```

-- FULL JOIN

SELECT countries.name, code, languages.name AS language
-- 3. From languages
FROM languages
  -- 4. Join to countries
  FULL JOIN countries
    -- 5. Match on code
    USING (code)
-- 1. Where countries.name starts with V or is null
WHERE countries.name LIKE 'V%' OR countries.name IS NULL
-- 2. Order by ascending countries.name
ORDER BY countries.name;


name	code	language
Vanuatu	VUT	Tribal Languages
Vanuatu	VUT	English
Vanuatu	VUT	French
Vanuatu	VUT	Other

-- 53 rows

```




```

-- LEFT JOIN

SELECT countries.name, code, languages.name AS language
FROM languages
  -- 1. Join to countries
  LEFT JOIN countries
    -- 2. Match using code
    USING (code)
-- 3. Where countries.name starts with V or is null
WHERE countries.name LIKE 'V%' OR countries.name IS NULL
ORDER BY countries.name;


name	code	language
Vanuatu	VUT	English
Vanuatu	VUT	Other
Vanuatu	VUT	French

-- 51 rows

```




```

-- LEFT JOIN

SELECT countries.name, code, languages.name AS language
FROM languages
  -- 1. Join to countries
  INNER JOIN countries
    USING (code)
-- 2. Where countries.name starts with V or is null
WHERE countries.name LIKE 'V%' OR countries.name IS NULL
ORDER BY countries.name;

name	code	language
Vanuatu	VUT	Tribal Languages
Vanuatu	VUT	Bislama
Vanuatu	VUT	English

-- 10 rows

```


####
 Full join (3)




```

-- 7. Select fields (with aliases)
SELECT c1.name AS country, region, l.name AS language,
       basic_unit, frac_unit
-- 1. From countries (alias as c1)
FROM countries AS c1
  -- 2. Join with languages (alias as l)
  FULL JOIN languages AS l
    -- 3. Match on code
    USING (code)
  -- 4. Join with currencies (alias as c2)
  FULL JOIN currencies AS c2
    -- 5. Match on code
    USING (code)
-- 6. Where region like Melanesia and Micronesia
WHERE region LIKE 'M%esia';


country	region	language	basic_unit	frac_unit
Kiribati	Micronesia	English	Australian dollar	Cent
Kiribati	Micronesia	Kiribati	Australian dollar	Cent
Marshall Islands	Micronesia	Other	United States dollar	Cent
Marshall Islands	Micronesia	Marshallese	United States dollar	Cent

```


###
 CROSSing the rubicon



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture2-17.png)

####
 A table of two cities

 CROSS JOIN




```

-- 4. Select fields
SELECT c.name AS city, l.name AS language
-- 1. From cities (alias as c)
FROM cities AS c
  -- 2. Join to languages (alias as l)
  CROSS JOIN languages AS l
-- 3. Where c.name like Hyderabad
WHERE c.name LIKE 'Hyder%';


city	language
Hyderabad (India)	Dari
Hyderabad	Dari
Hyderabad (India)	Pashto
Hyderabad	Pashto

```




```

-- 5. Select fields
SELECT c.name AS city, l.name AS language
-- 1. From cities (alias as c)
FROM cities AS c
  -- 2. Join to languages (alias as l)
  INNER JOIN languages AS l
    -- 3. Match on country code
    ON c.country_code = l.code
-- 4. Where c.name like Hyderabad
WHERE c.name LIKE 'Hyder%';


city	language
Hyderabad (India)	Hindi
Hyderabad (India)	Bengali
Hyderabad (India)	Telugu
Hyderabad (India)	Marathi

```


####
 Outer challenge




```

-- Select fields
SELECT c.name AS country, region, life_expectancy AS life_exp
-- From countries (alias as c)
FROM countries as c
  -- Join to populations (alias as p)
  LEFT JOIN populations as p
    -- Match on country code
    ON c.code = p.country_code
-- Focus on 2010
WHERE year = 2010
-- Order by life_exp
ORDER BY life_exp
-- Limit to 5 records
LIMIT 5;

```




---



**Set theory clauses**
-----------------------


###
 State of the UNION



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture1-18.png)

####
 Union



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture5-13.png)



```

-- Select fields from 2010 table
SELECT *
  -- From 2010 table
  FROM economies2010
    -- Set theory clause
    UNION
-- Select fields from 2015 table
SELECT *
  -- From 2015 table
  FROM economies2015
-- Order by code and year
ORDER BY code, year;


code	year	income_group	gross_savings
AFG	2010	Low income	37.133
AFG	2015	Low income	21.466
AGO	2010	Upper middle income	23.534
AGO	2015	Upper middle income	-0.425

```


####
 Union (2)




```

-- Select field
SELECT country_code
  -- From cities
  FROM cities
	-- Set theory clause
	UNION
-- Select field
SELECT code AS country_code
  -- From currencies
  FROM currencies
-- Order by country_code
ORDER BY country_code;


country_code
ABW
AFG
AGO
AIA

```


####
 Union all



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture6-10.png)



```

-- Select fields
SELECT code, year
  -- From economies
  FROM economies
	-- Set theory clause
	UNION ALL
-- Select fields
SELECT country_code AS code, year
  -- From populations
  FROM populations
-- Order by code, year
ORDER BY code, year;


code	year
ABW	2010
ABW	2015
AFG	2010
AFG	2010

```


###
 INTERSECTional data science



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture2-18.png)

####
 Intersect




```

-- Select fields
SELECT code, year
  -- From economies
  FROM economies
	-- Set theory clause
	INTERSECT
-- Select fields
SELECT country_code AS code, year
  -- From populations
  FROM populations
-- Order by code and year
ORDER BY code, year;


code	year
AFG	2010
AFG	2015
AGO	2010

```


####
 Intersect (2)




```

-- Select fields
SELECT name
  -- From countries
  FROM countries
	-- Set theory clause
	INTERSECT
-- Select fields
SELECT name
  -- From cities
  FROM cities;


name
Singapore
Hong Kong

```



 Hong Kong is part of China, but it appears separately here because it has its own ISO country code. Depending upon your analysis, treating Hong Kong separately could be useful or a mistake. Always check your dataset closely before you perform an analysis!



###
 EXCEPTional



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture3-16.png)

####
 Except




```

-- Get the names of cities in cities which are not noted as capital cities in countries as a single field result.

-- Select field
SELECT name
  -- From cities
  FROM cities
	-- Set theory clause
	EXCEPT
-- Select field
SELECT capital
  -- From countries
  FROM countries
-- Order by result
ORDER BY name;


name
Abidjan
Ahmedabad
Alexandria

```


####
 Except (2)




```

-- Determine the names of capital cities that are not listed in the cities table.

-- Select field
SELECT capital
  -- From countries
  FROM countries
	-- Set theory clause
	EXCEPT
-- Select field
SELECT name
  -- From cities
  FROM cities
-- Order by ascending capital
ORDER BY capital;


capital
Agana
Amman
Amsterdam
...

```


###
 Semi-joins and Anti-joins



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture4-15.png)

####
 Semi-join




```

-- You are now going to use the concept of a semi-join to identify languages spoken in the Middle East.

-- Select distinct fields
SELECT DISTINCT name
  -- From languages
  FROM languages
-- Where in statement
WHERE code IN
  -- Subquery
  (SELECT code
    FROM countries
        WHERE region = 'Middle East')
-- Order by name
ORDER BY name;

```


####
 Relating semi-join to a tweaked inner join




```

SELECT DISTINCT name
FROM languages
WHERE code IN
  (SELECT code
   FROM countries
   WHERE region = 'Middle East')
ORDER BY name;

-- is equal to

SELECT DISTINCT languages.name AS language
FROM languages
INNER JOIN countries
ON languages.code = countries.code
WHERE region = 'Middle East'
ORDER BY language;

```


####
 Diagnosing problems using anti-join



 Your goal is to identify the currencies used in Oceanian countries!





```

-- Begin by determining the number of countries in countries that are listed in Oceania using SELECT, FROM, and WHERE.


-- Select statement
SELECT COUNT(*)
  -- From countries
  FROM countries
-- Where continent is Oceania
WHERE continent = 'Oceania';


count
19

```




```

-- 5. Select fields (with aliases)
SELECT c1.code, name, basic_unit AS currency
  -- 1. From countries (alias as c1)
  FROM countries AS c1
  	-- 2. Join with currencies (alias as c2)
  	INNER JOIN currencies c2
    -- 3. Match on code
    USING (code)
-- 4. Where continent is Oceania
WHERE continent = 'Oceania';


code	name	currency
AUS	Australia	Australian dollar
PYF	French Polynesia	CFP franc
KIR	Kiribati	Australian dollar

```




```

-- 3. Select fields
SELECT code, name
  -- 4. From Countries
  FROM countries
  -- 5. Where continent is Oceania
  WHERE continent = 'Oceania'
  	-- 1. And code not in
  	AND code NOT IN
  	-- 2. Subquery
  	(SELECT code
  	 FROM currencies);


code	name
ASM	American Samoa
FJI	Fiji Islands
GUM	Guam
FSM	Micronesia, Federated States of
MNP	Northern Mariana Islands

```


####
 Set theory challenge


* Identify the country codes that are included in either
 `economies`
 or
 `currencies`
 but not in
 `populations`
 .
* Use that result to determine the names of cities in the countries that match the specification in the previous instruction.




```

-- Select the city name
SELECT name
  -- Alias the table where city name resides
  FROM cities AS c1
  -- Choose only records matching the result of multiple set theory clauses
  WHERE country_code IN
(
    -- Select appropriate field from economies AS e
    SELECT e.code
    FROM economies AS e
    -- Get all additional (unique) values of the field from currencies AS c2
    UNION
    SELECT c2.code
    FROM currencies AS c2
    -- Exclude those appearing in populations AS p
    EXCEPT
    SELECT p.country_code
    FROM populations AS p
);

```




---



**Subqueries**
---------------


###
 Subqueries inside WHERE and SELECT clauses


####
 Subquery inside where



 You’ll now try to figure out which countries had high average life expectancies (at the country level) in 2015.





```

-- Select average life_expectancy
SELECT AVG(life_expectancy)
  -- From populations
  FROM populations
-- Where year is 2015
WHERE year = 2015


avg
71.6763415481105

```




```

-- Select fields
SELECT *
  -- From populations
  FROM populations
-- Where life_expectancy is greater than
WHERE life_expectancy >
  -- 1.15 * subquery
  1.15 * (SELECT AVG(life_expectancy)
   FROM populations
   WHERE year = 2015) AND
  year = 2015;


pop_id	country_code	year	fertility_rate	life_expectancy	size
21	AUS	2015	1.833	82.4512	23789800
376	CHE	2015	1.54	83.1976	8281430
356	ESP	2015	1.32	83.3805	46444000
134	FRA	2015	2.01	82.6707	66538400

```


####
 Subquery inside where (2)




```

-- 2. Select fields
SELECT name, country_code, urbanarea_pop
  -- 3. From cities
  FROM cities
-- 4. Where city name in the field of capital cities
WHERE name IN
  -- 1. Subquery
  (SELECT capital
   FROM countries)
ORDER BY urbanarea_pop DESC;


name	country_code	urbanarea_pop
Beijing	CHN	21516000
Dhaka	BGD	14543100
Tokyo	JPN	13513700

```


####
 Subquery inside select



 The code selects the top 9 countries in terms of number of cities appearing in the
 `cities`
 table.





```

SELECT countries.name AS country, COUNT(*) AS cities_num
  FROM cities
    INNER JOIN countries
    ON countries.code = cities.country_code
GROUP BY country
ORDER BY cities_num DESC, country
LIMIT 9;

-- is equal to

SELECT countries.name AS country,
  (SELECT COUNT(*)
   FROM cities
   WHERE countries.code = cities.country_code) AS cities_num
FROM countries
ORDER BY cities_num DESC, country
LIMIT 9;

country	cities_num
China	36
India	18
Japan	11

```


###
 Subquery inside FROM clause



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture7-11.png)

####
 Subquery inside from



 You will use this to determine the number of languages spoken for each country, identified by the country’s local name!





```

-- Select fields (with aliases)
SELECT code, COUNT(*) AS lang_num
  -- From languages
  From languages
-- Group by code
GROUP BY code;


code	lang_num
BLZ	9
BGD	2
ITA	4

```




```

-- Select fields
SELECT local_name, subquery.lang_num
  -- From countries
  FROM countries,
  	-- Subquery (alias as subquery)
  	(SELECT code, COUNT(*) AS lang_num
  	 From languages
  	 GROUP BY code) AS subquery
  -- Where codes match
  WHERE countries.code = subquery.code
-- Order by descending number of languages
ORDER BY lang_num DESC;


local_name	lang_num
Zambia	19
Zimbabwe	16
YeItyop´iya	16
Bharat/India	14

```


####
 Advanced subquery



 You can also nest multiple subqueries to answer even more specific questions.




 In this exercise, for each of the six continents listed in 2015, you’ll identify which country had the maximum inflation rate (and how high it was) using multiple subqueries. The table result of your query in
 **Task 3**
 should look something like the following, where anything between
 `<`
`>`
 will be filled in with appropriate values:





```

+------------+---------------+-------------------+
| name       | continent     | inflation_rate    |
|------------+---------------+-------------------|
| <country1> | North America | <max_inflation1>  |
| <country2> | Africa        | <max_inflation2>  |
| <country3> | Oceania       | <max_inflation3>  |
| <country4> | Europe        | <max_inflation4>  |
| <country5> | South America | <max_inflation5>  |
| <country6> | Asia          | <max_inflation6>  |
+------------+---------------+-------------------+

```



 Again, there are multiple ways to get to this solution using only joins, but the focus here is on showing you an introduction into advanced subqueries.





```

-- step 1

-- Select fields
SELECT name, continent, inflation_rate
  -- From countries
  FROM countries
  	-- Join to economies
  	INNER JOIN economies
    -- Match on code
    USING (code)
-- Where year is 2015
WHERE year = 2015;


name	continent	inflation_rate
Afghanistan	Asia	-1.549
Angola	Africa	10.287
Albania	Europe	1.896
United Arab Emirates	Asia	4.07

```




```

-- step 2

-- Select fields
SELECT MAX(inflation_rate) AS max_inf
  -- Subquery using FROM (alias as subquery)
  FROM (
      SELECT name, continent, inflation_rate
      FROM countries
        INNER JOIN economies
        USING (code)
      WHERE year = 2015) AS subquery
-- Group by continent
GROUP BY continent;


max_inf
48.684
9.784
39.403
21.858

```




```

-- step 3

-- Select fields
SELECT name, continent, inflation_rate
  -- From countries
  FROM countries
	-- Join to economies
	INNER JOIN economies
	-- Match on code
	ON countries.code = economies.code
  -- Where year is 2015
  WHERE year = 2015 AND inflation_rate
    -- And inflation rate in subquery (alias as subquery)
    IN (
        SELECT MAX(inflation_rate) AS max_inf
        FROM (
             SELECT name, continent, inflation_rate
             FROM countries
                INNER JOIN economies
                ON countries.code = economies.code
             WHERE year = 2015) AS subquery
        GROUP BY continent);


name	continent	inflation_rate
Haiti	North America	7.524
Malawi	Africa	21.858
Nauru	Oceania	9.784

```


####
 Subquery challenge



 Let’s test your understanding of the subqueries with a challenge problem! Use a subquery to get 2015 economic data for countries that do
 **not**
 have



* `gov_form`
 of
 `'Constitutional Monarchy'`
 or
* `'Republic'`
 in their
 `gov_form`
 .



 Here,
 `gov_form`
 stands for the form of the government for each country. Review the different entries for
 `gov_form`
 in the
 `countries`
 table.





```

-- Select fields
SELECT code, inflation_rate, unemployment_rate
  -- From economies
  FROM economies
  -- Where year is 2015 and code is not in
  WHERE year = 2015 AND code NOT IN
  	-- Subquery
  	(SELECT code
  	 FROM countries
  	 WHERE (gov_form = 'Constitutional Monarchy' OR gov_form LIKE '%Republic%'))
-- Order by inflation rate
ORDER BY inflation_rate;


code	inflation_rate	unemployment_rate
AFG	-1.549	null
CHE	-1.14	3.178
PRI	-0.751	12
ROU	-0.596	6.812

```


###
 Course review



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture8-8.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture9-7.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture10-7.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture11-7.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture12-6.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture13-5.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture14-4.png)

####
 Final challenge



 In this exercise, you’ll need to get the country names and other 2015 data in the
 `economies`
 table and the
 `countries`
 table for
 **Central American countries with an official language**
 .





```

-- Select fields
SELECT DISTINCT c.name, e.total_investment, e.imports
  -- From table (with alias)
  FROM countries AS c
    -- Join with table (with alias)
    LEFT JOIN economies AS e
      -- Match on code
      ON (c.code = e.code
      -- and code in Subquery
        AND c.code IN (
          SELECT l.code
          FROM languages AS l
          WHERE official = 'true'
        ) )
  -- Where region and year are correct
  WHERE region = 'Central America' AND year = 2015
-- Order by field
ORDER BY name;


name	total_investment	imports
Belize	22.014	6.743
Costa Rica	20.218	4.629
El Salvador	13.983	8.193

```


####
 Final challenge (2)



 Let’s ease up a bit and calculate the average fertility rate for each region in 2015.





```

-- Select fields
SELECT region, continent, AVG(fertility_rate) AS avg_fert_rate
  -- From left table
  FROM countries AS c
    -- Join to right table
    INNER JOIN populations AS p
      -- Match on join condition
      ON c.code = p.country_code
  -- Where specific records matching some condition
  WHERE year = 2015
-- Group appropriately
GROUP BY region, continent
-- Order appropriately
ORDER BY avg_fert_rate;


region	continent	avg_fert_rate
Southern Europe	Europe	1.42610000371933
Eastern Europe	Europe	1.49088890022702
Baltic Countries	Europe	1.60333331425985
Eastern Asia	Asia	1.62071430683136

```


####
 Final challenge (3)



 You are now tasked with determining the top 10 capital cities in Europe and the Americas in terms of a calculated percentage using
 `city_proper_pop`
 and
 `metroarea_pop`
 in
 `cities`
 .





```

-- Select fields
SELECT name, country_code, city_proper_pop, metroarea_pop,
      -- Calculate city_perc
      city_proper_pop / metroarea_pop * 100 AS city_perc
  -- From appropriate table
  FROM cities
  -- Where
  WHERE name IN
    -- Subquery
    (SELECT capital
     FROM countries
     WHERE (continent = 'Europe'
        OR continent LIKE '%America'))
       AND metroarea_pop IS NOT NULL
-- Order appropriately
ORDER BY city_perc DESC
-- Limit amount
LIMIT 10;


name	country_code	city_proper_pop	metroarea_pop	city_perc
Lima	PER	8852000	10750000	82.3441863059998
Bogota	COL	7878780	9800000	80.3957462310791
Moscow	RUS	12197600	16170000	75.4334926605225

```



 This is the memo of the 18th course of ‘Data Scientist with Python’ track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/joining-data-in-postgresql)**
 .




 Further Reading:


[More dangerous subtleties of JOINs in SQL](https://alexpetralia.com/posts/2017/7/19/more-dangerous-subtleties-of-joins-in-sql)
 — Be careful when JOIN tables with duplications or NULLs





---



**Introduction to joins**
--------------------------


###
 Introduction to INNER JOIN


####
 Inner join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture-19.png)





```

SELECT table_name
FROM information_schema.tables
-- Specify the correct table_schema value
WHERE table_schema = 'public';


table_name
cities
countries
languages
economies
currencies
populations

```




```

SELECT *
FROM left_table
INNER JOIN right_table
ON left_table.id = right_table.id;

```




```

-- 1. Select name fields (with alias) and region
SELECT cities.name AS city, countries.name AS country, region
FROM cities
  INNER JOIN countries
    ON cities.country_code = countries.code;

city	country	region
Abidjan	Cote d'Ivoire	Western Africa
Abu Dhabi	United Arab Emirates	Middle East
Abuja	Nigeria	Western Africa

```


####
 Inner join (2)




```

SELECT c1.name AS city, c2.name AS country
FROM cities AS c1
INNER JOIN countries AS c2
ON c1.country_code = c2.code;

```




```

-- 3. Select fields with aliases
SELECT c.code AS country_code, name, year, inflation_rate
FROM countries AS c
  -- 1. Join to economies (alias e)
  INNER JOIN economies AS e
    -- 2. Match on code
    ON c.code = e.code;

```


####
 Inner join (3)




```

SELECT *
FROM left_table
  INNER JOIN right_table
    ON left_table.id = right_table.id
  INNER JOIN another_table
    ON left_table.id = another_table.id;

```




```

-- 6. Select fields
SELECT c.code, name, region, e.year, fertility_rate, unemployment_rate
  -- 1. From countries (alias as c)
  FROM countries AS c
  -- 2. Join to populations (as p)
  INNER JOIN populations AS p
    -- 3. Match on country code
    ON c.code = p.country_code
  -- 4. Join to economies (as e)
  INNER JOIN economies AS e
    -- 5. Match on country code
    ON c.code = e.code;


```




```

-- countries INNER JOIN populations table
code	name	fertility_rate
ABW	Aruba	1.704
ABW	Aruba	1.647
AFG	Afghanistan	5.746
AFG	Afghanistan	4.653

-- economies table
econ_id	code	year
1	AFG	2010
2	AFG	2015


code	name	region	year	fertility_rate	unemployment_rate
AFG	Afghanistan	Southern and Central Asia	2010	4.653	null
AFG	Afghanistan	Southern and Central Asia	2010	5.746	null
AFG	Afghanistan	Southern and Central Asia	2015	4.653	null
AFG	Afghanistan	Southern and Central Asia	2015	5.746	null
AGO	Angola	Central Africa	2010	5.996	null

```




```

-- 6. Select fields
SELECT c.code, name, region, e.year, fertility_rate, unemployment_rate
  -- 1. From countries (alias as c)
  FROM countries AS c
  -- 2. Join to populations (as p)
  INNER JOIN populations AS p
    -- 3. Match on country code
    ON c.code = p.country_code
  -- 4. Join to economies (as e)
  INNER JOIN economies AS e
    -- 5. Match on country code and year
    ON c.code = e.code AND p.year = e.year;

code	name	region	year	fertility_rate	unemployment_rate
AFG	Afghanistan	Southern and Central Asia	2010	5.746	null
AFG	Afghanistan	Southern and Central Asia	2015	4.653	null

```


###
 INNER JOIN via USING


####
 Inner join with using




```

SELECT *
FROM countries
  INNER JOIN economies
    ON countries.code = economies.code

-- is equal to

SELECT *
FROM countries
  INNER JOIN economies
    USING(code)

```




```

-- 4. Select fields
SELECT c.name AS country, continent, l.name AS language, official
  -- 1. From countries (alias as c)
  FROM countries as c
  -- 2. Join to languages (as l)
  INNER JOIN languages as l
    -- 3. Match using code
    USING (code)


country	continent	language	official
Afghanistan	Asia	Dari	true
Afghanistan	Asia	Pashto	true

```


###
 Self-ish joins, just in CASE


####
 Self-join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture1-19.png)



```

pop_id	country_code	year	fertility_rate	life_expectancy	size
20	ABW	2010	1.704	74.9535	101597
19	ABW	2015	1.647	75.5736	103889

```




```

-- 4. Select fields with aliases
SELECT p1.country_code, p1.size AS size2010, p2.size AS size2015
-- 1. From populations (alias as p1)
FROM populations AS p1
  -- 2. Join to itself (alias as p2)
  INNER JOIN populations AS p2
    -- 3. Match on country code
    ON p1.country_code = p2.country_code


country_code	size2010	size2015
ABW	101597	103889
ABW	101597	101597
ABW	103889	103889
ABW	103889	101597

```




```

-- 5. Select fields with aliases
SELECT p1.country_code,
       p1.size AS size2010,
       p2.size AS size2015
-- 1. From populations (alias as p1)
FROM populations as p1
  -- 2. Join to itself (alias as p2)
  INNER JOIN populations as p2
    -- 3. Match on country code
    ON p1.country_code = p2.country_code
        -- 4. and year (with calculation)
        AND p1.year = p2.year - 5


country_code	size2010	size2015
ABW	101597	103889
AFG	27962200	32526600
AGO	21220000	25022000
ALB	2913020	2889170

```




```

-- With two numeric fields A and B, the percentage growth from A to B can be calculated as (B−A)/A∗100.0.

SELECT p1.country_code,
       p1.size AS size2010,
       p2.size AS size2015,
       -- 1. calculate growth_perc
       ((p2.size - p1.size)/p1.size * 100.0) AS growth_perc
-- 2. From populations (alias as p1)
FROM populations AS p1
  -- 3. Join to itself (alias as p2)
  INNER JOIN populations AS p2
    -- 4. Match on country code
    ON p1.country_code = p2.country_code
        -- 5. and year (with calculation)
        AND p1.year = p2.year - 5;


country_code	size2010	size2015	growth_perc
ABW	101597	103889	2.25597210228443
AFG	27962200	32526600	16.32329672575
AGO	21220000	25022000	17.9171919822693
ALB	2913020	2889170	-0.818874966353178

```


####
 Case when and then




```

SELECT name, continent, code, surface_area,
    -- 1. First case
    CASE WHEN surface_area > 2000000 THEN 'large'
        -- 2. Second case
        WHEN surface_area > 350000 THEN 'medium'
        -- 3. Else clause + end
        ELSE 'small' END
        -- 4. Alias name
        AS geosize_group
-- 5. From table
FROM countries;


name	continent	code	surface_area	geosize_group
Afghanistan	Asia	AFG	652090	medium
Netherlands	Europe	NLD	41526	small
Albania	Europe	ALB	28748	small

```


####
 Inner challenge




```

SELECT name, continent, code, surface_area,
    CASE WHEN surface_area > 2000000
            THEN 'large'
       WHEN surface_area > 350000
            THEN 'medium'
       ELSE 'small' END
       AS geosize_group
INTO countries_plus
FROM countries;


name	continent	code	surface_area	geosize_group
Afghanistan	Asia	AFG	652090	medium
Netherlands	Europe	NLD	41526	small
Albania	Europe	ALB	28748	small
Algeria	Africa	DZA	2381740	large

```




```

SELECT country_code, size,
    -- 1. First case
    CASE WHEN size > 50000000 THEN 'large'
        -- 2. Second case
        WHEN size > 1000000 THEN 'medium'
        -- 3. Else clause + end
        ELSE 'small' END
        -- 4. Alias name
        AS popsize_group
-- 5. From table
FROM populations
-- 6. Focus on 2015
WHERE year = 2015;


country_code	size	popsize_group
ABW	103889	small
AFG	32526600	medium
AGO	25022000	medium
ALB	2889170	medium

```




```

SELECT country_code, size,
    CASE WHEN size > 50000000 THEN 'large'
        WHEN size > 1000000 THEN 'medium'
        ELSE 'small' END
        AS popsize_group
-- 1. Into table
INTO pop_plus
FROM populations
WHERE year = 2015;

-- 2. Select all columns of pop_plus
SELECT * FROM pop_plus;


country_code	size	popsize_group
ABW	103889	small
AFG	32526600	medium
AGO	25022000	medium
ALB	2889170	medium

```




```

SELECT country_code, size,
  CASE WHEN size > 50000000
            THEN 'large'
       WHEN size > 1000000
            THEN 'medium'
       ELSE 'small' END
       AS popsize_group
INTO pop_plus
FROM populations
WHERE year = 2015;

-- 5. Select fields
SELECT name, continent, geosize_group, popsize_group
-- 1. From countries_plus (alias as c)
FROM countries_plus AS c
  -- 2. Join to pop_plus (alias as p)
  INNER JOIN pop_plus AS p
    -- 3. Match on country code
    ON c.code = p.country_code
-- 4. Order the table
ORDER BY geosize_group;


name	continent	geosize_group	popsize_group
India	Asia	large	large
United States	North America	large	large
Saudi Arabia	Asia	large	medium
China	Asia	large	large

```




---



**Outer joins and cross joins**
--------------------------------


###
 LEFT and RIGHT JOINs



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture-18.png)

####
 Left Join




```

-- Select the city name (with alias), the country code,
-- the country name (with alias), the region,
-- and the city proper population
SELECT c1.name AS city, code, c2.name AS country,
       region, city_proper_pop
-- From left table (with alias)
FROM cities AS c1
  -- Join to right table (with alias)
  INNER JOIN countries AS c2
    -- Match on country code
    ON c1.country_code = c2.code
-- Order by descending country code
ORDER BY code DESC;


city	code	country	region	city_proper_pop
Harare	ZWE	Zimbabwe	Eastern Africa	1606000
Lusaka	ZMB	Zambia	Eastern Africa	1742980
Cape Town	ZAF	South Africa	Southern Africa	3740030

-- 230 rows

```




```

SELECT c1.name AS city, code, c2.name AS country,
       region, city_proper_pop
FROM cities AS c1
  -- 1. Join right table (with alias)
  LEFT JOIN countries AS c2
    -- 2. Match on country code
    ON c1.country_code = c2.code
-- 3. Order by descending country code
ORDER BY code DESC;


city	code	country	region	city_proper_pop
Taichung	null	null	null	2752410
Tainan	null	null	null	1885250
Kaohsiung	null	null	null	2778920
Bucharest	null	null	null	1883420

-- 236 rows

```


####
 Left join (2)




```

/*
5. Select country name AS country, the country's local name,
the language name AS language, and
the percent of the language spoken in the country
*/
SELECT c.name AS country, local_name, l.name AS language, percent
-- 1. From left table (alias as c)
FROM countries AS c
  -- 2. Join to right table (alias as l)
  INNER JOIN languages AS l
    -- 3. Match on fields
    ON c.code = l.code
-- 4. Order by descending country
ORDER BY country DESC;


country	local_name	language	percent
Zimbabwe	Zimbabwe	Shona	null
Zimbabwe	Zimbabwe	Tonga	null
Zimbabwe	Zimbabwe	Tswana	null

-- 914 rows

```




```

/*
5. Select country name AS country, the country's local name,
the language name AS language, and
the percent of the language spoken in the country
*/
SELECT c.name AS country, local_name, l.name AS language, percent
-- 1. From left table (alias as c)
FROM countries AS c
  -- 2. Join to right table (alias as l)
  LEFT JOIN languages AS l
    -- 3. Match on fields
    ON c.code = l.code
-- 4. Order by descending country
ORDER BY country DESC;


country	local_name	language	percent
Zimbabwe	Zimbabwe	Chibarwe	null
Zimbabwe	Zimbabwe	Shona	null
Zimbabwe	Zimbabwe	Ndebele	null
Zimbabwe	Zimbabwe	English	null

-- 921 rows

```


####
 Left join (3)




```

-- 5. Select name, region, and gdp_percapita
SELECT name, region, gdp_percapita
-- 1. From countries (alias as c)
FROM countries AS c
  -- 2. Left join with economies (alias as e)
  LEFT JOIN economies AS e
    -- 3. Match on code fields
    ON c.code = e.code
-- 4. Focus on 2010
WHERE year = 2010;


name	region	gdp_percapita
Afghanistan	Southern and Central Asia	539.667
Angola	Central Africa	3599.27
Albania	Southern Europe	4098.13
United Arab Emirates	Middle East	34628.6

```




```

-- Select fields
SELECT region, AVG(gdp_percapita) AS avg_gdp
-- From countries (alias as c)
FROM countries AS c
  -- Left join with economies (alias as e)
  LEFT JOIN economies AS e
    -- Match on code fields
    ON c.code = e.code
-- Focus on 2010
WHERE year = 2010
-- Group by region
GROUP BY region;


region	avg_gdp
Southern Africa	5051.59797363281
Australia and New Zealand	44792.384765625
Southeast Asia	10547.1541320801

```




```

-- Select fields
SELECT region, AVG(gdp_percapita) AS avg_gdp
-- From countries (alias as c)
FROM countries AS c
  -- Left join with economies (alias as e)
  LEFT JOIN economies AS e
    -- Match on code fields
    ON c.code = e.code
-- Focus on 2010
WHERE year = 2010
-- Group by region
GROUP BY region
-- Order by descending avg_gdp
ORDER BY avg_gdp DESC;


region	avg_gdp
Western Europe	58130.9614955357
Nordic Countries	57073.99765625
North America	47911.509765625
Australia and New Zealand	44792.384765625

```


####
 Right join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture3-17.png)



```

-- convert this code to use RIGHT JOINs instead of LEFT JOINs
/*
SELECT cities.name AS city, urbanarea_pop, countries.name AS country,
       indep_year, languages.name AS language, percent
FROM cities
  LEFT JOIN countries
    ON cities.country_code = countries.code
  LEFT JOIN languages
    ON countries.code = languages.code
ORDER BY city, language;
*/

SELECT cities.name AS city, urbanarea_pop, countries.name AS country,
       indep_year, languages.name AS language, percent
FROM languages
  RIGHT JOIN countries
    ON languages.code = countries.code
  RIGHT JOIN cities
    ON countries.code = cities.country_code
ORDER BY city, language;


city	urbanarea_pop	country	indep_year	language	percent
Abidjan	4765000	Cote d'Ivoire	1960	French	null
Abidjan	4765000	Cote d'Ivoire	1960	Other	null
Abu Dhabi	1145000	United Arab Emirates	1971	Arabic	null

```


###
 FULL JOINs


####
 Full join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture4-16.png)



```

SELECT name AS country, code, region, basic_unit
-- 3. From countries
FROM countries
  -- 4. Join to currencies
  FULL JOIN currencies
    -- 5. Match on code
    USING (code)
-- 1. Where region is North America or null
WHERE region = 'North America' OR region IS NULL
-- 2. Order by region
ORDER BY region;


country	code	region	basic_unit
Greenland	GRL	North America	null
null	TMP	null	United States dollar
null	FLK	null	Falkland Islands pound
null	AIA	null	East Caribbean dollar
null	NIU	null	New Zealand dollar
null	ROM	null	Romanian leu
null	SHN	null	Saint Helena pound
null	SGS	null	British pound
null	TWN	null	New Taiwan dollar
null	WLF	null	CFP franc
null	MSR	null	East Caribbean dollar
null	IOT	null	United States dollar
null	CCK	null	Australian dollar
null	COK	null	New Zealand dollar

```




```

SELECT name AS country, code, region, basic_unit
-- 1. From countries
FROM countries
  -- 2. Join to currencies
  LEFT JOIN currencies
    -- 3. Match on code
    USING (code)
-- 4. Where region is North America or null
WHERE region = 'North America' OR region IS NULL
-- 5. Order by region
ORDER BY region;


country	code	region	basic_unit
Bermuda	BMU	North America	Bermudian dollar
Canada	CAN	North America	Canadian dollar
United States	USA	North America	United States dollar
Greenland	GRL	North America	null

```




```

SELECT name AS country, code, region, basic_unit
FROM countries
  -- 1. Join to currencies
  INNER JOIN currencies
    USING (code)
-- 2. Where region is North America or null
WHERE region = 'North America' OR region IS NULL
-- 3. Order by region
ORDER BY region;


country	code	region	basic_unit
Bermuda	BMU	North America	Bermudian dollar
Canada	CAN	North America	Canadian dollar
United States	USA	North America	United States dollar

```


####
 Full join (2)




```

-- FULL JOIN

SELECT countries.name, code, languages.name AS language
-- 3. From languages
FROM languages
  -- 4. Join to countries
  FULL JOIN countries
    -- 5. Match on code
    USING (code)
-- 1. Where countries.name starts with V or is null
WHERE countries.name LIKE 'V%' OR countries.name IS NULL
-- 2. Order by ascending countries.name
ORDER BY countries.name;


name	code	language
Vanuatu	VUT	Tribal Languages
Vanuatu	VUT	English
Vanuatu	VUT	French
Vanuatu	VUT	Other

-- 53 rows

```




```

-- LEFT JOIN

SELECT countries.name, code, languages.name AS language
FROM languages
  -- 1. Join to countries
  LEFT JOIN countries
    -- 2. Match using code
    USING (code)
-- 3. Where countries.name starts with V or is null
WHERE countries.name LIKE 'V%' OR countries.name IS NULL
ORDER BY countries.name;


name	code	language
Vanuatu	VUT	English
Vanuatu	VUT	Other
Vanuatu	VUT	French

-- 51 rows

```




```

-- LEFT JOIN

SELECT countries.name, code, languages.name AS language
FROM languages
  -- 1. Join to countries
  INNER JOIN countries
    USING (code)
-- 2. Where countries.name starts with V or is null
WHERE countries.name LIKE 'V%' OR countries.name IS NULL
ORDER BY countries.name;

name	code	language
Vanuatu	VUT	Tribal Languages
Vanuatu	VUT	Bislama
Vanuatu	VUT	English

-- 10 rows

```


####
 Full join (3)




```

-- 7. Select fields (with aliases)
SELECT c1.name AS country, region, l.name AS language,
       basic_unit, frac_unit
-- 1. From countries (alias as c1)
FROM countries AS c1
  -- 2. Join with languages (alias as l)
  FULL JOIN languages AS l
    -- 3. Match on code
    USING (code)
  -- 4. Join with currencies (alias as c2)
  FULL JOIN currencies AS c2
    -- 5. Match on code
    USING (code)
-- 6. Where region like Melanesia and Micronesia
WHERE region LIKE 'M%esia';


country	region	language	basic_unit	frac_unit
Kiribati	Micronesia	English	Australian dollar	Cent
Kiribati	Micronesia	Kiribati	Australian dollar	Cent
Marshall Islands	Micronesia	Other	United States dollar	Cent
Marshall Islands	Micronesia	Marshallese	United States dollar	Cent

```


###
 CROSSing the rubicon



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture2-17.png)

####
 A table of two cities

 CROSS JOIN




```

-- 4. Select fields
SELECT c.name AS city, l.name AS language
-- 1. From cities (alias as c)
FROM cities AS c
  -- 2. Join to languages (alias as l)
  CROSS JOIN languages AS l
-- 3. Where c.name like Hyderabad
WHERE c.name LIKE 'Hyder%';


city	language
Hyderabad (India)	Dari
Hyderabad	Dari
Hyderabad (India)	Pashto
Hyderabad	Pashto

```




```

-- 5. Select fields
SELECT c.name AS city, l.name AS language
-- 1. From cities (alias as c)
FROM cities AS c
  -- 2. Join to languages (alias as l)
  INNER JOIN languages AS l
    -- 3. Match on country code
    ON c.country_code = l.code
-- 4. Where c.name like Hyderabad
WHERE c.name LIKE 'Hyder%';


city	language
Hyderabad (India)	Hindi
Hyderabad (India)	Bengali
Hyderabad (India)	Telugu
Hyderabad (India)	Marathi

```


####
 Outer challenge




```

-- Select fields
SELECT c.name AS country, region, life_expectancy AS life_exp
-- From countries (alias as c)
FROM countries as c
  -- Join to populations (alias as p)
  LEFT JOIN populations as p
    -- Match on country code
    ON c.code = p.country_code
-- Focus on 2010
WHERE year = 2010
-- Order by life_exp
ORDER BY life_exp
-- Limit to 5 records
LIMIT 5;

```




---



**Set theory clauses**
-----------------------


###
 State of the UNION



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture1-18.png)

####
 Union



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture5-13.png)



```

-- Select fields from 2010 table
SELECT *
  -- From 2010 table
  FROM economies2010
    -- Set theory clause
    UNION
-- Select fields from 2015 table
SELECT *
  -- From 2015 table
  FROM economies2015
-- Order by code and year
ORDER BY code, year;


code	year	income_group	gross_savings
AFG	2010	Low income	37.133
AFG	2015	Low income	21.466
AGO	2010	Upper middle income	23.534
AGO	2015	Upper middle income	-0.425

```


####
 Union (2)




```

-- Select field
SELECT country_code
  -- From cities
  FROM cities
	-- Set theory clause
	UNION
-- Select field
SELECT code AS country_code
  -- From currencies
  FROM currencies
-- Order by country_code
ORDER BY country_code;


country_code
ABW
AFG
AGO
AIA

```


####
 Union all



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture6-10.png)



```

-- Select fields
SELECT code, year
  -- From economies
  FROM economies
	-- Set theory clause
	UNION ALL
-- Select fields
SELECT country_code AS code, year
  -- From populations
  FROM populations
-- Order by code, year
ORDER BY code, year;


code	year
ABW	2010
ABW	2015
AFG	2010
AFG	2010

```


###
 INTERSECTional data science



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture2-18.png)

####
 Intersect




```

-- Select fields
SELECT code, year
  -- From economies
  FROM economies
	-- Set theory clause
	INTERSECT
-- Select fields
SELECT country_code AS code, year
  -- From populations
  FROM populations
-- Order by code and year
ORDER BY code, year;


code	year
AFG	2010
AFG	2015
AGO	2010

```


####
 Intersect (2)




```

-- Select fields
SELECT name
  -- From countries
  FROM countries
	-- Set theory clause
	INTERSECT
-- Select fields
SELECT name
  -- From cities
  FROM cities;


name
Singapore
Hong Kong

```



 Hong Kong is part of China, but it appears separately here because it has its own ISO country code. Depending upon your analysis, treating Hong Kong separately could be useful or a mistake. Always check your dataset closely before you perform an analysis!



###
 EXCEPTional



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture3-16.png)

####
 Except




```

-- Get the names of cities in cities which are not noted as capital cities in countries as a single field result.

-- Select field
SELECT name
  -- From cities
  FROM cities
	-- Set theory clause
	EXCEPT
-- Select field
SELECT capital
  -- From countries
  FROM countries
-- Order by result
ORDER BY name;


name
Abidjan
Ahmedabad
Alexandria

```


####
 Except (2)




```

-- Determine the names of capital cities that are not listed in the cities table.

-- Select field
SELECT capital
  -- From countries
  FROM countries
	-- Set theory clause
	EXCEPT
-- Select field
SELECT name
  -- From cities
  FROM cities
-- Order by ascending capital
ORDER BY capital;


capital
Agana
Amman
Amsterdam
...

```


###
 Semi-joins and Anti-joins



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture4-15.png)

####
 Semi-join




```

-- You are now going to use the concept of a semi-join to identify languages spoken in the Middle East.

-- Select distinct fields
SELECT DISTINCT name
  -- From languages
  FROM languages
-- Where in statement
WHERE code IN
  -- Subquery
  (SELECT code
    FROM countries
        WHERE region = 'Middle East')
-- Order by name
ORDER BY name;

```


####
 Relating semi-join to a tweaked inner join




```

SELECT DISTINCT name
FROM languages
WHERE code IN
  (SELECT code
   FROM countries
   WHERE region = 'Middle East')
ORDER BY name;

-- is equal to

SELECT DISTINCT languages.name AS language
FROM languages
INNER JOIN countries
ON languages.code = countries.code
WHERE region = 'Middle East'
ORDER BY language;

```


####
 Diagnosing problems using anti-join



 Your goal is to identify the currencies used in Oceanian countries!





```

-- Begin by determining the number of countries in countries that are listed in Oceania using SELECT, FROM, and WHERE.


-- Select statement
SELECT COUNT(*)
  -- From countries
  FROM countries
-- Where continent is Oceania
WHERE continent = 'Oceania';


count
19

```




```

-- 5. Select fields (with aliases)
SELECT c1.code, name, basic_unit AS currency
  -- 1. From countries (alias as c1)
  FROM countries AS c1
  	-- 2. Join with currencies (alias as c2)
  	INNER JOIN currencies c2
    -- 3. Match on code
    USING (code)
-- 4. Where continent is Oceania
WHERE continent = 'Oceania';


code	name	currency
AUS	Australia	Australian dollar
PYF	French Polynesia	CFP franc
KIR	Kiribati	Australian dollar

```




```

-- 3. Select fields
SELECT code, name
  -- 4. From Countries
  FROM countries
  -- 5. Where continent is Oceania
  WHERE continent = 'Oceania'
  	-- 1. And code not in
  	AND code NOT IN
  	-- 2. Subquery
  	(SELECT code
  	 FROM currencies);


code	name
ASM	American Samoa
FJI	Fiji Islands
GUM	Guam
FSM	Micronesia, Federated States of
MNP	Northern Mariana Islands

```


####
 Set theory challenge


* Identify the country codes that are included in either
 `economies`
 or
 `currencies`
 but not in
 `populations`
 .
* Use that result to determine the names of cities in the countries that match the specification in the previous instruction.




```

-- Select the city name
SELECT name
  -- Alias the table where city name resides
  FROM cities AS c1
  -- Choose only records matching the result of multiple set theory clauses
  WHERE country_code IN
(
    -- Select appropriate field from economies AS e
    SELECT e.code
    FROM economies AS e
    -- Get all additional (unique) values of the field from currencies AS c2
    UNION
    SELECT c2.code
    FROM currencies AS c2
    -- Exclude those appearing in populations AS p
    EXCEPT
    SELECT p.country_code
    FROM populations AS p
);

```




---



**Subqueries**
---------------


###
 Subqueries inside WHERE and SELECT clauses


####
 Subquery inside where



 You’ll now try to figure out which countries had high average life expectancies (at the country level) in 2015.





```

-- Select average life_expectancy
SELECT AVG(life_expectancy)
  -- From populations
  FROM populations
-- Where year is 2015
WHERE year = 2015


avg
71.6763415481105

```




```

-- Select fields
SELECT *
  -- From populations
  FROM populations
-- Where life_expectancy is greater than
WHERE life_expectancy >
  -- 1.15 * subquery
  1.15 * (SELECT AVG(life_expectancy)
   FROM populations
   WHERE year = 2015) AND
  year = 2015;


pop_id	country_code	year	fertility_rate	life_expectancy	size
21	AUS	2015	1.833	82.4512	23789800
376	CHE	2015	1.54	83.1976	8281430
356	ESP	2015	1.32	83.3805	46444000
134	FRA	2015	2.01	82.6707	66538400

```


####
 Subquery inside where (2)




```

-- 2. Select fields
SELECT name, country_code, urbanarea_pop
  -- 3. From cities
  FROM cities
-- 4. Where city name in the field of capital cities
WHERE name IN
  -- 1. Subquery
  (SELECT capital
   FROM countries)
ORDER BY urbanarea_pop DESC;


name	country_code	urbanarea_pop
Beijing	CHN	21516000
Dhaka	BGD	14543100
Tokyo	JPN	13513700

```


####
 Subquery inside select



 The code selects the top 9 countries in terms of number of cities appearing in the
 `cities`
 table.





```

SELECT countries.name AS country, COUNT(*) AS cities_num
  FROM cities
    INNER JOIN countries
    ON countries.code = cities.country_code
GROUP BY country
ORDER BY cities_num DESC, country
LIMIT 9;

-- is equal to

SELECT countries.name AS country,
  (SELECT COUNT(*)
   FROM cities
   WHERE countries.code = cities.country_code) AS cities_num
FROM countries
ORDER BY cities_num DESC, country
LIMIT 9;

country	cities_num
China	36
India	18
Japan	11

```


###
 Subquery inside FROM clause



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture7-11.png)

####
 Subquery inside from



 You will use this to determine the number of languages spoken for each country, identified by the country’s local name!





```

-- Select fields (with aliases)
SELECT code, COUNT(*) AS lang_num
  -- From languages
  From languages
-- Group by code
GROUP BY code;


code	lang_num
BLZ	9
BGD	2
ITA	4

```




```

-- Select fields
SELECT local_name, subquery.lang_num
  -- From countries
  FROM countries,
  	-- Subquery (alias as subquery)
  	(SELECT code, COUNT(*) AS lang_num
  	 From languages
  	 GROUP BY code) AS subquery
  -- Where codes match
  WHERE countries.code = subquery.code
-- Order by descending number of languages
ORDER BY lang_num DESC;


local_name	lang_num
Zambia	19
Zimbabwe	16
YeItyop´iya	16
Bharat/India	14

```


####
 Advanced subquery



 You can also nest multiple subqueries to answer even more specific questions.




 In this exercise, for each of the six continents listed in 2015, you’ll identify which country had the maximum inflation rate (and how high it was) using multiple subqueries. The table result of your query in
 **Task 3**
 should look something like the following, where anything between
 `<`
`>`
 will be filled in with appropriate values:





```

+------------+---------------+-------------------+
| name       | continent     | inflation_rate    |
|------------+---------------+-------------------|
| <country1> | North America | <max_inflation1>  |
| <country2> | Africa        | <max_inflation2>  |
| <country3> | Oceania       | <max_inflation3>  |
| <country4> | Europe        | <max_inflation4>  |
| <country5> | South America | <max_inflation5>  |
| <country6> | Asia          | <max_inflation6>  |
+------------+---------------+-------------------+

```



 Again, there are multiple ways to get to this solution using only joins, but the focus here is on showing you an introduction into advanced subqueries.





```

-- step 1

-- Select fields
SELECT name, continent, inflation_rate
  -- From countries
  FROM countries
  	-- Join to economies
  	INNER JOIN economies
    -- Match on code
    USING (code)
-- Where year is 2015
WHERE year = 2015;


name	continent	inflation_rate
Afghanistan	Asia	-1.549
Angola	Africa	10.287
Albania	Europe	1.896
United Arab Emirates	Asia	4.07

```




```

-- step 2

-- Select fields
SELECT MAX(inflation_rate) AS max_inf
  -- Subquery using FROM (alias as subquery)
  FROM (
      SELECT name, continent, inflation_rate
      FROM countries
        INNER JOIN economies
        USING (code)
      WHERE year = 2015) AS subquery
-- Group by continent
GROUP BY continent;


max_inf
48.684
9.784
39.403
21.858

```




```

-- step 3

-- Select fields
SELECT name, continent, inflation_rate
  -- From countries
  FROM countries
	-- Join to economies
	INNER JOIN economies
	-- Match on code
	ON countries.code = economies.code
  -- Where year is 2015
  WHERE year = 2015 AND inflation_rate
    -- And inflation rate in subquery (alias as subquery)
    IN (
        SELECT MAX(inflation_rate) AS max_inf
        FROM (
             SELECT name, continent, inflation_rate
             FROM countries
                INNER JOIN economies
                ON countries.code = economies.code
             WHERE year = 2015) AS subquery
        GROUP BY continent);


name	continent	inflation_rate
Haiti	North America	7.524
Malawi	Africa	21.858
Nauru	Oceania	9.784

```


####
 Subquery challenge



 Let’s test your understanding of the subqueries with a challenge problem! Use a subquery to get 2015 economic data for countries that do
 **not**
 have



* `gov_form`
 of
 `'Constitutional Monarchy'`
 or
* `'Republic'`
 in their
 `gov_form`
 .



 Here,
 `gov_form`
 stands for the form of the government for each country. Review the different entries for
 `gov_form`
 in the
 `countries`
 table.





```

-- Select fields
SELECT code, inflation_rate, unemployment_rate
  -- From economies
  FROM economies
  -- Where year is 2015 and code is not in
  WHERE year = 2015 AND code NOT IN
  	-- Subquery
  	(SELECT code
  	 FROM countries
  	 WHERE (gov_form = 'Constitutional Monarchy' OR gov_form LIKE '%Republic%'))
-- Order by inflation rate
ORDER BY inflation_rate;


code	inflation_rate	unemployment_rate
AFG	-1.549	null
CHE	-1.14	3.178
PRI	-0.751	12
ROU	-0.596	6.812

```


###
 Course review



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture8-8.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture9-7.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture10-7.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture11-7.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture12-6.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture13-5.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture14-4.png)

####
 Final challenge



 In this exercise, you’ll need to get the country names and other 2015 data in the
 `economies`
 table and the
 `countries`
 table for
 **Central American countries with an official language**
 .





```

-- Select fields
SELECT DISTINCT c.name, e.total_investment, e.imports
  -- From table (with alias)
  FROM countries AS c
    -- Join with table (with alias)
    LEFT JOIN economies AS e
      -- Match on code
      ON (c.code = e.code
      -- and code in Subquery
        AND c.code IN (
          SELECT l.code
          FROM languages AS l
          WHERE official = 'true'
        ) )
  -- Where region and year are correct
  WHERE region = 'Central America' AND year = 2015
-- Order by field
ORDER BY name;


name	total_investment	imports
Belize	22.014	6.743
Costa Rica	20.218	4.629
El Salvador	13.983	8.193

```


####
 Final challenge (2)



 Let’s ease up a bit and calculate the average fertility rate for each region in 2015.





```

-- Select fields
SELECT region, continent, AVG(fertility_rate) AS avg_fert_rate
  -- From left table
  FROM countries AS c
    -- Join to right table
    INNER JOIN populations AS p
      -- Match on join condition
      ON c.code = p.country_code
  -- Where specific records matching some condition
  WHERE year = 2015
-- Group appropriately
GROUP BY region, continent
-- Order appropriately
ORDER BY avg_fert_rate;


region	continent	avg_fert_rate
Southern Europe	Europe	1.42610000371933
Eastern Europe	Europe	1.49088890022702
Baltic Countries	Europe	1.60333331425985
Eastern Asia	Asia	1.62071430683136

```


####
 Final challenge (3)



 You are now tasked with determining the top 10 capital cities in Europe and the Americas in terms of a calculated percentage using
 `city_proper_pop`
 and
 `metroarea_pop`
 in
 `cities`
 .





```

-- Select fields
SELECT name, country_code, city_proper_pop, metroarea_pop,
      -- Calculate city_perc
      city_proper_pop / metroarea_pop * 100 AS city_perc
  -- From appropriate table
  FROM cities
  -- Where
  WHERE name IN
    -- Subquery
    (SELECT capital
     FROM countries
     WHERE (continent = 'Europe'
        OR continent LIKE '%America'))
       AND metroarea_pop IS NOT NULL
-- Order appropriately
ORDER BY city_perc DESC
-- Limit amount
LIMIT 10;


name	country_code	city_proper_pop	metroarea_pop	city_perc
Lima	PER	8852000	10750000	82.3441863059998
Bogota	COL	7878780	9800000	80.3957462310791
Moscow	RUS	12197600	16170000	75.4334926605225

```



 This is the memo of the 18th course of ‘Data Scientist with Python’ track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/joining-data-in-postgresql)**
 .




 Further Reading:


[More dangerous subtleties of JOINs in SQL](https://alexpetralia.com/posts/2017/7/19/more-dangerous-subtleties-of-joins-in-sql)
 — Be careful when JOIN tables with duplications or NULLs





---



**Introduction to joins**
--------------------------


###
 Introduction to INNER JOIN


####
 Inner join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture-19.png)





```

SELECT table_name
FROM information_schema.tables
-- Specify the correct table_schema value
WHERE table_schema = 'public';


table_name
cities
countries
languages
economies
currencies
populations

```




```

SELECT *
FROM left_table
INNER JOIN right_table
ON left_table.id = right_table.id;

```




```

-- 1. Select name fields (with alias) and region
SELECT cities.name AS city, countries.name AS country, region
FROM cities
  INNER JOIN countries
    ON cities.country_code = countries.code;

city	country	region
Abidjan	Cote d'Ivoire	Western Africa
Abu Dhabi	United Arab Emirates	Middle East
Abuja	Nigeria	Western Africa

```


####
 Inner join (2)




```

SELECT c1.name AS city, c2.name AS country
FROM cities AS c1
INNER JOIN countries AS c2
ON c1.country_code = c2.code;

```




```

-- 3. Select fields with aliases
SELECT c.code AS country_code, name, year, inflation_rate
FROM countries AS c
  -- 1. Join to economies (alias e)
  INNER JOIN economies AS e
    -- 2. Match on code
    ON c.code = e.code;

```


####
 Inner join (3)




```

SELECT *
FROM left_table
  INNER JOIN right_table
    ON left_table.id = right_table.id
  INNER JOIN another_table
    ON left_table.id = another_table.id;

```




```

-- 6. Select fields
SELECT c.code, name, region, e.year, fertility_rate, unemployment_rate
  -- 1. From countries (alias as c)
  FROM countries AS c
  -- 2. Join to populations (as p)
  INNER JOIN populations AS p
    -- 3. Match on country code
    ON c.code = p.country_code
  -- 4. Join to economies (as e)
  INNER JOIN economies AS e
    -- 5. Match on country code
    ON c.code = e.code;


```




```

-- countries INNER JOIN populations table
code	name	fertility_rate
ABW	Aruba	1.704
ABW	Aruba	1.647
AFG	Afghanistan	5.746
AFG	Afghanistan	4.653

-- economies table
econ_id	code	year
1	AFG	2010
2	AFG	2015


code	name	region	year	fertility_rate	unemployment_rate
AFG	Afghanistan	Southern and Central Asia	2010	4.653	null
AFG	Afghanistan	Southern and Central Asia	2010	5.746	null
AFG	Afghanistan	Southern and Central Asia	2015	4.653	null
AFG	Afghanistan	Southern and Central Asia	2015	5.746	null
AGO	Angola	Central Africa	2010	5.996	null

```




```

-- 6. Select fields
SELECT c.code, name, region, e.year, fertility_rate, unemployment_rate
  -- 1. From countries (alias as c)
  FROM countries AS c
  -- 2. Join to populations (as p)
  INNER JOIN populations AS p
    -- 3. Match on country code
    ON c.code = p.country_code
  -- 4. Join to economies (as e)
  INNER JOIN economies AS e
    -- 5. Match on country code and year
    ON c.code = e.code AND p.year = e.year;

code	name	region	year	fertility_rate	unemployment_rate
AFG	Afghanistan	Southern and Central Asia	2010	5.746	null
AFG	Afghanistan	Southern and Central Asia	2015	4.653	null

```


###
 INNER JOIN via USING


####
 Inner join with using




```

SELECT *
FROM countries
  INNER JOIN economies
    ON countries.code = economies.code

-- is equal to

SELECT *
FROM countries
  INNER JOIN economies
    USING(code)

```




```

-- 4. Select fields
SELECT c.name AS country, continent, l.name AS language, official
  -- 1. From countries (alias as c)
  FROM countries as c
  -- 2. Join to languages (as l)
  INNER JOIN languages as l
    -- 3. Match using code
    USING (code)


country	continent	language	official
Afghanistan	Asia	Dari	true
Afghanistan	Asia	Pashto	true

```


###
 Self-ish joins, just in CASE


####
 Self-join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture1-19.png)



```

pop_id	country_code	year	fertility_rate	life_expectancy	size
20	ABW	2010	1.704	74.9535	101597
19	ABW	2015	1.647	75.5736	103889

```




```

-- 4. Select fields with aliases
SELECT p1.country_code, p1.size AS size2010, p2.size AS size2015
-- 1. From populations (alias as p1)
FROM populations AS p1
  -- 2. Join to itself (alias as p2)
  INNER JOIN populations AS p2
    -- 3. Match on country code
    ON p1.country_code = p2.country_code


country_code	size2010	size2015
ABW	101597	103889
ABW	101597	101597
ABW	103889	103889
ABW	103889	101597

```




```

-- 5. Select fields with aliases
SELECT p1.country_code,
       p1.size AS size2010,
       p2.size AS size2015
-- 1. From populations (alias as p1)
FROM populations as p1
  -- 2. Join to itself (alias as p2)
  INNER JOIN populations as p2
    -- 3. Match on country code
    ON p1.country_code = p2.country_code
        -- 4. and year (with calculation)
        AND p1.year = p2.year - 5


country_code	size2010	size2015
ABW	101597	103889
AFG	27962200	32526600
AGO	21220000	25022000
ALB	2913020	2889170

```




```

-- With two numeric fields A and B, the percentage growth from A to B can be calculated as (B−A)/A∗100.0.

SELECT p1.country_code,
       p1.size AS size2010,
       p2.size AS size2015,
       -- 1. calculate growth_perc
       ((p2.size - p1.size)/p1.size * 100.0) AS growth_perc
-- 2. From populations (alias as p1)
FROM populations AS p1
  -- 3. Join to itself (alias as p2)
  INNER JOIN populations AS p2
    -- 4. Match on country code
    ON p1.country_code = p2.country_code
        -- 5. and year (with calculation)
        AND p1.year = p2.year - 5;


country_code	size2010	size2015	growth_perc
ABW	101597	103889	2.25597210228443
AFG	27962200	32526600	16.32329672575
AGO	21220000	25022000	17.9171919822693
ALB	2913020	2889170	-0.818874966353178

```


####
 Case when and then




```

SELECT name, continent, code, surface_area,
    -- 1. First case
    CASE WHEN surface_area > 2000000 THEN 'large'
        -- 2. Second case
        WHEN surface_area > 350000 THEN 'medium'
        -- 3. Else clause + end
        ELSE 'small' END
        -- 4. Alias name
        AS geosize_group
-- 5. From table
FROM countries;


name	continent	code	surface_area	geosize_group
Afghanistan	Asia	AFG	652090	medium
Netherlands	Europe	NLD	41526	small
Albania	Europe	ALB	28748	small

```


####
 Inner challenge




```

SELECT name, continent, code, surface_area,
    CASE WHEN surface_area > 2000000
            THEN 'large'
       WHEN surface_area > 350000
            THEN 'medium'
       ELSE 'small' END
       AS geosize_group
INTO countries_plus
FROM countries;


name	continent	code	surface_area	geosize_group
Afghanistan	Asia	AFG	652090	medium
Netherlands	Europe	NLD	41526	small
Albania	Europe	ALB	28748	small
Algeria	Africa	DZA	2381740	large

```




```

SELECT country_code, size,
    -- 1. First case
    CASE WHEN size > 50000000 THEN 'large'
        -- 2. Second case
        WHEN size > 1000000 THEN 'medium'
        -- 3. Else clause + end
        ELSE 'small' END
        -- 4. Alias name
        AS popsize_group
-- 5. From table
FROM populations
-- 6. Focus on 2015
WHERE year = 2015;


country_code	size	popsize_group
ABW	103889	small
AFG	32526600	medium
AGO	25022000	medium
ALB	2889170	medium

```




```

SELECT country_code, size,
    CASE WHEN size > 50000000 THEN 'large'
        WHEN size > 1000000 THEN 'medium'
        ELSE 'small' END
        AS popsize_group
-- 1. Into table
INTO pop_plus
FROM populations
WHERE year = 2015;

-- 2. Select all columns of pop_plus
SELECT * FROM pop_plus;


country_code	size	popsize_group
ABW	103889	small
AFG	32526600	medium
AGO	25022000	medium
ALB	2889170	medium

```




```

SELECT country_code, size,
  CASE WHEN size > 50000000
            THEN 'large'
       WHEN size > 1000000
            THEN 'medium'
       ELSE 'small' END
       AS popsize_group
INTO pop_plus
FROM populations
WHERE year = 2015;

-- 5. Select fields
SELECT name, continent, geosize_group, popsize_group
-- 1. From countries_plus (alias as c)
FROM countries_plus AS c
  -- 2. Join to pop_plus (alias as p)
  INNER JOIN pop_plus AS p
    -- 3. Match on country code
    ON c.code = p.country_code
-- 4. Order the table
ORDER BY geosize_group;


name	continent	geosize_group	popsize_group
India	Asia	large	large
United States	North America	large	large
Saudi Arabia	Asia	large	medium
China	Asia	large	large

```




---



**Outer joins and cross joins**
--------------------------------


###
 LEFT and RIGHT JOINs



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture-18.png)

####
 Left Join




```

-- Select the city name (with alias), the country code,
-- the country name (with alias), the region,
-- and the city proper population
SELECT c1.name AS city, code, c2.name AS country,
       region, city_proper_pop
-- From left table (with alias)
FROM cities AS c1
  -- Join to right table (with alias)
  INNER JOIN countries AS c2
    -- Match on country code
    ON c1.country_code = c2.code
-- Order by descending country code
ORDER BY code DESC;


city	code	country	region	city_proper_pop
Harare	ZWE	Zimbabwe	Eastern Africa	1606000
Lusaka	ZMB	Zambia	Eastern Africa	1742980
Cape Town	ZAF	South Africa	Southern Africa	3740030

-- 230 rows

```




```

SELECT c1.name AS city, code, c2.name AS country,
       region, city_proper_pop
FROM cities AS c1
  -- 1. Join right table (with alias)
  LEFT JOIN countries AS c2
    -- 2. Match on country code
    ON c1.country_code = c2.code
-- 3. Order by descending country code
ORDER BY code DESC;


city	code	country	region	city_proper_pop
Taichung	null	null	null	2752410
Tainan	null	null	null	1885250
Kaohsiung	null	null	null	2778920
Bucharest	null	null	null	1883420

-- 236 rows

```


####
 Left join (2)




```

/*
5. Select country name AS country, the country's local name,
the language name AS language, and
the percent of the language spoken in the country
*/
SELECT c.name AS country, local_name, l.name AS language, percent
-- 1. From left table (alias as c)
FROM countries AS c
  -- 2. Join to right table (alias as l)
  INNER JOIN languages AS l
    -- 3. Match on fields
    ON c.code = l.code
-- 4. Order by descending country
ORDER BY country DESC;


country	local_name	language	percent
Zimbabwe	Zimbabwe	Shona	null
Zimbabwe	Zimbabwe	Tonga	null
Zimbabwe	Zimbabwe	Tswana	null

-- 914 rows

```




```

/*
5. Select country name AS country, the country's local name,
the language name AS language, and
the percent of the language spoken in the country
*/
SELECT c.name AS country, local_name, l.name AS language, percent
-- 1. From left table (alias as c)
FROM countries AS c
  -- 2. Join to right table (alias as l)
  LEFT JOIN languages AS l
    -- 3. Match on fields
    ON c.code = l.code
-- 4. Order by descending country
ORDER BY country DESC;


country	local_name	language	percent
Zimbabwe	Zimbabwe	Chibarwe	null
Zimbabwe	Zimbabwe	Shona	null
Zimbabwe	Zimbabwe	Ndebele	null
Zimbabwe	Zimbabwe	English	null

-- 921 rows

```


####
 Left join (3)




```

-- 5. Select name, region, and gdp_percapita
SELECT name, region, gdp_percapita
-- 1. From countries (alias as c)
FROM countries AS c
  -- 2. Left join with economies (alias as e)
  LEFT JOIN economies AS e
    -- 3. Match on code fields
    ON c.code = e.code
-- 4. Focus on 2010
WHERE year = 2010;


name	region	gdp_percapita
Afghanistan	Southern and Central Asia	539.667
Angola	Central Africa	3599.27
Albania	Southern Europe	4098.13
United Arab Emirates	Middle East	34628.6

```




```

-- Select fields
SELECT region, AVG(gdp_percapita) AS avg_gdp
-- From countries (alias as c)
FROM countries AS c
  -- Left join with economies (alias as e)
  LEFT JOIN economies AS e
    -- Match on code fields
    ON c.code = e.code
-- Focus on 2010
WHERE year = 2010
-- Group by region
GROUP BY region;


region	avg_gdp
Southern Africa	5051.59797363281
Australia and New Zealand	44792.384765625
Southeast Asia	10547.1541320801

```




```

-- Select fields
SELECT region, AVG(gdp_percapita) AS avg_gdp
-- From countries (alias as c)
FROM countries AS c
  -- Left join with economies (alias as e)
  LEFT JOIN economies AS e
    -- Match on code fields
    ON c.code = e.code
-- Focus on 2010
WHERE year = 2010
-- Group by region
GROUP BY region
-- Order by descending avg_gdp
ORDER BY avg_gdp DESC;


region	avg_gdp
Western Europe	58130.9614955357
Nordic Countries	57073.99765625
North America	47911.509765625
Australia and New Zealand	44792.384765625

```


####
 Right join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture3-17.png)



```

-- convert this code to use RIGHT JOINs instead of LEFT JOINs
/*
SELECT cities.name AS city, urbanarea_pop, countries.name AS country,
       indep_year, languages.name AS language, percent
FROM cities
  LEFT JOIN countries
    ON cities.country_code = countries.code
  LEFT JOIN languages
    ON countries.code = languages.code
ORDER BY city, language;
*/

SELECT cities.name AS city, urbanarea_pop, countries.name AS country,
       indep_year, languages.name AS language, percent
FROM languages
  RIGHT JOIN countries
    ON languages.code = countries.code
  RIGHT JOIN cities
    ON countries.code = cities.country_code
ORDER BY city, language;


city	urbanarea_pop	country	indep_year	language	percent
Abidjan	4765000	Cote d'Ivoire	1960	French	null
Abidjan	4765000	Cote d'Ivoire	1960	Other	null
Abu Dhabi	1145000	United Arab Emirates	1971	Arabic	null

```


###
 FULL JOINs


####
 Full join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture4-16.png)



```

SELECT name AS country, code, region, basic_unit
-- 3. From countries
FROM countries
  -- 4. Join to currencies
  FULL JOIN currencies
    -- 5. Match on code
    USING (code)
-- 1. Where region is North America or null
WHERE region = 'North America' OR region IS NULL
-- 2. Order by region
ORDER BY region;


country	code	region	basic_unit
Greenland	GRL	North America	null
null	TMP	null	United States dollar
null	FLK	null	Falkland Islands pound
null	AIA	null	East Caribbean dollar
null	NIU	null	New Zealand dollar
null	ROM	null	Romanian leu
null	SHN	null	Saint Helena pound
null	SGS	null	British pound
null	TWN	null	New Taiwan dollar
null	WLF	null	CFP franc
null	MSR	null	East Caribbean dollar
null	IOT	null	United States dollar
null	CCK	null	Australian dollar
null	COK	null	New Zealand dollar

```




```

SELECT name AS country, code, region, basic_unit
-- 1. From countries
FROM countries
  -- 2. Join to currencies
  LEFT JOIN currencies
    -- 3. Match on code
    USING (code)
-- 4. Where region is North America or null
WHERE region = 'North America' OR region IS NULL
-- 5. Order by region
ORDER BY region;


country	code	region	basic_unit
Bermuda	BMU	North America	Bermudian dollar
Canada	CAN	North America	Canadian dollar
United States	USA	North America	United States dollar
Greenland	GRL	North America	null

```




```

SELECT name AS country, code, region, basic_unit
FROM countries
  -- 1. Join to currencies
  INNER JOIN currencies
    USING (code)
-- 2. Where region is North America or null
WHERE region = 'North America' OR region IS NULL
-- 3. Order by region
ORDER BY region;


country	code	region	basic_unit
Bermuda	BMU	North America	Bermudian dollar
Canada	CAN	North America	Canadian dollar
United States	USA	North America	United States dollar

```


####
 Full join (2)




```

-- FULL JOIN

SELECT countries.name, code, languages.name AS language
-- 3. From languages
FROM languages
  -- 4. Join to countries
  FULL JOIN countries
    -- 5. Match on code
    USING (code)
-- 1. Where countries.name starts with V or is null
WHERE countries.name LIKE 'V%' OR countries.name IS NULL
-- 2. Order by ascending countries.name
ORDER BY countries.name;


name	code	language
Vanuatu	VUT	Tribal Languages
Vanuatu	VUT	English
Vanuatu	VUT	French
Vanuatu	VUT	Other

-- 53 rows

```




```

-- LEFT JOIN

SELECT countries.name, code, languages.name AS language
FROM languages
  -- 1. Join to countries
  LEFT JOIN countries
    -- 2. Match using code
    USING (code)
-- 3. Where countries.name starts with V or is null
WHERE countries.name LIKE 'V%' OR countries.name IS NULL
ORDER BY countries.name;


name	code	language
Vanuatu	VUT	English
Vanuatu	VUT	Other
Vanuatu	VUT	French

-- 51 rows

```




```

-- LEFT JOIN

SELECT countries.name, code, languages.name AS language
FROM languages
  -- 1. Join to countries
  INNER JOIN countries
    USING (code)
-- 2. Where countries.name starts with V or is null
WHERE countries.name LIKE 'V%' OR countries.name IS NULL
ORDER BY countries.name;

name	code	language
Vanuatu	VUT	Tribal Languages
Vanuatu	VUT	Bislama
Vanuatu	VUT	English

-- 10 rows

```


####
 Full join (3)




```

-- 7. Select fields (with aliases)
SELECT c1.name AS country, region, l.name AS language,
       basic_unit, frac_unit
-- 1. From countries (alias as c1)
FROM countries AS c1
  -- 2. Join with languages (alias as l)
  FULL JOIN languages AS l
    -- 3. Match on code
    USING (code)
  -- 4. Join with currencies (alias as c2)
  FULL JOIN currencies AS c2
    -- 5. Match on code
    USING (code)
-- 6. Where region like Melanesia and Micronesia
WHERE region LIKE 'M%esia';


country	region	language	basic_unit	frac_unit
Kiribati	Micronesia	English	Australian dollar	Cent
Kiribati	Micronesia	Kiribati	Australian dollar	Cent
Marshall Islands	Micronesia	Other	United States dollar	Cent
Marshall Islands	Micronesia	Marshallese	United States dollar	Cent

```


###
 CROSSing the rubicon



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture2-17.png)

####
 A table of two cities

 CROSS JOIN




```

-- 4. Select fields
SELECT c.name AS city, l.name AS language
-- 1. From cities (alias as c)
FROM cities AS c
  -- 2. Join to languages (alias as l)
  CROSS JOIN languages AS l
-- 3. Where c.name like Hyderabad
WHERE c.name LIKE 'Hyder%';


city	language
Hyderabad (India)	Dari
Hyderabad	Dari
Hyderabad (India)	Pashto
Hyderabad	Pashto

```




```

-- 5. Select fields
SELECT c.name AS city, l.name AS language
-- 1. From cities (alias as c)
FROM cities AS c
  -- 2. Join to languages (alias as l)
  INNER JOIN languages AS l
    -- 3. Match on country code
    ON c.country_code = l.code
-- 4. Where c.name like Hyderabad
WHERE c.name LIKE 'Hyder%';


city	language
Hyderabad (India)	Hindi
Hyderabad (India)	Bengali
Hyderabad (India)	Telugu
Hyderabad (India)	Marathi

```


####
 Outer challenge




```

-- Select fields
SELECT c.name AS country, region, life_expectancy AS life_exp
-- From countries (alias as c)
FROM countries as c
  -- Join to populations (alias as p)
  LEFT JOIN populations as p
    -- Match on country code
    ON c.code = p.country_code
-- Focus on 2010
WHERE year = 2010
-- Order by life_exp
ORDER BY life_exp
-- Limit to 5 records
LIMIT 5;

```




---



**Set theory clauses**
-----------------------


###
 State of the UNION



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture1-18.png)

####
 Union



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture5-13.png)



```

-- Select fields from 2010 table
SELECT *
  -- From 2010 table
  FROM economies2010
    -- Set theory clause
    UNION
-- Select fields from 2015 table
SELECT *
  -- From 2015 table
  FROM economies2015
-- Order by code and year
ORDER BY code, year;


code	year	income_group	gross_savings
AFG	2010	Low income	37.133
AFG	2015	Low income	21.466
AGO	2010	Upper middle income	23.534
AGO	2015	Upper middle income	-0.425

```


####
 Union (2)




```

-- Select field
SELECT country_code
  -- From cities
  FROM cities
	-- Set theory clause
	UNION
-- Select field
SELECT code AS country_code
  -- From currencies
  FROM currencies
-- Order by country_code
ORDER BY country_code;


country_code
ABW
AFG
AGO
AIA

```


####
 Union all



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture6-10.png)



```

-- Select fields
SELECT code, year
  -- From economies
  FROM economies
	-- Set theory clause
	UNION ALL
-- Select fields
SELECT country_code AS code, year
  -- From populations
  FROM populations
-- Order by code, year
ORDER BY code, year;


code	year
ABW	2010
ABW	2015
AFG	2010
AFG	2010

```


###
 INTERSECTional data science



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture2-18.png)

####
 Intersect




```

-- Select fields
SELECT code, year
  -- From economies
  FROM economies
	-- Set theory clause
	INTERSECT
-- Select fields
SELECT country_code AS code, year
  -- From populations
  FROM populations
-- Order by code and year
ORDER BY code, year;


code	year
AFG	2010
AFG	2015
AGO	2010

```


####
 Intersect (2)




```

-- Select fields
SELECT name
  -- From countries
  FROM countries
	-- Set theory clause
	INTERSECT
-- Select fields
SELECT name
  -- From cities
  FROM cities;


name
Singapore
Hong Kong

```



 Hong Kong is part of China, but it appears separately here because it has its own ISO country code. Depending upon your analysis, treating Hong Kong separately could be useful or a mistake. Always check your dataset closely before you perform an analysis!



###
 EXCEPTional



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture3-16.png)

####
 Except




```

-- Get the names of cities in cities which are not noted as capital cities in countries as a single field result.

-- Select field
SELECT name
  -- From cities
  FROM cities
	-- Set theory clause
	EXCEPT
-- Select field
SELECT capital
  -- From countries
  FROM countries
-- Order by result
ORDER BY name;


name
Abidjan
Ahmedabad
Alexandria

```


####
 Except (2)




```

-- Determine the names of capital cities that are not listed in the cities table.

-- Select field
SELECT capital
  -- From countries
  FROM countries
	-- Set theory clause
	EXCEPT
-- Select field
SELECT name
  -- From cities
  FROM cities
-- Order by ascending capital
ORDER BY capital;


capital
Agana
Amman
Amsterdam
...

```


###
 Semi-joins and Anti-joins



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture4-15.png)

####
 Semi-join




```

-- You are now going to use the concept of a semi-join to identify languages spoken in the Middle East.

-- Select distinct fields
SELECT DISTINCT name
  -- From languages
  FROM languages
-- Where in statement
WHERE code IN
  -- Subquery
  (SELECT code
    FROM countries
        WHERE region = 'Middle East')
-- Order by name
ORDER BY name;

```


####
 Relating semi-join to a tweaked inner join




```

SELECT DISTINCT name
FROM languages
WHERE code IN
  (SELECT code
   FROM countries
   WHERE region = 'Middle East')
ORDER BY name;

-- is equal to

SELECT DISTINCT languages.name AS language
FROM languages
INNER JOIN countries
ON languages.code = countries.code
WHERE region = 'Middle East'
ORDER BY language;

```


####
 Diagnosing problems using anti-join



 Your goal is to identify the currencies used in Oceanian countries!





```

-- Begin by determining the number of countries in countries that are listed in Oceania using SELECT, FROM, and WHERE.


-- Select statement
SELECT COUNT(*)
  -- From countries
  FROM countries
-- Where continent is Oceania
WHERE continent = 'Oceania';


count
19

```




```

-- 5. Select fields (with aliases)
SELECT c1.code, name, basic_unit AS currency
  -- 1. From countries (alias as c1)
  FROM countries AS c1
  	-- 2. Join with currencies (alias as c2)
  	INNER JOIN currencies c2
    -- 3. Match on code
    USING (code)
-- 4. Where continent is Oceania
WHERE continent = 'Oceania';


code	name	currency
AUS	Australia	Australian dollar
PYF	French Polynesia	CFP franc
KIR	Kiribati	Australian dollar

```




```

-- 3. Select fields
SELECT code, name
  -- 4. From Countries
  FROM countries
  -- 5. Where continent is Oceania
  WHERE continent = 'Oceania'
  	-- 1. And code not in
  	AND code NOT IN
  	-- 2. Subquery
  	(SELECT code
  	 FROM currencies);


code	name
ASM	American Samoa
FJI	Fiji Islands
GUM	Guam
FSM	Micronesia, Federated States of
MNP	Northern Mariana Islands

```


####
 Set theory challenge


* Identify the country codes that are included in either
 `economies`
 or
 `currencies`
 but not in
 `populations`
 .
* Use that result to determine the names of cities in the countries that match the specification in the previous instruction.




```

-- Select the city name
SELECT name
  -- Alias the table where city name resides
  FROM cities AS c1
  -- Choose only records matching the result of multiple set theory clauses
  WHERE country_code IN
(
    -- Select appropriate field from economies AS e
    SELECT e.code
    FROM economies AS e
    -- Get all additional (unique) values of the field from currencies AS c2
    UNION
    SELECT c2.code
    FROM currencies AS c2
    -- Exclude those appearing in populations AS p
    EXCEPT
    SELECT p.country_code
    FROM populations AS p
);

```




---



**Subqueries**
---------------


###
 Subqueries inside WHERE and SELECT clauses


####
 Subquery inside where



 You’ll now try to figure out which countries had high average life expectancies (at the country level) in 2015.





```

-- Select average life_expectancy
SELECT AVG(life_expectancy)
  -- From populations
  FROM populations
-- Where year is 2015
WHERE year = 2015


avg
71.6763415481105

```




```

-- Select fields
SELECT *
  -- From populations
  FROM populations
-- Where life_expectancy is greater than
WHERE life_expectancy >
  -- 1.15 * subquery
  1.15 * (SELECT AVG(life_expectancy)
   FROM populations
   WHERE year = 2015) AND
  year = 2015;


pop_id	country_code	year	fertility_rate	life_expectancy	size
21	AUS	2015	1.833	82.4512	23789800
376	CHE	2015	1.54	83.1976	8281430
356	ESP	2015	1.32	83.3805	46444000
134	FRA	2015	2.01	82.6707	66538400

```


####
 Subquery inside where (2)




```

-- 2. Select fields
SELECT name, country_code, urbanarea_pop
  -- 3. From cities
  FROM cities
-- 4. Where city name in the field of capital cities
WHERE name IN
  -- 1. Subquery
  (SELECT capital
   FROM countries)
ORDER BY urbanarea_pop DESC;


name	country_code	urbanarea_pop
Beijing	CHN	21516000
Dhaka	BGD	14543100
Tokyo	JPN	13513700

```


####
 Subquery inside select



 The code selects the top 9 countries in terms of number of cities appearing in the
 `cities`
 table.





```

SELECT countries.name AS country, COUNT(*) AS cities_num
  FROM cities
    INNER JOIN countries
    ON countries.code = cities.country_code
GROUP BY country
ORDER BY cities_num DESC, country
LIMIT 9;

-- is equal to

SELECT countries.name AS country,
  (SELECT COUNT(*)
   FROM cities
   WHERE countries.code = cities.country_code) AS cities_num
FROM countries
ORDER BY cities_num DESC, country
LIMIT 9;

country	cities_num
China	36
India	18
Japan	11

```


###
 Subquery inside FROM clause



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture7-11.png)

####
 Subquery inside from



 You will use this to determine the number of languages spoken for each country, identified by the country’s local name!





```

-- Select fields (with aliases)
SELECT code, COUNT(*) AS lang_num
  -- From languages
  From languages
-- Group by code
GROUP BY code;


code	lang_num
BLZ	9
BGD	2
ITA	4

```




```

-- Select fields
SELECT local_name, subquery.lang_num
  -- From countries
  FROM countries,
  	-- Subquery (alias as subquery)
  	(SELECT code, COUNT(*) AS lang_num
  	 From languages
  	 GROUP BY code) AS subquery
  -- Where codes match
  WHERE countries.code = subquery.code
-- Order by descending number of languages
ORDER BY lang_num DESC;


local_name	lang_num
Zambia	19
Zimbabwe	16
YeItyop´iya	16
Bharat/India	14

```


####
 Advanced subquery



 You can also nest multiple subqueries to answer even more specific questions.




 In this exercise, for each of the six continents listed in 2015, you’ll identify which country had the maximum inflation rate (and how high it was) using multiple subqueries. The table result of your query in
 **Task 3**
 should look something like the following, where anything between
 `<`
`>`
 will be filled in with appropriate values:





```

+------------+---------------+-------------------+
| name       | continent     | inflation_rate    |
|------------+---------------+-------------------|
| <country1> | North America | <max_inflation1>  |
| <country2> | Africa        | <max_inflation2>  |
| <country3> | Oceania       | <max_inflation3>  |
| <country4> | Europe        | <max_inflation4>  |
| <country5> | South America | <max_inflation5>  |
| <country6> | Asia          | <max_inflation6>  |
+------------+---------------+-------------------+

```



 Again, there are multiple ways to get to this solution using only joins, but the focus here is on showing you an introduction into advanced subqueries.





```

-- step 1

-- Select fields
SELECT name, continent, inflation_rate
  -- From countries
  FROM countries
  	-- Join to economies
  	INNER JOIN economies
    -- Match on code
    USING (code)
-- Where year is 2015
WHERE year = 2015;


name	continent	inflation_rate
Afghanistan	Asia	-1.549
Angola	Africa	10.287
Albania	Europe	1.896
United Arab Emirates	Asia	4.07

```




```

-- step 2

-- Select fields
SELECT MAX(inflation_rate) AS max_inf
  -- Subquery using FROM (alias as subquery)
  FROM (
      SELECT name, continent, inflation_rate
      FROM countries
        INNER JOIN economies
        USING (code)
      WHERE year = 2015) AS subquery
-- Group by continent
GROUP BY continent;


max_inf
48.684
9.784
39.403
21.858

```




```

-- step 3

-- Select fields
SELECT name, continent, inflation_rate
  -- From countries
  FROM countries
	-- Join to economies
	INNER JOIN economies
	-- Match on code
	ON countries.code = economies.code
  -- Where year is 2015
  WHERE year = 2015 AND inflation_rate
    -- And inflation rate in subquery (alias as subquery)
    IN (
        SELECT MAX(inflation_rate) AS max_inf
        FROM (
             SELECT name, continent, inflation_rate
             FROM countries
                INNER JOIN economies
                ON countries.code = economies.code
             WHERE year = 2015) AS subquery
        GROUP BY continent);


name	continent	inflation_rate
Haiti	North America	7.524
Malawi	Africa	21.858
Nauru	Oceania	9.784

```


####
 Subquery challenge



 Let’s test your understanding of the subqueries with a challenge problem! Use a subquery to get 2015 economic data for countries that do
 **not**
 have



* `gov_form`
 of
 `'Constitutional Monarchy'`
 or
* `'Republic'`
 in their
 `gov_form`
 .



 Here,
 `gov_form`
 stands for the form of the government for each country. Review the different entries for
 `gov_form`
 in the
 `countries`
 table.





```

-- Select fields
SELECT code, inflation_rate, unemployment_rate
  -- From economies
  FROM economies
  -- Where year is 2015 and code is not in
  WHERE year = 2015 AND code NOT IN
  	-- Subquery
  	(SELECT code
  	 FROM countries
  	 WHERE (gov_form = 'Constitutional Monarchy' OR gov_form LIKE '%Republic%'))
-- Order by inflation rate
ORDER BY inflation_rate;


code	inflation_rate	unemployment_rate
AFG	-1.549	null
CHE	-1.14	3.178
PRI	-0.751	12
ROU	-0.596	6.812

```


###
 Course review



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture8-8.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture9-7.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture10-7.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture11-7.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture12-6.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture13-5.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture14-4.png)

####
 Final challenge



 In this exercise, you’ll need to get the country names and other 2015 data in the
 `economies`
 table and the
 `countries`
 table for
 **Central American countries with an official language**
 .





```

-- Select fields
SELECT DISTINCT c.name, e.total_investment, e.imports
  -- From table (with alias)
  FROM countries AS c
    -- Join with table (with alias)
    LEFT JOIN economies AS e
      -- Match on code
      ON (c.code = e.code
      -- and code in Subquery
        AND c.code IN (
          SELECT l.code
          FROM languages AS l
          WHERE official = 'true'
        ) )
  -- Where region and year are correct
  WHERE region = 'Central America' AND year = 2015
-- Order by field
ORDER BY name;


name	total_investment	imports
Belize	22.014	6.743
Costa Rica	20.218	4.629
El Salvador	13.983	8.193

```


####
 Final challenge (2)



 Let’s ease up a bit and calculate the average fertility rate for each region in 2015.





```

-- Select fields
SELECT region, continent, AVG(fertility_rate) AS avg_fert_rate
  -- From left table
  FROM countries AS c
    -- Join to right table
    INNER JOIN populations AS p
      -- Match on join condition
      ON c.code = p.country_code
  -- Where specific records matching some condition
  WHERE year = 2015
-- Group appropriately
GROUP BY region, continent
-- Order appropriately
ORDER BY avg_fert_rate;


region	continent	avg_fert_rate
Southern Europe	Europe	1.42610000371933
Eastern Europe	Europe	1.49088890022702
Baltic Countries	Europe	1.60333331425985
Eastern Asia	Asia	1.62071430683136

```


####
 Final challenge (3)



 You are now tasked with determining the top 10 capital cities in Europe and the Americas in terms of a calculated percentage using
 `city_proper_pop`
 and
 `metroarea_pop`
 in
 `cities`
 .





```

-- Select fields
SELECT name, country_code, city_proper_pop, metroarea_pop,
      -- Calculate city_perc
      city_proper_pop / metroarea_pop * 100 AS city_perc
  -- From appropriate table
  FROM cities
  -- Where
  WHERE name IN
    -- Subquery
    (SELECT capital
     FROM countries
     WHERE (continent = 'Europe'
        OR continent LIKE '%America'))
       AND metroarea_pop IS NOT NULL
-- Order appropriately
ORDER BY city_perc DESC
-- Limit amount
LIMIT 10;


name	country_code	city_proper_pop	metroarea_pop	city_perc
Lima	PER	8852000	10750000	82.3441863059998
Bogota	COL	7878780	9800000	80.3957462310791
Moscow	RUS	12197600	16170000	75.4334926605225

```



 This is the memo of the 18th course of ‘Data Scientist with Python’ track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/joining-data-in-postgresql)**
 .




 Further Reading:


[More dangerous subtleties of JOINs in SQL](https://alexpetralia.com/posts/2017/7/19/more-dangerous-subtleties-of-joins-in-sql)
 — Be careful when JOIN tables with duplications or NULLs





---



**Introduction to joins**
--------------------------


###
 Introduction to INNER JOIN


####
 Inner join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture-19.png)





```

SELECT table_name
FROM information_schema.tables
-- Specify the correct table_schema value
WHERE table_schema = 'public';


table_name
cities
countries
languages
economies
currencies
populations

```




```

SELECT *
FROM left_table
INNER JOIN right_table
ON left_table.id = right_table.id;

```




```

-- 1. Select name fields (with alias) and region
SELECT cities.name AS city, countries.name AS country, region
FROM cities
  INNER JOIN countries
    ON cities.country_code = countries.code;

city	country	region
Abidjan	Cote d'Ivoire	Western Africa
Abu Dhabi	United Arab Emirates	Middle East
Abuja	Nigeria	Western Africa

```


####
 Inner join (2)




```

SELECT c1.name AS city, c2.name AS country
FROM cities AS c1
INNER JOIN countries AS c2
ON c1.country_code = c2.code;

```




```

-- 3. Select fields with aliases
SELECT c.code AS country_code, name, year, inflation_rate
FROM countries AS c
  -- 1. Join to economies (alias e)
  INNER JOIN economies AS e
    -- 2. Match on code
    ON c.code = e.code;

```


####
 Inner join (3)




```

SELECT *
FROM left_table
  INNER JOIN right_table
    ON left_table.id = right_table.id
  INNER JOIN another_table
    ON left_table.id = another_table.id;

```




```

-- 6. Select fields
SELECT c.code, name, region, e.year, fertility_rate, unemployment_rate
  -- 1. From countries (alias as c)
  FROM countries AS c
  -- 2. Join to populations (as p)
  INNER JOIN populations AS p
    -- 3. Match on country code
    ON c.code = p.country_code
  -- 4. Join to economies (as e)
  INNER JOIN economies AS e
    -- 5. Match on country code
    ON c.code = e.code;


```




```

-- countries INNER JOIN populations table
code	name	fertility_rate
ABW	Aruba	1.704
ABW	Aruba	1.647
AFG	Afghanistan	5.746
AFG	Afghanistan	4.653

-- economies table
econ_id	code	year
1	AFG	2010
2	AFG	2015


code	name	region	year	fertility_rate	unemployment_rate
AFG	Afghanistan	Southern and Central Asia	2010	4.653	null
AFG	Afghanistan	Southern and Central Asia	2010	5.746	null
AFG	Afghanistan	Southern and Central Asia	2015	4.653	null
AFG	Afghanistan	Southern and Central Asia	2015	5.746	null
AGO	Angola	Central Africa	2010	5.996	null

```




```

-- 6. Select fields
SELECT c.code, name, region, e.year, fertility_rate, unemployment_rate
  -- 1. From countries (alias as c)
  FROM countries AS c
  -- 2. Join to populations (as p)
  INNER JOIN populations AS p
    -- 3. Match on country code
    ON c.code = p.country_code
  -- 4. Join to economies (as e)
  INNER JOIN economies AS e
    -- 5. Match on country code and year
    ON c.code = e.code AND p.year = e.year;

code	name	region	year	fertility_rate	unemployment_rate
AFG	Afghanistan	Southern and Central Asia	2010	5.746	null
AFG	Afghanistan	Southern and Central Asia	2015	4.653	null

```


###
 INNER JOIN via USING


####
 Inner join with using




```

SELECT *
FROM countries
  INNER JOIN economies
    ON countries.code = economies.code

-- is equal to

SELECT *
FROM countries
  INNER JOIN economies
    USING(code)

```




```

-- 4. Select fields
SELECT c.name AS country, continent, l.name AS language, official
  -- 1. From countries (alias as c)
  FROM countries as c
  -- 2. Join to languages (as l)
  INNER JOIN languages as l
    -- 3. Match using code
    USING (code)


country	continent	language	official
Afghanistan	Asia	Dari	true
Afghanistan	Asia	Pashto	true

```


###
 Self-ish joins, just in CASE


####
 Self-join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture1-19.png)



```

pop_id	country_code	year	fertility_rate	life_expectancy	size
20	ABW	2010	1.704	74.9535	101597
19	ABW	2015	1.647	75.5736	103889

```




```

-- 4. Select fields with aliases
SELECT p1.country_code, p1.size AS size2010, p2.size AS size2015
-- 1. From populations (alias as p1)
FROM populations AS p1
  -- 2. Join to itself (alias as p2)
  INNER JOIN populations AS p2
    -- 3. Match on country code
    ON p1.country_code = p2.country_code


country_code	size2010	size2015
ABW	101597	103889
ABW	101597	101597
ABW	103889	103889
ABW	103889	101597

```




```

-- 5. Select fields with aliases
SELECT p1.country_code,
       p1.size AS size2010,
       p2.size AS size2015
-- 1. From populations (alias as p1)
FROM populations as p1
  -- 2. Join to itself (alias as p2)
  INNER JOIN populations as p2
    -- 3. Match on country code
    ON p1.country_code = p2.country_code
        -- 4. and year (with calculation)
        AND p1.year = p2.year - 5


country_code	size2010	size2015
ABW	101597	103889
AFG	27962200	32526600
AGO	21220000	25022000
ALB	2913020	2889170

```




```

-- With two numeric fields A and B, the percentage growth from A to B can be calculated as (B−A)/A∗100.0.

SELECT p1.country_code,
       p1.size AS size2010,
       p2.size AS size2015,
       -- 1. calculate growth_perc
       ((p2.size - p1.size)/p1.size * 100.0) AS growth_perc
-- 2. From populations (alias as p1)
FROM populations AS p1
  -- 3. Join to itself (alias as p2)
  INNER JOIN populations AS p2
    -- 4. Match on country code
    ON p1.country_code = p2.country_code
        -- 5. and year (with calculation)
        AND p1.year = p2.year - 5;


country_code	size2010	size2015	growth_perc
ABW	101597	103889	2.25597210228443
AFG	27962200	32526600	16.32329672575
AGO	21220000	25022000	17.9171919822693
ALB	2913020	2889170	-0.818874966353178

```


####
 Case when and then




```

SELECT name, continent, code, surface_area,
    -- 1. First case
    CASE WHEN surface_area > 2000000 THEN 'large'
        -- 2. Second case
        WHEN surface_area > 350000 THEN 'medium'
        -- 3. Else clause + end
        ELSE 'small' END
        -- 4. Alias name
        AS geosize_group
-- 5. From table
FROM countries;


name	continent	code	surface_area	geosize_group
Afghanistan	Asia	AFG	652090	medium
Netherlands	Europe	NLD	41526	small
Albania	Europe	ALB	28748	small

```


####
 Inner challenge




```

SELECT name, continent, code, surface_area,
    CASE WHEN surface_area > 2000000
            THEN 'large'
       WHEN surface_area > 350000
            THEN 'medium'
       ELSE 'small' END
       AS geosize_group
INTO countries_plus
FROM countries;


name	continent	code	surface_area	geosize_group
Afghanistan	Asia	AFG	652090	medium
Netherlands	Europe	NLD	41526	small
Albania	Europe	ALB	28748	small
Algeria	Africa	DZA	2381740	large

```




```

SELECT country_code, size,
    -- 1. First case
    CASE WHEN size > 50000000 THEN 'large'
        -- 2. Second case
        WHEN size > 1000000 THEN 'medium'
        -- 3. Else clause + end
        ELSE 'small' END
        -- 4. Alias name
        AS popsize_group
-- 5. From table
FROM populations
-- 6. Focus on 2015
WHERE year = 2015;


country_code	size	popsize_group
ABW	103889	small
AFG	32526600	medium
AGO	25022000	medium
ALB	2889170	medium

```




```

SELECT country_code, size,
    CASE WHEN size > 50000000 THEN 'large'
        WHEN size > 1000000 THEN 'medium'
        ELSE 'small' END
        AS popsize_group
-- 1. Into table
INTO pop_plus
FROM populations
WHERE year = 2015;

-- 2. Select all columns of pop_plus
SELECT * FROM pop_plus;


country_code	size	popsize_group
ABW	103889	small
AFG	32526600	medium
AGO	25022000	medium
ALB	2889170	medium

```




```

SELECT country_code, size,
  CASE WHEN size > 50000000
            THEN 'large'
       WHEN size > 1000000
            THEN 'medium'
       ELSE 'small' END
       AS popsize_group
INTO pop_plus
FROM populations
WHERE year = 2015;

-- 5. Select fields
SELECT name, continent, geosize_group, popsize_group
-- 1. From countries_plus (alias as c)
FROM countries_plus AS c
  -- 2. Join to pop_plus (alias as p)
  INNER JOIN pop_plus AS p
    -- 3. Match on country code
    ON c.code = p.country_code
-- 4. Order the table
ORDER BY geosize_group;


name	continent	geosize_group	popsize_group
India	Asia	large	large
United States	North America	large	large
Saudi Arabia	Asia	large	medium
China	Asia	large	large

```




---



**Outer joins and cross joins**
--------------------------------


###
 LEFT and RIGHT JOINs



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture-18.png)

####
 Left Join




```

-- Select the city name (with alias), the country code,
-- the country name (with alias), the region,
-- and the city proper population
SELECT c1.name AS city, code, c2.name AS country,
       region, city_proper_pop
-- From left table (with alias)
FROM cities AS c1
  -- Join to right table (with alias)
  INNER JOIN countries AS c2
    -- Match on country code
    ON c1.country_code = c2.code
-- Order by descending country code
ORDER BY code DESC;


city	code	country	region	city_proper_pop
Harare	ZWE	Zimbabwe	Eastern Africa	1606000
Lusaka	ZMB	Zambia	Eastern Africa	1742980
Cape Town	ZAF	South Africa	Southern Africa	3740030

-- 230 rows

```




```

SELECT c1.name AS city, code, c2.name AS country,
       region, city_proper_pop
FROM cities AS c1
  -- 1. Join right table (with alias)
  LEFT JOIN countries AS c2
    -- 2. Match on country code
    ON c1.country_code = c2.code
-- 3. Order by descending country code
ORDER BY code DESC;


city	code	country	region	city_proper_pop
Taichung	null	null	null	2752410
Tainan	null	null	null	1885250
Kaohsiung	null	null	null	2778920
Bucharest	null	null	null	1883420

-- 236 rows

```


####
 Left join (2)




```

/*
5. Select country name AS country, the country's local name,
the language name AS language, and
the percent of the language spoken in the country
*/
SELECT c.name AS country, local_name, l.name AS language, percent
-- 1. From left table (alias as c)
FROM countries AS c
  -- 2. Join to right table (alias as l)
  INNER JOIN languages AS l
    -- 3. Match on fields
    ON c.code = l.code
-- 4. Order by descending country
ORDER BY country DESC;


country	local_name	language	percent
Zimbabwe	Zimbabwe	Shona	null
Zimbabwe	Zimbabwe	Tonga	null
Zimbabwe	Zimbabwe	Tswana	null

-- 914 rows

```




```

/*
5. Select country name AS country, the country's local name,
the language name AS language, and
the percent of the language spoken in the country
*/
SELECT c.name AS country, local_name, l.name AS language, percent
-- 1. From left table (alias as c)
FROM countries AS c
  -- 2. Join to right table (alias as l)
  LEFT JOIN languages AS l
    -- 3. Match on fields
    ON c.code = l.code
-- 4. Order by descending country
ORDER BY country DESC;


country	local_name	language	percent
Zimbabwe	Zimbabwe	Chibarwe	null
Zimbabwe	Zimbabwe	Shona	null
Zimbabwe	Zimbabwe	Ndebele	null
Zimbabwe	Zimbabwe	English	null

-- 921 rows

```


####
 Left join (3)




```

-- 5. Select name, region, and gdp_percapita
SELECT name, region, gdp_percapita
-- 1. From countries (alias as c)
FROM countries AS c
  -- 2. Left join with economies (alias as e)
  LEFT JOIN economies AS e
    -- 3. Match on code fields
    ON c.code = e.code
-- 4. Focus on 2010
WHERE year = 2010;


name	region	gdp_percapita
Afghanistan	Southern and Central Asia	539.667
Angola	Central Africa	3599.27
Albania	Southern Europe	4098.13
United Arab Emirates	Middle East	34628.6

```




```

-- Select fields
SELECT region, AVG(gdp_percapita) AS avg_gdp
-- From countries (alias as c)
FROM countries AS c
  -- Left join with economies (alias as e)
  LEFT JOIN economies AS e
    -- Match on code fields
    ON c.code = e.code
-- Focus on 2010
WHERE year = 2010
-- Group by region
GROUP BY region;


region	avg_gdp
Southern Africa	5051.59797363281
Australia and New Zealand	44792.384765625
Southeast Asia	10547.1541320801

```




```

-- Select fields
SELECT region, AVG(gdp_percapita) AS avg_gdp
-- From countries (alias as c)
FROM countries AS c
  -- Left join with economies (alias as e)
  LEFT JOIN economies AS e
    -- Match on code fields
    ON c.code = e.code
-- Focus on 2010
WHERE year = 2010
-- Group by region
GROUP BY region
-- Order by descending avg_gdp
ORDER BY avg_gdp DESC;


region	avg_gdp
Western Europe	58130.9614955357
Nordic Countries	57073.99765625
North America	47911.509765625
Australia and New Zealand	44792.384765625

```


####
 Right join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture3-17.png)



```

-- convert this code to use RIGHT JOINs instead of LEFT JOINs
/*
SELECT cities.name AS city, urbanarea_pop, countries.name AS country,
       indep_year, languages.name AS language, percent
FROM cities
  LEFT JOIN countries
    ON cities.country_code = countries.code
  LEFT JOIN languages
    ON countries.code = languages.code
ORDER BY city, language;
*/

SELECT cities.name AS city, urbanarea_pop, countries.name AS country,
       indep_year, languages.name AS language, percent
FROM languages
  RIGHT JOIN countries
    ON languages.code = countries.code
  RIGHT JOIN cities
    ON countries.code = cities.country_code
ORDER BY city, language;


city	urbanarea_pop	country	indep_year	language	percent
Abidjan	4765000	Cote d'Ivoire	1960	French	null
Abidjan	4765000	Cote d'Ivoire	1960	Other	null
Abu Dhabi	1145000	United Arab Emirates	1971	Arabic	null

```


###
 FULL JOINs


####
 Full join



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture4-16.png)



```

SELECT name AS country, code, region, basic_unit
-- 3. From countries
FROM countries
  -- 4. Join to currencies
  FULL JOIN currencies
    -- 5. Match on code
    USING (code)
-- 1. Where region is North America or null
WHERE region = 'North America' OR region IS NULL
-- 2. Order by region
ORDER BY region;


country	code	region	basic_unit
Greenland	GRL	North America	null
null	TMP	null	United States dollar
null	FLK	null	Falkland Islands pound
null	AIA	null	East Caribbean dollar
null	NIU	null	New Zealand dollar
null	ROM	null	Romanian leu
null	SHN	null	Saint Helena pound
null	SGS	null	British pound
null	TWN	null	New Taiwan dollar
null	WLF	null	CFP franc
null	MSR	null	East Caribbean dollar
null	IOT	null	United States dollar
null	CCK	null	Australian dollar
null	COK	null	New Zealand dollar

```




```

SELECT name AS country, code, region, basic_unit
-- 1. From countries
FROM countries
  -- 2. Join to currencies
  LEFT JOIN currencies
    -- 3. Match on code
    USING (code)
-- 4. Where region is North America or null
WHERE region = 'North America' OR region IS NULL
-- 5. Order by region
ORDER BY region;


country	code	region	basic_unit
Bermuda	BMU	North America	Bermudian dollar
Canada	CAN	North America	Canadian dollar
United States	USA	North America	United States dollar
Greenland	GRL	North America	null

```




```

SELECT name AS country, code, region, basic_unit
FROM countries
  -- 1. Join to currencies
  INNER JOIN currencies
    USING (code)
-- 2. Where region is North America or null
WHERE region = 'North America' OR region IS NULL
-- 3. Order by region
ORDER BY region;


country	code	region	basic_unit
Bermuda	BMU	North America	Bermudian dollar
Canada	CAN	North America	Canadian dollar
United States	USA	North America	United States dollar

```


####
 Full join (2)




```

-- FULL JOIN

SELECT countries.name, code, languages.name AS language
-- 3. From languages
FROM languages
  -- 4. Join to countries
  FULL JOIN countries
    -- 5. Match on code
    USING (code)
-- 1. Where countries.name starts with V or is null
WHERE countries.name LIKE 'V%' OR countries.name IS NULL
-- 2. Order by ascending countries.name
ORDER BY countries.name;


name	code	language
Vanuatu	VUT	Tribal Languages
Vanuatu	VUT	English
Vanuatu	VUT	French
Vanuatu	VUT	Other

-- 53 rows

```




```

-- LEFT JOIN

SELECT countries.name, code, languages.name AS language
FROM languages
  -- 1. Join to countries
  LEFT JOIN countries
    -- 2. Match using code
    USING (code)
-- 3. Where countries.name starts with V or is null
WHERE countries.name LIKE 'V%' OR countries.name IS NULL
ORDER BY countries.name;


name	code	language
Vanuatu	VUT	English
Vanuatu	VUT	Other
Vanuatu	VUT	French

-- 51 rows

```




```

-- LEFT JOIN

SELECT countries.name, code, languages.name AS language
FROM languages
  -- 1. Join to countries
  INNER JOIN countries
    USING (code)
-- 2. Where countries.name starts with V or is null
WHERE countries.name LIKE 'V%' OR countries.name IS NULL
ORDER BY countries.name;

name	code	language
Vanuatu	VUT	Tribal Languages
Vanuatu	VUT	Bislama
Vanuatu	VUT	English

-- 10 rows

```


####
 Full join (3)




```

-- 7. Select fields (with aliases)
SELECT c1.name AS country, region, l.name AS language,
       basic_unit, frac_unit
-- 1. From countries (alias as c1)
FROM countries AS c1
  -- 2. Join with languages (alias as l)
  FULL JOIN languages AS l
    -- 3. Match on code
    USING (code)
  -- 4. Join with currencies (alias as c2)
  FULL JOIN currencies AS c2
    -- 5. Match on code
    USING (code)
-- 6. Where region like Melanesia and Micronesia
WHERE region LIKE 'M%esia';


country	region	language	basic_unit	frac_unit
Kiribati	Micronesia	English	Australian dollar	Cent
Kiribati	Micronesia	Kiribati	Australian dollar	Cent
Marshall Islands	Micronesia	Other	United States dollar	Cent
Marshall Islands	Micronesia	Marshallese	United States dollar	Cent

```


###
 CROSSing the rubicon



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture2-17.png)

####
 A table of two cities

 CROSS JOIN




```

-- 4. Select fields
SELECT c.name AS city, l.name AS language
-- 1. From cities (alias as c)
FROM cities AS c
  -- 2. Join to languages (alias as l)
  CROSS JOIN languages AS l
-- 3. Where c.name like Hyderabad
WHERE c.name LIKE 'Hyder%';


city	language
Hyderabad (India)	Dari
Hyderabad	Dari
Hyderabad (India)	Pashto
Hyderabad	Pashto

```




```

-- 5. Select fields
SELECT c.name AS city, l.name AS language
-- 1. From cities (alias as c)
FROM cities AS c
  -- 2. Join to languages (alias as l)
  INNER JOIN languages AS l
    -- 3. Match on country code
    ON c.country_code = l.code
-- 4. Where c.name like Hyderabad
WHERE c.name LIKE 'Hyder%';


city	language
Hyderabad (India)	Hindi
Hyderabad (India)	Bengali
Hyderabad (India)	Telugu
Hyderabad (India)	Marathi

```


####
 Outer challenge




```

-- Select fields
SELECT c.name AS country, region, life_expectancy AS life_exp
-- From countries (alias as c)
FROM countries as c
  -- Join to populations (alias as p)
  LEFT JOIN populations as p
    -- Match on country code
    ON c.code = p.country_code
-- Focus on 2010
WHERE year = 2010
-- Order by life_exp
ORDER BY life_exp
-- Limit to 5 records
LIMIT 5;

```




---



**Set theory clauses**
-----------------------


###
 State of the UNION



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture1-18.png)

####
 Union



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture5-13.png)



```

-- Select fields from 2010 table
SELECT *
  -- From 2010 table
  FROM economies2010
    -- Set theory clause
    UNION
-- Select fields from 2015 table
SELECT *
  -- From 2015 table
  FROM economies2015
-- Order by code and year
ORDER BY code, year;


code	year	income_group	gross_savings
AFG	2010	Low income	37.133
AFG	2015	Low income	21.466
AGO	2010	Upper middle income	23.534
AGO	2015	Upper middle income	-0.425

```


####
 Union (2)




```

-- Select field
SELECT country_code
  -- From cities
  FROM cities
	-- Set theory clause
	UNION
-- Select field
SELECT code AS country_code
  -- From currencies
  FROM currencies
-- Order by country_code
ORDER BY country_code;


country_code
ABW
AFG
AGO
AIA

```


####
 Union all



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture6-10.png)



```

-- Select fields
SELECT code, year
  -- From economies
  FROM economies
	-- Set theory clause
	UNION ALL
-- Select fields
SELECT country_code AS code, year
  -- From populations
  FROM populations
-- Order by code, year
ORDER BY code, year;


code	year
ABW	2010
ABW	2015
AFG	2010
AFG	2010

```


###
 INTERSECTional data science



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture2-18.png)

####
 Intersect




```

-- Select fields
SELECT code, year
  -- From economies
  FROM economies
	-- Set theory clause
	INTERSECT
-- Select fields
SELECT country_code AS code, year
  -- From populations
  FROM populations
-- Order by code and year
ORDER BY code, year;


code	year
AFG	2010
AFG	2015
AGO	2010

```


####
 Intersect (2)




```

-- Select fields
SELECT name
  -- From countries
  FROM countries
	-- Set theory clause
	INTERSECT
-- Select fields
SELECT name
  -- From cities
  FROM cities;


name
Singapore
Hong Kong

```



 Hong Kong is part of China, but it appears separately here because it has its own ISO country code. Depending upon your analysis, treating Hong Kong separately could be useful or a mistake. Always check your dataset closely before you perform an analysis!



###
 EXCEPTional



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture3-16.png)

####
 Except




```

-- Get the names of cities in cities which are not noted as capital cities in countries as a single field result.

-- Select field
SELECT name
  -- From cities
  FROM cities
	-- Set theory clause
	EXCEPT
-- Select field
SELECT capital
  -- From countries
  FROM countries
-- Order by result
ORDER BY name;


name
Abidjan
Ahmedabad
Alexandria

```


####
 Except (2)




```

-- Determine the names of capital cities that are not listed in the cities table.

-- Select field
SELECT capital
  -- From countries
  FROM countries
	-- Set theory clause
	EXCEPT
-- Select field
SELECT name
  -- From cities
  FROM cities
-- Order by ascending capital
ORDER BY capital;


capital
Agana
Amman
Amsterdam
...

```


###
 Semi-joins and Anti-joins



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture4-15.png)

####
 Semi-join




```

-- You are now going to use the concept of a semi-join to identify languages spoken in the Middle East.

-- Select distinct fields
SELECT DISTINCT name
  -- From languages
  FROM languages
-- Where in statement
WHERE code IN
  -- Subquery
  (SELECT code
    FROM countries
        WHERE region = 'Middle East')
-- Order by name
ORDER BY name;

```


####
 Relating semi-join to a tweaked inner join




```

SELECT DISTINCT name
FROM languages
WHERE code IN
  (SELECT code
   FROM countries
   WHERE region = 'Middle East')
ORDER BY name;

-- is equal to

SELECT DISTINCT languages.name AS language
FROM languages
INNER JOIN countries
ON languages.code = countries.code
WHERE region = 'Middle East'
ORDER BY language;

```


####
 Diagnosing problems using anti-join



 Your goal is to identify the currencies used in Oceanian countries!





```

-- Begin by determining the number of countries in countries that are listed in Oceania using SELECT, FROM, and WHERE.


-- Select statement
SELECT COUNT(*)
  -- From countries
  FROM countries
-- Where continent is Oceania
WHERE continent = 'Oceania';


count
19

```




```

-- 5. Select fields (with aliases)
SELECT c1.code, name, basic_unit AS currency
  -- 1. From countries (alias as c1)
  FROM countries AS c1
  	-- 2. Join with currencies (alias as c2)
  	INNER JOIN currencies c2
    -- 3. Match on code
    USING (code)
-- 4. Where continent is Oceania
WHERE continent = 'Oceania';


code	name	currency
AUS	Australia	Australian dollar
PYF	French Polynesia	CFP franc
KIR	Kiribati	Australian dollar

```




```

-- 3. Select fields
SELECT code, name
  -- 4. From Countries
  FROM countries
  -- 5. Where continent is Oceania
  WHERE continent = 'Oceania'
  	-- 1. And code not in
  	AND code NOT IN
  	-- 2. Subquery
  	(SELECT code
  	 FROM currencies);


code	name
ASM	American Samoa
FJI	Fiji Islands
GUM	Guam
FSM	Micronesia, Federated States of
MNP	Northern Mariana Islands

```


####
 Set theory challenge


* Identify the country codes that are included in either
 `economies`
 or
 `currencies`
 but not in
 `populations`
 .
* Use that result to determine the names of cities in the countries that match the specification in the previous instruction.




```

-- Select the city name
SELECT name
  -- Alias the table where city name resides
  FROM cities AS c1
  -- Choose only records matching the result of multiple set theory clauses
  WHERE country_code IN
(
    -- Select appropriate field from economies AS e
    SELECT e.code
    FROM economies AS e
    -- Get all additional (unique) values of the field from currencies AS c2
    UNION
    SELECT c2.code
    FROM currencies AS c2
    -- Exclude those appearing in populations AS p
    EXCEPT
    SELECT p.country_code
    FROM populations AS p
);

```




---



**Subqueries**
---------------


###
 Subqueries inside WHERE and SELECT clauses


####
 Subquery inside where



 You’ll now try to figure out which countries had high average life expectancies (at the country level) in 2015.





```

-- Select average life_expectancy
SELECT AVG(life_expectancy)
  -- From populations
  FROM populations
-- Where year is 2015
WHERE year = 2015


avg
71.6763415481105

```




```

-- Select fields
SELECT *
  -- From populations
  FROM populations
-- Where life_expectancy is greater than
WHERE life_expectancy >
  -- 1.15 * subquery
  1.15 * (SELECT AVG(life_expectancy)
   FROM populations
   WHERE year = 2015) AND
  year = 2015;


pop_id	country_code	year	fertility_rate	life_expectancy	size
21	AUS	2015	1.833	82.4512	23789800
376	CHE	2015	1.54	83.1976	8281430
356	ESP	2015	1.32	83.3805	46444000
134	FRA	2015	2.01	82.6707	66538400

```


####
 Subquery inside where (2)




```

-- 2. Select fields
SELECT name, country_code, urbanarea_pop
  -- 3. From cities
  FROM cities
-- 4. Where city name in the field of capital cities
WHERE name IN
  -- 1. Subquery
  (SELECT capital
   FROM countries)
ORDER BY urbanarea_pop DESC;


name	country_code	urbanarea_pop
Beijing	CHN	21516000
Dhaka	BGD	14543100
Tokyo	JPN	13513700

```


####
 Subquery inside select



 The code selects the top 9 countries in terms of number of cities appearing in the
 `cities`
 table.





```

SELECT countries.name AS country, COUNT(*) AS cities_num
  FROM cities
    INNER JOIN countries
    ON countries.code = cities.country_code
GROUP BY country
ORDER BY cities_num DESC, country
LIMIT 9;

-- is equal to

SELECT countries.name AS country,
  (SELECT COUNT(*)
   FROM cities
   WHERE countries.code = cities.country_code) AS cities_num
FROM countries
ORDER BY cities_num DESC, country
LIMIT 9;

country	cities_num
China	36
India	18
Japan	11

```


###
 Subquery inside FROM clause



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture7-11.png)

####
 Subquery inside from



 You will use this to determine the number of languages spoken for each country, identified by the country’s local name!





```

-- Select fields (with aliases)
SELECT code, COUNT(*) AS lang_num
  -- From languages
  From languages
-- Group by code
GROUP BY code;


code	lang_num
BLZ	9
BGD	2
ITA	4

```




```

-- Select fields
SELECT local_name, subquery.lang_num
  -- From countries
  FROM countries,
  	-- Subquery (alias as subquery)
  	(SELECT code, COUNT(*) AS lang_num
  	 From languages
  	 GROUP BY code) AS subquery
  -- Where codes match
  WHERE countries.code = subquery.code
-- Order by descending number of languages
ORDER BY lang_num DESC;


local_name	lang_num
Zambia	19
Zimbabwe	16
YeItyop´iya	16
Bharat/India	14

```


####
 Advanced subquery



 You can also nest multiple subqueries to answer even more specific questions.




 In this exercise, for each of the six continents listed in 2015, you’ll identify which country had the maximum inflation rate (and how high it was) using multiple subqueries. The table result of your query in
 **Task 3**
 should look something like the following, where anything between
 `<`
`>`
 will be filled in with appropriate values:





```

+------------+---------------+-------------------+
| name       | continent     | inflation_rate    |
|------------+---------------+-------------------|
| <country1> | North America | <max_inflation1>  |
| <country2> | Africa        | <max_inflation2>  |
| <country3> | Oceania       | <max_inflation3>  |
| <country4> | Europe        | <max_inflation4>  |
| <country5> | South America | <max_inflation5>  |
| <country6> | Asia          | <max_inflation6>  |
+------------+---------------+-------------------+

```



 Again, there are multiple ways to get to this solution using only joins, but the focus here is on showing you an introduction into advanced subqueries.





```

-- step 1

-- Select fields
SELECT name, continent, inflation_rate
  -- From countries
  FROM countries
  	-- Join to economies
  	INNER JOIN economies
    -- Match on code
    USING (code)
-- Where year is 2015
WHERE year = 2015;


name	continent	inflation_rate
Afghanistan	Asia	-1.549
Angola	Africa	10.287
Albania	Europe	1.896
United Arab Emirates	Asia	4.07

```




```

-- step 2

-- Select fields
SELECT MAX(inflation_rate) AS max_inf
  -- Subquery using FROM (alias as subquery)
  FROM (
      SELECT name, continent, inflation_rate
      FROM countries
        INNER JOIN economies
        USING (code)
      WHERE year = 2015) AS subquery
-- Group by continent
GROUP BY continent;


max_inf
48.684
9.784
39.403
21.858

```




```

-- step 3

-- Select fields
SELECT name, continent, inflation_rate
  -- From countries
  FROM countries
	-- Join to economies
	INNER JOIN economies
	-- Match on code
	ON countries.code = economies.code
  -- Where year is 2015
  WHERE year = 2015 AND inflation_rate
    -- And inflation rate in subquery (alias as subquery)
    IN (
        SELECT MAX(inflation_rate) AS max_inf
        FROM (
             SELECT name, continent, inflation_rate
             FROM countries
                INNER JOIN economies
                ON countries.code = economies.code
             WHERE year = 2015) AS subquery
        GROUP BY continent);


name	continent	inflation_rate
Haiti	North America	7.524
Malawi	Africa	21.858
Nauru	Oceania	9.784

```


####
 Subquery challenge



 Let’s test your understanding of the subqueries with a challenge problem! Use a subquery to get 2015 economic data for countries that do
 **not**
 have



* `gov_form`
 of
 `'Constitutional Monarchy'`
 or
* `'Republic'`
 in their
 `gov_form`
 .



 Here,
 `gov_form`
 stands for the form of the government for each country. Review the different entries for
 `gov_form`
 in the
 `countries`
 table.





```

-- Select fields
SELECT code, inflation_rate, unemployment_rate
  -- From economies
  FROM economies
  -- Where year is 2015 and code is not in
  WHERE year = 2015 AND code NOT IN
  	-- Subquery
  	(SELECT code
  	 FROM countries
  	 WHERE (gov_form = 'Constitutional Monarchy' OR gov_form LIKE '%Republic%'))
-- Order by inflation rate
ORDER BY inflation_rate;


code	inflation_rate	unemployment_rate
AFG	-1.549	null
CHE	-1.14	3.178
PRI	-0.751	12
ROU	-0.596	6.812

```


###
 Course review



![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture8-8.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture9-7.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture10-7.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture11-7.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture12-6.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture13-5.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/joining-data-in-sql/capture14-4.png)

####
 Final challenge



 In this exercise, you’ll need to get the country names and other 2015 data in the
 `economies`
 table and the
 `countries`
 table for
 **Central American countries with an official language**
 .





```

-- Select fields
SELECT DISTINCT c.name, e.total_investment, e.imports
  -- From table (with alias)
  FROM countries AS c
    -- Join with table (with alias)
    LEFT JOIN economies AS e
      -- Match on code
      ON (c.code = e.code
      -- and code in Subquery
        AND c.code IN (
          SELECT l.code
          FROM languages AS l
          WHERE official = 'true'
        ) )
  -- Where region and year are correct
  WHERE region = 'Central America' AND year = 2015
-- Order by field
ORDER BY name;


name	total_investment	imports
Belize	22.014	6.743
Costa Rica	20.218	4.629
El Salvador	13.983	8.193

```


####
 Final challenge (2)



 Let’s ease up a bit and calculate the average fertility rate for each region in 2015.





```

-- Select fields
SELECT region, continent, AVG(fertility_rate) AS avg_fert_rate
  -- From left table
  FROM countries AS c
    -- Join to right table
    INNER JOIN populations AS p
      -- Match on join condition
      ON c.code = p.country_code
  -- Where specific records matching some condition
  WHERE year = 2015
-- Group appropriately
GROUP BY region, continent
-- Order appropriately
ORDER BY avg_fert_rate;


region	continent	avg_fert_rate
Southern Europe	Europe	1.42610000371933
Eastern Europe	Europe	1.49088890022702
Baltic Countries	Europe	1.60333331425985
Eastern Asia	Asia	1.62071430683136

```


####
 Final challenge (3)



 You are now tasked with determining the top 10 capital cities in Europe and the Americas in terms of a calculated percentage using
 `city_proper_pop`
 and
 `metroarea_pop`
 in
 `cities`
 .





```

-- Select fields
SELECT name, country_code, city_proper_pop, metroarea_pop,
      -- Calculate city_perc
      city_proper_pop / metroarea_pop * 100 AS city_perc
  -- From appropriate table
  FROM cities
  -- Where
  WHERE name IN
    -- Subquery
    (SELECT capital
     FROM countries
     WHERE (continent = 'Europe'
        OR continent LIKE '%America'))
       AND metroarea_pop IS NOT NULL
-- Order appropriately
ORDER BY city_perc DESC
-- Limit amount
LIMIT 10;


name	country_code	city_proper_pop	metroarea_pop	city_perc
Lima	PER	8852000	10750000	82.3441863059998
Bogota	COL	7878780	9800000	80.3957462310791
Moscow	RUS	12197600	16170000	75.4334926605225

```






