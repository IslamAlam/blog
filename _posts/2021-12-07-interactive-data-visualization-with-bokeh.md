---
title: Interactive Data Visualization with Bokeh
date: 2021-12-07 11:22:07 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Interactive Data Visualization with Bokeh
=============================================







 Basic plotting with Bokeh
---------------------------


###
 Plotting with glyphs


####
 What are glyphs?



 In Bokeh, visual properties of shapes are called glyphs.


 Multiple glyphs can be drawn by setting glyph properties to ordered sequences of values.





####
 A simple scatter plot




```python

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Import output_file and show from bokeh.io
from bokeh.io import output_file, show

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility, female_literacy)

# Call the output_file() function and specify the name of the file
output_file('fert_lit.html')

# Display the plot
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture-9.png)

####
 A scatter plot with different shapes




```python

# Create the figure: p
p = figure(x_axis_label='fertility', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica)

# Add an x glyph to the figure p
p.x(fertility_africa, female_literacy_africa)

# Specify the name of the file
output_file('fert_lit_separate.html')

# Display the plot
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture1-8.png)

####
 Customizing your scatter plots




```python

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a blue circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica, color='blue', size=10, alpha=0.8)

# Add a red circle glyph to the figure p
p.circle(fertility_africa, female_literacy_africa, color='red', size=10, alpha=0.8)

# Specify the name of the file
output_file('fert_lit_separate_colors.html')

# Display the plot
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture2-7.png)


[CSS color names](http://www.colors.commutercreative.com/grid/)



###
 Additional glyphs


####
 Lines




```python

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create a figure with x_axis_type="datetime": p
p = figure(x_axis_type="datetime", x_axis_label='Date', y_axis_label='US Dollars')

# Plot date along the x axis and price along the y axis
p.line(date,price)

# Specify the name of the output file and show the result
output_file('line.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture3-6.png)

####
 Lines and markers




```python

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create a figure with x_axis_type='datetime': p
p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')

# Plot date along the x-axis and price along the y-axis
p.line(date, price)

# With date on the x-axis and price on the y-axis, add a white circle glyph of size 4
p.circle(date, price, fill_color='white', size=4)

# Specify the name of the output file and show the result
output_file('line.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture4-6.png)

####
 Patches




```python

# Create a list of az_lons, co_lons, nm_lons and ut_lons: x
x = [az_lons, co_lons, nm_lons, ut_lons]

# Create a list of az_lats, co_lats, nm_lats and ut_lats: y
y = [az_lats, co_lats, nm_lats, ut_lats]

# Add patches to figure p with line_color=white for x and y
p.patches(x, y, line_color = 'white')

# Specify the name of the output file and show the result
output_file('four_corners.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture5-5.png)

###
 Data formats


####
 Plotting data from NumPy arrays




```python

# Import numpy as np
import numpy as np

# Create array using np.linspace: x
x = np.linspace(0,5,100)

# Create array using np.cos: y
y = np.cos(x)

# Add circles at x and y
p.circle(x,y)

# Specify the name of the output file and show the result
output_file('numpy.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture6-5.png)

####
 Plotting data from Pandas DataFrames




```python

# Import pandas as pd
import pandas as pd

# Read in the CSV file: df
df = pd.read_csv('auto.csv')

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create the figure: p
p = figure(x_axis_label='HP', y_axis_label='MPG')

# Plot mpg vs hp by color
p.circle(df['hp'], df['mpg'], color=df['color'], size=10)

# Specify the name of the output file and show the result
output_file('auto-df.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture7-5.png)

####
 The Bokeh ColumnDataSource



 The
 `ColumnDataSource`
 is a table-like data object that maps string column names to sequences (columns) of data. It is the central and most common data structure in Bokeh.




 All columns in a
 `ColumnDataSource`
 must have the same length.press



####
 The Bokeh ColumnDataSource (continued)




```

df.head()
               Name Country   Medal  Time  Year        color
0        Usain Bolt     JAM    GOLD  9.63  2012    goldenrod
1       Yohan Blake     JAM  SILVER  9.75  2012       silver
2     Justin Gatlin     USA  BRONZE  9.79  2012  saddlebrown
3        Usain Bolt     JAM    GOLD  9.69  2008    goldenrod
4  Richard Thompson     TRI  SILVER  9.89  2008       silver

```




```python

# Import the ColumnDataSource class from bokeh.plotting
from bokeh.plotting import ColumnDataSource

# Create a ColumnDataSource from df: source
source = ColumnDataSource(df)

# Add circle glyphs to the figure p
p.circle('Year', 'Time', source=source, color='color', size=8)

# Specify the name of the output file and show the result
output_file('sprint.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture8-4.png)

####
 Selection and non-selection glyphs




```python

# Create a figure with the "box_select" tool: p
p = figure(x_axis_label='Year', y_axis_label='Time', tools='box_select')

# Add circle glyphs to the figure p with the selected and non-selected properties
p.circle('Year', 'Time', source=source, selection_color = 'red', nonselection_alpha = 0.1)

# Specify the name of the output file and show the result
output_file('selection_glyph.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture9-3.png)

####
 Hover glyphs




```python

# import the HoverTool
from bokeh.models import HoverTool

# Add circle glyphs to figure p
p.circle(x, y, size=10,
         fill_color='grey', alpha=0.1, line_color=None,
         hover_fill_color='firebrick', hover_alpha=0.5,
         hover_line_color='white')

# Create a HoverTool: hover
hover = HoverTool(tooltips=None, mode='vline')

# Add the hover tool to the figure p
p.add_tools(hover)

# Specify the name of the output file and show the result
output_file('hover_glyph.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture10-3.png)

####
 Colormapping




```

#Import CategoricalColorMapper from bokeh.models
from bokeh.models import CategoricalColorMapper

# Convert df to a ColumnDataSource: source
source = ColumnDataSource(df)

# Make a CategoricalColorMapper object: color_mapper
color_mapper = CategoricalColorMapper(factors=['Europe', 'Asia', 'US'],
                                      palette=['red', 'green', 'blue'])

# Add a circle glyph to the figure p
p.circle('weight', 'mpg', source=source,
            color=dict(field='origin', transform=color_mapper),
            legend='origin')

# Specify the name of the output file and show the result
output_file('colormap.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture11-3.png)


 Layouts, Interactions, and Annotations
----------------------------------------


###
 Introduction to layouts


####
 Creating rows of plots




```python

# Import row from bokeh.layouts
from bokeh.layouts import row

# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to p1
p1.circle('fertility', 'female_literacy', source=source)

# Create the second figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')

# Add a circle glyph to p2
p2.circle('population', 'female_literacy', source=source)


# Put p1 and p2 into a horizontal row: layout
layout = row(p1, p2)

# Specify the name of the output_file and show the result
output_file('fert_row.html')
show(layout)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture12-2.png)

####
 Creating columns of plots




```python

# Import column from the bokeh.layouts module
from bokeh.layouts import column

# Create a blank figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add circle scatter to the figure p1
p1.circle('fertility', 'female_literacy', source=source)

# Create a new blank figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')

# Add circle scatter to the figure p2
p2.circle('population', 'female_literacy', source=source)

# Put plots p1 and p2 in a column: layout
layout = column(p1, p2)

# Specify the name of the output_file and show the result
output_file('fert_column.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture13-2.png)

####
 Nesting rows and columns of plots




```python

# Import column and row from bokeh.layouts
from bokeh.layouts import row, column

# Make a column layout that will be used as the second row: row2
row2 = column([mpg_hp, mpg_weight], sizing_mode='scale_width')

# Make a row layout that includes the above column layout: layout
layout = row([avg_mpg, row2], sizing_mode='scale_width')

# Specify the name of the output_file and show the result
output_file('layout_custom.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture14-1.png)

###
 Advanced layouts


####
 Creating gridded layouts




```python

# Import gridplot from bokeh.layouts
from bokeh.layouts import gridplot

# Create a list containing plots p1 and p2: row1
row1 = [p1, p2]

# Create a list containing plots p3 and p4: row2
row2 = [p3, p4]

# Create a gridplot using row1 and row2: layout
layout = gridplot([row1, row2])

# Specify the name of the output_file and show the result
output_file('grid.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture15-1.png)

####
 Starting tabbed layouts




```python

# Import Panel from bokeh.models.widgets
from bokeh.models.widgets import Panel

# Create tab1 from plot p1: tab1
tab1 = Panel(child=p1, title='Latin America')

# Create tab2 from plot p2: tab2
tab2 = Panel(child=p2, title='Africa')

# Create tab3 from plot p3: tab3
tab3 = Panel(child=p3, title='Asia')

# Create tab4 from plot p4: tab4
tab4 = Panel(child=p4, title='Europe')


```


####
 Displaying tabbed layouts




```python

# Import Tabs from bokeh.models.widgets
from bokeh.models.widgets import Tabs

# Create a Tabs layout: layout
layout = Tabs(tabs=[tab1, tab2, tab3, tab4])

# Specify the name of the output_file and show the result
output_file('tabs.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture16-1.png)

###
 Linking plots together


####
 Linked axes




```python

# Link the x_range of p2 to p1: p2.x_range
p2.x_range = p1.x_range

# Link the y_range of p2 to p1: p2.y_range
p2.y_range = p1.y_range

# Link the x_range of p3 to p1: p3.x_range
p3.x_range = p1.x_range

# Link the y_range of p4 to p1: p4.y_range
p4.y_range = p1.y_range

# Specify the name of the output_file and show the result
output_file('linked_range.html')
show(layout)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture-10.png)

####
 Linked brushing




```

    Country  Continent female literacy fertility   population
0      Chine       ASI            90.5     1.769  1324.655000
1       Inde       ASI            50.8     2.682  1139.964932
2        USA       NAM              99     2.077   304.060000
3  Indonésie       ASI            88.8     2.132   227.345082
4     Brésil       LAT            90.2     1.827   191.971506

```




```python

# Create ColumnDataSource: source
source = ColumnDataSource(data)

# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female literacy (% population)',
            tools='box_select,lasso_select')

# Add a circle glyph to p1
p1.circle('fertility', 'female literacy', source=source)

# Create the second figure: p2
p2 = figure(x_axis_label='fertility (children per woman)', y_axis_label='population (millions)',
            tools='box_select,lasso_select')

# Add a circle glyph to p2
p2.circle('fertility', 'population', source=source)


# Create row layout of figures p1 and p2: layout
layout = row(p1, p2)

# Specify the name of the output_file and show the result
output_file('linked_brush.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture1-9.png)

###
 Annotations and guides


####
 How to create legends




```python

# Add the first circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=latin_america, size=10, color='red', legend='Latin America')

# Add the second circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=africa, size=10, color='blue', legend='Africa')

# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture2-8.png)

####
 Positioning and styling legends




```python

# Assign the legend to the bottom left: p.legend.location
p.legend.location = 'bottom_left'

# Fill the legend background with the color 'lightgray': p.legend.background_fill_color
p.legend.background_fill_color = 'lightgray'

# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture3-7.png)

####
 Adding a hover tooltip




```python

# Import HoverTool from bokeh.models
from bokeh.models import HoverTool

# Create a HoverTool object: hover
hover = HoverTool(tooltips = [('Country','@Country')])

# Add the HoverTool object to figure p
p.add_tools(hover)

# Specify the name of the output_file and show the result
output_file('hover.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture4-7.png)


 Building interactive apps with Bokeh
--------------------------------------


###
 Introducing the Bokeh Server


####
 Understanding Bokeh apps



 The main purpose of the Bokeh server is to synchronize python objects with web applications in a browser, so that rich, interactive data applications can be connected to powerful PyData libraries such as NumPy, SciPy, Pandas, and scikit-learn.




 The Bokeh server can automatically keep in sync any property of any Bokeh object.





```

bokeh serve myapp.py

```


####

 Using the current document




```python

# Perform necessary imports
from bokeh.io import curdoc
from bokeh.plotting import figure

# Create a new plot: plot
plot = figure()

# Add a line to the plot
plot.line(x = [1,2,3,4,5], y = [2,5,4,6,7])

# Add the plot to the current document
curdoc().add_root(plot)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture-11.png)

####
 Add a single slider




```python

# Perform the necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider

# Create a slider: slider
slider = Slider(title='my slider', start=0, end=10, step=0.1, value=2)

# Create a widgetbox layout: layout
layout = widgetbox(slider)

# Add the layout to the current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture1-10.png)

####
 Multiple sliders in one document




```python

# Perform necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider

# Create first slider: slider1
slider1 = Slider(title = 'slider1', start = 0, end = 10, step = 0.1, value = 2)

# Create second slider: slider2
slider2 = Slider(title = 'slider2', start = 10, end = 100, step = 1, value = 20)

# Add slider1 and slider2 to a widgetbox
layout = widgetbox(slider1, slider2)

# Add the layout to the current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture2-9.png)

###
 Connecting sliders to plots


####
 Adding callbacks to sliders



 Callbacks are functions that a user can define, like
 `def callback(attr, old, new)`
 , that can be called automatically when some property of a Bokeh object (e.g., the
 `value`
 of a
 `Slider`
 ) changes.




 For the
 `value`
 property of
 `Slider`
 objects, callbacks are added by passing a callback function to the
 `on_change`
 method.





```

myslider.on_change('value', callback)

```


####
 How to combine Bokeh models into layouts




```python

# Create ColumnDataSource: source
source = ColumnDataSource(data = {'x': x, 'y': y})

# Add a line to the plot
plot.line('x', 'y', source=source)

# Create a column layout: layout
layout = column(widgetbox(slider), plot)

# Add the layout to the current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture3-8.png)

####
 Learn about widget callbacks




```python

# Define a callback function: callback
def callback(attr, old, new):

    # Read the current value of the slider: scale
    scale = slider.value

    # Compute the updated y using np.sin(scale/x): new_y
    new_y = np.sin(scale/x)

    # Update source with the new data values
    source.data = {'x': x, 'y': new_y}

# Attach the callback to the 'value' property of slider
slider.on_change('value', callback)

# Create layout and add to current document
layout = column(widgetbox(slider), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture4-8.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture5-6.png)

###
 Updating plots from dropdowns


####
 Updating data sources from dropdown callbacks




```python

# Perform necessary imports
from bokeh.models import ColumnDataSource, Select

# Create ColumnDataSource: source
source = ColumnDataSource(data={
    'x' : fertility,
    'y' : female_literacy
})

# Create a new plot: plot
plot = figure()

# Add circles to the plot
plot.circle('x', 'y', source=source)

# Define a callback function: update_plot
def update_plot(attr, old, new):
    # If the new Selection is 'female_literacy', update 'y' to female_literacy
    if new == 'female_literacy':
        source.data = {
            'x' : fertility,
            'y' : female_literacy
        }
    # Else, update 'y' to population
    else:
        source.data = {
            'x' : fertility,
            'y' : population
        }

# Create a dropdown Select widget: select
select = Select(title="distribution", options=['female_literacy', 'population'], value='female_literacy')

# Attach the update_plot callback to the 'value' property of select
select.on_change('value', update_plot)

# Create layout and add to current document
layout = row(select, plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture6-6.png)

####
 Synchronize two dropdowns




```python

# Create two dropdown Select widgets: select1, select2
select1 = Select(title='First', options=['A', 'B'], value='A')
select2 = Select(title='Second', options=['1', '2', '3'], value='1')

# Define a callback function: callback
def callback(attr, old, new):
    # If select1 is 'A'
    if select1.value == 'A':
        # Set select2 options to ['1', '2', '3']
        select2.options = ['1', '2', '3']

        # Set select2 value to '1'
        select2.value = '1'
    else:
        # Set select2 options to ['100', '200', '300']
        select2.options = ['100', '200', '300']

        # Set select2 value to '100'
        select2.value = '100'

# Attach the callback to the 'value' property of select1
select1.on_change('value', callback)

# Create layout and add to current document
layout = widgetbox(select1, select2)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture8-5.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture7-6.png)

###
 Buttons


####
 Button widgets




```python

# Create a Button with label 'Update Data'
button = Button(label='Update Data')

# Define an update callback with no arguments: update
def update():

    # Compute new y values: y
    y = np.sin(x) + np.random.random(N)

    # Update the ColumnDataSource data dictionary
    source.data = {'x':x,'y':y}

# Add the update callback to the button
button.on_click(update)

# Create layout and add to current document
layout = column(widgetbox(button), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture9-4.png)

####
 Button styles




```python

# Import CheckboxGroup, RadioGroup, Toggle from bokeh.models
from bokeh.models import CheckboxGroup, RadioGroup, Toggle

# Add a Toggle: toggle
toggle = Toggle(button_type = 'success', label = 'Toggle button')

# Add a CheckboxGroup: checkbox
checkbox = CheckboxGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add a RadioGroup: radio
radio = RadioGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add widgetbox(toggle, checkbox, radio) to the current document
curdoc().add_root(widgetbox(toggle, checkbox, radio))

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture10-4.png)


 Putting It All Together! A Case Study
---------------------------------------


###
 Time to put it all together!


####
 Introducing the project dataset




```

data.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 10111 entries, 1964 to 2006
Data columns (total 7 columns):
Country            10111 non-null object
fertility          10100 non-null float64
life               10111 non-null float64
population         10108 non-null float64
child_mortality    9210 non-null float64
gdp                9000 non-null float64
region             10111 non-null object
dtypes: float64(5), object(2)
memory usage: 631.9+ KB


data.head()
          Country  fertility    life  population  child_mortality     gdp  \
Year
1964  Afghanistan      7.671  33.639  10474903.0            339.7  1182.0
1965  Afghanistan      7.671  34.152  10697983.0            334.1  1182.0
1966  Afghanistan      7.671  34.662  10927724.0            328.7  1168.0
1967  Afghanistan      7.671  35.170  11163656.0            323.3  1173.0
1968  Afghanistan      7.671  35.674  11411022.0            318.1  1187.0

          region
Year
1964  South Asia
1965  South Asia
1966  South Asia
1967  South Asia
1968  South Asia

```


####
 Some exploratory plots of the data




```python

# Perform necessary imports
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource

# Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : data.loc[1970].fertility,
    'y'       : data.loc[1970].life,
    'country' : data.loc[1970].Country,
})

# Create the figure: p
p = figure(title='1970', x_axis_label='Fertility (children per woman)', y_axis_label='Life Expectancy (years)',
           plot_height=400, plot_width=700,
           tools=[HoverTool(tooltips='@country')])

# Add a circle glyph to the figure p
p.circle(x='x', y='y', source=source)

# Output the file and show the figure
output_file('gapminder.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture11-4.png)

###
 Starting the app



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture12-3.png)

####
 Beginning with just a plot




```python

# Import the necessary modules
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

# Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : data.loc[1970].fertility,
    'y'       : data.loc[1970].life,
    'country'      : data.loc[1970].Country,
    'pop'      : (data.loc[1970].population / 20000000) + 2,
    'region'      : data.loc[1970].region,
})

# Save the minimum and maximum values of the fertility column: xmin, xmax
xmin, xmax = min(data.fertility), max(data.fertility)

# Save the minimum and maximum values of the life expectancy column: ymin, ymax
ymin, ymax = min(data.life), max(data.life)

# Create the figure: plot
plot = figure(title='Gapminder Data for 1970', plot_height=400, plot_width=700,
              x_range=(xmin, xmax), y_range=(ymin, ymax))

# Add circle glyphs to the plot
plot.circle(x='x', y='y', fill_alpha=0.8, source=source)

# Set the x-axis label
plot.xaxis.axis_label ='Fertility (children per woman)'

# Set the y-axis label
plot.yaxis.axis_label = 'Life Expectancy (years)'

# Add the plot to the current document and add a title
curdoc().add_root(plot)
curdoc().title = 'Gapminder'



```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture13-3.png)

####
 Enhancing the plot with some shading




```python

# Make a list of the unique values from the region column: regions_list
regions_list = data.region.unique().tolist()

# Import CategoricalColorMapper from bokeh.models and the Spectral6 palette from bokeh.palettes
from bokeh.models import CategoricalColorMapper
from bokeh.palettes import Spectral6

# Make a color mapper: color_mapper
color_mapper = CategoricalColorMapper(factors=regions_list, palette=Spectral6)

# Add the color mapper to the circle glyph
plot.circle(x='x', y='y', fill_alpha=0.8, source=source,
            color=dict(field='region', transform=color_mapper), legend='region')

# Set the legend.location attribute of the plot to 'top_right'
plot.legend.location = 'top_right'

# Add the plot to the current document and add the title
curdoc().add_root(plot)
curdoc().title = 'Gapminder'

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture14-2.png)

####
 Adding a slider to vary the year




```python

# Import the necessary modules
from bokeh.layouts import row, widgetbox
from bokeh.models import Slider

# Define the callback function: update_plot
def update_plot(attr, old, new):
    # Set the yr name to slider.value and new_data to source.data
    yr = slider.value
    new_data = {
        'x'       : data.loc[yr].fertility,
        'y'       : data.loc[yr].life,
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    source.data = new_data


# Make a slider object: slider
slider = Slider(start = 1970, end = 2010, step = 1, value = 1970, title = 'Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Make a row layout of widgetbox(slider) and plot and add it to the current document
layout = row(widgetbox(slider), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture15-2.png)

###
 Customizing based on user input




```python

# Define the callback function: update_plot
def update_plot(attr, old, new):
    # Assign the value of the slider: yr
    yr = slider.value
    # Set new_data
    new_data = {
        'x'       : data.loc[yr].fertility,
        'y'       : data.loc[yr].life,
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    # Assign new_data to: source.data
    source.data = new_data

    # Add title to figure: plot.title.text
    plot.title.text = 'Gapminder data for %d' % yr

# Make a slider object: slider
slider = Slider(start = 1970, end = 2010, step = 1, value = 1970, title = 'Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Make a row layout of widgetbox(slider) and plot and add it to the current document
layout = row(widgetbox(slider), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture16-2.png)

###
 Adding more interactivity to the app


####
 Adding a hover tool



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture17-1.png)



```python

# Import HoverTool from bokeh.models
from bokeh.models import HoverTool

# Create a HoverTool: hover
hover = HoverTool(tooltips=[('Country', '@country')])

# Add the HoverTool to the plot
plot.add_tools(hover)

# Create layout: layout
layout = row(widgetbox(slider), plot)

# Add layout to current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture19-1.png)

####
 Adding dropdowns to the app



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture18-1.png)



```python

# Define the callback: update_plot
def update_plot(attr, old, new):
    # Read the current value off the slider and 2 dropdowns: yr, x, y
    yr = slider.value
    x = x_select.value
    y = y_select.value
    # Label axes of plot
    plot.xaxis.axis_label = x
    plot.yaxis.axis_label = y
    # Set new_data
    new_data = {
        'x'       : data.loc[yr][x],
        'y'       : data.loc[yr][y],
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    # Assign new_data to source.data
    source.data = new_data

    # Set the range of all axes
    plot.x_range.start = min(data[x])
    plot.x_range.end = max(data[x])
    plot.y_range.start = min(data[y])
    plot.y_range.end = max(data[y])

    # Add title to plot
    plot.title.text = 'Gapminder data for %d' % yr

# Create a dropdown slider widget: slider
slider = Slider(start=1970, end=2010, step=1, value=1970, title='Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Create a dropdown Select widget for the x data: x_select
x_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='fertility',
    title='x-axis data'
)

# Attach the update_plot callback to the 'value' property of x_select
x_select.on_change('value', update_plot)

# Create a dropdown Select widget for the y data: y_select
y_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='life',
    title='y-axis data'
)

# Attach the update_plot callback to the 'value' property of y_select
y_select.on_change('value', update_plot)

# Create layout and add to current document
layout = row(widgetbox(slider, x_select, y_select), plot)
curdoc().add_root(layout)

```




 Basic plotting with Bokeh
---------------------------


###
 Plotting with glyphs


####
 What are glyphs?



 In Bokeh, visual properties of shapes are called glyphs.


 Multiple glyphs can be drawn by setting glyph properties to ordered sequences of values.





####
 A simple scatter plot




```python

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Import output_file and show from bokeh.io
from bokeh.io import output_file, show

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility, female_literacy)

# Call the output_file() function and specify the name of the file
output_file('fert_lit.html')

# Display the plot
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture-9.png)

####
 A scatter plot with different shapes




```python

# Create the figure: p
p = figure(x_axis_label='fertility', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica)

# Add an x glyph to the figure p
p.x(fertility_africa, female_literacy_africa)

# Specify the name of the file
output_file('fert_lit_separate.html')

# Display the plot
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture1-8.png)

####
 Customizing your scatter plots




```python

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a blue circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica, color='blue', size=10, alpha=0.8)

# Add a red circle glyph to the figure p
p.circle(fertility_africa, female_literacy_africa, color='red', size=10, alpha=0.8)

# Specify the name of the file
output_file('fert_lit_separate_colors.html')

# Display the plot
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture2-7.png)


[CSS color names](http://www.colors.commutercreative.com/grid/)



###
 Additional glyphs


####
 Lines




```python

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create a figure with x_axis_type="datetime": p
p = figure(x_axis_type="datetime", x_axis_label='Date', y_axis_label='US Dollars')

# Plot date along the x axis and price along the y axis
p.line(date,price)

# Specify the name of the output file and show the result
output_file('line.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture3-6.png)

####
 Lines and markers




```python

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create a figure with x_axis_type='datetime': p
p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')

# Plot date along the x-axis and price along the y-axis
p.line(date, price)

# With date on the x-axis and price on the y-axis, add a white circle glyph of size 4
p.circle(date, price, fill_color='white', size=4)

# Specify the name of the output file and show the result
output_file('line.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture4-6.png)

####
 Patches




```python

# Create a list of az_lons, co_lons, nm_lons and ut_lons: x
x = [az_lons, co_lons, nm_lons, ut_lons]

# Create a list of az_lats, co_lats, nm_lats and ut_lats: y
y = [az_lats, co_lats, nm_lats, ut_lats]

# Add patches to figure p with line_color=white for x and y
p.patches(x, y, line_color = 'white')

# Specify the name of the output file and show the result
output_file('four_corners.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture5-5.png)

###
 Data formats


####
 Plotting data from NumPy arrays




```python

# Import numpy as np
import numpy as np

# Create array using np.linspace: x
x = np.linspace(0,5,100)

# Create array using np.cos: y
y = np.cos(x)

# Add circles at x and y
p.circle(x,y)

# Specify the name of the output file and show the result
output_file('numpy.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture6-5.png)

####
 Plotting data from Pandas DataFrames




```python

# Import pandas as pd
import pandas as pd

# Read in the CSV file: df
df = pd.read_csv('auto.csv')

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create the figure: p
p = figure(x_axis_label='HP', y_axis_label='MPG')

# Plot mpg vs hp by color
p.circle(df['hp'], df['mpg'], color=df['color'], size=10)

# Specify the name of the output file and show the result
output_file('auto-df.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture7-5.png)

####
 The Bokeh ColumnDataSource



 The
 `ColumnDataSource`
 is a table-like data object that maps string column names to sequences (columns) of data. It is the central and most common data structure in Bokeh.




 All columns in a
 `ColumnDataSource`
 must have the same length.press



####
 The Bokeh ColumnDataSource (continued)




```

df.head()
               Name Country   Medal  Time  Year        color
0        Usain Bolt     JAM    GOLD  9.63  2012    goldenrod
1       Yohan Blake     JAM  SILVER  9.75  2012       silver
2     Justin Gatlin     USA  BRONZE  9.79  2012  saddlebrown
3        Usain Bolt     JAM    GOLD  9.69  2008    goldenrod
4  Richard Thompson     TRI  SILVER  9.89  2008       silver

```




```python

# Import the ColumnDataSource class from bokeh.plotting
from bokeh.plotting import ColumnDataSource

# Create a ColumnDataSource from df: source
source = ColumnDataSource(df)

# Add circle glyphs to the figure p
p.circle('Year', 'Time', source=source, color='color', size=8)

# Specify the name of the output file and show the result
output_file('sprint.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture8-4.png)

####
 Selection and non-selection glyphs




```python

# Create a figure with the "box_select" tool: p
p = figure(x_axis_label='Year', y_axis_label='Time', tools='box_select')

# Add circle glyphs to the figure p with the selected and non-selected properties
p.circle('Year', 'Time', source=source, selection_color = 'red', nonselection_alpha = 0.1)

# Specify the name of the output file and show the result
output_file('selection_glyph.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture9-3.png)

####
 Hover glyphs




```python

# import the HoverTool
from bokeh.models import HoverTool

# Add circle glyphs to figure p
p.circle(x, y, size=10,
         fill_color='grey', alpha=0.1, line_color=None,
         hover_fill_color='firebrick', hover_alpha=0.5,
         hover_line_color='white')

# Create a HoverTool: hover
hover = HoverTool(tooltips=None, mode='vline')

# Add the hover tool to the figure p
p.add_tools(hover)

# Specify the name of the output file and show the result
output_file('hover_glyph.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture10-3.png)

####
 Colormapping




```

#Import CategoricalColorMapper from bokeh.models
from bokeh.models import CategoricalColorMapper

# Convert df to a ColumnDataSource: source
source = ColumnDataSource(df)

# Make a CategoricalColorMapper object: color_mapper
color_mapper = CategoricalColorMapper(factors=['Europe', 'Asia', 'US'],
                                      palette=['red', 'green', 'blue'])

# Add a circle glyph to the figure p
p.circle('weight', 'mpg', source=source,
            color=dict(field='origin', transform=color_mapper),
            legend='origin')

# Specify the name of the output file and show the result
output_file('colormap.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture11-3.png)


 Layouts, Interactions, and Annotations
----------------------------------------


###
 Introduction to layouts


####
 Creating rows of plots




```python

# Import row from bokeh.layouts
from bokeh.layouts import row

# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to p1
p1.circle('fertility', 'female_literacy', source=source)

# Create the second figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')

# Add a circle glyph to p2
p2.circle('population', 'female_literacy', source=source)


# Put p1 and p2 into a horizontal row: layout
layout = row(p1, p2)

# Specify the name of the output_file and show the result
output_file('fert_row.html')
show(layout)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture12-2.png)

####
 Creating columns of plots




```python

# Import column from the bokeh.layouts module
from bokeh.layouts import column

# Create a blank figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add circle scatter to the figure p1
p1.circle('fertility', 'female_literacy', source=source)

# Create a new blank figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')

# Add circle scatter to the figure p2
p2.circle('population', 'female_literacy', source=source)

# Put plots p1 and p2 in a column: layout
layout = column(p1, p2)

# Specify the name of the output_file and show the result
output_file('fert_column.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture13-2.png)

####
 Nesting rows and columns of plots




```python

# Import column and row from bokeh.layouts
from bokeh.layouts import row, column

# Make a column layout that will be used as the second row: row2
row2 = column([mpg_hp, mpg_weight], sizing_mode='scale_width')

# Make a row layout that includes the above column layout: layout
layout = row([avg_mpg, row2], sizing_mode='scale_width')

# Specify the name of the output_file and show the result
output_file('layout_custom.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture14-1.png)

###
 Advanced layouts


####
 Creating gridded layouts




```python

# Import gridplot from bokeh.layouts
from bokeh.layouts import gridplot

# Create a list containing plots p1 and p2: row1
row1 = [p1, p2]

# Create a list containing plots p3 and p4: row2
row2 = [p3, p4]

# Create a gridplot using row1 and row2: layout
layout = gridplot([row1, row2])

# Specify the name of the output_file and show the result
output_file('grid.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture15-1.png)

####
 Starting tabbed layouts




```python

# Import Panel from bokeh.models.widgets
from bokeh.models.widgets import Panel

# Create tab1 from plot p1: tab1
tab1 = Panel(child=p1, title='Latin America')

# Create tab2 from plot p2: tab2
tab2 = Panel(child=p2, title='Africa')

# Create tab3 from plot p3: tab3
tab3 = Panel(child=p3, title='Asia')

# Create tab4 from plot p4: tab4
tab4 = Panel(child=p4, title='Europe')


```


####
 Displaying tabbed layouts




```python

# Import Tabs from bokeh.models.widgets
from bokeh.models.widgets import Tabs

# Create a Tabs layout: layout
layout = Tabs(tabs=[tab1, tab2, tab3, tab4])

# Specify the name of the output_file and show the result
output_file('tabs.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture16-1.png)

###
 Linking plots together


####
 Linked axes




```python

# Link the x_range of p2 to p1: p2.x_range
p2.x_range = p1.x_range

# Link the y_range of p2 to p1: p2.y_range
p2.y_range = p1.y_range

# Link the x_range of p3 to p1: p3.x_range
p3.x_range = p1.x_range

# Link the y_range of p4 to p1: p4.y_range
p4.y_range = p1.y_range

# Specify the name of the output_file and show the result
output_file('linked_range.html')
show(layout)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture-10.png)

####
 Linked brushing




```

    Country  Continent female literacy fertility   population
0      Chine       ASI            90.5     1.769  1324.655000
1       Inde       ASI            50.8     2.682  1139.964932
2        USA       NAM              99     2.077   304.060000
3  Indonésie       ASI            88.8     2.132   227.345082
4     Brésil       LAT            90.2     1.827   191.971506

```




```python

# Create ColumnDataSource: source
source = ColumnDataSource(data)

# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female literacy (% population)',
            tools='box_select,lasso_select')

# Add a circle glyph to p1
p1.circle('fertility', 'female literacy', source=source)

# Create the second figure: p2
p2 = figure(x_axis_label='fertility (children per woman)', y_axis_label='population (millions)',
            tools='box_select,lasso_select')

# Add a circle glyph to p2
p2.circle('fertility', 'population', source=source)


# Create row layout of figures p1 and p2: layout
layout = row(p1, p2)

# Specify the name of the output_file and show the result
output_file('linked_brush.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture1-9.png)

###
 Annotations and guides


####
 How to create legends




```python

# Add the first circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=latin_america, size=10, color='red', legend='Latin America')

# Add the second circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=africa, size=10, color='blue', legend='Africa')

# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture2-8.png)

####
 Positioning and styling legends




```python

# Assign the legend to the bottom left: p.legend.location
p.legend.location = 'bottom_left'

# Fill the legend background with the color 'lightgray': p.legend.background_fill_color
p.legend.background_fill_color = 'lightgray'

# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture3-7.png)

####
 Adding a hover tooltip




```python

# Import HoverTool from bokeh.models
from bokeh.models import HoverTool

# Create a HoverTool object: hover
hover = HoverTool(tooltips = [('Country','@Country')])

# Add the HoverTool object to figure p
p.add_tools(hover)

# Specify the name of the output_file and show the result
output_file('hover.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture4-7.png)


 Building interactive apps with Bokeh
--------------------------------------


###
 Introducing the Bokeh Server


####
 Understanding Bokeh apps



 The main purpose of the Bokeh server is to synchronize python objects with web applications in a browser, so that rich, interactive data applications can be connected to powerful PyData libraries such as NumPy, SciPy, Pandas, and scikit-learn.




 The Bokeh server can automatically keep in sync any property of any Bokeh object.





```

bokeh serve myapp.py

```


####

 Using the current document




```python

# Perform necessary imports
from bokeh.io import curdoc
from bokeh.plotting import figure

# Create a new plot: plot
plot = figure()

# Add a line to the plot
plot.line(x = [1,2,3,4,5], y = [2,5,4,6,7])

# Add the plot to the current document
curdoc().add_root(plot)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture-11.png)

####
 Add a single slider




```python

# Perform the necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider

# Create a slider: slider
slider = Slider(title='my slider', start=0, end=10, step=0.1, value=2)

# Create a widgetbox layout: layout
layout = widgetbox(slider)

# Add the layout to the current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture1-10.png)

####
 Multiple sliders in one document




```python

# Perform necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider

# Create first slider: slider1
slider1 = Slider(title = 'slider1', start = 0, end = 10, step = 0.1, value = 2)

# Create second slider: slider2
slider2 = Slider(title = 'slider2', start = 10, end = 100, step = 1, value = 20)

# Add slider1 and slider2 to a widgetbox
layout = widgetbox(slider1, slider2)

# Add the layout to the current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture2-9.png)

###
 Connecting sliders to plots


####
 Adding callbacks to sliders



 Callbacks are functions that a user can define, like
 `def callback(attr, old, new)`
 , that can be called automatically when some property of a Bokeh object (e.g., the
 `value`
 of a
 `Slider`
 ) changes.




 For the
 `value`
 property of
 `Slider`
 objects, callbacks are added by passing a callback function to the
 `on_change`
 method.





```

myslider.on_change('value', callback)

```


####
 How to combine Bokeh models into layouts




```python

# Create ColumnDataSource: source
source = ColumnDataSource(data = {'x': x, 'y': y})

# Add a line to the plot
plot.line('x', 'y', source=source)

# Create a column layout: layout
layout = column(widgetbox(slider), plot)

# Add the layout to the current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture3-8.png)

####
 Learn about widget callbacks




```python

# Define a callback function: callback
def callback(attr, old, new):

    # Read the current value of the slider: scale
    scale = slider.value

    # Compute the updated y using np.sin(scale/x): new_y
    new_y = np.sin(scale/x)

    # Update source with the new data values
    source.data = {'x': x, 'y': new_y}

# Attach the callback to the 'value' property of slider
slider.on_change('value', callback)

# Create layout and add to current document
layout = column(widgetbox(slider), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture4-8.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture5-6.png)

###
 Updating plots from dropdowns


####
 Updating data sources from dropdown callbacks




```python

# Perform necessary imports
from bokeh.models import ColumnDataSource, Select

# Create ColumnDataSource: source
source = ColumnDataSource(data={
    'x' : fertility,
    'y' : female_literacy
})

# Create a new plot: plot
plot = figure()

# Add circles to the plot
plot.circle('x', 'y', source=source)

# Define a callback function: update_plot
def update_plot(attr, old, new):
    # If the new Selection is 'female_literacy', update 'y' to female_literacy
    if new == 'female_literacy':
        source.data = {
            'x' : fertility,
            'y' : female_literacy
        }
    # Else, update 'y' to population
    else:
        source.data = {
            'x' : fertility,
            'y' : population
        }

# Create a dropdown Select widget: select
select = Select(title="distribution", options=['female_literacy', 'population'], value='female_literacy')

# Attach the update_plot callback to the 'value' property of select
select.on_change('value', update_plot)

# Create layout and add to current document
layout = row(select, plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture6-6.png)

####
 Synchronize two dropdowns




```python

# Create two dropdown Select widgets: select1, select2
select1 = Select(title='First', options=['A', 'B'], value='A')
select2 = Select(title='Second', options=['1', '2', '3'], value='1')

# Define a callback function: callback
def callback(attr, old, new):
    # If select1 is 'A'
    if select1.value == 'A':
        # Set select2 options to ['1', '2', '3']
        select2.options = ['1', '2', '3']

        # Set select2 value to '1'
        select2.value = '1'
    else:
        # Set select2 options to ['100', '200', '300']
        select2.options = ['100', '200', '300']

        # Set select2 value to '100'
        select2.value = '100'

# Attach the callback to the 'value' property of select1
select1.on_change('value', callback)

# Create layout and add to current document
layout = widgetbox(select1, select2)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture8-5.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture7-6.png)

###
 Buttons


####
 Button widgets




```python

# Create a Button with label 'Update Data'
button = Button(label='Update Data')

# Define an update callback with no arguments: update
def update():

    # Compute new y values: y
    y = np.sin(x) + np.random.random(N)

    # Update the ColumnDataSource data dictionary
    source.data = {'x':x,'y':y}

# Add the update callback to the button
button.on_click(update)

# Create layout and add to current document
layout = column(widgetbox(button), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture9-4.png)

####
 Button styles




```python

# Import CheckboxGroup, RadioGroup, Toggle from bokeh.models
from bokeh.models import CheckboxGroup, RadioGroup, Toggle

# Add a Toggle: toggle
toggle = Toggle(button_type = 'success', label = 'Toggle button')

# Add a CheckboxGroup: checkbox
checkbox = CheckboxGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add a RadioGroup: radio
radio = RadioGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add widgetbox(toggle, checkbox, radio) to the current document
curdoc().add_root(widgetbox(toggle, checkbox, radio))

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture10-4.png)


 Putting It All Together! A Case Study
---------------------------------------


###
 Time to put it all together!


####
 Introducing the project dataset




```

data.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 10111 entries, 1964 to 2006
Data columns (total 7 columns):
Country            10111 non-null object
fertility          10100 non-null float64
life               10111 non-null float64
population         10108 non-null float64
child_mortality    9210 non-null float64
gdp                9000 non-null float64
region             10111 non-null object
dtypes: float64(5), object(2)
memory usage: 631.9+ KB


data.head()
          Country  fertility    life  population  child_mortality     gdp  \
Year
1964  Afghanistan      7.671  33.639  10474903.0            339.7  1182.0
1965  Afghanistan      7.671  34.152  10697983.0            334.1  1182.0
1966  Afghanistan      7.671  34.662  10927724.0            328.7  1168.0
1967  Afghanistan      7.671  35.170  11163656.0            323.3  1173.0
1968  Afghanistan      7.671  35.674  11411022.0            318.1  1187.0

          region
Year
1964  South Asia
1965  South Asia
1966  South Asia
1967  South Asia
1968  South Asia

```


####
 Some exploratory plots of the data




```python

# Perform necessary imports
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource

# Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : data.loc[1970].fertility,
    'y'       : data.loc[1970].life,
    'country' : data.loc[1970].Country,
})

# Create the figure: p
p = figure(title='1970', x_axis_label='Fertility (children per woman)', y_axis_label='Life Expectancy (years)',
           plot_height=400, plot_width=700,
           tools=[HoverTool(tooltips='@country')])

# Add a circle glyph to the figure p
p.circle(x='x', y='y', source=source)

# Output the file and show the figure
output_file('gapminder.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture11-4.png)

###
 Starting the app



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture12-3.png)

####
 Beginning with just a plot




```python

# Import the necessary modules
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

# Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : data.loc[1970].fertility,
    'y'       : data.loc[1970].life,
    'country'      : data.loc[1970].Country,
    'pop'      : (data.loc[1970].population / 20000000) + 2,
    'region'      : data.loc[1970].region,
})

# Save the minimum and maximum values of the fertility column: xmin, xmax
xmin, xmax = min(data.fertility), max(data.fertility)

# Save the minimum and maximum values of the life expectancy column: ymin, ymax
ymin, ymax = min(data.life), max(data.life)

# Create the figure: plot
plot = figure(title='Gapminder Data for 1970', plot_height=400, plot_width=700,
              x_range=(xmin, xmax), y_range=(ymin, ymax))

# Add circle glyphs to the plot
plot.circle(x='x', y='y', fill_alpha=0.8, source=source)

# Set the x-axis label
plot.xaxis.axis_label ='Fertility (children per woman)'

# Set the y-axis label
plot.yaxis.axis_label = 'Life Expectancy (years)'

# Add the plot to the current document and add a title
curdoc().add_root(plot)
curdoc().title = 'Gapminder'



```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture13-3.png)

####
 Enhancing the plot with some shading




```python

# Make a list of the unique values from the region column: regions_list
regions_list = data.region.unique().tolist()

# Import CategoricalColorMapper from bokeh.models and the Spectral6 palette from bokeh.palettes
from bokeh.models import CategoricalColorMapper
from bokeh.palettes import Spectral6

# Make a color mapper: color_mapper
color_mapper = CategoricalColorMapper(factors=regions_list, palette=Spectral6)

# Add the color mapper to the circle glyph
plot.circle(x='x', y='y', fill_alpha=0.8, source=source,
            color=dict(field='region', transform=color_mapper), legend='region')

# Set the legend.location attribute of the plot to 'top_right'
plot.legend.location = 'top_right'

# Add the plot to the current document and add the title
curdoc().add_root(plot)
curdoc().title = 'Gapminder'

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture14-2.png)

####
 Adding a slider to vary the year




```python

# Import the necessary modules
from bokeh.layouts import row, widgetbox
from bokeh.models import Slider

# Define the callback function: update_plot
def update_plot(attr, old, new):
    # Set the yr name to slider.value and new_data to source.data
    yr = slider.value
    new_data = {
        'x'       : data.loc[yr].fertility,
        'y'       : data.loc[yr].life,
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    source.data = new_data


# Make a slider object: slider
slider = Slider(start = 1970, end = 2010, step = 1, value = 1970, title = 'Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Make a row layout of widgetbox(slider) and plot and add it to the current document
layout = row(widgetbox(slider), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture15-2.png)

###
 Customizing based on user input




```python

# Define the callback function: update_plot
def update_plot(attr, old, new):
    # Assign the value of the slider: yr
    yr = slider.value
    # Set new_data
    new_data = {
        'x'       : data.loc[yr].fertility,
        'y'       : data.loc[yr].life,
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    # Assign new_data to: source.data
    source.data = new_data

    # Add title to figure: plot.title.text
    plot.title.text = 'Gapminder data for %d' % yr

# Make a slider object: slider
slider = Slider(start = 1970, end = 2010, step = 1, value = 1970, title = 'Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Make a row layout of widgetbox(slider) and plot and add it to the current document
layout = row(widgetbox(slider), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture16-2.png)

###
 Adding more interactivity to the app


####
 Adding a hover tool



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture17-1.png)



```python

# Import HoverTool from bokeh.models
from bokeh.models import HoverTool

# Create a HoverTool: hover
hover = HoverTool(tooltips=[('Country', '@country')])

# Add the HoverTool to the plot
plot.add_tools(hover)

# Create layout: layout
layout = row(widgetbox(slider), plot)

# Add layout to current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture19-1.png)

####
 Adding dropdowns to the app



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture18-1.png)



```python

# Define the callback: update_plot
def update_plot(attr, old, new):
    # Read the current value off the slider and 2 dropdowns: yr, x, y
    yr = slider.value
    x = x_select.value
    y = y_select.value
    # Label axes of plot
    plot.xaxis.axis_label = x
    plot.yaxis.axis_label = y
    # Set new_data
    new_data = {
        'x'       : data.loc[yr][x],
        'y'       : data.loc[yr][y],
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    # Assign new_data to source.data
    source.data = new_data

    # Set the range of all axes
    plot.x_range.start = min(data[x])
    plot.x_range.end = max(data[x])
    plot.y_range.start = min(data[y])
    plot.y_range.end = max(data[y])

    # Add title to plot
    plot.title.text = 'Gapminder data for %d' % yr

# Create a dropdown slider widget: slider
slider = Slider(start=1970, end=2010, step=1, value=1970, title='Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Create a dropdown Select widget for the x data: x_select
x_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='fertility',
    title='x-axis data'
)

# Attach the update_plot callback to the 'value' property of x_select
x_select.on_change('value', update_plot)

# Create a dropdown Select widget for the y data: y_select
y_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='life',
    title='y-axis data'
)

# Attach the update_plot callback to the 'value' property of y_select
y_select.on_change('value', update_plot)

# Create layout and add to current document
layout = row(widgetbox(slider, x_select, y_select), plot)
curdoc().add_root(layout)

```




 Basic plotting with Bokeh
---------------------------


###
 Plotting with glyphs


####
 What are glyphs?



 In Bokeh, visual properties of shapes are called glyphs.


 Multiple glyphs can be drawn by setting glyph properties to ordered sequences of values.





####
 A simple scatter plot




```python

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Import output_file and show from bokeh.io
from bokeh.io import output_file, show

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility, female_literacy)

# Call the output_file() function and specify the name of the file
output_file('fert_lit.html')

# Display the plot
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture-9.png)

####
 A scatter plot with different shapes




```python

# Create the figure: p
p = figure(x_axis_label='fertility', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica)

# Add an x glyph to the figure p
p.x(fertility_africa, female_literacy_africa)

# Specify the name of the file
output_file('fert_lit_separate.html')

# Display the plot
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture1-8.png)

####
 Customizing your scatter plots




```python

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a blue circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica, color='blue', size=10, alpha=0.8)

# Add a red circle glyph to the figure p
p.circle(fertility_africa, female_literacy_africa, color='red', size=10, alpha=0.8)

# Specify the name of the file
output_file('fert_lit_separate_colors.html')

# Display the plot
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture2-7.png)


[CSS color names](http://www.colors.commutercreative.com/grid/)



###
 Additional glyphs


####
 Lines




```python

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create a figure with x_axis_type="datetime": p
p = figure(x_axis_type="datetime", x_axis_label='Date', y_axis_label='US Dollars')

# Plot date along the x axis and price along the y axis
p.line(date,price)

# Specify the name of the output file and show the result
output_file('line.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture3-6.png)

####
 Lines and markers




```python

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create a figure with x_axis_type='datetime': p
p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')

# Plot date along the x-axis and price along the y-axis
p.line(date, price)

# With date on the x-axis and price on the y-axis, add a white circle glyph of size 4
p.circle(date, price, fill_color='white', size=4)

# Specify the name of the output file and show the result
output_file('line.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture4-6.png)

####
 Patches




```python

# Create a list of az_lons, co_lons, nm_lons and ut_lons: x
x = [az_lons, co_lons, nm_lons, ut_lons]

# Create a list of az_lats, co_lats, nm_lats and ut_lats: y
y = [az_lats, co_lats, nm_lats, ut_lats]

# Add patches to figure p with line_color=white for x and y
p.patches(x, y, line_color = 'white')

# Specify the name of the output file and show the result
output_file('four_corners.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture5-5.png)

###
 Data formats


####
 Plotting data from NumPy arrays




```python

# Import numpy as np
import numpy as np

# Create array using np.linspace: x
x = np.linspace(0,5,100)

# Create array using np.cos: y
y = np.cos(x)

# Add circles at x and y
p.circle(x,y)

# Specify the name of the output file and show the result
output_file('numpy.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture6-5.png)

####
 Plotting data from Pandas DataFrames




```python

# Import pandas as pd
import pandas as pd

# Read in the CSV file: df
df = pd.read_csv('auto.csv')

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create the figure: p
p = figure(x_axis_label='HP', y_axis_label='MPG')

# Plot mpg vs hp by color
p.circle(df['hp'], df['mpg'], color=df['color'], size=10)

# Specify the name of the output file and show the result
output_file('auto-df.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture7-5.png)

####
 The Bokeh ColumnDataSource



 The
 `ColumnDataSource`
 is a table-like data object that maps string column names to sequences (columns) of data. It is the central and most common data structure in Bokeh.




 All columns in a
 `ColumnDataSource`
 must have the same length.press



####
 The Bokeh ColumnDataSource (continued)




```

df.head()
               Name Country   Medal  Time  Year        color
0        Usain Bolt     JAM    GOLD  9.63  2012    goldenrod
1       Yohan Blake     JAM  SILVER  9.75  2012       silver
2     Justin Gatlin     USA  BRONZE  9.79  2012  saddlebrown
3        Usain Bolt     JAM    GOLD  9.69  2008    goldenrod
4  Richard Thompson     TRI  SILVER  9.89  2008       silver

```




```python

# Import the ColumnDataSource class from bokeh.plotting
from bokeh.plotting import ColumnDataSource

# Create a ColumnDataSource from df: source
source = ColumnDataSource(df)

# Add circle glyphs to the figure p
p.circle('Year', 'Time', source=source, color='color', size=8)

# Specify the name of the output file and show the result
output_file('sprint.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture8-4.png)

####
 Selection and non-selection glyphs




```python

# Create a figure with the "box_select" tool: p
p = figure(x_axis_label='Year', y_axis_label='Time', tools='box_select')

# Add circle glyphs to the figure p with the selected and non-selected properties
p.circle('Year', 'Time', source=source, selection_color = 'red', nonselection_alpha = 0.1)

# Specify the name of the output file and show the result
output_file('selection_glyph.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture9-3.png)

####
 Hover glyphs




```python

# import the HoverTool
from bokeh.models import HoverTool

# Add circle glyphs to figure p
p.circle(x, y, size=10,
         fill_color='grey', alpha=0.1, line_color=None,
         hover_fill_color='firebrick', hover_alpha=0.5,
         hover_line_color='white')

# Create a HoverTool: hover
hover = HoverTool(tooltips=None, mode='vline')

# Add the hover tool to the figure p
p.add_tools(hover)

# Specify the name of the output file and show the result
output_file('hover_glyph.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture10-3.png)

####
 Colormapping




```

#Import CategoricalColorMapper from bokeh.models
from bokeh.models import CategoricalColorMapper

# Convert df to a ColumnDataSource: source
source = ColumnDataSource(df)

# Make a CategoricalColorMapper object: color_mapper
color_mapper = CategoricalColorMapper(factors=['Europe', 'Asia', 'US'],
                                      palette=['red', 'green', 'blue'])

# Add a circle glyph to the figure p
p.circle('weight', 'mpg', source=source,
            color=dict(field='origin', transform=color_mapper),
            legend='origin')

# Specify the name of the output file and show the result
output_file('colormap.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture11-3.png)


 Layouts, Interactions, and Annotations
----------------------------------------


###
 Introduction to layouts


####
 Creating rows of plots




```python

# Import row from bokeh.layouts
from bokeh.layouts import row

# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to p1
p1.circle('fertility', 'female_literacy', source=source)

# Create the second figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')

# Add a circle glyph to p2
p2.circle('population', 'female_literacy', source=source)


# Put p1 and p2 into a horizontal row: layout
layout = row(p1, p2)

# Specify the name of the output_file and show the result
output_file('fert_row.html')
show(layout)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture12-2.png)

####
 Creating columns of plots




```python

# Import column from the bokeh.layouts module
from bokeh.layouts import column

# Create a blank figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add circle scatter to the figure p1
p1.circle('fertility', 'female_literacy', source=source)

# Create a new blank figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')

# Add circle scatter to the figure p2
p2.circle('population', 'female_literacy', source=source)

# Put plots p1 and p2 in a column: layout
layout = column(p1, p2)

# Specify the name of the output_file and show the result
output_file('fert_column.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture13-2.png)

####
 Nesting rows and columns of plots




```python

# Import column and row from bokeh.layouts
from bokeh.layouts import row, column

# Make a column layout that will be used as the second row: row2
row2 = column([mpg_hp, mpg_weight], sizing_mode='scale_width')

# Make a row layout that includes the above column layout: layout
layout = row([avg_mpg, row2], sizing_mode='scale_width')

# Specify the name of the output_file and show the result
output_file('layout_custom.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture14-1.png)

###
 Advanced layouts


####
 Creating gridded layouts




```python

# Import gridplot from bokeh.layouts
from bokeh.layouts import gridplot

# Create a list containing plots p1 and p2: row1
row1 = [p1, p2]

# Create a list containing plots p3 and p4: row2
row2 = [p3, p4]

# Create a gridplot using row1 and row2: layout
layout = gridplot([row1, row2])

# Specify the name of the output_file and show the result
output_file('grid.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture15-1.png)

####
 Starting tabbed layouts




```python

# Import Panel from bokeh.models.widgets
from bokeh.models.widgets import Panel

# Create tab1 from plot p1: tab1
tab1 = Panel(child=p1, title='Latin America')

# Create tab2 from plot p2: tab2
tab2 = Panel(child=p2, title='Africa')

# Create tab3 from plot p3: tab3
tab3 = Panel(child=p3, title='Asia')

# Create tab4 from plot p4: tab4
tab4 = Panel(child=p4, title='Europe')


```


####
 Displaying tabbed layouts




```python

# Import Tabs from bokeh.models.widgets
from bokeh.models.widgets import Tabs

# Create a Tabs layout: layout
layout = Tabs(tabs=[tab1, tab2, tab3, tab4])

# Specify the name of the output_file and show the result
output_file('tabs.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture16-1.png)

###
 Linking plots together


####
 Linked axes




```python

# Link the x_range of p2 to p1: p2.x_range
p2.x_range = p1.x_range

# Link the y_range of p2 to p1: p2.y_range
p2.y_range = p1.y_range

# Link the x_range of p3 to p1: p3.x_range
p3.x_range = p1.x_range

# Link the y_range of p4 to p1: p4.y_range
p4.y_range = p1.y_range

# Specify the name of the output_file and show the result
output_file('linked_range.html')
show(layout)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture-10.png)

####
 Linked brushing




```

    Country  Continent female literacy fertility   population
0      Chine       ASI            90.5     1.769  1324.655000
1       Inde       ASI            50.8     2.682  1139.964932
2        USA       NAM              99     2.077   304.060000
3  Indonésie       ASI            88.8     2.132   227.345082
4     Brésil       LAT            90.2     1.827   191.971506

```




```python

# Create ColumnDataSource: source
source = ColumnDataSource(data)

# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female literacy (% population)',
            tools='box_select,lasso_select')

# Add a circle glyph to p1
p1.circle('fertility', 'female literacy', source=source)

# Create the second figure: p2
p2 = figure(x_axis_label='fertility (children per woman)', y_axis_label='population (millions)',
            tools='box_select,lasso_select')

# Add a circle glyph to p2
p2.circle('fertility', 'population', source=source)


# Create row layout of figures p1 and p2: layout
layout = row(p1, p2)

# Specify the name of the output_file and show the result
output_file('linked_brush.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture1-9.png)

###
 Annotations and guides


####
 How to create legends




```python

# Add the first circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=latin_america, size=10, color='red', legend='Latin America')

# Add the second circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=africa, size=10, color='blue', legend='Africa')

# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture2-8.png)

####
 Positioning and styling legends




```python

# Assign the legend to the bottom left: p.legend.location
p.legend.location = 'bottom_left'

# Fill the legend background with the color 'lightgray': p.legend.background_fill_color
p.legend.background_fill_color = 'lightgray'

# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture3-7.png)

####
 Adding a hover tooltip




```python

# Import HoverTool from bokeh.models
from bokeh.models import HoverTool

# Create a HoverTool object: hover
hover = HoverTool(tooltips = [('Country','@Country')])

# Add the HoverTool object to figure p
p.add_tools(hover)

# Specify the name of the output_file and show the result
output_file('hover.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture4-7.png)


 Building interactive apps with Bokeh
--------------------------------------


###
 Introducing the Bokeh Server


####
 Understanding Bokeh apps



 The main purpose of the Bokeh server is to synchronize python objects with web applications in a browser, so that rich, interactive data applications can be connected to powerful PyData libraries such as NumPy, SciPy, Pandas, and scikit-learn.




 The Bokeh server can automatically keep in sync any property of any Bokeh object.





```

bokeh serve myapp.py

```


####

 Using the current document




```python

# Perform necessary imports
from bokeh.io import curdoc
from bokeh.plotting import figure

# Create a new plot: plot
plot = figure()

# Add a line to the plot
plot.line(x = [1,2,3,4,5], y = [2,5,4,6,7])

# Add the plot to the current document
curdoc().add_root(plot)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture-11.png)

####
 Add a single slider




```python

# Perform the necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider

# Create a slider: slider
slider = Slider(title='my slider', start=0, end=10, step=0.1, value=2)

# Create a widgetbox layout: layout
layout = widgetbox(slider)

# Add the layout to the current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture1-10.png)

####
 Multiple sliders in one document




```python

# Perform necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider

# Create first slider: slider1
slider1 = Slider(title = 'slider1', start = 0, end = 10, step = 0.1, value = 2)

# Create second slider: slider2
slider2 = Slider(title = 'slider2', start = 10, end = 100, step = 1, value = 20)

# Add slider1 and slider2 to a widgetbox
layout = widgetbox(slider1, slider2)

# Add the layout to the current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture2-9.png)

###
 Connecting sliders to plots


####
 Adding callbacks to sliders



 Callbacks are functions that a user can define, like
 `def callback(attr, old, new)`
 , that can be called automatically when some property of a Bokeh object (e.g., the
 `value`
 of a
 `Slider`
 ) changes.




 For the
 `value`
 property of
 `Slider`
 objects, callbacks are added by passing a callback function to the
 `on_change`
 method.





```

myslider.on_change('value', callback)

```


####
 How to combine Bokeh models into layouts




```python

# Create ColumnDataSource: source
source = ColumnDataSource(data = {'x': x, 'y': y})

# Add a line to the plot
plot.line('x', 'y', source=source)

# Create a column layout: layout
layout = column(widgetbox(slider), plot)

# Add the layout to the current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture3-8.png)

####
 Learn about widget callbacks




```python

# Define a callback function: callback
def callback(attr, old, new):

    # Read the current value of the slider: scale
    scale = slider.value

    # Compute the updated y using np.sin(scale/x): new_y
    new_y = np.sin(scale/x)

    # Update source with the new data values
    source.data = {'x': x, 'y': new_y}

# Attach the callback to the 'value' property of slider
slider.on_change('value', callback)

# Create layout and add to current document
layout = column(widgetbox(slider), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture4-8.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture5-6.png)

###
 Updating plots from dropdowns


####
 Updating data sources from dropdown callbacks




```python

# Perform necessary imports
from bokeh.models import ColumnDataSource, Select

# Create ColumnDataSource: source
source = ColumnDataSource(data={
    'x' : fertility,
    'y' : female_literacy
})

# Create a new plot: plot
plot = figure()

# Add circles to the plot
plot.circle('x', 'y', source=source)

# Define a callback function: update_plot
def update_plot(attr, old, new):
    # If the new Selection is 'female_literacy', update 'y' to female_literacy
    if new == 'female_literacy':
        source.data = {
            'x' : fertility,
            'y' : female_literacy
        }
    # Else, update 'y' to population
    else:
        source.data = {
            'x' : fertility,
            'y' : population
        }

# Create a dropdown Select widget: select
select = Select(title="distribution", options=['female_literacy', 'population'], value='female_literacy')

# Attach the update_plot callback to the 'value' property of select
select.on_change('value', update_plot)

# Create layout and add to current document
layout = row(select, plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture6-6.png)

####
 Synchronize two dropdowns




```python

# Create two dropdown Select widgets: select1, select2
select1 = Select(title='First', options=['A', 'B'], value='A')
select2 = Select(title='Second', options=['1', '2', '3'], value='1')

# Define a callback function: callback
def callback(attr, old, new):
    # If select1 is 'A'
    if select1.value == 'A':
        # Set select2 options to ['1', '2', '3']
        select2.options = ['1', '2', '3']

        # Set select2 value to '1'
        select2.value = '1'
    else:
        # Set select2 options to ['100', '200', '300']
        select2.options = ['100', '200', '300']

        # Set select2 value to '100'
        select2.value = '100'

# Attach the callback to the 'value' property of select1
select1.on_change('value', callback)

# Create layout and add to current document
layout = widgetbox(select1, select2)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture8-5.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture7-6.png)

###
 Buttons


####
 Button widgets




```python

# Create a Button with label 'Update Data'
button = Button(label='Update Data')

# Define an update callback with no arguments: update
def update():

    # Compute new y values: y
    y = np.sin(x) + np.random.random(N)

    # Update the ColumnDataSource data dictionary
    source.data = {'x':x,'y':y}

# Add the update callback to the button
button.on_click(update)

# Create layout and add to current document
layout = column(widgetbox(button), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture9-4.png)

####
 Button styles




```python

# Import CheckboxGroup, RadioGroup, Toggle from bokeh.models
from bokeh.models import CheckboxGroup, RadioGroup, Toggle

# Add a Toggle: toggle
toggle = Toggle(button_type = 'success', label = 'Toggle button')

# Add a CheckboxGroup: checkbox
checkbox = CheckboxGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add a RadioGroup: radio
radio = RadioGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add widgetbox(toggle, checkbox, radio) to the current document
curdoc().add_root(widgetbox(toggle, checkbox, radio))

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture10-4.png)


 Putting It All Together! A Case Study
---------------------------------------


###
 Time to put it all together!


####
 Introducing the project dataset




```

data.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 10111 entries, 1964 to 2006
Data columns (total 7 columns):
Country            10111 non-null object
fertility          10100 non-null float64
life               10111 non-null float64
population         10108 non-null float64
child_mortality    9210 non-null float64
gdp                9000 non-null float64
region             10111 non-null object
dtypes: float64(5), object(2)
memory usage: 631.9+ KB


data.head()
          Country  fertility    life  population  child_mortality     gdp  \
Year
1964  Afghanistan      7.671  33.639  10474903.0            339.7  1182.0
1965  Afghanistan      7.671  34.152  10697983.0            334.1  1182.0
1966  Afghanistan      7.671  34.662  10927724.0            328.7  1168.0
1967  Afghanistan      7.671  35.170  11163656.0            323.3  1173.0
1968  Afghanistan      7.671  35.674  11411022.0            318.1  1187.0

          region
Year
1964  South Asia
1965  South Asia
1966  South Asia
1967  South Asia
1968  South Asia

```


####
 Some exploratory plots of the data




```python

# Perform necessary imports
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource

# Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : data.loc[1970].fertility,
    'y'       : data.loc[1970].life,
    'country' : data.loc[1970].Country,
})

# Create the figure: p
p = figure(title='1970', x_axis_label='Fertility (children per woman)', y_axis_label='Life Expectancy (years)',
           plot_height=400, plot_width=700,
           tools=[HoverTool(tooltips='@country')])

# Add a circle glyph to the figure p
p.circle(x='x', y='y', source=source)

# Output the file and show the figure
output_file('gapminder.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture11-4.png)

###
 Starting the app



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture12-3.png)

####
 Beginning with just a plot




```python

# Import the necessary modules
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

# Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : data.loc[1970].fertility,
    'y'       : data.loc[1970].life,
    'country'      : data.loc[1970].Country,
    'pop'      : (data.loc[1970].population / 20000000) + 2,
    'region'      : data.loc[1970].region,
})

# Save the minimum and maximum values of the fertility column: xmin, xmax
xmin, xmax = min(data.fertility), max(data.fertility)

# Save the minimum and maximum values of the life expectancy column: ymin, ymax
ymin, ymax = min(data.life), max(data.life)

# Create the figure: plot
plot = figure(title='Gapminder Data for 1970', plot_height=400, plot_width=700,
              x_range=(xmin, xmax), y_range=(ymin, ymax))

# Add circle glyphs to the plot
plot.circle(x='x', y='y', fill_alpha=0.8, source=source)

# Set the x-axis label
plot.xaxis.axis_label ='Fertility (children per woman)'

# Set the y-axis label
plot.yaxis.axis_label = 'Life Expectancy (years)'

# Add the plot to the current document and add a title
curdoc().add_root(plot)
curdoc().title = 'Gapminder'



```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture13-3.png)

####
 Enhancing the plot with some shading




```python

# Make a list of the unique values from the region column: regions_list
regions_list = data.region.unique().tolist()

# Import CategoricalColorMapper from bokeh.models and the Spectral6 palette from bokeh.palettes
from bokeh.models import CategoricalColorMapper
from bokeh.palettes import Spectral6

# Make a color mapper: color_mapper
color_mapper = CategoricalColorMapper(factors=regions_list, palette=Spectral6)

# Add the color mapper to the circle glyph
plot.circle(x='x', y='y', fill_alpha=0.8, source=source,
            color=dict(field='region', transform=color_mapper), legend='region')

# Set the legend.location attribute of the plot to 'top_right'
plot.legend.location = 'top_right'

# Add the plot to the current document and add the title
curdoc().add_root(plot)
curdoc().title = 'Gapminder'

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture14-2.png)

####
 Adding a slider to vary the year




```python

# Import the necessary modules
from bokeh.layouts import row, widgetbox
from bokeh.models import Slider

# Define the callback function: update_plot
def update_plot(attr, old, new):
    # Set the yr name to slider.value and new_data to source.data
    yr = slider.value
    new_data = {
        'x'       : data.loc[yr].fertility,
        'y'       : data.loc[yr].life,
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    source.data = new_data


# Make a slider object: slider
slider = Slider(start = 1970, end = 2010, step = 1, value = 1970, title = 'Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Make a row layout of widgetbox(slider) and plot and add it to the current document
layout = row(widgetbox(slider), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture15-2.png)

###
 Customizing based on user input




```python

# Define the callback function: update_plot
def update_plot(attr, old, new):
    # Assign the value of the slider: yr
    yr = slider.value
    # Set new_data
    new_data = {
        'x'       : data.loc[yr].fertility,
        'y'       : data.loc[yr].life,
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    # Assign new_data to: source.data
    source.data = new_data

    # Add title to figure: plot.title.text
    plot.title.text = 'Gapminder data for %d' % yr

# Make a slider object: slider
slider = Slider(start = 1970, end = 2010, step = 1, value = 1970, title = 'Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Make a row layout of widgetbox(slider) and plot and add it to the current document
layout = row(widgetbox(slider), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture16-2.png)

###
 Adding more interactivity to the app


####
 Adding a hover tool



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture17-1.png)



```python

# Import HoverTool from bokeh.models
from bokeh.models import HoverTool

# Create a HoverTool: hover
hover = HoverTool(tooltips=[('Country', '@country')])

# Add the HoverTool to the plot
plot.add_tools(hover)

# Create layout: layout
layout = row(widgetbox(slider), plot)

# Add layout to current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture19-1.png)

####
 Adding dropdowns to the app



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture18-1.png)



```python

# Define the callback: update_plot
def update_plot(attr, old, new):
    # Read the current value off the slider and 2 dropdowns: yr, x, y
    yr = slider.value
    x = x_select.value
    y = y_select.value
    # Label axes of plot
    plot.xaxis.axis_label = x
    plot.yaxis.axis_label = y
    # Set new_data
    new_data = {
        'x'       : data.loc[yr][x],
        'y'       : data.loc[yr][y],
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    # Assign new_data to source.data
    source.data = new_data

    # Set the range of all axes
    plot.x_range.start = min(data[x])
    plot.x_range.end = max(data[x])
    plot.y_range.start = min(data[y])
    plot.y_range.end = max(data[y])

    # Add title to plot
    plot.title.text = 'Gapminder data for %d' % yr

# Create a dropdown slider widget: slider
slider = Slider(start=1970, end=2010, step=1, value=1970, title='Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Create a dropdown Select widget for the x data: x_select
x_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='fertility',
    title='x-axis data'
)

# Attach the update_plot callback to the 'value' property of x_select
x_select.on_change('value', update_plot)

# Create a dropdown Select widget for the y data: y_select
y_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='life',
    title='y-axis data'
)

# Attach the update_plot callback to the 'value' property of y_select
y_select.on_change('value', update_plot)

# Create layout and add to current document
layout = row(widgetbox(slider, x_select, y_select), plot)
curdoc().add_root(layout)

```




 Basic plotting with Bokeh
---------------------------


###
 Plotting with glyphs


####
 What are glyphs?



 In Bokeh, visual properties of shapes are called glyphs.


 Multiple glyphs can be drawn by setting glyph properties to ordered sequences of values.





####
 A simple scatter plot




```python

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Import output_file and show from bokeh.io
from bokeh.io import output_file, show

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility, female_literacy)

# Call the output_file() function and specify the name of the file
output_file('fert_lit.html')

# Display the plot
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture-9.png)

####
 A scatter plot with different shapes




```python

# Create the figure: p
p = figure(x_axis_label='fertility', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica)

# Add an x glyph to the figure p
p.x(fertility_africa, female_literacy_africa)

# Specify the name of the file
output_file('fert_lit_separate.html')

# Display the plot
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture1-8.png)

####
 Customizing your scatter plots




```python

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a blue circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica, color='blue', size=10, alpha=0.8)

# Add a red circle glyph to the figure p
p.circle(fertility_africa, female_literacy_africa, color='red', size=10, alpha=0.8)

# Specify the name of the file
output_file('fert_lit_separate_colors.html')

# Display the plot
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture2-7.png)


[CSS color names](http://www.colors.commutercreative.com/grid/)



###
 Additional glyphs


####
 Lines




```python

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create a figure with x_axis_type="datetime": p
p = figure(x_axis_type="datetime", x_axis_label='Date', y_axis_label='US Dollars')

# Plot date along the x axis and price along the y axis
p.line(date,price)

# Specify the name of the output file and show the result
output_file('line.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture3-6.png)

####
 Lines and markers




```python

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create a figure with x_axis_type='datetime': p
p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')

# Plot date along the x-axis and price along the y-axis
p.line(date, price)

# With date on the x-axis and price on the y-axis, add a white circle glyph of size 4
p.circle(date, price, fill_color='white', size=4)

# Specify the name of the output file and show the result
output_file('line.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture4-6.png)

####
 Patches




```python

# Create a list of az_lons, co_lons, nm_lons and ut_lons: x
x = [az_lons, co_lons, nm_lons, ut_lons]

# Create a list of az_lats, co_lats, nm_lats and ut_lats: y
y = [az_lats, co_lats, nm_lats, ut_lats]

# Add patches to figure p with line_color=white for x and y
p.patches(x, y, line_color = 'white')

# Specify the name of the output file and show the result
output_file('four_corners.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture5-5.png)

###
 Data formats


####
 Plotting data from NumPy arrays




```python

# Import numpy as np
import numpy as np

# Create array using np.linspace: x
x = np.linspace(0,5,100)

# Create array using np.cos: y
y = np.cos(x)

# Add circles at x and y
p.circle(x,y)

# Specify the name of the output file and show the result
output_file('numpy.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture6-5.png)

####
 Plotting data from Pandas DataFrames




```python

# Import pandas as pd
import pandas as pd

# Read in the CSV file: df
df = pd.read_csv('auto.csv')

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create the figure: p
p = figure(x_axis_label='HP', y_axis_label='MPG')

# Plot mpg vs hp by color
p.circle(df['hp'], df['mpg'], color=df['color'], size=10)

# Specify the name of the output file and show the result
output_file('auto-df.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture7-5.png)

####
 The Bokeh ColumnDataSource



 The
 `ColumnDataSource`
 is a table-like data object that maps string column names to sequences (columns) of data. It is the central and most common data structure in Bokeh.




 All columns in a
 `ColumnDataSource`
 must have the same length.press



####
 The Bokeh ColumnDataSource (continued)




```

df.head()
               Name Country   Medal  Time  Year        color
0        Usain Bolt     JAM    GOLD  9.63  2012    goldenrod
1       Yohan Blake     JAM  SILVER  9.75  2012       silver
2     Justin Gatlin     USA  BRONZE  9.79  2012  saddlebrown
3        Usain Bolt     JAM    GOLD  9.69  2008    goldenrod
4  Richard Thompson     TRI  SILVER  9.89  2008       silver

```




```python

# Import the ColumnDataSource class from bokeh.plotting
from bokeh.plotting import ColumnDataSource

# Create a ColumnDataSource from df: source
source = ColumnDataSource(df)

# Add circle glyphs to the figure p
p.circle('Year', 'Time', source=source, color='color', size=8)

# Specify the name of the output file and show the result
output_file('sprint.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture8-4.png)

####
 Selection and non-selection glyphs




```python

# Create a figure with the "box_select" tool: p
p = figure(x_axis_label='Year', y_axis_label='Time', tools='box_select')

# Add circle glyphs to the figure p with the selected and non-selected properties
p.circle('Year', 'Time', source=source, selection_color = 'red', nonselection_alpha = 0.1)

# Specify the name of the output file and show the result
output_file('selection_glyph.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture9-3.png)

####
 Hover glyphs




```python

# import the HoverTool
from bokeh.models import HoverTool

# Add circle glyphs to figure p
p.circle(x, y, size=10,
         fill_color='grey', alpha=0.1, line_color=None,
         hover_fill_color='firebrick', hover_alpha=0.5,
         hover_line_color='white')

# Create a HoverTool: hover
hover = HoverTool(tooltips=None, mode='vline')

# Add the hover tool to the figure p
p.add_tools(hover)

# Specify the name of the output file and show the result
output_file('hover_glyph.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture10-3.png)

####
 Colormapping




```

#Import CategoricalColorMapper from bokeh.models
from bokeh.models import CategoricalColorMapper

# Convert df to a ColumnDataSource: source
source = ColumnDataSource(df)

# Make a CategoricalColorMapper object: color_mapper
color_mapper = CategoricalColorMapper(factors=['Europe', 'Asia', 'US'],
                                      palette=['red', 'green', 'blue'])

# Add a circle glyph to the figure p
p.circle('weight', 'mpg', source=source,
            color=dict(field='origin', transform=color_mapper),
            legend='origin')

# Specify the name of the output file and show the result
output_file('colormap.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture11-3.png)


 Layouts, Interactions, and Annotations
----------------------------------------


###
 Introduction to layouts


####
 Creating rows of plots




```python

# Import row from bokeh.layouts
from bokeh.layouts import row

# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to p1
p1.circle('fertility', 'female_literacy', source=source)

# Create the second figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')

# Add a circle glyph to p2
p2.circle('population', 'female_literacy', source=source)


# Put p1 and p2 into a horizontal row: layout
layout = row(p1, p2)

# Specify the name of the output_file and show the result
output_file('fert_row.html')
show(layout)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture12-2.png)

####
 Creating columns of plots




```python

# Import column from the bokeh.layouts module
from bokeh.layouts import column

# Create a blank figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add circle scatter to the figure p1
p1.circle('fertility', 'female_literacy', source=source)

# Create a new blank figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')

# Add circle scatter to the figure p2
p2.circle('population', 'female_literacy', source=source)

# Put plots p1 and p2 in a column: layout
layout = column(p1, p2)

# Specify the name of the output_file and show the result
output_file('fert_column.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture13-2.png)

####
 Nesting rows and columns of plots




```python

# Import column and row from bokeh.layouts
from bokeh.layouts import row, column

# Make a column layout that will be used as the second row: row2
row2 = column([mpg_hp, mpg_weight], sizing_mode='scale_width')

# Make a row layout that includes the above column layout: layout
layout = row([avg_mpg, row2], sizing_mode='scale_width')

# Specify the name of the output_file and show the result
output_file('layout_custom.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture14-1.png)

###
 Advanced layouts


####
 Creating gridded layouts




```python

# Import gridplot from bokeh.layouts
from bokeh.layouts import gridplot

# Create a list containing plots p1 and p2: row1
row1 = [p1, p2]

# Create a list containing plots p3 and p4: row2
row2 = [p3, p4]

# Create a gridplot using row1 and row2: layout
layout = gridplot([row1, row2])

# Specify the name of the output_file and show the result
output_file('grid.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture15-1.png)

####
 Starting tabbed layouts




```python

# Import Panel from bokeh.models.widgets
from bokeh.models.widgets import Panel

# Create tab1 from plot p1: tab1
tab1 = Panel(child=p1, title='Latin America')

# Create tab2 from plot p2: tab2
tab2 = Panel(child=p2, title='Africa')

# Create tab3 from plot p3: tab3
tab3 = Panel(child=p3, title='Asia')

# Create tab4 from plot p4: tab4
tab4 = Panel(child=p4, title='Europe')


```


####
 Displaying tabbed layouts




```python

# Import Tabs from bokeh.models.widgets
from bokeh.models.widgets import Tabs

# Create a Tabs layout: layout
layout = Tabs(tabs=[tab1, tab2, tab3, tab4])

# Specify the name of the output_file and show the result
output_file('tabs.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture16-1.png)

###
 Linking plots together


####
 Linked axes




```python

# Link the x_range of p2 to p1: p2.x_range
p2.x_range = p1.x_range

# Link the y_range of p2 to p1: p2.y_range
p2.y_range = p1.y_range

# Link the x_range of p3 to p1: p3.x_range
p3.x_range = p1.x_range

# Link the y_range of p4 to p1: p4.y_range
p4.y_range = p1.y_range

# Specify the name of the output_file and show the result
output_file('linked_range.html')
show(layout)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture-10.png)

####
 Linked brushing




```

    Country  Continent female literacy fertility   population
0      Chine       ASI            90.5     1.769  1324.655000
1       Inde       ASI            50.8     2.682  1139.964932
2        USA       NAM              99     2.077   304.060000
3  Indonésie       ASI            88.8     2.132   227.345082
4     Brésil       LAT            90.2     1.827   191.971506

```




```python

# Create ColumnDataSource: source
source = ColumnDataSource(data)

# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female literacy (% population)',
            tools='box_select,lasso_select')

# Add a circle glyph to p1
p1.circle('fertility', 'female literacy', source=source)

# Create the second figure: p2
p2 = figure(x_axis_label='fertility (children per woman)', y_axis_label='population (millions)',
            tools='box_select,lasso_select')

# Add a circle glyph to p2
p2.circle('fertility', 'population', source=source)


# Create row layout of figures p1 and p2: layout
layout = row(p1, p2)

# Specify the name of the output_file and show the result
output_file('linked_brush.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture1-9.png)

###
 Annotations and guides


####
 How to create legends




```python

# Add the first circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=latin_america, size=10, color='red', legend='Latin America')

# Add the second circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=africa, size=10, color='blue', legend='Africa')

# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture2-8.png)

####
 Positioning and styling legends




```python

# Assign the legend to the bottom left: p.legend.location
p.legend.location = 'bottom_left'

# Fill the legend background with the color 'lightgray': p.legend.background_fill_color
p.legend.background_fill_color = 'lightgray'

# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture3-7.png)

####
 Adding a hover tooltip




```python

# Import HoverTool from bokeh.models
from bokeh.models import HoverTool

# Create a HoverTool object: hover
hover = HoverTool(tooltips = [('Country','@Country')])

# Add the HoverTool object to figure p
p.add_tools(hover)

# Specify the name of the output_file and show the result
output_file('hover.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture4-7.png)


 Building interactive apps with Bokeh
--------------------------------------


###
 Introducing the Bokeh Server


####
 Understanding Bokeh apps



 The main purpose of the Bokeh server is to synchronize python objects with web applications in a browser, so that rich, interactive data applications can be connected to powerful PyData libraries such as NumPy, SciPy, Pandas, and scikit-learn.




 The Bokeh server can automatically keep in sync any property of any Bokeh object.





```

bokeh serve myapp.py

```


####

 Using the current document




```python

# Perform necessary imports
from bokeh.io import curdoc
from bokeh.plotting import figure

# Create a new plot: plot
plot = figure()

# Add a line to the plot
plot.line(x = [1,2,3,4,5], y = [2,5,4,6,7])

# Add the plot to the current document
curdoc().add_root(plot)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture-11.png)

####
 Add a single slider




```python

# Perform the necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider

# Create a slider: slider
slider = Slider(title='my slider', start=0, end=10, step=0.1, value=2)

# Create a widgetbox layout: layout
layout = widgetbox(slider)

# Add the layout to the current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture1-10.png)

####
 Multiple sliders in one document




```python

# Perform necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider

# Create first slider: slider1
slider1 = Slider(title = 'slider1', start = 0, end = 10, step = 0.1, value = 2)

# Create second slider: slider2
slider2 = Slider(title = 'slider2', start = 10, end = 100, step = 1, value = 20)

# Add slider1 and slider2 to a widgetbox
layout = widgetbox(slider1, slider2)

# Add the layout to the current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture2-9.png)

###
 Connecting sliders to plots


####
 Adding callbacks to sliders



 Callbacks are functions that a user can define, like
 `def callback(attr, old, new)`
 , that can be called automatically when some property of a Bokeh object (e.g., the
 `value`
 of a
 `Slider`
 ) changes.




 For the
 `value`
 property of
 `Slider`
 objects, callbacks are added by passing a callback function to the
 `on_change`
 method.





```

myslider.on_change('value', callback)

```


####
 How to combine Bokeh models into layouts




```python

# Create ColumnDataSource: source
source = ColumnDataSource(data = {'x': x, 'y': y})

# Add a line to the plot
plot.line('x', 'y', source=source)

# Create a column layout: layout
layout = column(widgetbox(slider), plot)

# Add the layout to the current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture3-8.png)

####
 Learn about widget callbacks




```python

# Define a callback function: callback
def callback(attr, old, new):

    # Read the current value of the slider: scale
    scale = slider.value

    # Compute the updated y using np.sin(scale/x): new_y
    new_y = np.sin(scale/x)

    # Update source with the new data values
    source.data = {'x': x, 'y': new_y}

# Attach the callback to the 'value' property of slider
slider.on_change('value', callback)

# Create layout and add to current document
layout = column(widgetbox(slider), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture4-8.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture5-6.png)

###
 Updating plots from dropdowns


####
 Updating data sources from dropdown callbacks




```python

# Perform necessary imports
from bokeh.models import ColumnDataSource, Select

# Create ColumnDataSource: source
source = ColumnDataSource(data={
    'x' : fertility,
    'y' : female_literacy
})

# Create a new plot: plot
plot = figure()

# Add circles to the plot
plot.circle('x', 'y', source=source)

# Define a callback function: update_plot
def update_plot(attr, old, new):
    # If the new Selection is 'female_literacy', update 'y' to female_literacy
    if new == 'female_literacy':
        source.data = {
            'x' : fertility,
            'y' : female_literacy
        }
    # Else, update 'y' to population
    else:
        source.data = {
            'x' : fertility,
            'y' : population
        }

# Create a dropdown Select widget: select
select = Select(title="distribution", options=['female_literacy', 'population'], value='female_literacy')

# Attach the update_plot callback to the 'value' property of select
select.on_change('value', update_plot)

# Create layout and add to current document
layout = row(select, plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture6-6.png)

####
 Synchronize two dropdowns




```python

# Create two dropdown Select widgets: select1, select2
select1 = Select(title='First', options=['A', 'B'], value='A')
select2 = Select(title='Second', options=['1', '2', '3'], value='1')

# Define a callback function: callback
def callback(attr, old, new):
    # If select1 is 'A'
    if select1.value == 'A':
        # Set select2 options to ['1', '2', '3']
        select2.options = ['1', '2', '3']

        # Set select2 value to '1'
        select2.value = '1'
    else:
        # Set select2 options to ['100', '200', '300']
        select2.options = ['100', '200', '300']

        # Set select2 value to '100'
        select2.value = '100'

# Attach the callback to the 'value' property of select1
select1.on_change('value', callback)

# Create layout and add to current document
layout = widgetbox(select1, select2)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture8-5.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture7-6.png)

###
 Buttons


####
 Button widgets




```python

# Create a Button with label 'Update Data'
button = Button(label='Update Data')

# Define an update callback with no arguments: update
def update():

    # Compute new y values: y
    y = np.sin(x) + np.random.random(N)

    # Update the ColumnDataSource data dictionary
    source.data = {'x':x,'y':y}

# Add the update callback to the button
button.on_click(update)

# Create layout and add to current document
layout = column(widgetbox(button), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture9-4.png)

####
 Button styles




```python

# Import CheckboxGroup, RadioGroup, Toggle from bokeh.models
from bokeh.models import CheckboxGroup, RadioGroup, Toggle

# Add a Toggle: toggle
toggle = Toggle(button_type = 'success', label = 'Toggle button')

# Add a CheckboxGroup: checkbox
checkbox = CheckboxGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add a RadioGroup: radio
radio = RadioGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add widgetbox(toggle, checkbox, radio) to the current document
curdoc().add_root(widgetbox(toggle, checkbox, radio))

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture10-4.png)


 Putting It All Together! A Case Study
---------------------------------------


###
 Time to put it all together!


####
 Introducing the project dataset




```

data.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 10111 entries, 1964 to 2006
Data columns (total 7 columns):
Country            10111 non-null object
fertility          10100 non-null float64
life               10111 non-null float64
population         10108 non-null float64
child_mortality    9210 non-null float64
gdp                9000 non-null float64
region             10111 non-null object
dtypes: float64(5), object(2)
memory usage: 631.9+ KB


data.head()
          Country  fertility    life  population  child_mortality     gdp  \
Year
1964  Afghanistan      7.671  33.639  10474903.0            339.7  1182.0
1965  Afghanistan      7.671  34.152  10697983.0            334.1  1182.0
1966  Afghanistan      7.671  34.662  10927724.0            328.7  1168.0
1967  Afghanistan      7.671  35.170  11163656.0            323.3  1173.0
1968  Afghanistan      7.671  35.674  11411022.0            318.1  1187.0

          region
Year
1964  South Asia
1965  South Asia
1966  South Asia
1967  South Asia
1968  South Asia

```


####
 Some exploratory plots of the data




```python

# Perform necessary imports
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource

# Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : data.loc[1970].fertility,
    'y'       : data.loc[1970].life,
    'country' : data.loc[1970].Country,
})

# Create the figure: p
p = figure(title='1970', x_axis_label='Fertility (children per woman)', y_axis_label='Life Expectancy (years)',
           plot_height=400, plot_width=700,
           tools=[HoverTool(tooltips='@country')])

# Add a circle glyph to the figure p
p.circle(x='x', y='y', source=source)

# Output the file and show the figure
output_file('gapminder.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture11-4.png)

###
 Starting the app



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture12-3.png)

####
 Beginning with just a plot




```python

# Import the necessary modules
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

# Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : data.loc[1970].fertility,
    'y'       : data.loc[1970].life,
    'country'      : data.loc[1970].Country,
    'pop'      : (data.loc[1970].population / 20000000) + 2,
    'region'      : data.loc[1970].region,
})

# Save the minimum and maximum values of the fertility column: xmin, xmax
xmin, xmax = min(data.fertility), max(data.fertility)

# Save the minimum and maximum values of the life expectancy column: ymin, ymax
ymin, ymax = min(data.life), max(data.life)

# Create the figure: plot
plot = figure(title='Gapminder Data for 1970', plot_height=400, plot_width=700,
              x_range=(xmin, xmax), y_range=(ymin, ymax))

# Add circle glyphs to the plot
plot.circle(x='x', y='y', fill_alpha=0.8, source=source)

# Set the x-axis label
plot.xaxis.axis_label ='Fertility (children per woman)'

# Set the y-axis label
plot.yaxis.axis_label = 'Life Expectancy (years)'

# Add the plot to the current document and add a title
curdoc().add_root(plot)
curdoc().title = 'Gapminder'



```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture13-3.png)

####
 Enhancing the plot with some shading




```python

# Make a list of the unique values from the region column: regions_list
regions_list = data.region.unique().tolist()

# Import CategoricalColorMapper from bokeh.models and the Spectral6 palette from bokeh.palettes
from bokeh.models import CategoricalColorMapper
from bokeh.palettes import Spectral6

# Make a color mapper: color_mapper
color_mapper = CategoricalColorMapper(factors=regions_list, palette=Spectral6)

# Add the color mapper to the circle glyph
plot.circle(x='x', y='y', fill_alpha=0.8, source=source,
            color=dict(field='region', transform=color_mapper), legend='region')

# Set the legend.location attribute of the plot to 'top_right'
plot.legend.location = 'top_right'

# Add the plot to the current document and add the title
curdoc().add_root(plot)
curdoc().title = 'Gapminder'

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture14-2.png)

####
 Adding a slider to vary the year




```python

# Import the necessary modules
from bokeh.layouts import row, widgetbox
from bokeh.models import Slider

# Define the callback function: update_plot
def update_plot(attr, old, new):
    # Set the yr name to slider.value and new_data to source.data
    yr = slider.value
    new_data = {
        'x'       : data.loc[yr].fertility,
        'y'       : data.loc[yr].life,
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    source.data = new_data


# Make a slider object: slider
slider = Slider(start = 1970, end = 2010, step = 1, value = 1970, title = 'Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Make a row layout of widgetbox(slider) and plot and add it to the current document
layout = row(widgetbox(slider), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture15-2.png)

###
 Customizing based on user input




```python

# Define the callback function: update_plot
def update_plot(attr, old, new):
    # Assign the value of the slider: yr
    yr = slider.value
    # Set new_data
    new_data = {
        'x'       : data.loc[yr].fertility,
        'y'       : data.loc[yr].life,
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    # Assign new_data to: source.data
    source.data = new_data

    # Add title to figure: plot.title.text
    plot.title.text = 'Gapminder data for %d' % yr

# Make a slider object: slider
slider = Slider(start = 1970, end = 2010, step = 1, value = 1970, title = 'Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Make a row layout of widgetbox(slider) and plot and add it to the current document
layout = row(widgetbox(slider), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture16-2.png)

###
 Adding more interactivity to the app


####
 Adding a hover tool



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture17-1.png)



```python

# Import HoverTool from bokeh.models
from bokeh.models import HoverTool

# Create a HoverTool: hover
hover = HoverTool(tooltips=[('Country', '@country')])

# Add the HoverTool to the plot
plot.add_tools(hover)

# Create layout: layout
layout = row(widgetbox(slider), plot)

# Add layout to current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture19-1.png)

####
 Adding dropdowns to the app



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture18-1.png)



```python

# Define the callback: update_plot
def update_plot(attr, old, new):
    # Read the current value off the slider and 2 dropdowns: yr, x, y
    yr = slider.value
    x = x_select.value
    y = y_select.value
    # Label axes of plot
    plot.xaxis.axis_label = x
    plot.yaxis.axis_label = y
    # Set new_data
    new_data = {
        'x'       : data.loc[yr][x],
        'y'       : data.loc[yr][y],
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    # Assign new_data to source.data
    source.data = new_data

    # Set the range of all axes
    plot.x_range.start = min(data[x])
    plot.x_range.end = max(data[x])
    plot.y_range.start = min(data[y])
    plot.y_range.end = max(data[y])

    # Add title to plot
    plot.title.text = 'Gapminder data for %d' % yr

# Create a dropdown slider widget: slider
slider = Slider(start=1970, end=2010, step=1, value=1970, title='Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Create a dropdown Select widget for the x data: x_select
x_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='fertility',
    title='x-axis data'
)

# Attach the update_plot callback to the 'value' property of x_select
x_select.on_change('value', update_plot)

# Create a dropdown Select widget for the y data: y_select
y_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='life',
    title='y-axis data'
)

# Attach the update_plot callback to the 'value' property of y_select
y_select.on_change('value', update_plot)

# Create layout and add to current document
layout = row(widgetbox(slider, x_select, y_select), plot)
curdoc().add_root(layout)

```




 Basic plotting with Bokeh
---------------------------


###
 Plotting with glyphs


####
 What are glyphs?



 In Bokeh, visual properties of shapes are called glyphs.


 Multiple glyphs can be drawn by setting glyph properties to ordered sequences of values.





####
 A simple scatter plot




```python

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Import output_file and show from bokeh.io
from bokeh.io import output_file, show

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility, female_literacy)

# Call the output_file() function and specify the name of the file
output_file('fert_lit.html')

# Display the plot
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture-9.png)

####
 A scatter plot with different shapes




```python

# Create the figure: p
p = figure(x_axis_label='fertility', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica)

# Add an x glyph to the figure p
p.x(fertility_africa, female_literacy_africa)

# Specify the name of the file
output_file('fert_lit_separate.html')

# Display the plot
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture1-8.png)

####
 Customizing your scatter plots




```python

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a blue circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica, color='blue', size=10, alpha=0.8)

# Add a red circle glyph to the figure p
p.circle(fertility_africa, female_literacy_africa, color='red', size=10, alpha=0.8)

# Specify the name of the file
output_file('fert_lit_separate_colors.html')

# Display the plot
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture2-7.png)


[CSS color names](http://www.colors.commutercreative.com/grid/)



###
 Additional glyphs


####
 Lines




```python

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create a figure with x_axis_type="datetime": p
p = figure(x_axis_type="datetime", x_axis_label='Date', y_axis_label='US Dollars')

# Plot date along the x axis and price along the y axis
p.line(date,price)

# Specify the name of the output file and show the result
output_file('line.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture3-6.png)

####
 Lines and markers




```python

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create a figure with x_axis_type='datetime': p
p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')

# Plot date along the x-axis and price along the y-axis
p.line(date, price)

# With date on the x-axis and price on the y-axis, add a white circle glyph of size 4
p.circle(date, price, fill_color='white', size=4)

# Specify the name of the output file and show the result
output_file('line.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture4-6.png)

####
 Patches




```python

# Create a list of az_lons, co_lons, nm_lons and ut_lons: x
x = [az_lons, co_lons, nm_lons, ut_lons]

# Create a list of az_lats, co_lats, nm_lats and ut_lats: y
y = [az_lats, co_lats, nm_lats, ut_lats]

# Add patches to figure p with line_color=white for x and y
p.patches(x, y, line_color = 'white')

# Specify the name of the output file and show the result
output_file('four_corners.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture5-5.png)

###
 Data formats


####
 Plotting data from NumPy arrays




```python

# Import numpy as np
import numpy as np

# Create array using np.linspace: x
x = np.linspace(0,5,100)

# Create array using np.cos: y
y = np.cos(x)

# Add circles at x and y
p.circle(x,y)

# Specify the name of the output file and show the result
output_file('numpy.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture6-5.png)

####
 Plotting data from Pandas DataFrames




```python

# Import pandas as pd
import pandas as pd

# Read in the CSV file: df
df = pd.read_csv('auto.csv')

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create the figure: p
p = figure(x_axis_label='HP', y_axis_label='MPG')

# Plot mpg vs hp by color
p.circle(df['hp'], df['mpg'], color=df['color'], size=10)

# Specify the name of the output file and show the result
output_file('auto-df.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture7-5.png)

####
 The Bokeh ColumnDataSource



 The
 `ColumnDataSource`
 is a table-like data object that maps string column names to sequences (columns) of data. It is the central and most common data structure in Bokeh.




 All columns in a
 `ColumnDataSource`
 must have the same length.press



####
 The Bokeh ColumnDataSource (continued)




```

df.head()
               Name Country   Medal  Time  Year        color
0        Usain Bolt     JAM    GOLD  9.63  2012    goldenrod
1       Yohan Blake     JAM  SILVER  9.75  2012       silver
2     Justin Gatlin     USA  BRONZE  9.79  2012  saddlebrown
3        Usain Bolt     JAM    GOLD  9.69  2008    goldenrod
4  Richard Thompson     TRI  SILVER  9.89  2008       silver

```




```python

# Import the ColumnDataSource class from bokeh.plotting
from bokeh.plotting import ColumnDataSource

# Create a ColumnDataSource from df: source
source = ColumnDataSource(df)

# Add circle glyphs to the figure p
p.circle('Year', 'Time', source=source, color='color', size=8)

# Specify the name of the output file and show the result
output_file('sprint.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture8-4.png)

####
 Selection and non-selection glyphs




```python

# Create a figure with the "box_select" tool: p
p = figure(x_axis_label='Year', y_axis_label='Time', tools='box_select')

# Add circle glyphs to the figure p with the selected and non-selected properties
p.circle('Year', 'Time', source=source, selection_color = 'red', nonselection_alpha = 0.1)

# Specify the name of the output file and show the result
output_file('selection_glyph.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture9-3.png)

####
 Hover glyphs




```python

# import the HoverTool
from bokeh.models import HoverTool

# Add circle glyphs to figure p
p.circle(x, y, size=10,
         fill_color='grey', alpha=0.1, line_color=None,
         hover_fill_color='firebrick', hover_alpha=0.5,
         hover_line_color='white')

# Create a HoverTool: hover
hover = HoverTool(tooltips=None, mode='vline')

# Add the hover tool to the figure p
p.add_tools(hover)

# Specify the name of the output file and show the result
output_file('hover_glyph.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture10-3.png)

####
 Colormapping




```

#Import CategoricalColorMapper from bokeh.models
from bokeh.models import CategoricalColorMapper

# Convert df to a ColumnDataSource: source
source = ColumnDataSource(df)

# Make a CategoricalColorMapper object: color_mapper
color_mapper = CategoricalColorMapper(factors=['Europe', 'Asia', 'US'],
                                      palette=['red', 'green', 'blue'])

# Add a circle glyph to the figure p
p.circle('weight', 'mpg', source=source,
            color=dict(field='origin', transform=color_mapper),
            legend='origin')

# Specify the name of the output file and show the result
output_file('colormap.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture11-3.png)


 Layouts, Interactions, and Annotations
----------------------------------------


###
 Introduction to layouts


####
 Creating rows of plots




```python

# Import row from bokeh.layouts
from bokeh.layouts import row

# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to p1
p1.circle('fertility', 'female_literacy', source=source)

# Create the second figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')

# Add a circle glyph to p2
p2.circle('population', 'female_literacy', source=source)


# Put p1 and p2 into a horizontal row: layout
layout = row(p1, p2)

# Specify the name of the output_file and show the result
output_file('fert_row.html')
show(layout)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture12-2.png)

####
 Creating columns of plots




```python

# Import column from the bokeh.layouts module
from bokeh.layouts import column

# Create a blank figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add circle scatter to the figure p1
p1.circle('fertility', 'female_literacy', source=source)

# Create a new blank figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')

# Add circle scatter to the figure p2
p2.circle('population', 'female_literacy', source=source)

# Put plots p1 and p2 in a column: layout
layout = column(p1, p2)

# Specify the name of the output_file and show the result
output_file('fert_column.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture13-2.png)

####
 Nesting rows and columns of plots




```python

# Import column and row from bokeh.layouts
from bokeh.layouts import row, column

# Make a column layout that will be used as the second row: row2
row2 = column([mpg_hp, mpg_weight], sizing_mode='scale_width')

# Make a row layout that includes the above column layout: layout
layout = row([avg_mpg, row2], sizing_mode='scale_width')

# Specify the name of the output_file and show the result
output_file('layout_custom.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture14-1.png)

###
 Advanced layouts


####
 Creating gridded layouts




```python

# Import gridplot from bokeh.layouts
from bokeh.layouts import gridplot

# Create a list containing plots p1 and p2: row1
row1 = [p1, p2]

# Create a list containing plots p3 and p4: row2
row2 = [p3, p4]

# Create a gridplot using row1 and row2: layout
layout = gridplot([row1, row2])

# Specify the name of the output_file and show the result
output_file('grid.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture15-1.png)

####
 Starting tabbed layouts




```python

# Import Panel from bokeh.models.widgets
from bokeh.models.widgets import Panel

# Create tab1 from plot p1: tab1
tab1 = Panel(child=p1, title='Latin America')

# Create tab2 from plot p2: tab2
tab2 = Panel(child=p2, title='Africa')

# Create tab3 from plot p3: tab3
tab3 = Panel(child=p3, title='Asia')

# Create tab4 from plot p4: tab4
tab4 = Panel(child=p4, title='Europe')


```


####
 Displaying tabbed layouts




```python

# Import Tabs from bokeh.models.widgets
from bokeh.models.widgets import Tabs

# Create a Tabs layout: layout
layout = Tabs(tabs=[tab1, tab2, tab3, tab4])

# Specify the name of the output_file and show the result
output_file('tabs.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture16-1.png)

###
 Linking plots together


####
 Linked axes




```python

# Link the x_range of p2 to p1: p2.x_range
p2.x_range = p1.x_range

# Link the y_range of p2 to p1: p2.y_range
p2.y_range = p1.y_range

# Link the x_range of p3 to p1: p3.x_range
p3.x_range = p1.x_range

# Link the y_range of p4 to p1: p4.y_range
p4.y_range = p1.y_range

# Specify the name of the output_file and show the result
output_file('linked_range.html')
show(layout)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture-10.png)

####
 Linked brushing




```

    Country  Continent female literacy fertility   population
0      Chine       ASI            90.5     1.769  1324.655000
1       Inde       ASI            50.8     2.682  1139.964932
2        USA       NAM              99     2.077   304.060000
3  Indonésie       ASI            88.8     2.132   227.345082
4     Brésil       LAT            90.2     1.827   191.971506

```




```python

# Create ColumnDataSource: source
source = ColumnDataSource(data)

# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female literacy (% population)',
            tools='box_select,lasso_select')

# Add a circle glyph to p1
p1.circle('fertility', 'female literacy', source=source)

# Create the second figure: p2
p2 = figure(x_axis_label='fertility (children per woman)', y_axis_label='population (millions)',
            tools='box_select,lasso_select')

# Add a circle glyph to p2
p2.circle('fertility', 'population', source=source)


# Create row layout of figures p1 and p2: layout
layout = row(p1, p2)

# Specify the name of the output_file and show the result
output_file('linked_brush.html')
show(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture1-9.png)

###
 Annotations and guides


####
 How to create legends




```python

# Add the first circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=latin_america, size=10, color='red', legend='Latin America')

# Add the second circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=africa, size=10, color='blue', legend='Africa')

# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture2-8.png)

####
 Positioning and styling legends




```python

# Assign the legend to the bottom left: p.legend.location
p.legend.location = 'bottom_left'

# Fill the legend background with the color 'lightgray': p.legend.background_fill_color
p.legend.background_fill_color = 'lightgray'

# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture3-7.png)

####
 Adding a hover tooltip




```python

# Import HoverTool from bokeh.models
from bokeh.models import HoverTool

# Create a HoverTool object: hover
hover = HoverTool(tooltips = [('Country','@Country')])

# Add the HoverTool object to figure p
p.add_tools(hover)

# Specify the name of the output_file and show the result
output_file('hover.html')
show(p)


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture4-7.png)


 Building interactive apps with Bokeh
--------------------------------------


###
 Introducing the Bokeh Server


####
 Understanding Bokeh apps



 The main purpose of the Bokeh server is to synchronize python objects with web applications in a browser, so that rich, interactive data applications can be connected to powerful PyData libraries such as NumPy, SciPy, Pandas, and scikit-learn.




 The Bokeh server can automatically keep in sync any property of any Bokeh object.





```

bokeh serve myapp.py

```


####

 Using the current document




```python

# Perform necessary imports
from bokeh.io import curdoc
from bokeh.plotting import figure

# Create a new plot: plot
plot = figure()

# Add a line to the plot
plot.line(x = [1,2,3,4,5], y = [2,5,4,6,7])

# Add the plot to the current document
curdoc().add_root(plot)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture-11.png)

####
 Add a single slider




```python

# Perform the necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider

# Create a slider: slider
slider = Slider(title='my slider', start=0, end=10, step=0.1, value=2)

# Create a widgetbox layout: layout
layout = widgetbox(slider)

# Add the layout to the current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture1-10.png)

####
 Multiple sliders in one document




```python

# Perform necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider

# Create first slider: slider1
slider1 = Slider(title = 'slider1', start = 0, end = 10, step = 0.1, value = 2)

# Create second slider: slider2
slider2 = Slider(title = 'slider2', start = 10, end = 100, step = 1, value = 20)

# Add slider1 and slider2 to a widgetbox
layout = widgetbox(slider1, slider2)

# Add the layout to the current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture2-9.png)

###
 Connecting sliders to plots


####
 Adding callbacks to sliders



 Callbacks are functions that a user can define, like
 `def callback(attr, old, new)`
 , that can be called automatically when some property of a Bokeh object (e.g., the
 `value`
 of a
 `Slider`
 ) changes.




 For the
 `value`
 property of
 `Slider`
 objects, callbacks are added by passing a callback function to the
 `on_change`
 method.





```

myslider.on_change('value', callback)

```


####
 How to combine Bokeh models into layouts




```python

# Create ColumnDataSource: source
source = ColumnDataSource(data = {'x': x, 'y': y})

# Add a line to the plot
plot.line('x', 'y', source=source)

# Create a column layout: layout
layout = column(widgetbox(slider), plot)

# Add the layout to the current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture3-8.png)

####
 Learn about widget callbacks




```python

# Define a callback function: callback
def callback(attr, old, new):

    # Read the current value of the slider: scale
    scale = slider.value

    # Compute the updated y using np.sin(scale/x): new_y
    new_y = np.sin(scale/x)

    # Update source with the new data values
    source.data = {'x': x, 'y': new_y}

# Attach the callback to the 'value' property of slider
slider.on_change('value', callback)

# Create layout and add to current document
layout = column(widgetbox(slider), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture4-8.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture5-6.png)

###
 Updating plots from dropdowns


####
 Updating data sources from dropdown callbacks




```python

# Perform necessary imports
from bokeh.models import ColumnDataSource, Select

# Create ColumnDataSource: source
source = ColumnDataSource(data={
    'x' : fertility,
    'y' : female_literacy
})

# Create a new plot: plot
plot = figure()

# Add circles to the plot
plot.circle('x', 'y', source=source)

# Define a callback function: update_plot
def update_plot(attr, old, new):
    # If the new Selection is 'female_literacy', update 'y' to female_literacy
    if new == 'female_literacy':
        source.data = {
            'x' : fertility,
            'y' : female_literacy
        }
    # Else, update 'y' to population
    else:
        source.data = {
            'x' : fertility,
            'y' : population
        }

# Create a dropdown Select widget: select
select = Select(title="distribution", options=['female_literacy', 'population'], value='female_literacy')

# Attach the update_plot callback to the 'value' property of select
select.on_change('value', update_plot)

# Create layout and add to current document
layout = row(select, plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture6-6.png)

####
 Synchronize two dropdowns




```python

# Create two dropdown Select widgets: select1, select2
select1 = Select(title='First', options=['A', 'B'], value='A')
select2 = Select(title='Second', options=['1', '2', '3'], value='1')

# Define a callback function: callback
def callback(attr, old, new):
    # If select1 is 'A'
    if select1.value == 'A':
        # Set select2 options to ['1', '2', '3']
        select2.options = ['1', '2', '3']

        # Set select2 value to '1'
        select2.value = '1'
    else:
        # Set select2 options to ['100', '200', '300']
        select2.options = ['100', '200', '300']

        # Set select2 value to '100'
        select2.value = '100'

# Attach the callback to the 'value' property of select1
select1.on_change('value', callback)

# Create layout and add to current document
layout = widgetbox(select1, select2)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture8-5.png)


![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture7-6.png)

###
 Buttons


####
 Button widgets




```python

# Create a Button with label 'Update Data'
button = Button(label='Update Data')

# Define an update callback with no arguments: update
def update():

    # Compute new y values: y
    y = np.sin(x) + np.random.random(N)

    # Update the ColumnDataSource data dictionary
    source.data = {'x':x,'y':y}

# Add the update callback to the button
button.on_click(update)

# Create layout and add to current document
layout = column(widgetbox(button), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture9-4.png)

####
 Button styles




```python

# Import CheckboxGroup, RadioGroup, Toggle from bokeh.models
from bokeh.models import CheckboxGroup, RadioGroup, Toggle

# Add a Toggle: toggle
toggle = Toggle(button_type = 'success', label = 'Toggle button')

# Add a CheckboxGroup: checkbox
checkbox = CheckboxGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add a RadioGroup: radio
radio = RadioGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add widgetbox(toggle, checkbox, radio) to the current document
curdoc().add_root(widgetbox(toggle, checkbox, radio))

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture10-4.png)


 Putting It All Together! A Case Study
---------------------------------------


###
 Time to put it all together!


####
 Introducing the project dataset




```

data.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 10111 entries, 1964 to 2006
Data columns (total 7 columns):
Country            10111 non-null object
fertility          10100 non-null float64
life               10111 non-null float64
population         10108 non-null float64
child_mortality    9210 non-null float64
gdp                9000 non-null float64
region             10111 non-null object
dtypes: float64(5), object(2)
memory usage: 631.9+ KB


data.head()
          Country  fertility    life  population  child_mortality     gdp  \
Year
1964  Afghanistan      7.671  33.639  10474903.0            339.7  1182.0
1965  Afghanistan      7.671  34.152  10697983.0            334.1  1182.0
1966  Afghanistan      7.671  34.662  10927724.0            328.7  1168.0
1967  Afghanistan      7.671  35.170  11163656.0            323.3  1173.0
1968  Afghanistan      7.671  35.674  11411022.0            318.1  1187.0

          region
Year
1964  South Asia
1965  South Asia
1966  South Asia
1967  South Asia
1968  South Asia

```


####
 Some exploratory plots of the data




```python

# Perform necessary imports
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource

# Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : data.loc[1970].fertility,
    'y'       : data.loc[1970].life,
    'country' : data.loc[1970].Country,
})

# Create the figure: p
p = figure(title='1970', x_axis_label='Fertility (children per woman)', y_axis_label='Life Expectancy (years)',
           plot_height=400, plot_width=700,
           tools=[HoverTool(tooltips='@country')])

# Add a circle glyph to the figure p
p.circle(x='x', y='y', source=source)

# Output the file and show the figure
output_file('gapminder.html')
show(p)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture11-4.png)

###
 Starting the app



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture12-3.png)

####
 Beginning with just a plot




```python

# Import the necessary modules
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

# Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : data.loc[1970].fertility,
    'y'       : data.loc[1970].life,
    'country'      : data.loc[1970].Country,
    'pop'      : (data.loc[1970].population / 20000000) + 2,
    'region'      : data.loc[1970].region,
})

# Save the minimum and maximum values of the fertility column: xmin, xmax
xmin, xmax = min(data.fertility), max(data.fertility)

# Save the minimum and maximum values of the life expectancy column: ymin, ymax
ymin, ymax = min(data.life), max(data.life)

# Create the figure: plot
plot = figure(title='Gapminder Data for 1970', plot_height=400, plot_width=700,
              x_range=(xmin, xmax), y_range=(ymin, ymax))

# Add circle glyphs to the plot
plot.circle(x='x', y='y', fill_alpha=0.8, source=source)

# Set the x-axis label
plot.xaxis.axis_label ='Fertility (children per woman)'

# Set the y-axis label
plot.yaxis.axis_label = 'Life Expectancy (years)'

# Add the plot to the current document and add a title
curdoc().add_root(plot)
curdoc().title = 'Gapminder'



```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture13-3.png)

####
 Enhancing the plot with some shading




```python

# Make a list of the unique values from the region column: regions_list
regions_list = data.region.unique().tolist()

# Import CategoricalColorMapper from bokeh.models and the Spectral6 palette from bokeh.palettes
from bokeh.models import CategoricalColorMapper
from bokeh.palettes import Spectral6

# Make a color mapper: color_mapper
color_mapper = CategoricalColorMapper(factors=regions_list, palette=Spectral6)

# Add the color mapper to the circle glyph
plot.circle(x='x', y='y', fill_alpha=0.8, source=source,
            color=dict(field='region', transform=color_mapper), legend='region')

# Set the legend.location attribute of the plot to 'top_right'
plot.legend.location = 'top_right'

# Add the plot to the current document and add the title
curdoc().add_root(plot)
curdoc().title = 'Gapminder'

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture14-2.png)

####
 Adding a slider to vary the year




```python

# Import the necessary modules
from bokeh.layouts import row, widgetbox
from bokeh.models import Slider

# Define the callback function: update_plot
def update_plot(attr, old, new):
    # Set the yr name to slider.value and new_data to source.data
    yr = slider.value
    new_data = {
        'x'       : data.loc[yr].fertility,
        'y'       : data.loc[yr].life,
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    source.data = new_data


# Make a slider object: slider
slider = Slider(start = 1970, end = 2010, step = 1, value = 1970, title = 'Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Make a row layout of widgetbox(slider) and plot and add it to the current document
layout = row(widgetbox(slider), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture15-2.png)

###
 Customizing based on user input




```python

# Define the callback function: update_plot
def update_plot(attr, old, new):
    # Assign the value of the slider: yr
    yr = slider.value
    # Set new_data
    new_data = {
        'x'       : data.loc[yr].fertility,
        'y'       : data.loc[yr].life,
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    # Assign new_data to: source.data
    source.data = new_data

    # Add title to figure: plot.title.text
    plot.title.text = 'Gapminder data for %d' % yr

# Make a slider object: slider
slider = Slider(start = 1970, end = 2010, step = 1, value = 1970, title = 'Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Make a row layout of widgetbox(slider) and plot and add it to the current document
layout = row(widgetbox(slider), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture16-2.png)

###
 Adding more interactivity to the app


####
 Adding a hover tool



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture17-1.png)



```python

# Import HoverTool from bokeh.models
from bokeh.models import HoverTool

# Create a HoverTool: hover
hover = HoverTool(tooltips=[('Country', '@country')])

# Add the HoverTool to the plot
plot.add_tools(hover)

# Create layout: layout
layout = row(widgetbox(slider), plot)

# Add layout to current document
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture19-1.png)

####
 Adding dropdowns to the app



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture18-1.png)



```python

# Define the callback: update_plot
def update_plot(attr, old, new):
    # Read the current value off the slider and 2 dropdowns: yr, x, y
    yr = slider.value
    x = x_select.value
    y = y_select.value
    # Label axes of plot
    plot.xaxis.axis_label = x
    plot.yaxis.axis_label = y
    # Set new_data
    new_data = {
        'x'       : data.loc[yr][x],
        'y'       : data.loc[yr][y],
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    # Assign new_data to source.data
    source.data = new_data

    # Set the range of all axes
    plot.x_range.start = min(data[x])
    plot.x_range.end = max(data[x])
    plot.y_range.start = min(data[y])
    plot.y_range.end = max(data[y])

    # Add title to plot
    plot.title.text = 'Gapminder data for %d' % yr

# Create a dropdown slider widget: slider
slider = Slider(start=1970, end=2010, step=1, value=1970, title='Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Create a dropdown Select widget for the x data: x_select
x_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='fertility',
    title='x-axis data'
)

# Attach the update_plot callback to the 'value' property of x_select
x_select.on_change('value', update_plot)

# Create a dropdown Select widget for the y data: y_select
y_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='life',
    title='y-axis data'
)

# Attach the update_plot callback to the 'value' property of y_select
y_select.on_change('value', update_plot)

# Create layout and add to current document
layout = row(widgetbox(slider, x_select, y_select), plot)
curdoc().add_root(layout)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/interactive-data-visualization-with-bokeh/capture20-1.png)

