---
title: Intro to Financial Concepts using Python
date: 2021-12-07 11:22:08 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Intro to Financial Concepts using Python
============================================







 This is a memo. This course does not have a track yet.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/intro-to-financial-concepts-using-python)**
 .





---



![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture-8.png?w=668)


# **1. The Time Value of Money**
-------------------------------




### **1.1 Fundamental Financial Concepts**


![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture1-10.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture2-14.png?w=614)


####
**Growth and Rate of Return**



**Growth**
 and
 **Rate of Return**
 are two concepts that are ubiquitous throughout the financial world.




 Calculate the future value (cumulative return) of a $100 investment which grows at a rate of 6% per year for 30 years in a row and assign it to
 `future_value`
 .





```python

# Calculate the future value of the investment and print it out
future_value = 100 * (1 + 0.06) ** 30
print("Future Value of Investment: " + str(round(future_value, 2)))
#  Future Value of Investment: 574.35

```


####
**Compound Interest**



 As you saw in the previous exercise, both time and the rate of return are very important variables when forecasting the future value of an investment.




 Another important variable is the number of compounding periods, which can greatly affect compounded returns over time.





```python

# Predefined variables
initial_investment = 100
growth_periods = 30
growth_rate = 0.06

# Calculate the value for the investment compounded once per year
compound_periods_1 = 1
investment_1 = initial_investment*(1 + growth_rate / compound_periods_1)**(compound_periods_1*growth_periods)
print("Investment 1: " + str(round(investment_1, 2)))

# Calculate the value for the investment compounded quarterly
compound_periods_2 = 4
investment_2 = initial_investment*(1 + growth_rate / compound_periods_2)**(compound_periods_2*growth_periods)
print("Investment 2: " + str(round(investment_2, 2)))

# Calculate the value for the investment compounded monthly
compound_periods_3 = 12
investment_3 = initial_investment*(1 + growth_rate / compound_periods_3)**(compound_periods_3*growth_periods)
print("Investment 3: " + str(round(investment_3, 2)))


# Investment 1: 574.35
# Investment 2: 596.93
# Investment 3: 602.26

```


####
**Discount Factors and Depreciation**



 Unfortunately, not everything grows in value over time.




 In fact, many assets
 **depreciate**
 , or lose value over time. To simulate this, you can simply assume a negative expected rate of return.




 Calculate the future value of a $100 investment that depreciates in value by 5% per year for 10 years and assign it to
 `future_value`
 .





```python

# Calculate the future value
initial_investment = 100
growth_rate = -0.05
growth_periods = 10
future_value = initial_investment*(1 + growth_rate)**(growth_periods)
print("Future value: " + str(round(future_value, 2)))

# Calculate the discount factor
discount_factor = 1/((1 + growth_rate)**(growth_periods))
print("Discount factor: " + str(round(discount_factor, 2)))

# Derive the initial value of the investment
initial_investment_again = future_value * discount_factor
print("Initial value: " + str(round(initial_investment_again, 2)))


# Future value: 59.87
# Discount factor: 1.67
# Initial value: 100.0

```




---


## **1.2 Present and Future Value**


####
**Present Value**



 Luckily for you, there is a module called
 `numpy`
 which contains many functions which will make your life much easier when working with financial values.




 The
 `.pv(rate, nper, pmt, fv)`
 function, for example, allows you to calculate the present value of an investment as before with a few simple parameters:



* **rate:**
 The rate of return of the investment
* **nper:**
 The lifespan of the investment
* **pmt:**
 The (fixed) payment at the beginning or end of each period (which is 0 in our example)
* **fv:**
 The future value of the investment



 You can use this formula in many ways. For example, you can calculate the present value of future investments in today’s dollars.




 Compute the present value of an investment which will yield $10,000 15 years from now at an inflation rate of 3% per year and assign it to
 `investment_1`
 .





```python

# Import numpy as np
import numpy as np

# Calculate investment_1
investment_1 = np.pv(rate=0.03, nper=15, pmt=0, fv=10000)

# Note that the present value returned is negative, so we multiply the result by -1
print("Investment 1 is worth " + str(round(-investment_1, 2)) + " in today's dollars")

# Calculate investment_2
investment_2 = np.pv(rate=0.05, nper=10, pmt=0, fv=10000)
print("Investment 2 is worth " + str(round(-investment_2, 2)) + " in today's dollars")

# Investment 1 is worth 6418.62 in today's dollars
# Investment 2 is worth 6139.13 in today's dollars

```



 Notice how a higher inflation rate leads to a lower present value.



####
**Future Value**



 The
 `numpy`
 module also contains a similar function,
 `.fv(rate, nper, pmt, pv)`
 , which allows you to calculate the future value of an investment as before with a few simple parameters:



* **rate:**
 The rate of return of the investment
* **nper:**
 The lifespan of the investment
* **pmt:**
 The (fixed) payment at the beginning or end of each period (which is 0 in our example)
* **pv:**
 The present value of the investment



 It is important to note that in this function call, you must pass a
 **negative**
 value into the
 `pv`
 parameter if it represents a
 **negative cash flow**
 (cash going out). In other words, if you were to compute the future value of an investment, requiring an up-front cash payment, you would need to pass a negative value to the
 `pv`
 parameter in the
 `.fv()`
 function.





```

import numpy as np

# Calculate investment_1
investment_1 = np.fv(rate=0.05, nper=15, pmt=0, pv=-10000)
print("Investment 1 will yield a total of $" + str(round(investment_1, 2)) + " in 15 years")

# Calculate investment_2
investment_2 = np.fv(rate=0.08, nper=15, pmt=0, pv=-10000)
print("Investment 2 will yield a total of $" + str(round(investment_2, 2)) + " in 15 years")

# Investment 1 will yield a total of $20789.28 in 15 years
# Investment 2 will yield a total of $31721.69 in 15 years

```



 Note how the growth rate dramatically affects the future value.



####
**Adjusting Future Values for Inflation**



 You can now put together what you learned in the previous exercises by following a simple methodology:



* First, forecast the future value of an investment given a rate of return
* Second, discount the future value of the investment by a projected inflation rate



 The methodology above will use both the
 `.fv()`
 and
 `.pv()`
 functions to arrive at the projected value of a given investment in today’s dollars, adjusted for inflation.





```

import numpy as np

# Calculate investment_1
investment_1 = np.fv(rate=0.08, nper=10, pmt=0, pv=-10000)
print("Investment 1 will yield a total of $" + str(round(investment_1, 2)) + " in 10 years")

# Calculate investment_2
investment_1_discounted = np.pv(rate=0.03, nper=10, pmt=0, fv=investment_1)
print("After adjusting for inflation, investment 1 is worth $" + str(round(-investment_1_discounted, 2)) + " in today's dollars")

# Investment 1 will yield a total of $21589.25 in 10 years
# After adjusting for inflation, investment 1 is worth $16064.43 in today's dollars

```



 You now know how to project the value of investments and adjust for inflation.



## **1.3 Net Present Value and Cash Flows**


####
**Discounting Cash Flows**



 You can use numpy’s net present value function
 `numpy.npv(rate, values)`
 to calculate the net present value of a series of cash flows.





```

import numpy as np

# Predefined array of cash flows
cash_flows = np.array([100, 100, 100, 100, 100])

# Calculate investment_1
investment_1 = np.npv(rate=0.03, values=cash_flows)
print("Investment 1's net present value is $" + str(round(investment_1, 2)) + " in today's dollars")

# Calculate investment_2
investment_2 = np.npv(rate=0.05, values=cash_flows)
print("Investment 2's net present value is $" + str(round(investment_2, 2)) + " in today's dollars")

# Calculate investment_3
investment_3 = np.npv(rate=0.07, values=cash_flows)
print("Investment 3's net present value is $" + str(round(investment_3, 2)) + " in today's dollars")

# Investment 1's net present value is $471.71 in today's dollars
# Investment 2's net present value is $454.6 in today's dollars
# Investment 3's net present value is $438.72 in today's dollars

```



 Notice how the higher discount rate leads to a lower NPV.



####
**Initial Project Costs**



 The
 `numpy.npv(rate, values)`
 function is very powerful because it allows you to pass in both positive and negative values.




 For this exercise, you will calculate the net present value of two potential projects with different cash flows:






|
 Year
  |
 Project 1
  |
 Project 2
  |
| --- | --- | --- |
|
 1
  |
 -$250 (initial investment)
  |
 -$250 (initial investment)
  |
|
 2
  |
 $100 cash flow
  |
 $300 cash flow
  |
|
 3
  |
 $200 cash flow
  |
 -$250 (net investment)
  |
|
 4
  |
 $300 cash flow
  |
 $300 cash flow
  |
|
 5
  |
 $400 cash flow
  |
 $300 cash flow
  |




 In this example, project 1 only requires an initial investment of $250, generating a slowly increasing series of cash flows over the next 4 years.




 Project 2, on the other hand, requires an initial investment of $250 and an additional investment of $250 in year 3. However, project 2 continues to generate larger cash flows.




 Assuming both projects don’t generate any more cash flows after the fifth year, which project would you decide to undertake? The best way to decide is by comparing the NPV of both projects.





```

import numpy as np

# Create an array of cash flows for project 1
cash_flows_1 = np.array([-250, 100, 200, 300, 400])

# Create an array of cash flows for project 2
cash_flows_2 = np.array([-250, 300, -250, 300, 300])

# Calculate the net present value of project 1
investment_1 = np.npv(rate=0.03, values=cash_flows_1)
print("The net present value of Investment 1 is worth $" + str(round(investment_1, 2)) + " in today's dollars")

# Calculate the net present value of project 2
investment_2 = np.npv(rate=0.03, values=cash_flows_2)
print("The net present value of Investment 2 is worth $" + str(round(investment_2, 2)) + " in today's dollars")

# The net present value of Investment 1 is worth $665.54 in today's dollars
# The net present value of Investment 2 is worth $346.7 in today's dollars

```


####
**Diminishing Cash Flows**



 Remember how compounded returns grow rapidly over time? Well, it works in the reverse, too. Compounded discount factors over time will quickly shrink a number towards zero.




 For example, $100 at a 3% annual discount for 1 year is still worth roughly $97.08:




![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture3-14.png?w=808)


 This means that the longer in the future your cash flows will be received (or paid), the close to 0 that number will be.





```

import numpy as np

# Calculate investment_1
investment_1 = np.pv(rate=0.03, nper=30, pmt=0, fv=100)
print("Investment 1 is worth $" + str(round(-investment_1, 2)) + " in today's dollars")

# Calculate investment_2
investment_2 = np.pv(rate=0.03, nper=50, pmt=0, fv=100)
print("Investment 2 is worth $" + str(round(-investment_2, 2)) + " in today's dollars")

# Calculate investment_3
investment_3 = np.pv(rate=0.03, nper=100, pmt=0, fv=100)
print("Investment 3 is worth $" + str(round(-investment_3, 2)) + " in today's dollars")


# Investment 1 is worth $41.2 in today's dollars
# Investment 2 is worth $22.81 in today's dollars
# Investment 3 is worth $5.2 in today's dollars

```



 The moral of the story? It’s generally better to have money now rather than later.




# **2. Making Data-Driven Financial Decisions**
----------------------------------------------


## **2.1 A Tale of Two Project Proposals**


![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture4-14.png?w=981)
![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture5-17.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture6-14.png?w=1024)


####
**Project Proposals and Cash Flows Projections**



 Your project managers have projected the cash flows for each of the proposals.




 Project 1 provides higher short term cash flows, but Project 2 becomes more profitable over time.




 The cash flow projections for both projects are as follows:






|
 Year
  |
 Project 1
  |
 Project 2
  |
| --- | --- | --- |
|
 1
  |
 -$1,000 (initial investment)
  |
 -$1,000 (initial investment)
  |
|
 2
  |
 $200 (cash flow)
  |
 $150 (cash flow)
  |
|
 3
  |
 $250
  |
 $225
  |
|
 4
  |
 $300
  |
 $300
  |
|
 5
  |
 $350
  |
 $375
  |
|
 6
  |
 $400
  |
 $425
  |
|
 7
  |
 $450
  |
 $500
  |
|
 8
  |
 $500
  |
 $575
  |
|
 9
  |
 $550
  |
 $600
  |
|
 10
  |
 $600
  |
 $625
  |




 Note: The projections are provided in thousands. For example, $1,000 = $1,000,000. We will use the smaller denominations to make everything easier to read. This is also commonly done in financial statements with thousands or even millions in order to represent millions or billions.





```

import numpy as np

# Create a numpy array of cash flows for Project 1
cf_project_1 = np.array([-1000, 200, 250, 300, 350, 400, 450, 500, 550, 600])

# Create a numpy array of cash flows for Project 2
cf_project_2 = np.array([-1000, 150, 225, 300, 375, 425, 500, 575, 600, 625])

# Scale the original objects by 1000x
cf_project1 = cf_project_1 * 1000
cf_project2 = cf_project_2 * 1000

```


####
**Internal Rate of Return**



 Now that you have the cash flow projections ready to go for each project, you want to compare the
 **internal rate of return**
 (IRR) of each project to help you decide which project would be most beneficial for your company in terms of yield (rate of return).





```

import numpy as np

# Calculate the internal rate of return for Project 1
irr_project1 = np.irr(cf_project1)
print("Project 1 IRR: " + str(round(100*irr_project1, 2)) + "%")

# Calculate the internal rate of return for Project 2
irr_project2 = np.irr(cf_project2)
print("Project 2 IRR: " + str(round(100*irr_project2, 2)) + "%")

# Project 1 IRR: 28.92%
# Project 2 IRR: 28.78%

```


####
**Make a Decision Based on IRR**



 If you were making the decision solely based on internal rate of return, which project would you be more interested in (assuming the IRR is greater than your required rate of return)?




 Assume your required rate of return is 10% for this example.




 Project 1! Because higher internal rates of return are preferable!





---


## **2.2 The Weighted Average Cost of Capital**
 (WACC)


![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture7-11.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture8-9.png?w=954)
![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture9-8.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture10-9.png?w=1024)


####
**Debt and Equity Financing**



 In the previous chapter, you were able to assume that your discount rate for the NPV calculation was solely based on a measure such as inflation.




 However, in this chapter, you are the CEO of a new company that has outstanding debt and financing costs, which you will have to adjust for.




 You will use the WACC as your discount rate in upcoming exercises.




 For this exercise, assume you take out a $1,000,000 loan to finance the project, which will be your company’s only outstanding debt. This loan will represent 50% of your company’s total financing of $2,000,000. The remaining funding comes from the market value of equity.





```python

# Set the market value of debt
mval_debt = 1000000

# Set the market value of equity
mval_equity = 1000000

# Compute the total market value of your company's financing
mval_total = mval_debt + mval_equity

# Compute the proportion of your company's financing via debt
percent_debt = mval_debt / mval_total
print("Debt Financing: " + str(round(100*percent_debt, 2)) + "%")

# Compute the proportion of your company's financing via equity
percent_equity = mval_equity / mval_total
print("Equity Financing: " + str(round(100*percent_equity, 2)) + "%")

# Debt Financing: 50.0%
# Equity Financing: 50.0%

```


####
**Calculating WACC**



 In addition to determining the proportion of both equity and debt financing, you will need to estimate the cost of financing via both debt and equity in order to estimate your WACC.




 The
 **cost of debt**
 financing can be estimated as the amount you will have to pay on a new loan. This can be estimated by looking at the interest rates of loans of similar sizes to similar companies, or could be based on previous loans your company may already have been issued.




 The
 **cost of equity**
 financing can be estimated as the return on equity of similar companies. Calculating the return on equity is a simple accounting exercise, but all you need to know is that essentially, investors will require a rate of return that is close to what could be earned by a similar investment.





```python

# The proportion of debt vs equity financing is predefined
percent_debt = 0.50
percent_equity = 0.50

# Set the cost of equity
cost_equity = 0.18

# Set the cost of debt
cost_debt = 0.12

# Set the corporate tax rate
tax_rate = 0.35

# Calculate the WACC
wacc = percent_equity * cost_equity + percent_debt * cost_debt * (1 - tax_rate)
print("WACC: " + str(round(100*wacc, 2)) + "%")

# WACC: 12.9%

```


####
**Comparing Project NPV with IRR**



 Companies use their WACC as the discount rate when calculating the net present value of potential projects.




 In the same way that you discounted values by inflation in the previous chapter to account for costs over time, companies adjust the cash flows of potential projects by their cost of financing (the WACC) to account for their investor’s required rate of return based on market conditions.




 Now that you calculated the WACC, you can determine the net present value (NPV) of each project’s cash flows. The cash flows for projects 1 and 2 are available as
 `cf_project1`
 and
 `cf_project2`
 .





```

import numpy as np

# Set your weighted average cost of capital equal to 12.9%
wacc = 0.129

# Calculate the net present value for Project 1
npv_project1 = np.npv(rate=wacc, values=cf_project1)
print("Project 1 NPV: " + str(round(npv_project1, 2)))

# Calculate the net present value for Project 2
npv_project2 = np.npv(rate=wacc, values=cf_project2)
print("Project 2 NPV: " + str(round(npv_project2, 2)))

# Project 1 NPV: 856073.18
# Project 2 NPV: 904741.35

```



 If you were making the decision solely based on net present value, which project would you be more interested in?




 Project 2! Higher net present value is a good thing.



####
**Two Project With Different Lifespans**



 The board of the company has decided to go a different direction, involving slightly shorter term projects and lower initial investments.




 Your project managers have come up with two new ideas, and projected the cash flows for each of the proposals.




 Project 1 has a lifespan of 8 years, but Project 2 only has a lifespan of 7 years. Project 1 requires an initial investment of $700,000, but Project 2 only requires $400,000.




 The cash flow projections for both projects are as follows:






|
 Year
  |
 Project 1
  |
 Project 2
  |
| --- | --- | --- |
|
 1
  |
 -$700 (initial investment)
  |
 -$400 (initial investment)
  |
|
 2
  |
 $100 (cash flow)
  |
 $50 (cash flow)
  |
|
 3
  |
 $150
  |
 $100
  |
|
 4
  |
 $200
  |
 $150
  |
|
 5
  |
 $250
  |
 $200
  |
|
 6
  |
 $300
  |
 $250
  |
|
 7
  |
 $350
  |
 $300
  |
|
 8
  |
 $400
  |
 N / A
  |





```

import numpy as np

# Create a numpy array of cash flows for Project 1
cf_project_1 = np.array([-700, 100, 150, 200, 250, 300, 350, 400])

# Create a numpy array of cash flows for Project 2
cf_project_2 = np.array([-400, 50, 100, 150, 200, 250, 300])

# Scale the original objects by 1000x
cf_project1 = cf_project_1 * 1000
cf_project2 = cf_project_2 * 1000

```


####
**Calculating IRR and NPV With Different Project Lifespans**



 Now that you calculated the WACC, you can calculate and compare the IRRs and NPVs of each project.




 While the IRR remains relatively comparable across projects, the NPV, on the other hand, will be much more difficult to compare given the additional year required for project 1.




 Luckily, in the next exercise, we will introduce another method to compare the NPVs of the projects, but we will first need to compute the NPVs as before.





```

import numpy as np

# Calculate the IRR for Project 1
irr_project1 = np.irr(cf_project1)
print("Project 1 IRR: " + str(round(100*irr_project1, 2)) + "%")

# Calculate the IRR for Project 2
irr_project2 = np.irr(cf_project2)
print("Project 2 IRR: " + str(round(100*irr_project2, 2)) + "%")

# Set the wacc equal to 12.9%
wacc = 0.129

# Calculate the NPV for Project 1
npv_project1 = np.npv(rate=wacc, values=cf_project1)
print("Project 1 NPV: " + str(round(npv_project1, 2)))

# Calculate the NPV for Project 2
npv_project2 = np.npv(rate=wacc, values=cf_project2)
print("Project 2 NPV: " + str(round(npv_project2, 2)))

# Project 1 IRR: 22.94%
# Project 2 IRR: 26.89%
# Project 1 NPV: 302744.98
# Project 2 NPV: 231228.39

```



 The NPVs really aren’t comparable.



####
**Using the Equivalent Annual Annuity Approach**



 Since the net present values of each project are not directly comparable given the different lifespans of each project, you will have to consider a different approach.




 The
 **equivalent annual annuity**
 (EAA) approach allows us to compare two projects by essentially assuming that each project is an investment generating a flat interest rate each year (an annuity), and calculating the annual payment you would receive from each project, discounted to present value.




 You can compute the EAA of each project using the
 `.pmt(rate, nper, pv, fv)`
 function in
 `numpy`
 .





```

import numpy as np

# Calculate the EAA for Project 1
eaa_project1 = np.pmt(rate=wacc, nper=8, pv=-npv_project1, fv=0)
print("Project 1 EAA: " + str(round(eaa_project1, 2)))

# Calculate the EAA for Project 2
eaa_project2 = np.pmt(rate=wacc, nper=7, pv=-npv_project2, fv=0)
print("Project 2 EAA: " + str(round(eaa_project2, 2)))

# Project 1 EAA: 62872.2
# Project 2 EAA: 52120.61

```



 This is one of a few ways to deal with this problem.



####
**Making a Data-Driven Decision on Projects of Different Lifespans**



 If you were making the decision solely based on the equivalent annual annuity analysis, which project would you be more interested in?




 Project 1!


 Higher EAA means higher annual returns.




# **3. Simulating a Mortgage Loan**
----------------------------------


## **3.1 Mortgage Basics**


![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture-9.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture1-11.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture2-15.png?w=1024)


####
**Taking Out a Mortgage Loan**



 You’re expecting a child soon, and its time to start looking for a home.




 You’re currently living out of an apartment in New York City, but your blossoming career as a Data Scientist has allowed you to save up a sizable sum and purchase a home in neighboring Hoboken, New Jersey.




 You have decided to purchase a beautiful brownstone home in the $800,000 range. While you do have a considerable amount of cash on hand, you don’t have enough to purchase the entire home outright, which means you will have to take the remaining balance out as a
 **mortgage loan**
 . From the sound of it, you’ll have to put about 20% down up-front to a mortgage loan of that size.




 This up-front payment is known as a
 **down payment**
 .





```

import numpy as np

# Set the value of the home you are looking to buy
home_value = 800000

# What percentage are you paying up-front?
down_payment_percent = 0.2

# Calculate the dollar value of the down payment
down_payment = home_value * down_payment_percent
print("Initial Down Payment: " + str(down_payment))

# Calculate the value of the mortgage loan required after the down payment
mortgage_loan = home_value - down_payment
print("Mortgage Loan: " + str(mortgage_loan))

# Initial Down Payment: 160000.0
# Mortgage Loan: 640000.0

```


####
**Calculating the Monthly Mortgage Payment**



 In order to make sure you can afford the home, you will have to calculate the monthly mortgage payment you will have to make on a loan that size.




 Now, since you will be paying a monthly mortgage, you will have to convert each of the parameters into their monthly equivalents. Be careful when adjusting the interest rate, which is compounding!




 In order to calculate the monthly mortgage payment, you will use the
 `numpy`
 function
 `.pmt(rate, nper, pv)`
 where:



* `rate`
 = The periodic (monthly) interest rate
* `nper`
 = The number of payment periods (months) in the lifespan of the mortgage loan
* `pv`
 = The total value of the mortgage loan



 You have been given a 30-year mortgage loan quote for your desired amount at 3.75%. The value of the mortgage loan is available as
 `mortgage_loan`
 .





```

import numpy as np

# Derive the equivalent monthly mortgage rate from the annual rate
mortgage_rate_periodic = (1 + mortgage_rate) ** (1/12) - 1

# How many monthly payment periods will there be over 30 years?
mortgage_payment_periods = 30 * 12

# Calculate the monthly mortgage payment (multiply by -1 to keep it positive)
periodic_mortgage_payment = -1*np.pmt(mortgage_rate_periodic, mortgage_payment_periods, mortgage_loan)
print("Monthly Mortgage Payment: " + str(round(periodic_mortgage_payment, 2)))

# Monthly Mortgage Payment: 2941.13

```




---


## **3.2 Amortization, Interest and Principal**



![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture3-15.png?w=1024)

####
**Calculating Interest and Principal Payments**



 Due to the size of the mortgage loan, you begin the mortgage in the initial period by paying mostly interest and retaining very little
 **principal**
 , or
 **equity**
 that goes towards the ownership of your home.




 This means that if you were to stop paying your mortgage and sell your home after only a few years, the bank would actually own most of the home because what you paid was mostly interest, and very little principal.





```

mortgage_loan
640000.0

periodic_mortgage_payment
2941.125363188976

mortgage_rate_periodic
0.003072541703255549

```




```python

# Calculate the amount of the first loan payment that will go towards interest
initial_interest_payment = mortgage_loan * mortgage_rate_periodic
print("Initial Interest Payment: " + str(round(initial_interest_payment, 2)))

# Calculate the amount of the first loan payment that will go towards principal
initial_principal_payment = periodic_mortgage_payment - initial_interest_payment
print("Initial Principal Payment: " + str(round(initial_principal_payment, 2)))

# Initial Interest Payment: 1966.43
# Initial Principal Payment: 974.7

```


####
**Simulating Periodic Payments (I)**



 You have all the tools you’ll need to simulate the mortgage payments over time.




 Every time a mortgage payment is made, the following payment will have a slightly lower percentage, which is used to pay off interest. This means that more of the remainder will go towards the portion of the home that you own instead of the bank. This is important to determine how much you will gain from selling the home before paying off your mortgage, or to determine when your mortgage is underwater. But more on that later.




 You will now write a simple program to calculate the interest and mortgage portions of each payment over time.





```

principal_remaining
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0.])

```




```python

# Loop through each mortgage payment period
for i in range(0, mortgage_payment_periods):

    # Handle the case for the first iteration
    if i == 0:
        previous_principal_remaining = mortgage_loan
    else:
        previous_principal_remaining = principal_remaining[i-1]

    # Calculate the interest and principal payments
    interest_payment = round(previous_principal_remaining*mortgage_rate_periodic, 2)
    principal_payment = round(periodic_mortgage_payment-interest_payment, 2)

    # Catch the case where all principal is paid off in the final period
    if previous_principal_remaining - principal_payment < 0:
        principal_payment = previous_principal_remaining

    # Collect the principal remaining values in an array
    principal_remaining[i] = previous_principal_remaining - principal_payment

    # Print the payments for the first few periods
    print_payments(i, interest_payment, principal_payment, principal_remaining)

```




```

Period 0: Interest Paid: 1966.43 | Principal Paid: 974.7 | Remaining Balance: 639025.3
Period 1: Interest Paid: 1963.43 | Principal Paid: 977.7 | Remaining Balance: 638047.6000000001
Period 2: Interest Paid: 1960.43 | Principal Paid: 980.7 | Remaining Balance: 637066.9000000001
Period 3: Interest Paid: 1957.41 | Principal Paid: 983.72 | Remaining Balance: 636083.1800000002
Period 4: Interest Paid: 1954.39 | Principal Paid: 986.74 | Remaining Balance: 635096.4400000002
Period 5: Interest Paid: 1951.36 | Principal Paid: 989.77 | Remaining Balance: 634106.6700000002

```


####
**Simulating Periodic Payments (II)**



 You have decided to extend your program from the previous exercise to store the principal and interest payments made at each period, and to plot the results instead of simply printing them.




 For this example, the plotting code is already done, so you just need to finish the logic inside the for loop and the initialization of the variables which will be updated at each iteration.





```python

# Loop through each mortgage payment period
for i in range(0, mortgage_payment_periods):

    # Handle the case for the first iteration
    if i == 0:
        previous_principal_remaining = mortgage_loan
    else:
        previous_principal_remaining = principal_remaining[i-1]

    # Calculate the interest based on the previous principal
    interest_payment = round(previous_principal_remaining*mortgage_rate_periodic, 2)
    principal_payment = round(periodic_mortgage_payment - interest_payment, 2)

    # Catch the case where all principal is paid off in the final period
    if previous_principal_remaining - principal_payment < 0:
        principal_payment = previous_principal_remaining

    # Collect the historical values
    interest_paid[i] = interest_payment
    principal_paid[i] = principal_payment
    principal_remaining[i] = previous_principal_remaining - principal_payment

# Plot the interest vs principal
plt.plot(interest_paid, color="red")
plt.plot(principal_paid, color="blue")
plt.legend(handles=[interest_plot, principal_plot], loc=2)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture4-15.png?w=1024)



---


## **3.3 Home Ownership, Home Prices and Recessions**


![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture5-18.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture6-15.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture7-12.png?w=830)


####
**Cumulative Payments and Home Equity**



 You are faithfully paying your mortgage each month, but it’s difficult to tell how much of the house you actually own and how much interest you have paid in total over the years.




`principal_paid`
 ,
 `interest_paid`
 ,
 `home_value`
 and
 `down_payment_percent`
 from the previous exercise are available.





```

principal_paid
array([ 974.7 ,  977.7 ,  980.7 ,  983.72,  986.74,  989.77,  992.81,
...
       2914.19, 2923.15, 2929.2 ])

interest_paid
array([1966.43, 1963.43, 1960.43, 1957.41, 1954.39, 1951.36, 1948.32,
...
         26.94,   17.98,    9.  ])

home_value
800000

down_payment_percent
0.2

```




```

import numpy as np

# Calculate the cumulative home equity (principal) over time
cumulative_home_equity = np.cumsum(principal_paid)

# Calculate the cumulative interest paid over time
cumulative_interest_paid = np.cumsum(interest_paid)

# Calculate your percentage home equity over time
cumulative_percent_owned = down_payment_percent + (cumulative_home_equity/home_value)

# Plot the cumulative interest paid vs equity accumulated
plt.plot(cumulative_interest_paid, color='red')
plt.plot(cumulative_home_equity, color='blue')
plt.legend(handles=[interest_plot, principal_plot], loc=2)
plt.show()

```




```

print(cumulative_percent_owned)

[0.20121838 0.2024405  0.20366638 0.20489603 0.20612945 0.20736666
...
 0.98178978 0.98541024 0.98904183 0.99268456 0.9963385  1.        ]

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture8-10.png?w=1024)

####
**Rising Housing Prices**



 Home values have been rising steadily each year, and this is a rather large investment for you.




 Calculate your home equity value over time given a steady growth rate of 0.25% per month. A repeated array of this growth rate (with a length equal to the number of mortgage payment periods) is already stored for you in an object called
 `growth_array`
 .




 The
 `home_value`
 and
 `cumulative_percent_owned`
 variables from the previous exercise are available.





```

growth_array
array([0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025,
...
       0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025])

```




```

import numpy as np

# Calculate the cumulative growth over time
cumulative_growth_forecast = np.cumprod(1 + growth_array)

# Forecast the home value over time
home_value_forecast = home_value * cumulative_growth_forecast

# Forecast the home equity value owned over time
cumulative_home_value_owned = home_value_forecast * cumulative_percent_owned

# Plot the home value vs equity accumulated
plt.plot(home_value_forecast, color='red')
plt.plot(cumulative_home_value_owned, color='blue')
plt.legend(handles=[homevalue_plot, homeequity_plot], loc=2)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture9-9.png?w=1024)


 Turned out to be a good investment. Now unfortunately, housing prices don’t always go up…



####
**Falling Housing Prices and Underwater Mortgages**



 Unfortunately, you are also well aware that home prices don’t always rise.




 An
 **underwater**
 mortgage is when the remaining amount you owe on your mortgage is actually higher than the value of the house itself.




 In this exercise, you will calculate the worst case scenario where home prices drop steadily at the rate of 0.45% per month. To speed things up, the cumulative drop in home prices has already been forecasted and stored for you in a variable called
 `cumulative_decline_forecast`
 , which is an array of multiplicative discount factors compared to today’s price – no need to add 1 to the rate array.





```

principal_remaining
array([639025.3 , 638047.6 , 637066.9 , 636083.18, 635096.44, 634106.67,
...
        14568.18,  11671.81,   8766.54,   5852.35,   2929.2 ,      0.  ])

```




```

import numpy as np
import pandas as pd

# Cumulative drop in home value over time as a ratio
cumulative_decline_forecast = np.cumprod(1+decline_array)

# Forecast the home value over time
home_value_forecast = home_value * cumulative_decline_forecast

# Find all periods where your mortgage is underwater
underwater = home_value_forecast < principal_remaining
pd.value_counts(underwater)

# Plot the home value vs principal remaining
plt.plot(home_value_forecast, color='red')
plt.plot(principal_remaining, color='blue')
plt.legend(handles=[homevalue_plot, principal_plot], loc=2)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture10-10.png?w=1024)


 When the blue line is above the red line, you are ‘underwater’. Putting more money down and taking a smaller mortgage in the first place will help you avoid this situation.




# **4. Budgeting Application**
-----------------------------


## **4.1 Budgeting Project Proposal**



![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture11-7.png?w=892)

####
**Salary and Taxes**



 You just got a new job as an entry-level Data Scientist at a technology company in New York City with a decent starting salary of $85,000 per year.




 Unfortunately, after state and local taxes, you can expect to be sending roughly 30% back to the government each year.




 You will need to calculate your monthly take home pay after taxes in order to begin budgeting.





```python

# Enter your annual salary
salary = 85000

# Assume a tax rate of 30%
tax_rate = 0.3

# Calculate your salary after taxes
salary_after_taxes = salary * (1 - tax_rate)
print("Salary after taxes: " + str(round(salary_after_taxes, 2)))

# Calculate your monthly salary after taxes
monthly_takehome_salary = salary_after_taxes / 12
print("Monthly takehome salary: " + str(round(monthly_takehome_salary, 2)))

# Salary after taxes: 59500.0
# Monthly takehome salary: 4958.33

```


####
**Monthly Expenses and Savings**



 In order to make it in New York City, you have decided to split a two-bedroom apartment with a friend. You will have to budget for rent, food and entertainment, but it’s also a good idea to allocate an amount for unforeseen expenses each month. This unforeseen expenses budget could be used for anything ranging from new clothes or electronics to doctor appointments.




 Set up your monthly budget as follows:



* Rent: $1200 / month (Includes utilities)
* Food: $30 / day (On average. Includes groceries and eating out.)
* Entertainment: $200 / month (Movies, drinks, museums, parties…)
* Unforeseen Expenses: 250 / month (Stay safe, and don’t drop your phone!)



 For this application, assume an average of 30 days per month. Whatever is left after your paying your monthly expenses will go into your savings account each month.





```python

# Enter your monthly rent
monthly_rent = 1200

# Enter your daily food budget
daily_food_budget = 30

# Calculate your monthly food budget assuming 30 days per month
monthly_food_budget = 30 * 30

# Set your monthly entertainment budget
monthly_entertainment_budget = 200

# Allocate funds for unforeseen expenses, just in case
monthly_unforeseen_expenses = 250

# Next, calculate your total monthly expenses
monthly_expenses = monthly_rent + monthly_food_budget + monthly_entertainment_budget + monthly_unforeseen_expenses
print("Monthly expenses: " + str(round(monthly_expenses, 2)))

# Finally, calculate your monthly take-home savings
monthly_savings = monthly_takehome_salary - monthly_expenses
print("Monthly savings: " + str(round(monthly_savings, 2)))

# Monthly expenses: 2550
# Monthly savings: 2408.33

```



 Expenses add up quickly, don’t they?



####
**Forecast Salary Growth and Cost of Living**



 Due to both inflation and increased productivity from experience, you can expect your salary to grow at different rates depending on your job. Now, since you are working in a growing and in-demand career field as a Data Scientist, you can assume a steady growth in your annual salary every year based on performance.




 You can assume an annual salary growth rate of 5%, which means if you start at $85,000 per year, you can expect to earn over $176,000 per year after 15 years. After taxes, assuming your tax rate hasn’t changed, that works out to roughly $125,000 per year, which is not unreasonable for a Data Scientist. In fact, you might even make it to that level in a few years! But just to be safe, you should be conservative with your projections.




 For this application, assume all inflation and salary growth happens in smaller increments on a monthly basis instead of just one large increase at the end of each year.





```

import numpy as np

# Create monthly forecasts up to 15 years from now
forecast_months = 12*15

# Set your annual salary growth rate
annual_salary_growth = 0.05

# Calculate your equivalent monthly salary growth rate
monthly_salary_growth = (1 + annual_salary_growth) ** (1/12) - 1

# Forecast the cumulative growth of your salary
cumulative_salary_growth_forecast = np.cumprod(np.repeat(1 + monthly_salary_growth, forecast_months))

# Calculate the actual salary forecast
salary_forecast = monthly_takehome_salary * cumulative_salary_growth_forecast

# Plot the forecasted salary
plt.plot(salary_forecast, color='blue')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture12-7.png?w=1024)


 That’s becomes a mighty fine salary very quickly.



####
**Forecast Growing Expenses Due to Inflation**



 You will also assume your monthly expenses will rise by an average of 2.5% per year due to
 **inflation**
 . This will lead to higher
 **cost of living**
 over time, paying for the same expenses each year but at a higher price. Luckily, your salary is growing faster than inflation, which means you should have more money going into savings each month.





```

import numpy as np

# Set the annual inflation rate
annual_inflation = 0.025

# Calculate the equivalent monthly inflation rate
monthly_inflation = (1+annual_inflation)**(1/12) - 1

# Forecast cumulative inflation over time
cumulative_inflation_forecast = np.cumprod(np.repeat(1 + monthly_inflation, forecast_months))

# Calculate your forecasted expenses
expenses_forecast = monthly_expenses*cumulative_inflation_forecast

# Plot the forecasted expenses
plt.plot(expenses_forecast, color='red')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture13-6.png?w=1024)


 Even though you’re making more money, you’re spending more, too!





---



**4.2 Net Worth, Saving, and Investing**
-----------------------------------------


####
**Calculate Your Net Worth**



 Now that you have forecasted your savings and salary over time while taking career progression and inflation into account, you have constructed a time-series which you can use to calculate your cash flows, just like in Chapter 1.




 For this example, all you need to do is subtract your forecasted monthly expenses from your forecasted monthly salary. The remaining cash flow will go straight into your savings account for each month.




 You want to project your cumulative savings over time to see how effective your budgeting process will be given your projections.




`salary_forecast`
 and
 `expenses_forecast`
 from the previous exercises are available.





```

import numpy as np

# Calculate your savings for each month
savings_forecast = salary_forecast - expenses_forecast

# Calculate your cumulative savings over time
cumulative_savings = np.cumsum(savings_forecast)

# Print the final cumulative savings after 15 years
final_net_worth = cumulative_savings[-1]
print("Your final net worth: " + str(round(final_net_worth, 2)))

# Plot the forecasted savings
plt.plot(cumulative_savings, color='blue')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture18-4.png?w=1024)


 Not bad! But there’s a better way to accumulate wealth over time.



####
**So You Want to Be a Millionaire?**



 Your projections show that you can accumulate over $700,000 in just 15 years by following a strict budget and growing your salary steadily over time.




 But you want to be a millionaire in 15 years, retire young, sip margaritas and travel for the rest of your life. In order to do that, you’re going to need to invest.




 Remember the
 `.pmt()`
 function from
 `numpy`
 ? You can use this function to calculate how much you need to save each month in order to accumulate your desired wealth over time.




 You still have a lot to learn about the stock market, but your financial advisor told you that you can earn anywhere from 5-10% per year on your capital on average by investing in a low cost index fund.




 You know that the stock market doesn’t always go up, but you will assume a modest 7% return per year, which has been the average annual return in the US stock market from 1950-2009.





```

import numpy as np

# Set the annual investment return to 7%
investment_rate_annual = 0.07

# Calculate the monthly investment return
investment_rate_monthly = (1 + investment_rate_annual) ** (1/12) - 1

# Calculate your required monthly investment to amass $1M
required_investment_monthly = np.pmt(rate=investment_rate_monthly, nper=forecast_months, pv=0, fv=-1000000)
print("You will have to invest $" + str(round(required_investment_monthly, 2)) + " per month to amass $1M over 15 years")

# You will have to invest $3214.35 per month to amass $1M over 15 years

```



 $3000 per month?! Let’s start slow, and build it up over time.



####
**Investing a Percentage of Your Income (I)**



 Unfortunately, you really can’t afford to save $3,000 per month in order to amass $1,000,000 after only 15 years.




 But what you can do is start slowly, investing a small percentage of your take-home income each month, which should grow over time as your income grows as well.




 In this exercise, you will lay the foundations to simulate this investing process over time.




 The
 `salary_forecast`
 and
 `expenses_forecast`
 variables are available from the previous exercise.




 The
 `cash_flow_forecast`
 is also available, and is an array of your forecasted salary minus your forecasted expenses. The
 `monthly_investment_percentage`
 variable is already set to 0.30.





```

import numpy as np

# Calculate your monthly deposit into your investment account
investment_deposit_forecast = cash_flow_forecast * monthly_investment_percentage

# The rest goes into your savings account
savings_forecast_new = cash_flow_forecast * (1 - monthly_investment_percentage)

# Calculate your cumulative savings over time
cumulative_savings_new = np.cumsum(savings_forecast_new)

# Plot your forecasted monthly savings vs investments
plt.plot(investment_deposit_forecast, color='red')
plt.plot(savings_forecast_new, color='blue')
plt.legend(handles=[investments_plot, savings_plot], loc=2)
plt.show()

```


####
**Investing a Percentage of Your Income (II)**



 To finish up your investment simulation, you will need to loop through each time period, calculate the growth of any investments you have already made, add your new monthly deposit, and calculate your net worth at each point in time.




 Cumulative savings (
 `cumulative_savings_new`
 ) from the previous exercise is available, and
 `investment_portfolio`
 and
 `net_worth`
 are pre-allocated empty numpy arrays of length equal to
 `forecast_months`
 .





```

import numpy as np

# Loop through each forecast period
for i in range(forecast_months):

    # Find the previous investment deposit amount
    if i == 0:
        previous_investment = 0
    else:
        previous_investment = investment_portfolio[i-1]

    # Calculate the value of your previous investments, which have grown
    previous_investment_growth = previous_investment*(1 + investment_rate_monthly)

    # Add your new deposit to your investment portfolio
    investment_portfolio[i] =  previous_investment_growth + investment_deposit_forecast[i]

    # Calculate your net worth at each point in time
    net_worth[i] = np.cumsum(cumulative_savings_new[i] + investment_portfolio[i])

# Plot your forecasted cumulative savings vs investments and net worth
plot_investments(investment_portfolio, cumulative_savings_new, net_worth)

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture20-4.png?w=1024)



---


## **4.3 The Power of Time and Compound Interest**


![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture21-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture22-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/intro-to-financial-concepts-using-python/capture23-2.png?w=819)


####
**Investing Over Time**



 If you would like to accumulate $1,000,000 over 15 years, at 7% per year, you will have to invest $3214.35 per month:





```

np.pmt(rate=((1+0.07)**(1/12) - 1),
       nper=15*12, pv=0, fv=1000000)
# -3214.351338524575

```



 But what if you were willing to wait an extra 15 years, for a total of 30 years? How much will you need to invest each month?





```

np.pmt(rate=((1+0.07)**(1/12) - 1),
       nper=30*12, pv=0, fv=1000000)

# -855.1009225937204

```



 Compounded returns mean you only need to save $855.10 per month.



####
**Inflation-Adjusted Net Worth**



 By saving 30% per year, your simulation shows that you can accumulate $896,962.66. Not quite a millionaire, but not bad at all!




 For the sake of simplicity, let’s assume you were able to save $900,000 by following your budget.




 But what if you retire 15 years from now? What is $900,000 going to be truly worth 15 years from now?





```

import numpy as np

# Set your future net worth
future_net_worth = 900000

# Set the annual inflation rate to 2.5%
annual_inflation = 0.025

# Calculate the present value of your terminal wealth over 15 years
inflation_adjusted_net_worth = np.pv(rate=annual_inflation, nper=15, pmt=0, fv=-1*future_net_worth)
print("Your inflation-adjusted net worth: $" + str(round(inflation_adjusted_net_worth, 2)))

# Your inflation-adjusted net worth: $621419.0

```



 The End.


 Thank you for reading and hope you’ve learned a lot.



