---
title: Statistical Simulation in Python
date: 2021-12-07 11:22:11 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Statistical Simulation in Python
===================================







 This is the memo of the 4th course (5 courses in all) of ‘Statistics Fundamentals with Python’ skill track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/statistical-simulation-in-python)**
 .



###
**Table of contents**


1. Basics of randomness & simulation
2. [Probability & data generation process](https://datascience103579984.wordpress.com/2019/09/26/statistical-simulation-in-python-from-datacamp/2/)
3. [Resampling methods](https://datascience103579984.wordpress.com/2019/09/26/statistical-simulation-in-python-from-datacamp/3/)
4. [Advanced Applications of Simulation](https://datascience103579984.wordpress.com/2019/09/26/statistical-simulation-in-python-from-datacamp/4/)





# **1. Basics of randomness & simulation**
-----------------------------------------


### **1.1 Introduction to random variables**



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/1-2.png?w=1024)


 A probability distribution is a mapping from the set of possible outcomes of a random variable to the probability of observing that outcome.


 It tells you how likely you are to observe a given outcome or a set of outcomes.



####
**np.random.choice()**



 In this exercise, you will be introduced to the
 `np.random.choice()`
 function. This is a remarkably useful function for simulations and you will be making extensive use of it later in the course. As a first step, let’s try to understand the basics of this function.





```

np.random.choice?
Docstring:
choice(a, size=None, replace=True, p=None)

Generates a random sample from a given 1-D array

        .. versionadded:: 1.7.0

Parameters
-----------
a : 1-D array-like or int
    If an ndarray, a random sample is generated from its elements.
    If an int, the random sample is generated as if a were np.arange(a)
size : int or tuple of ints, optional
    Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
    ``m * n * k`` samples are drawn.  Default is None, in which case a
    single value is returned.
replace : boolean, optional
    Whether the sample is with or without replacement
p : 1-D array-like, optional
    The probabilities associated with each entry in a.
    If not given the sample assumes a uniform distribution over all
    entries in a.

Returns
--------
samples : single item or ndarray
    The generated random samples

Raises
-------
ValueError
    If a is an int and less than zero, if a or p are not 1-dimensional,
    if a is an array-like of size 0, if p is not a vector of
    probabilities, if a and p have different lengths, or if
    replace=False and the sample size is greater than the population
    size

See Also
---------
randint, shuffle, permutation

```



 The
 `np.random.choice()`
 function will work even if you only provide the input array
 `a`
 . Make sure to make use of the
 `help()`
 function throughout the course, it will help get you through some tough exercises!



####
**Poisson random variable**



 The
 `numpy.random`
 module also has a number of useful probability distributions for both discrete and continuous random variables. In this exercise, you will learn how to draw samples from a probability distribution.




 In particular, you will draw samples from a very important discrete probability distribution, the Poisson distribution, which is typically used for modeling the average rate at which events occur.




 Following the exercise, you should be able to apply these steps to any of the probability distributions found in
 `numpy.random`
 . In addition, you will also see how the sample mean changes as we draw more samples from a distribution.





```python

# Initialize seed and parameters
np.random.seed(123)
lam, size_1, size_2 = 5, 3, 1000

# Draw samples & calculate absolute difference between lambda and sample mean
samples_1 = np.random.poisson(lam, size_1)
samples_2 = np.random.poisson(lam, size_2)
answer_1 = abs(np.mean(samples_1) - lam)
answer_2 = abs(np.mean(samples_2) - lam)

print("|Lambda - sample mean| with {} samples is {} and with {} samples is {}. ".format(size_1, answer_1, size_2, answer_2))

# |Lambda - sample mean| with 3 samples is 0.33333333333333304 and with 1000 samples is 0.07699999999999996.

```




```

import matplotlib.pyplot as plt
plt.hist(samples_2)
plt.xlabel('samples_2 value')
plt.ylabel('count')
plt.title('np.random.poisson result\nlamda:5')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/2-2.png?w=640)


 Why do you think the larger size gives us a better result?



####
**Shuffling a deck of cards**



 Often times we are interested in randomizing the order of a set of items. Consider a game of cards where you first shuffle the deck of cards or a game of scrabble where the letters are first mixed in a bag. As the final exercise of this section, you will learn another useful function –
 `np.random.shuffle()`
 . This function allows you to randomly shuffle a sequence in place. At the end of this exercise, you will know how to shuffle a deck of cards or any sequence of items.




 Examine
 `deck_of_cards`
 in the shell.





```

print(deck_of_cards)

# Shuffle the deck
np.random.shuffle(deck_of_cards)

# Print out the top three cards
card_choices_after_shuffle = deck_of_cards[:3]
print(card_choices_after_shuffle)

```




```

[('Heart', 0), ('Heart', 1), ('Heart', 2), ..., ('Diamond', 10), ('Diamond', 11), ('Diamond', 12)]
[('Diamond', 9), ('Spade', 9), ('Spade', 4)]

```




---


## **1.2 Simulation basics**



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/3-1.png?w=903)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/4-1.png?w=1024)



####
**Throwing a fair die**



 Once you grasp the basics of designing a simulation, you can apply it to any system or process. Next, we will learn how each step is implemented using some basic examples.




 As we have learned, simulation involves repeated random sampling. The first step then is to get
 **one**
 random sample. Once we have that, all we do is repeat the process multiple times. This exercise will focus on understanding how we get one random sample. We will study this in the context of throwing a fair six-sided die.




 By the end of this exercise, you will be familiar with how to implement the first two steps of running a simulation – defining a random variable and assigning probabilities.




 For the rest of the course, look to the IPython shell to find out what seed has been set.





```python

# The seed for this exercise is set to 123
# Define die outcomes and probabilities
die, probabilities, throws = [1,2,3,4,5,6], [1/6]*6, 1

# Use np.random.choice to throw the die once and record the outcome
outcome = np.random.choice(die, size=1, p=probabilities)
print("Outcome of the throw: {}".format(outcome[0]))

# Outcome of the throw: 5

```


####
**Throwing two fair dice**



 We now know how to implement the first two steps of a simulation. Now let’s implement the next step – defining the relationship between random variables.




 Often times, our simulation will involve not just one, but multiple random variables. Consider a game where throw you two dice and win if each die shows the same number. Here we have two random variables – the two dice – and a relationship between each of them – we win if they show the same number, lose if they don’t. In reality, the relationship between random variables can be much more complex, especially when simulating things like weather patterns.




 By the end of this exercise, you will be familiar with how to implement the third step of running a simulation – defining relationships between random variables.





```python

# The seed for this exercise is set to 223
# Initialize number of dice, simulate & record outcome
die, probabilities, num_dice = [1,2,3,4,5,6], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], 2
outcomes = np.random.choice(die, size=num_dice, p=probabilities)

# Win if the two dice show the same number
if outcomes[0] == outcomes[1]:
    answer = 'win'
else:
    answer = 'lose'

print("The dice show {} and {}. You {}!".format(outcomes[0], outcomes[1], answer))
# The dice show 5 and 5. You win!

```


####
**Simulating the dice game**



 We now know how to implement the first three steps of a simulation. Now let’s consider the next step – repeated random sampling.




 Simulating an outcome once doesn’t tell us much about how often we can expect to see that outcome. In the case of the dice game from the previous exercise, it’s great that we won once. But suppose we want to see how many times we can expect to win if we played this game multiple times, we need to repeat the random sampling process many times. Repeating the process of random sampling is helpful to understand and visualize inherent uncertainty and deciding next steps.




 Following this exercise, you will be familiar with implementing the fourth step of running a simulation – sampling repeatedly and generating outcomes.





```python

# The seed for this exercise is set to 223
# Initialize model parameters & simulate dice throw
die, probabilities, num_dice = [1,2,3,4,5,6], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], 2
sims, wins = 100, 0

for i in range(sims):
    outcomes = np.random.choice(die, size=num_dice, p=probabilities)
    # Increment `wins` by 1 if the dice show same number
    if outcomes[0] == outcomes[1]:
        wins = wins + 1

print("In {} games, you win {} times".format(sims, wins))
# In 100 games, you win 25 times

```




---


## **1.3 Using simulation for decision-making**



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/5-1.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/9-1.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/6-1.png?w=909)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/7-1.png?w=929)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/8-1.png?w=929)



####
**Simulating one lottery drawing**



 In the last three exercises of this chapter, we will be bringing together everything you’ve learned so far. We will run a complete simulation, take a decision based on our observed outcomes, and learn to modify inputs to the simulation model.




 We will use simulations to figure out whether or not we want to buy a lottery ticket. Suppose you have the opportunity to buy a lottery ticket which gives you a shot at a grand prize of $1 Million. Since there are 1000 tickets in total, your probability of winning is 1 in 1000. Each ticket costs $10. Let’s use our understanding of basic simulations to first simulate one drawing of the lottery.





```python

# The seed for this exercise is set to 123
# Pre-defined constant variables
lottery_ticket_cost, num_tickets, grand_prize = 10, 1000, 1000000

# Probability of winning
chance_of_winning = 1/num_tickets

# Simulate a single drawing of the lottery
gains = [-lottery_ticket_cost, grand_prize-lottery_ticket_cost]
probability = [1-chance_of_winning, chance_of_winning]
outcome = np.random.choice(a=gains, size=1, p=probability, replace=True)

print("Outcome of one drawing of the lottery is {}".format(outcome))
# Outcome of one drawing of the lottery is [-10]

```


####
**Should we buy?**



 In the last exercise, we simulated the random drawing of the lottery ticket once. In this exercise, we complete the simulation process by repeating the process multiple times.




 Repeating the process gives us multiple outcomes. We can think of this as multiple universes where the same lottery drawing occurred. We can then determine the average winnings across all these universes. If the average winnings are greater than what we pay for the ticket then it makes sense to buy it, otherwise, we might not want to buy the ticket.




 This is typically how simulations are used for evaluating business investments. After completing this exercise, you will have the basic tools required to use simulations for decision-making.





```python

# Initialize size and simulate outcome
lottery_ticket_cost, num_tickets, grand_prize = 10, 1000, 1000000
chance_of_winning = 1/num_tickets
size = 2000
payoffs = [-lottery_ticket_cost, grand_prize-lottery_ticket_cost]
probs = [1-chance_of_winning,chance_of_winning]

outcomes = np.random.choice(a=payoffs, size=size, p=probs, replace=True)

# Mean of outcomes.
answer = np.mean(outcomes)
print("Average payoff from {} simulations = {}".format(size, answer))
# Average payoff from 2000 simulations = 1990.0

```



 Is it worth spending $10 on the ticket for this average payoff?



####
**Calculating a break-even lottery price**



 Simulations allow us to ask more nuanced questions that might not necessarily have an easy analytical solution. Rather than solving a complex mathematical formula, we directly get multiple sample outcomes. We can run experiments by modifying inputs and studying how those changes impact the system. For example, once we have a moderately reasonable model of global weather patterns, we could evaluate the impact of increased greenhouse gas emissions.




 In the lottery example, we might want to know how expensive the ticket needs to be for it to not make sense to buy it. To understand this, we need to modify the ticket cost to see when the expected payoff is negative.




`grand_prize`
 ,
 `num_tickets`
 , and
 `chance_of_winning`
 are loaded in the environment.





```python

# The seed for this exercise is set to 333
# Initialize simulations and cost of ticket
sims, lottery_ticket_cost = 3000, 0

# Use a while loop to increment `lottery_ticket_cost` till average value of outcomes falls below zero
while 1:
    outcomes = np.random.choice([-lottery_ticket_cost, grand_prize-lottery_ticket_cost],
                 size=sims, p=[1-chance_of_winning, chance_of_winning], replace=True)
    if outcomes.mean() < 0:
        break
    else:
        lottery_ticket_cost += 1
answer = lottery_ticket_cost - 1

print("The highest price at which it makes sense to buy the ticket is {}".format(answer))

# The highest price at which it makes sense to buy the ticket is 9

```python

# **2. Probability & data generation process**
---------------------------------------------


## **2.1 Probability basics**



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/10-1.png?w=1024)

####
**Queen or spade**



 In this example, you’ll use the generalized probability formula




 P(A∪B)=P(A)+P(B)−P(A∩B)




 to calculate the probability of two events. Consider a deck of cards (13 cards x 4 suites = 52 cards in total). One card is drawn at random. What is the probability of getting a queen or a spade? Here event A is the card being a queen and event B is the card being a spade. Think carefully about whether the two events have anything in common.





```python

# P(Queens) + P(Spades) - P(Queen of Spades)
4/52 + 13/52 - 1/52
= 16/52

```


####
**Two of a kind**



 Now let’s use simulation to estimate probabilities. Suppose you’ve been invited to a game of poker at your friend’s home. In this variation of the game, you are dealt five cards and the player with the better hand wins. You will use a simulation to estimate the probabilities of getting certain hands. Let’s work on estimating the probability of getting at least two of a kind. Two of a kind is when you get two cards of different suites but having the same numeric value (e.g., 2 of hearts, 2 of spades, and 3 other cards).




 By the end of this exercise, you will know how to use simulation to calculate probabilities for card games.





```python

# The seed for this exercise is set to 123
# Shuffle deck & count card occurrences in the hand
n_sims, two_kind = 10000, 0
for i in range(n_sims):
    np.random.shuffle(deck_of_cards)
    hand, cards_in_hand = deck_of_cards[0:5], {}
    for card in hand:
        # Use .get() method on cards_in_hand
        cards_in_hand[card[1]] = cards_in_hand.get(card[1], 0) + 1

    # Condition for getting at least 2 of a kind
    highest_card = max(cards_in_hand.values())
    if highest_card>=2:
        two_kind += 1

print("Probability of seeing at least two of a kind = {} ".format(two_kind/n_sims))

# Probability of seeing at least two of a kind = 0.4952

```




```

np.random.shuffle(deck_of_cards)
        hand, cards_in_hand = deck_of_cards[0:5], {}
        for card in hand:
            # Use .get() method on cards_in_hand
            cards_in_hand[card[1]] = cards_in_hand.get(card[1], 0) + 1

hand
[('Club', 1), ('Diamond', 4), ('Heart', 8), ('Spade', 4), ('Spade', 8)]

cards_in_hand
{1: 1, 4: 2, 8: 2}

cards_in_hand.values()
dict_values([1, 2, 2])

```


####
**Game of thirteen**



 A famous French mathematician Pierre Raymond De Montmart, who was known for his work in combinatorics, proposed a simple game called as Game of Thirteen. You have a deck of 13 cards, each numbered from 1 through 13. Shuffle this deck and draw cards one by one. A coincidence is when the number on the card matches the order in which the card is drawn. For instance, if the 5th card you draw happens to be a 5, it’s a coincidence. You win the game if you get through all the cards without any coincidences. Let’s calculate the probability of winning at this game using simulation.




 By completing this exercise, you will further strengthen your ability to cast abstract problems into the simulation framework for estimating probabilities.





```python

# Pre-set constant variables
deck, sims, coincidences = np.arange(1, 14), 10000, 0

for _ in range(sims):
    # Draw all the cards without replacement to simulate one game
    draw = np.random.choice(deck, size=len(deck), replace=False)
    # Check if there are any coincidences
    coincidence = (draw == list(np.arange(1, 14))).any()
    if coincidence == True:
        coincidences += 1

# Calculate probability of winning
prob_of_winning = 1 - coincidences / sims
print("Probability of winning = {}".format(prob_of_winning))
# Probability of winning = 0.36950000000000005

```




---


## **2.2 More probability concepts**



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/1-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/2-4.png?w=687)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/3-2.png?w=1024)



####
**The conditional urn**



 As we’ve learned, conditional probability is defined as the probability of an event given another event. To illustrate this concept, let’s turn to an urn problem.




 We have an urn that contains 7 white and 6 black balls. Four balls are drawn at random. We’d like to know the probability that the first and third balls are white, while the second and the fourth balls are black.




 Upon completion, you will learn to manipulate simulations to calculate simple conditional probabilities.





```python

# Initialize success, sims and urn
success, sims = 0, 5000
urn = ['w'] * 7 + ['b'] * 6

for _ in range(sims):
    # Draw 4 balls without replacement
    draw = np.random.choice(urn, replace=False, size=4)
    # Count the number of successes
    if (draw == ['w','b','w','b']).all():
        success +=1

print("Probability of success = {}".format(success/sims))

# Probability of success = 0.0722

```


####
**Birthday problem**



 Now we’ll use simulation to solve a famous probability puzzle – the birthday problem. It sounds quite straightforward –
 *How many people do you need in a room to ensure at least a 50% chance that two of them share the same birthday?*




 With 366 people in a 365-day year, we are 100% sure that at least two have the same birthday, but we only need to be 50% sure. Simulation gives us an elegant way of solving this problem.




 Upon completion of this exercise, you will begin to understand how to cast problems in a simulation framework.





```python

# Draw a sample of birthdays & check if each birthday is unique
days = np.arange(1,366)
people = 2

def birthday_sim(people):
    sims, unique_birthdays = 2000, 0
    for _ in range(sims):
        draw = np.random.choice(days, size=people, replace=True)
        if len(draw) == len(set(draw)):
            unique_birthdays += 1
    out = 1 - unique_birthdays / sims
    return out

# Break out of the loop if probability greater than 0.5
while (people > 0):
    prop_bds = birthday_sim(people)
    if prop_bds > 0.5:
        break
    people += 1

print("With {} people, there's a 50% chance that two share a birthday.".format(people))
# With 23 people, there's a 50% chance that two share a birthday.

```



 23 seems surprisingly low, but it’s enough to have a 50% chance!



####
**Full house**



 Let’s return to our poker game. Last time, we calculated the probability of getting at least two of a kind. This time we are interested in a full house. A full house is when you get two cards of different suits that share the same numeric value and three other cards that have the same numeric value (e.g., 2 of hearts & spades, jacks of clubs, diamonds, & spades).




 Thus, a full house is the probability of getting exactly three of a kind conditional on getting exactly two of a kind of another value. Using the same code as before, modify the success condition to get the desired output. This exercise will teach you to estimate conditional probabilities in card games and build your foundation in framing abstract problems for simulation.





```

#Shuffle deck & count card occurrences in the hand
n_sims, full_house, deck_of_cards = 50000, 0, deck.copy()
for i in range(n_sims):
    np.random.shuffle(deck_of_cards)
    hand, cards_in_hand = deck_of_cards[0:5], {}
    for card in hand:
        # Use .get() method to count occurrences of each card
        cards_in_hand[card[1]] = cards_in_hand.get(card[1], 0) + 1

    # Condition for getting full house
    condition = (max(cards_in_hand.values()) ==3) & (min(cards_in_hand.values())==2)
    if  condition == True:
        full_house += 1
print("Probability of seeing a full house = {}".format(full_house/n_sims))

# Probability of seeing a full house = 0.0014

```



 Look at how small this probability is compared to that of at least two of a kind.





---


## **2.3 Data generating process**



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/4-2.png?w=851)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/5-2.png?w=975)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/6-2.png?w=907)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/7-2.png?w=805)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/8-2.png?w=903)



####
**Driving test**



 Through the next exercises, we will learn how to build a data generating process (DGP) through progressively complex examples.




 In this exercise, you will simulate a very simple DGP. Suppose that you are about to take a driving test tomorrow. Based on your own practice and based on data you have gathered, you know that the probability of you passing the test is 90% when it’s sunny and only 30% when it’s raining. Your local weather station forecasts that there’s a 40% chance of rain tomorrow. Based on this information, you want to know what is the probability of you passing the driving test tomorrow.




 This is a simple problem and can be solved analytically. Here, you will learn how to model a simple DGP and see how it can be used for simulation.





```

sims, outcomes, p_rain, p_pass = 1000, [], 0.40, {'sun':0.9, 'rain':0.3}

def test_outcome(p_rain):
    # Simulate whether it will rain or not
    weather = np.random.choice(['sun', 'rain'], p=[1-p_rain, p_rain])
    # Simulate and return whether you will pass or fail
    return np.random.choice(['pass', 'fail'], p=[p_pass[weather], 1-p_pass[weather]])


for _ in range(sims):
    outcomes.append(test_outcome(p_rain))

# Calculate fraction of outcomes where you pass
pass_outcomes_frac = outcomes.count('pass') / len(outcomes)
print("Probability of Passing the driving test = {}".format(pass_outcomes_frac))

# Probability of Passing the driving test = 0.654

```


####
**National elections**



 This exercise will give you a taste of how you can model a DGP at different levels of complexity.




 Consider national elections in a country with two political parties – Red and Blue. This country has 50 states and the party that wins the most states wins the elections. You have the probability pp of Red winning in each individual state and want to know the probability of Red winning nationally.




 Let’s model the DGP to understand the distribution. Suppose the election outcome in each state follows a binomial distribution with probability pp such that 00 indicates a loss for Red and 11 indicates a win. We then simulate a number of election outcomes. Finally, we can ask rich questions like
 **what is the probability of Red winning less than 45% of the states?**





```python

# probabilities of red win in a state
p
array([0.52076814, 0.67846401, 0.82731745, 0.64722761, 0.03665174,
...
       0.96263926, 0.0548948 , 0.14092758, 0.54739446, 0.54555576])

```




```

outcomes, sims, probs = [], 1000, p

for _ in range(sims):
    # Simulate elections in the 50 states
    election = np.random.binomial(p=probs, n=1)
    # Get average of Red wins and add to `outcomes`
    outcomes.append(np.sum(election)/len(election))

# Calculate probability of Red winning in less than 45% of the states
prob_red_wins = sum(x<0.45 for x in outcomes)/ len(outcomes)
print("Probability of Red winning in less than 45% of the states = {}".format(prob_red_wins))

# Probability of Red winning in less than 45% of the states = 0.196

```



 Now think about what you would do if you were given the probabilities of winning in each county within a state.



####
**Fitness goals**



 Let’s model how activity levels impact weight loss using modern fitness trackers. On days when you go to the gym, you average around 15k steps, and around 5k steps otherwise. You go to the gym 40% of the time. Let’s model the step counts in a day as a Poisson random variable with a mean λ dependent on whether or not you go to the gym.




 For simplicity, let’s say you have an 80% chance of losing 1lb and a 20% chance of gaining 1lb when you get more than 10k steps. The probabilities are reversed when you get less than 8k steps. Otherwise, there’s an even chance of gaining or losing 1lb. Given all this information, find the probability of losing weight in a month.





```python

# Simulate steps & choose prob
for _ in range(sims):
    w = []
    for i in range(days):
        lam = np.random.choice([5000, 15000], p=[0.6, 0.4], size=1)
        steps = np.random.poisson(lam)
        if steps > 10000:
            prob = [0.2, 0.8]
        elif steps < 8000:
            prob = [0.8, 0.2]
        else:
            prob = [0.5, 0.5]
        w.append(np.random.choice([1, -1], p=prob))
    outcomes.append(sum(w))

# Calculate fraction of outcomes where there was a weight loss
weight_loss_outcomes_frac = sum(x<0 for x in outcomes)/ len(outcomes)
print("Probability of Weight Loss = {}".format(weight_loss_outcomes_frac))

# Probability of Weight Loss = 0.215

```



 You now know how easy it is to break down complex DGPs into simple components.





---


## **2.4 eCommerce Ad Simulation**



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/1-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/2-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/3-3.png?w=1024)



####
**Sign up Flow**



 We will now model the DGP of an eCommerce ad flow starting with sign-ups.




 On any day, we get many ad impressions, which can be modeled as Poisson random variables (RV). You are told that λλ is normally distributed with a mean of 100k visitors and standard deviation 2000.




 During the signup journey, the customer sees an ad, decides whether or not to click, and then whether or not to signup. Thus both clicks and signups are binary, modeled using binomial RVs. What about probability pp of success? Our current low-cost option gives us a click-through rate of 1% and a sign-up rate of 20%. A higher cost option could increase the clickthrough and signup rate by up to 20%, but we are unsure of the level of improvement, so we model it as a uniform RV.





```python

# Initialize click-through rate and signup rate dictionaries
ct_rate = {'low':0.01, 'high':np.random.uniform(low=0.01, high=1.2*0.01)}
su_rate = {'low':0.2, 'high':np.random.uniform(low=0.2, high=1.2*0.2)}

def get_signups(cost, ct_rate, su_rate, sims):
    lam = np.random.normal(loc=100000, scale=2000, size=sims)
    # Simulate impressions(poisson), clicks(binomial) and signups(binomial)
    impressions = np.random.poisson(lam=lam)
    clicks = np.random.binomial(n=impressions, p=ct_rate[cost])
    signups = np.random.binomial(n=clicks, p=su_rate[cost])
    return signups

print("Simulated Signups = {}".format(get_signups('high', ct_rate, su_rate, 1)))
# Simulated Signups = [268]

```



 Now that we have signups, let’s see how to model the purchases from these signups.



####
**Purchase Flow**



 After signups, let’s model the revenue generation process. Once the customer has signed up, they decide whether or not to purchase – a natural candidate for a binomial RV. Let’s assume that 10% of signups result in a purchase.




 Although customers can make many purchases, let’s assume one purchase. The purchase value could be modeled by any continuous RV, but one nice candidate is the exponential RV. Suppose we know that purchase value per customer has averaged around $1000. We use this information to create the
 `purchase_values`
 RV. The revenue, then, is simply the sum of all purchase values.




 The variables
 `ct_rate`
 ,
 `su_rate`
 and the function
 `get_signups()`
 from the last exercise are pre-loaded for you.





```

def get_revenue(signups):
    rev = []
    np.random.seed(123)
    for s in signups:
        # Model purchases as binomial, purchase_values as exponential
        purchases = np.random.binomial(s, p=0.1)
        purchase_values = np.random.exponential(size=purchases, scale=1000)

        # Append to revenue the sum of all purchase values.
        rev.append(np.sum(purchase_values))
    return rev

print("Simulated Revenue = ${}".format(get_revenue(get_signups('low', ct_rate, su_rate, 1))[0]))

# Simulated Revenue = $22404.217742298042

```



 What are some other distributions you could have used in place of exponential?



####
**Probability of losing money**



 In this exercise, we will use the DGP model to estimate probability.




 As seen earlier, this company has the option of spending extra money, let’s say $3000, to redesign the ad. This could potentially get them higher clickthrough and signup rates, but this is not guaranteed. We would like to know whether or not to spend this extra $3000 by calculating the probability of losing money. In other words, the probability that the revenue from the high-cost option minus the revenue from the low-cost option is lesser than the cost.




 Once we have simulated revenue outcomes, we can ask a rich set of questions that might not have been accessible using traditional analytical methods.




 This simple yet powerful framework forms the basis of Bayesian methods for getting probabilities.





```python

# Initialize cost_diff
sims, cost_diff = 10000, 3000

# Get revenue when the cost is 'low' and when the cost is 'high'
rev_low = get_revenue(get_signups('low', ct_rate, su_rate, sims))
rev_high = get_revenue(get_signups('high', ct_rate, su_rate, sims))

# calculate fraction of times rev_high - rev_low is less than cost_diff
frac = np.sum(np.array(rev_high) - np.array(rev_low) < cost_diff) / len(rev_low)
print("Probability of losing money = {}".format(frac))

# Probability of losing money = 0.4659

```python

# **3. Resampling methods**
--------------------------


## **3.1 Introduction to resampling methods**



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/4-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/5-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/6-3.png?w=984)



####
**Probability example**



 In this exercise, we will review the difference between sampling with and without replacement. We will calculate the probability of an event using simulation, but vary our sampling method to see how it impacts probability.




 Consider a bowl filled with colored candies – three blue, two green, and five yellow. Draw three candies at random, with replacement and without replacement. You want to know the probability of
 **drawing a yellow candy on the third draw given that the first candy was blue and the second candy was green.**





```python

# Set up the bowl
success_rep, success_no_rep, sims = 0, 0, 10000
bowl = ['b'] * 3 + ['g'] * 2 + ['y'] * 5

for i in range(sims):
    # Sample with and without replacement & increment success counters
    sample_rep = np.random.choice(bowl, replace=True, size=3)
    sample_no_rep = np.random.choice(bowl, replace=False, size=3)
    if (sample_rep[0] == 'b') & (sample_rep[1] == 'g') & (sample_rep[2] == 'y'):
        success_rep += 1
    if (sample_no_rep[0] == 'b') & (sample_no_rep[1] == 'g') & (sample_no_rep[2] == 'y'):
        success_no_rep += 1

# Calculate probabilities
prob_with_replacement = success_rep / sims
prob_without_replacement = success_no_rep / sims
print("Probability with replacement = {}, without replacement = {}".format(prob_with_replacement, prob_without_replacement))

# Probability with replacement = 0.0266, without replacement = 0.0415

```



 Does the difference between sampling with and without replacement make sense now?





---


### **3.2 Bootstrapping**



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/7-3.png?w=648)

####
**Running a simple bootstrap**



 Welcome to the first exercise in the bootstrapping section. We will work through an example where we learn to run a simple bootstrap. As we saw in the video, the main idea behind bootstrapping is sampling with replacement.




 Suppose you own a factory that produces wrenches. You want to be able to characterize the average length of the wrenches and ensure that they meet some specifications. Your factory produces thousands of wrenches every day, but it’s infeasible to measure the length of each wrench. However, you have access to a representative sample of 100 wrenches. Let’s use bootstrapping to get the 95% confidence interval (CI) for the average lengths.




 Examine the list
 `wrench_lengths`
 , which has 100 observed lengths of wrenches, in the shell.





```python

# Draw some random sample with replacement and append mean to mean_lengths.
mean_lengths, sims = [], 1000
for i in range(sims):
    temp_sample = np.random.choice(wrench_lengths, replace=True, size=100)
    sample_mean = np.mean(temp_sample)
    mean_lengths.append(sample_mean)

# Calculate bootstrapped mean and 95% confidence interval.
boot_mean = np.mean(mean_lengths)
boot_95_ci = np.percentile(mean_lengths, [2.5, 97.5])
print("Bootstrapped Mean Length = {}, 95% CI = {}".format(boot_mean, boot_95_ci))

# Bootstrapped Mean Length = 10.027059690070363, 95% CI = [ 9.78662216 10.24854356]

```


####
**Non-standard estimators**



 In the last exercise, you ran a simple bootstrap that we will now modify for more complicated estimators.




 Suppose you are studying the health of students. You are given the height and weight of 1000 students and are interested in the median height as well as the correlation between height and weight and the associated 95% CI for these quantities. Let’s use bootstrapping.




 Examine the
 `pandas`
 DataFrame
 `df`
 with the heights and weights of 1000 students.
 *Using this, calculate the 95% CI for both the
 **median height**
 as well as the
 **correlation**
 between height and weight.*





```python

# Sample with replacement and calculate quantities of interest
sims, data_size, height_medians, hw_corr = 1000, df.shape[0], [], []
for i in range(sims):
    tmp_df = df.sample(n=data_size, replace=True)
    height_medians.append(np.median(tmp_df.heights))
    hw_corr.append(tmp_df.weights.corr(tmp_df.heights))

# Calculate confidence intervals
height_median_ci = np.percentile(height_medians, [2.5, 97.5])
height_weight_corr_ci = np.percentile(hw_corr, [2.5, 97.5])
print("Height Median CI = {} \nHeight Weight Correlation CI = {}".format( height_median_ci, height_weight_corr_ci))

# Height Median CI = [5.25262253 5.55928686]
# Height Weight Correlation CI = [0.93892136 0.95103152]

```


####
**Bootstrapping regression**



 Now let’s see how bootstrapping works with regression. Bootstrapping helps estimate the uncertainty of non-standard estimators. Consider the R2 statistic associated with a regression. When you run a simple least squares regression, you get a value for R2. But let’s see how can we get a 95% CI for R2.




 Examine the DataFrame
 `df`
 with a dependent variable yy and two independent variables X1 and X2 using
 `df.head()`
 . We’ve already fit this regression with
 `statsmodels`
 (
 `sm`
 ) using:





```

reg_fit = sm.OLS(df['y'], df.iloc[:,1:]).fit()

```



 Examine the result using
 `reg_fit.summary()`
 to find that R2=0.3504. Use bootstrapping to calculate the 95% CI.





```

df.head()
          y  Intercept        X1        X2
0  1.217851        1.0  0.696469  0.286139
1  1.555250        1.0  0.226851  0.551315
2  0.888520        1.0  0.719469  0.423106
3  1.736052        1.0  0.980764  0.684830
4  1.632073        1.0  0.480932  0.392118

```




```

rsquared_boot, coefs_boot, sims = [], [], 1000
reg_fit = sm.OLS(df['y'], df.iloc[:,1:]).fit()

# Run 1K iterations
for i in range(sims):
    # First create a bootstrap sample with replacement with n=df.shape[0]
    bootstrap = df.sample(n=df.shape[0], replace=True)
    # Fit the regression and append the r square to rsquared_boot
    rsquared_boot.append(sm.OLS(bootstrap['y'],bootstrap.iloc[:,1:]).fit().rsquared)

# Calculate 95% CI on rsquared_boot
r_sq_95_ci = np.percentile(rsquared_boot, [2.5, 97.5])
print("R Squared 95% CI = {}".format(r_sq_95_ci))

# R Squared 95% CI = [0.31089312 0.40543591]

```




---


## **3.3 Jackknife resampling**



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/8-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/9-2.png?w=1024)



####
**Basic jackknife estimation – mean**



 Jackknife resampling is an older procedure, which isn’t used as often compared as bootstrapping. However, it’s still useful to know how to run a basic jackknife estimation procedure. In this first exercise, we will calculate the jackknife estimate for the mean. Let’s return to the wrench factory.




 You own a wrench factory and want to measure the average length of the wrenches to ensure that they meet some specifications. Your factory produces thousands of wrenches every day, but it’s infeasible to measure the length of each wrench. However, you have access to a representative sample of 100 wrenches. Let’s use jackknife estimation to get the average lengths.




 Examine the variable
 `wrench_lengths`
 in the shell.





```python

# Leave one observation out from wrench_lengths to get the jackknife sample and store the mean length
mean_lengths, n = [], len(wrench_lengths)
index = np.arange(n)

for i in range(n):
    jk_sample = wrench_lengths[index != i]
    mean_lengths.append(np.mean(jk_sample))

# The jackknife estimate is the mean of the mean lengths from each sample
mean_lengths_jk = np.mean(np.array(mean_lengths))
print("Jackknife estimate of the mean = {}".format(mean_lengths_jk))

```


####
**Jackknife confidence interval for the median**



 In this exercise, we will calculate the jackknife 95% CI for a non-standard estimator. Here, we will look at the median. Keep in mind that the variance of a jackknife estimator is
 `n-1`
 times the variance of the individual jackknife sample estimates where
 `n`
 is the number of observations in the original sample.




 Returning to the wrench factory, you are now interested in estimating the median length of the wrenches along with a 95% CI to ensure that the wrenches are within tolerance.




 Let’s revisit the code from the previous exercise, but this time in the context of median lengths. By the end of this exercise, you will have a much better idea of how to use jackknife resampling to calculate confidence intervals for non-standard estimators.





```

n = 100
# Leave one observation out to get the jackknife sample and store the median length
median_lengths = []
for i in range(n):
    jk_sample = wrench_lengths[index != i]
    median_lengths.append(np.median(jk_sample))

median_lengths = np.array(median_lengths)

# Calculate jackknife estimate and it's variance
jk_median_length = np.mean(median_lengths)
jk_var = (n-1)*np.var(median_lengths)

# Assuming normality, calculate lower and upper 95% confidence intervals
jk_lower_ci = jk_median_length - 1.96*np.sqrt(jk_var)
jk_upper_ci = jk_median_length + 1.96*np.sqrt(jk_var)
print("Jackknife 95% CI lower = {}, upper = {}".format(jk_lower_ci, jk_upper_ci))

# Jackknife 95% CI lower = 9.138592467547202, upper = 10.754868069037098

```




---


## **3.4 Permutation testing**



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/11.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/12.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/13.png?w=1024)



####
**Generating a single permutation**



 In the next few exercises, we will run a significance test using permutation testing. As discussed in the video, we want to see if there’s any difference in the donations generated by the two designs – A and B. Suppose that you have been running both the versions for a few days and have generated 500 donations on A and 700 donations on B, stored in the variables
 `donations_A`
 and
 `donations_B`
 .




 We first need to generate a null distribution for the difference in means. We will achieve this by generating multiple permutations of the dataset and calculating the difference in means for each case.




 First, let’s generate one permutation and calculate the difference in means for the permuted dataset.





```python

# Concatenate the two arrays donations_A and donations_B into data
len_A, len_B = len(donations_A), len(donations_B)
data = np.concatenate([donations_A, donations_B])

# Get a single permutation of the concatenated length
perm = np.random.permutation(len(donations_A) + len(donations_B))

# Calculate the permutated datasets and difference in means
permuted_A = data[perm[:len(donations_A)]]
permuted_B = data[perm[len(donations_A):]]
diff_in_means = np.mean(permuted_A) - np.mean(permuted_B)
print("Difference in the permuted mean values = {}.".format(diff_in_means))

# Difference in the permuted mean values = -0.13886241452516757.

```


####
**Hypothesis testing – Difference of means**



 We want to test the hypothesis that there is a difference in the average donations received from A and B. Previously, you learned how to generate one permutation of the data. Now, we will generate a null distribution of the difference in means and then calculate the p-value.




 For the null distribution, we first generate multiple permuted datasets and store the difference in means for each case. We then calculate the test statistic as the difference in means with the original dataset. Finally, we calculate the p-value as twice the fraction of cases where the difference is greater than or equal to the absolute value of the test statistic (2-sided hypothesis). A p-value of less than say 0.05 could then determine statistical significance.





```

perm
array([[ 850,  473, 1067, ...,  592,  644,  944],
       [ 594,  535,  125, ..., 1198, 1052,  233],
       [ 494,  246,  809, ...,  179,  448,  953],
       ...,
       [ 923,  888,  393, ...,  314,   40,  258],
       [ 915, 1140,  953, ...,   20,  526,  272],
       [ 925,  480, 1102, ...,  371,   85,  379]])

permuted_A_datasets
array([[2.49959605e+00, 1.94959571e+00, 4.10970395e+00, ...,
        1.28042313e+00, 3.09147765e+00, 3.00804073e-01],
       ...,
       [6.33883845e-01, 5.64755610e-01, 1.75817616e+00, ...,
        5.46401636e-01, 1.97500056e+00, 5.43297027e+00]])

reps
1000

```




```python

# Generate permutations equal to the number of repetitions
perm = np.array([np.random.permutation(len(donations_A) + len(donations_B)) for i in range(reps)])
permuted_A_datasets = data[perm[:, :len(donations_A)]]
permuted_B_datasets = data[perm[:, len(donations_A):]]

# Calculate the difference in means for each of the datasets
samples = np.mean(permuted_A_datasets, axis=1) - np.mean(permuted_B_datasets, axis=1)

# Calculate the test statistic and p-value
test_stat = np.mean(donations_A) - np.mean(donations_B)
p_val = 2*np.sum(samples >= np.abs(test_stat))/reps
print("p-value = {}".format(p_val))

# p-value = 0.002

```


####
**Hypothesis testing – Non-standard statistics**



 In the previous two exercises, we ran a permutation test for the difference in mean values. Now let’s look at non-standard statistics.




 Suppose that you’re interested in understanding the distribution of the donations received from websites A and B. For this, you want to see if there’s a statistically significant difference in the median and the 80th percentile of the donations. Permutation testing gives you a wonderfully flexible framework for attacking such problems.




 Let’s go through running a test to see if there’s a difference in the median and the 80th percentile of the distribution of donations. As before, you’re given the donations from the websites A and B in the variables
 `donations_A`
 and
 `donations_B`
 respectively.





```python

# Calculate the difference in 80th percentile and median for each of the permuted datasets (A and B)
samples_percentile = np.percentile(permuted_A_datasets, 80, axis=1) - np.percentile(permuted_B_datasets, 80, axis=1)
samples_median = np.median(permuted_A_datasets, axis=1) - np.median(permuted_B_datasets, axis=1)

# Calculate the test statistic from the original dataset and corresponding p-values
test_stat_percentile = np.percentile(donations_A, 80) - np.percentile(donations_B, 80)
test_stat_median = np.median(donations_A) - np.median(donations_B)

p_val_percentile = 2*np.sum(samples_percentile >= np.abs(test_stat_percentile))/reps
p_val_median = 2*np.sum(samples_median >= np.abs(test_stat_median))/reps

print("80th Percentile: test statistic = {}, p-value = {}".format(test_stat_percentile, p_val_percentile))
print("Median: test statistic = {}, p-value = {}".format(test_stat_median, p_val_median))

```




```python

# 80th Percentile: test statistic = 1.6951624543447839, p-value = 0.026
# Median: test statistic = 0.6434965714975927, p-value = 0.014

```python

# **4. Advanced Applications of Simulation**
-------------------------------------------


## **4.1 Simulation for Business Planning**



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/1-5.png?w=582)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/2-6.png?w=989)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/3-4.png?w=1024)



####
**Modeling Corn Production**



 Suppose that you manage a small corn farm and are interested in optimizing your costs. In this exercise, we will model the production of corn.




 For simplicity, let’s assume that corn production depends on only two factors: rain, which you don’t control, and cost, which you control. Rain is normally distributed with mean 50 and standard deviation 15. For now, let’s fix cost at 5,000. Corn produced in any season is a Poisson random variable while the average corn production is governed by the equation:




 100×(cost)^0.1×(rain)^0.2




 Let’s model this production function and simulate one outcome.





```python

# Initialize variables
cost = 5000
rain = np.random.normal(loc=50, scale=15)

# Corn Production Model
def corn_produced(rain, cost):
  mean_corn = 100 * cost ** 0.1 * rain ** 0.2
  corn = np.random.poisson(mean_corn)
  return corn

# Simulate and print corn production
corn_result = corn_produced(rain, cost)
print("Simulated Corn Production = {}".format(corn_result))

# Simulated Corn Production = 560

```


####
**Modeling Profits**



 In the previous exercise, you built a model of corn production. For a small farm, you typically have no control over the price or demand for corn. Suppose that price is normally distributed with mean 40 and standard deviation 10. You are given a function
 `corn_demanded()`
 , which takes the price and determines the demand for corn. This is reasonable because demand is usually determined by the market and is not in your control.




 In this exercise, you will work on a function to calculate the profit by pulling together all the other simulated variables. The only input to this function will be the cost. Upon completion, you will have a function that will give you one simulated profit outcome for a given cost. This function can then be used for planning your costs.





```python

# Function to calculate profits
def profits(cost):
    rain = np.random.normal(50, 15)
    price = np.random.normal(40, 10)
    supply = corn_produced(rain, cost)
    demand = corn_demanded(price)
    equil_short = supply <= demand
    if equil_short == True:
        tmp = supply*price - cost
        return tmp
    else:
        tmp2 = demand*price - cost
        return tmp2
result = profits(cost)
print("Simulated profit = {}".format(result))

# Simulated profit = 20675.3291075312

```


####
**Optimizing Costs**



 Now we will use the functions you’ve built to optimize our cost of production. We are interested in maximizing average profits. However, our profits depend on a number of factors, but we only control cost. Thus, we can simulate the uncertainty in the other factors and vary cost to see how our profits are impacted.




 Since you manage the small corn farm, you have the ability to choose your cost – from $100 to $5,000. You want to choose the cost that gives you the maximum average profit. In this exercise, we will simulate multiple outcomes for each cost level and calculate an average. We will then choose the cost that gives us the maximum mean profit. Upon completion, you will have a framework for selecting optimal inputs for business decisions.





```python

# Initialize results and cost_levels variables
sims, results = 1000, {}
cost_levels = np.arange(100, 5100, 100)

# For each cost level, simulate profits and store mean profit
for cost in cost_levels:
    tmp_profits = []
    for i in range(sims):
        tmp_profits.append(profits(cost))
    results[cost] = np.mean(tmp_profits)

# Get the cost that maximizes average profit
cost_max = [x for x in results.keys() if results[x] == max(results.values())][0]
print("Average profit is maximized when cost = {}".format(cost_max))

# Average profit is maximized when cost = 1400

```



 Businesses use a similar framework with more details to help in a number of decisions.



## **4.2 Monte Carlo Integration**



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/4-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/5-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/6-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/7-4.png?w=1024)



####
**Integrating a Simple Function**



 This is a simple exercise introducing the concept of Monte Carlo Integration.




 Here we will evaluate a simple integral ∫10xexdx∫01xexdx. We know that the exact answer is 11, but simulation will give us an approximate solution, so we can expect an answer close to 11. As we saw in the video, it’s a simple process. For a function of a single variable f(x):



1. Get the limits of the x-axis (xmin,xmax) and y-axis (max(f(x)),min(min(f(x)),0)).
2. Generate a number of uniformly distributed point in this box.
3. Multiply the area of the box ((max(f(x)−min(f(x))×(xmax−xmin) by the fraction of points that lie below f(x).



 Upon completion, you will have a framework for handling definite integrals using Monte Carlo Integration.





```python

# Define the sim_integrate function
def sim_integrate(func, xmin, xmax, sims):
    x = np.random.uniform(xmin, xmax, sims)
    y = np.random.uniform(min(min(func(x)), 0), max(func(x)), sims)
    area = (max(y) - min(y))*(xmax-xmin)
    result = area * sum(abs(y) < abs(func(x)))/sims
    return result

# Call the sim_integrate function and print results
result = sim_integrate(func = lambda x: x * np.exp(x), xmin = 0, xmax = 1, sims = 50)
print("Simulated answer = {}, Actual Answer = 1".format(result))

# Simulated answer = 0.7240166789450252, Actual Answer = 1

```



 Try seeing what happens to the answer when you increase or decrease
 `sims`
 .



####
**Calculating the value of pi**



 Now we work through a classic example – estimating the value of π.




 Imagine a square of side 2 with the origin (0,0) as its center and the four corners having coordinates (1,1),(1,−1),(−1,1),(−1,−1). The area of this square is 2×2=4. Now imagine a circle of radius 1 with its center at the origin fitting perfectly inside this square. The area of the circle will be π×radius^2=π.




 To estimate π, we randomly sample multiple points in this square & get the fraction of points inside the circle (x^2+y^2<=1). The area of the circle then is 4 times this fraction, which gives us our estimate of π.




 After this exercise, you’ll have a grasp of how to use simulation for computation.





```python

# Initialize sims and circle_points
sims, circle_points = 10000, 0

for i in range(sims):
    # Generate the two coordinates of a point
    point = np.random.uniform(-1,1,size=2)
    # if the point lies within the unit circle, increment counter
    within_circle = point[0]**2 + point[1]**2 <= 1
    if within_circle == True:
        circle_points +=1

# Estimate pi as 4 times the avg number of points in the circle.
pi_sim = 4*circle_points/sims
print("Simulated value of pi = {}".format(pi_sim))

# Simulated value of pi = 3.1468

```




---


## **4.3 Simulation for Power Analysis**



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/1-7.png?w=832)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/2-8.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/3-6.png?w=1024)



####
**Factors influencing Statistical Power**



 In this exercise, you will refresh some basic concepts to test your understanding of statistical power. It is very important to understand statistical power, especially if you are designing an
 [A/B test](https://en.wikipedia.org/wiki/A/B_testing)
 .




 Consider the following four options and select the factor that influence the statistical power of an experiment:



* Magnitude of the effect / Effect size
* Sample Size
* Statistical Significance Level (α)



 Note: The number of simulations doesn’t really impact the statistical power of an experiment.



####
**Power Analysis – Part I**



 Now we turn to power analysis. You typically want to ensure that any experiment or A/B test you run has at least 80% power. One way to ensure this is to calculate the sample size required to achieve 80% power.




 Suppose that you are in charge of a news media website and you are interested in increasing the amount of time users spend on your website. Currently, the time users spend on your website is normally distributed with a mean of 1 minute and a variance of 0.5 minutes. Suppose that you are introducing a feature that loads pages faster and want to know the sample size required to measure a 10% increase in time spent on the website.




 In this exercise, we will set up the framework to run one simulation, run a t-test, & calculate the p-value.





```

import scipy.stats as st
# Initialize effect_size, control_mean, control_sd
effect_size, sample_size, control_mean, control_sd = 0.05, 50, 1, 0.5

# Simulate control_time_spent and treatment_time_spent, assuming equal variance
control_time_spent = np.random.normal(loc=control_mean, scale=control_sd, size=sample_size)
treatment_time_spent = np.random.normal(loc=control_mean*(1+effect_size), scale=control_sd, size=sample_size)

# Run the t-test and get the p_value
t_stat, p_value = st.ttest_ind(control_time_spent, treatment_time_spent)
stat_sig = p_value < 0.05
print("P-value: {}, Statistically Significant? {}".format(p_value, stat_sig))

# P-value: 0.5766409395002308, Statistically Significant? False

```


####
**Power Analysis – Part II**



 Previously, we simulated one instance of the experiment & generated a p-value. We will now use this framework to calculate statistical power. Power of an experiment is the experiment’s ability to detect a difference between treatment & control if the difference really exists. It’s good statistical hygiene to strive for 80% power.




 For our website, we want to know how many people need to visit each variant, such that we can detect a 10% increase in time spent with 80% power. For this, we start with a small sample (50), simulate multiple instances of this experiment & check power. If 80% power is reached, we stop. If not, we increase the sample size & try again.





```

sample_size = 50

# Keep incrementing sample size by 10 till we reach required power
while 1:
    control_time_spent = np.random.normal(loc=control_mean, scale=control_sd, size=(sample_size, sims))
    treatment_time_spent = np.random.normal(loc=control_mean*(1+effect_size), scale=control_sd, size=(sample_size, sims))
    t, p = st.ttest_ind(treatment_time_spent, control_time_spent)

    # Power is the fraction of times in the simulation when the p-value was less than 0.05
    power = (p < 0.05).sum()/sims
    if power >= 0.8:
        break
    else:
        sample_size += 10
print("For 80% power, sample size required = {}".format(sample_size))

# For 80% power, sample size required = 360

```




---


## **4.4 Applications in Finance**



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/4-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/5-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/6-5.png?w=1024)



####
**Portfolio Simulation – Part I**



 In the next few exercises, you will calculate the expected returns of a stock portfolio & characterize its uncertainty.




 Suppose you have invested $10,000 in your portfolio comprising of multiple stocks. You want to evaluate the portfolio’s performance over 10 years. You can tweak your overall expected rate of return and volatility (standard deviation of the rate of return). Assume the rate of return follows a normal distribution.




 First, let’s write a function that takes the principal (initial investment), number of years, expected rate of return and volatility as inputs and returns the portfolio’s total value after 10 years.




 Upon completion of this exercise, you will have a function you can call to determine portfolio performance.





```python

# rates is a Normal random variable and has size equal to number of years
def portfolio_return(yrs, avg_return, sd_of_return, principal):
    np.random.seed(123)
    rates = np.random.normal(loc=avg_return, scale=sd_of_return, size=yrs)
    # Calculate the return at the end of the period
    end_return = principal
    for x in rates:
        end_return = end_return * (1 + x)
    return end_return

result = portfolio_return(yrs = 5, avg_return = 0.07, sd_of_return = 0.15, principal = 1000)
print("Portfolio return after 5 years = {}".format(result))

# Portfolio return after 5 years = 1021.4013412039292

```


####
**Portfolio Simulation – Part II**



 Now we will use the simulation function you built to evaluate 10-year returns.




 Your stock-heavy portfolio has an initial investment of $10,000, an expected return of 7% and a volatility of 30%. You want to get a 95% confidence interval of what your investment will be worth in 10 years. We will simulate multiple samples of 10-year returns and calculate the confidence intervals on the distribution of returns.




 By the end of this exercise, you will have run a complete portfolio simulation.




 The function
 `portfolio_return()`
 from the previous exercise is already initialized in the environment.





```python

# Run 1,000 iterations and store the results
sims, rets = 1000, []

for i in range(sims):
    rets.append(portfolio_return(yrs = 10, avg_return = 0.07,
                                 volatility = 0.3, principal = 10000))

# Calculate the 95% CI
lower_ci = np.percentile(rets, 2.5)
upper_ci = np.percentile(rets, 97.5)
print("95% CI of Returns: Lower = {}, Upper = {}".format(lower_ci, upper_ci))

# 95% CI of Returns: Lower = 1236.4468015417674, Upper = 79510.31743325583

```


####
**Portfolio Simulation – Part III**



 Previously, we ran a complete simulation to get a distribution for 10-year returns. Now we will use simulation for decision making.




 Let’s go back to your stock-heavy portfolio with an expected return of 7% and a volatility of 30%. You have the choice of rebalancing your portfolio with some bonds such that the expected return is 4% & volatility is 10%. You have a principal of $10,000. You want to select a strategy based on how much your portfolio will be worth in 10 years. Let’s simulate returns for both the portfolios and choose based on the least amount you can expect with 75% probability (25th percentile).




 Upon completion, you will know how to use a portfolio simulation for investment decisions.




 The
 `portfolio_return()`
 function is again pre-loaded in the environment.





```

for i in range(sims):
    rets_stock.append(portfolio_return(yrs = 10, avg_return = 0.07, volatility = 0.3, principal = 10000))
    rets_bond.append(portfolio_return(yrs = 10, avg_return = 0.04, volatility = 0.1, principal = 10000))

# Calculate the 25th percentile of the distributions and the amount you'd lose or gain
rets_stock_perc = np.percentile(rets_stock, 25)
rets_bond_perc = np.percentile(rets_bond, 25)
additional_returns = rets_stock_perc - rets_bond_perc
print("Sticking to stocks gets you an additional return of {}".format(additional_returns))

# Sticking to stocks gets you an additional return of -5518.530403193416

```


####
**Summary**



![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/7-6.png?w=716)
![Desktop View]({{ site.baseurl }}/assets/datacamp/statistical-simulation-in-python/8-5.png?w=824)




 Thank you for reading and hope you’ve learned a lot.



