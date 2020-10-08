#!/usr/bin/env python
# coding: utf-8

# ## Mobile App for Lottery Addiction
# 
# A medical institute that aims to prevent and treat gambling addictions wants to build a dedicated mobile app to help lottery addicts better estimate their chances of winning. The institute has a team of engineers that will build the app, but they need us to create the logical core of the app and calculate probabilities.
# 
# They want us to focus on the 6/49 lottery and build functions that enable users to answer questions like:
# 
# * What is the probability of winning the big prize with a single ticket?
# * What is the probability of winning the big prize if we play 40 different tickets (or any other number)?
# * What is the probability of having at least five (or four, or three, or two) winning numbers on a single ticket?
# 
# In this project, we'll try to figure out the questions from the historical data of the national 6/49 lottery game in Canada. The data set has data for 3,665 drawings, dating from 1982 to 2018.
# 
# ## Create required functions

# In[1]:


# create two functions that calculates factorial and combinations
def factorial(n):
    result = 1
    for i in range(1,n+1):
        result *= i
    
    return result

print(factorial(5))

def combinations(n,k):
    return factorial(n) / (factorial(k)*factorial(n-k))

print(combinations(5,3))


# ## One ticket probability
# For the first version of the app, we want players to be able to calculate the probability of winning the big prize with the various numbers they play on a single ticket (for each ticket a player chooses six numbers out of 49). So, we'll start by building a function that calculates the probability of winning the big prize for any given ticket.
# 
# We discussed with the engineering team of the medical institute, and they told us we need to be aware of the following details when we write the function:
# 
# * Inside the app, the user inputs six different numbers from 1 to 49.
# * Under the hood, the six numbers will come as a Python list, which will serve as the single input to our function.
# * The engineering team wants the function to print the probability value in a friendly way — in a way that people without any probability training are able to understand.

# In[2]:


def one_ticket_probability(user_numbers):
    
    # the total number of combinations
    # There are 49 possible numbers
    # and six numbers are sampled without replacement
    n_combinations = combinations(49, 6)
    probability = 1 / n_combinations * 100
    
    print('''Your chances to win the big prize with the numbers {} are {:.7f}%.
In other words, you have a 1 in {:,} chances to win.'''.format(user_numbers,
                    probability, int(n_combinations)))

# test
one_ticket_probability([1,2,3,4,5,6])


# ## Compare tickets against the historical data
# 
# Users should also be able to compare their ticket against the historical lottery data in Canada and determine whether they would have ever won by now. We'll need to explore the historical data coming from the Canada 6/49 lottery. 

# In[3]:


import pandas as pd

history = pd.read_csv('649.csv')
history.shape


# In[4]:


history.head(3)


# In[5]:


history.tail(3)


# To enable users to compare their ticket against the historical lottery data. we need to create a new function.
# 
# The function should prints:
# 
# * the number of times the combination selected occurred in the Canada data set; and
# * the probability of winning the big prize in the next drawing with that combination.
# 
# We need to note that, inside the app, the user inputs six different numbers from 1 to 49. Under the hood, the six numbers will come as a Python list and serve as an input to our function.

# In[6]:


# write a function that takes as input a row of the lottery dataframe 
# and returns a set containing all the six winning numbers

def extract_numbers(row):
    
    row = row[4:10]
    row = set(row.values)
    return row

# extract all the winning numbers
winning_set = history.apply(extract_numbers, axis=1)
winning_set.head()


# In[7]:


def check_historical_occurence(user_numbers, history_set):

    user_set = set(user_numbers)
    occurrence = user_set == history_set
    n_occurrences = occurrence.sum()
    
    if n_occurrences == 0:
        
        print('''The combination {} has never occured.
This doesn't mean it's more likely to occur now. Your chances to win the big prize in the next drawing using the combination {} are 0.0000072%.
In other words, you have a 1 in 13,983,816 chances to win.'''.format(user_numbers, user_numbers))
        
    else:
        print('''The number of times combination {} has occured in the past is {}.
Your chances to win the big prize in the next drawing using the combination {} are 0.0000072%.
In other words, you have a 1 in 13,983,816 chances to win.'''.format(user_numbers, n_occurrences, user_numbers))

        
#test 
check_historical_occurence([6,7,15,20,45,48], winning_set)


# ## Multi-ticket probability
# 
# Lottery addicts usually play more than one ticket on a single drawing, thinking that this might increase their chances of winning significantly. Our purpose is to help them better estimate their chances of winning. Hence, we're going to write a function that will allow the users to calculate the chances of winning for any number of different tickets.
# 
# * The user will input the number of different tickets they want to play (without inputting the specific combinations they intend to play).
# * Our function will see an integer between 1 and 13,983,816 (the maximum number of different tickets).
# * The function should print information about the probability of winning the big prize depending on the number of different tickets played.

# In[8]:


def multi_ticket_probability(n_tickets):
    
    n_combinations = combinations(49,6)
    probability = (n_tickets / n_combinations) * 100
    
    if n_tickets == 1:
        print('''Your chances to win the big prize with one ticket are {:.6f}%.
In other words, you have a 1 in {:,} chances to win.'''.format(probability, int(n_combinations)))
    
    else:
        combinations_simplified = round(n_combinations / n_tickets)   
        print('''Your chances to win the big prize with {:,} different tickets are {:.6f}%.
In other words, you have a 1 in {:,} chances to win.'''.format(n_tickets, probability,
                                                               combinations_simplified))


test_inputs = [1, 10, 100, 10000, 1000000, 6991908, 13983816]
for num in test_inputs:
    multi_ticket_probability(num)


# ## Probabilities for two, three, four, or five winning numbers
# 
# In most 6/49 lotteries there are smaller prizes if a player's ticket match two, three, four, or five of the six numbers drawn. As a consequence, the users might be interested in knowing the probability of having two, three, four, or five winning numbers.
# 
# We need to create a function to allow the users to calculate probabilities for two, three, four, or five winning numbers.
# 
# * the user inputs six different numbers from 1 to 49; and an integer between 2 and 5 that represents the number of winning numbers expected
# * Our function prints information about the probability of having the inputted number of winning numbers.

# In[9]:


def probability_less_6(n_winning_numbers):
    
    n_combinations_ticket = combinations(6, n_winning_numbers)
    n_combinations_remaining = combinations(43, 6 - n_winning_numbers)
    successful_outcomes = n_combinations_ticket * n_combinations_remaining
    
    n_combinations_total = combinations(49, 6)    
    probability = successful_outcomes / n_combinations_total
    
    probability_percentage = probability * 100    
    combinations_simplified = round(n_combinations_total/successful_outcomes)    
    print('''Your chances of having {} winning numbers with this ticket are {:.6f}%.
In other words, you have a 1 in {:,} chances to win.'''.format(n_winning_numbers, probability_percentage,
                                                               int(combinations_simplified)))


# In[10]:


# test 
for n in {2,3,4,5}:
    probability_less_6(n)

