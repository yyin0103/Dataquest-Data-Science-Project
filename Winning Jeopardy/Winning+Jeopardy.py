#!/usr/bin/env python
# coding: utf-8

# # Winning Jeopardy
# 
# Jeopardy is a popular TV show in the US where participants answer questions to win money. It's been running for a few decades, and is a major force in popular culture.
# 
# In this project, we'll work with a dataset of Jeopardy questions to figure out some patterns in the questions that could help one win.
# 
# Here are explanations of each column:
# 
# * Show Number -- the Jeopardy episode number of the show this question was in.
# * Air Date -- the date the episode aired.
# * Round -- the round of Jeopardy that the question was asked in. Jeopardy has several rounds as each episode progresses.
# * Category -- the category of the question.
# * Value -- the number of dollars answering the question correctly is worth.
# * Question -- the text of the question.
# * Answer -- the text of the answer.

# ## Reorganize the data

# In[1]:


import pandas as pd
jeopardy = pd.read_csv('jeopardy.csv')
print(jeopardy.head())


# In[2]:


# fix the column names that have spaces in front
jeopardy.columns = jeopardy.columns.str.strip()
print(jeopardy.columns)


# There's a need to normalize the questions and answers. We can lowercase words and remove punctuation.

# In[3]:


import re
def normalized_text(string):
    string = str.lower(string)
    string = re.sub('[^A-Za-z0-9]+', ' ', string)
    return string

jeopardy['clean_question'] = jeopardy['Question'].apply(normalized_text)
jeopardy['clean_answer'] = jeopardy['Answer'].apply(normalized_text)


# Write a function to normalize dollar values.

# In[4]:


def normalize_values(text):
    text = re.sub("[^A-Za-z0-9\s]", "", text)
    try:
        text = int(text)
    except Exception:
        text = 0
    return text

jeopardy['clean_value'] = jeopardy['Value'].apply(normalize_values)


# Convert the Air Date column to a datetime column.

# In[5]:


import datetime as dt
jeopardy['Air Date'] = pd.to_datetime(jeopardy['Air Date'],format='%Y-%m-%d')


# In[6]:


jeopardy.head()


# In order to figure out whether to study past questions, study general knowledge, or not study it all, it would be helpful to figure out two things:
# 
# * How often the answer is deducible from the question?
# * How often new questions are repeats of older questions?

# ## Answers deducible from questions
# 
# To figure out how often the answer is deducible from the question, we can see how many times words in the answer also occur in the question.

# In[7]:


def repeat_words(row):
    split_answer = row['clean_answer'].split()
    split_question = row['clean_question'].split()
    
    match_count = 0
    
    if 'the' in split_answer:
        split_answer.remove('the')
    
    if len(split_answer) == 0:
        return 0
    
    for word in split_answer:
        if word in split_question:
            match_count += 1
    
    return match_count / len(split_answer)


# In[8]:


answer_in_question = jeopardy.apply(repeat_words, axis=1)
print(answer_in_question.mean())


# Only around 6% of the answers are implied in the questions.

# ##  Recycled Questions

# In[9]:


# sort jeopardy in order of ascending air date
jeopardy.sort_values('Air Date', ascending=True, inplace=True)
jeopardy.head()


# In[10]:


#Investigate how often new questions are repeats of older ones.
question_overlap = []
terms_used = set()

for i, row in jeopardy.iterrows():
    split_question = row['clean_question'].split(' ')
    split_question = [q for q in split_question if len(q) > 5]

    match_count = 0
    for word in split_question:
        if word in terms_used:
            match_count += 1        
        terms_used.add(word)
        
    if len(split_question) > 0:
        p_match = match_count / len(split_question)
    
    question_overlap.append(p_match)

jeopardy['question_overlap'] = question_overlap

print(jeopardy['question_overlap'].mean())
print(len(terms_used))


# ## High-value questions
# 
# Studying questions that pertain to high value questions instead of low value questions will help one earn more money. We can figure out which terms correspond to high-value questions using a chi-squared test.
# 
# First, we need to narrow down the questions into two categories:
# 
# * Low value -- Any row where Value is less than 800
# * High value -- Any row where Value is greater than 800

# In[11]:


def is_high_value(row):
    value = 0
    
    if row['clean_value'] > 800:
        value = 1
    
    return value

jeopardy['high_value'] = jeopardy.apply(is_high_value, axis=1)


# In[12]:


def count_usage(word):
    low_count = 0
    high_count = 0
    
    for i,row in jeopardy.iterrows():
        split_question = row['clean_question'].split(' ')
        if word in split_question:
            if row['high_value'] == 1:
                high_count += 1
            else:
                low_count += 1
    
    return high_count, low_count 


# In[13]:


import random
comparison_terms = random.sample(terms_used, 10)

observed_expected = []
for term in comparison_terms:
    counts = count_usage(term)
    observed_expected.append(counts)

print(observed_expected)


# ## Expected counts and chi-squared value

# In[14]:


from scipy.stats import chisquare
import numpy as np

high_value_count = len(jeopardy[jeopardy['high_value'] == 1])
low_value_count = len(jeopardy[jeopardy['high_value'] == 0])

chi_squared = []

for obs in observed_expected:
    total = sum(obs)
    total_prop = total / len(jeopardy)
    high_value_exp = total_prop * high_value_count
    low_value_exp = total_prop * low_value_count
    
    observed = np.array([obs[0], obs[1]])
    expected = np.array([high_value_exp, low_value_exp])
    chi_squared.append(chisquare(observed, expected))
    
print(chi_squared)


# There isn't a significant difference in usage between high value and low value rows. Additionally, the frequencies were all lower than 5, so the chi-squared test isn't as valid. 
