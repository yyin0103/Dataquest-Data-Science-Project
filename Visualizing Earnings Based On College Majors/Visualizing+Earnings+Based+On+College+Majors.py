#!/usr/bin/env python
# coding: utf-8

# # Visualizing Earnings Based On College Majors
# 
# In this project, we'll be working with a dataset on job outcomes of studentss who graduated from coleege between 2010 aand 2012. The original data on job outcomes was released by American Community Survey, FiveThirtyEight cleaned the dataset and released it on their Github repo.
# 
# Here are some columns in the dataset:
# * Rank - Rank by median earnings (the dataset is ordered by this column).
# * Major_code - Major code.
# * Major - Major description.
# * Major_category - Category of major.
# * Total - Total number of people with major.
# * Sample_size - Sample size (unweighted) of full-time.
# * Men - Male graduates.
# * Women - Female graduates.
# * ShareWomen - Women as share of total.
# * Employed - Number employed.
# * Median - Median salary of full-time, year-round workers.
# * Low_wage_jobs - Number in low-wage service jobs.
# * Full_time - Number employed 35 hours or more.
# * Part_time - Number employed less than 35 hours.
# 
# Our goal is to determine:
# 1. Do students in more popular majors make more money?
# 2. How many majors are predominantly male? Predominantly female?
# 3. Which category of majors have the most students?

# ## Understanding data

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


#check the column labels and a few rows to see how the data is structured
recent_grads = pd.read_csv('recent-grads.csv')
recent_grads.info()


# In[3]:


recent_grads.head()


# In[4]:


#Use dataframe.describe() to generate the summary statistics
recent_grads.describe()


# In[5]:


#drop the rows containing missing values (one row)
recent_grads = recent_grads.dropna(axis=0)


# ## Scatter plots

# In[17]:


recent_grads.plot(x='Total', y='Median', kind='scatter')


# In[7]:


recent_grads.plot(x='Sample_size', y='Unemployment_rate', kind='scatter')


# In[8]:


recent_grads.plot(x='Full_time', y='Median', kind='scatter')


# In[9]:


recent_grads.plot(x='ShareWomen', y='Unemployment_rate', kind='scatter')


# In[10]:


recent_grads.plot(x='Men', y='Median', kind='scatter')


# In[11]:


recent_grads.plot(x='Women', y='Median', kind='scatter')


# ## Histogram

# In[18]:


cols = ["Sample_size", "Median", "Employed", "Full_time", "ShareWomen", "Unemployment_rate", "Men", "Women"]

fig = plt.figure(figsize=(5,12))
for r in range(0,4):
    ax = fig.add_subplot(4,1,r+1)
    ax = recent_grads[cols[r]].plot(kind='hist', rot=30)


# cols = ["Sample_size", "Median", "Employed", "Full_time", "ShareWomen", "Unemployment_rate", "Men", "Women"]
# 
# fig = plt.figure(figsize=(5,12))
# for r in range(4,8):
#     ax = fig.add_subplot(4,1,r-3)
#     ax = recent_grads[cols[r]].plot(kind='hist', rot=30)

# ## Scatter Matrix Plot

# In[14]:


from pandas.plotting import scatter_matrix
scatter_matrix(recent_grads[['Sample_size', 'Median']], figsize=(10,10))


# In[15]:


scatter_matrix(recent_grads[['Sample_size', 'Median','Unemployment_rate']], figsize=(15,15))


# ## Bar Plots

# In[21]:


recent_grads[:10].plot.bar(x='Major', y='ShareWomen', legend=False)
recent_grads[163:].plot.bar(x='Major', y='ShareWomen', legend=False)
plt.show()


# #### Do students in more popular majors make more money?
# Students in more popular majors actually made less money.
# 
# #### Do students that majored in subjects that were majority female make more money?
# No, we can't see the relevance.
# 
# #### Is there any link between the number of full-time employees and median salary?
# The more number of full-time emplyees, the less median salary.
