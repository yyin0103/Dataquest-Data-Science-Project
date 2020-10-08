#!/usr/bin/env python
# coding: utf-8

# # Predicting Bike Rentals
# 
# Many American cities have communal bike sharing stations where you can rent bicycles by the hour or day. Washington, D.C. is one of these cities. The District collects detailed data on the number of bicycles people rent by the hour and day.
# 
# Hadi Fanaee-T at the University of Porto compiled this data into a CSV file. In this project, we'll try to predict the total number of bikes people rented in a given hour.
# 
# The dataset contains columns as below:
# 
# * instant - A unique sequential ID number for each row
# * dteday - The date of the rentals
# * season - The season in which the rentals occurred
# * yr - The year the rentals occurred
# * mnth - The month the rentals occurred
# * hr - The hour the rentals occurred
# * holiday - Whether or not the day was a holiday
# * weekday - The day of the week (as a number, 0 to 7)
# * workingday - Whether or not the day was a working day
# * weathersit - The weather (as a categorical variable)
# * temp - The temperature, on a 0-1 scale
# * atemp - The adjusted temperature
# * hum - The humidity, on a 0-1 scale
# * windspeed - The wind speed, on a 0-1 scale
# * casual - The number of casual riders (people who hadn't previously signed up with the bike sharing program)
# * registered - The number of registered riders (people who had already signed up)
# * cnt - The total number of bike rentals (casual + registered)

# In[1]:


import pandas as pd
bike_rentals = pd.read_csv('bike_rental_hour.csv')
bike_rentals.head(3)


# In[2]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# take a look at the distribution of total rentals
plt.hist(bike_rentals['cnt'])


# In[3]:


# explore how each column is correlated with cnt
bike_rentals.corr()["cnt"]


# We can introduce some order into the process by creating a new column with labels for morning, afternoon, evening, and night. This will bundle similar times together, enabling the model to make better decisions.

# In[4]:


def assign_label(hr):
    if (hr >= 6 and hr < 12):
        return 1
    elif (hr >= 12 and hr < 18):
        return 2
    elif (hr >= 18 and hr < 24):
        return 3
    else:
        return 4

bike_rentals['time_label'] = bike_rentals['hr'].apply(assign_label)
bike_rentals.head()


# ## Linear Regression

# In[5]:


# Select 80% of the rows to be part of the training set
train = bike_rentals.sample(frac=.8)
test = bike_rentals[~bike_rentals.index.isin(train.index)]


# In[6]:


# drop the unnecessary columns 
to_drop = ['cnt', 'casual', 'dteday', 'registered']
features = train.columns.drop(to_drop)


# In[7]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

lr = LinearRegression()
lr.fit(train[features], train['cnt'])
prediction = lr.predict(test[features])
mse = mean_squared_error(test['cnt'], prediction)

mse


# ## Decision Tree Regressor

# In[8]:


from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(min_samples_leaf=5)
dtr.fit(train[features], train['cnt'])
prediction = dtr.predict(test[features])
mse = mean_squared_error(test['cnt'], prediction)

mse


# In[9]:


from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(min_samples_leaf=2)
dtr.fit(train[features], train['cnt'])
prediction = dtr.predict(test[features])
mse = mean_squared_error(test['cnt'], prediction)

mse


# Decision tree regressor appears to have much higher accuracy than linear regression.

# ##  Random Forest Regressor

# In[10]:


from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(min_samples_leaf=5)
rfr.fit(train[features], train['cnt'])
prediction = rfr.predict(test[features])
mse = mean_squared_error(test['cnt'], prediction)

mse


# In[11]:


rfr = RandomForestRegressor(min_samples_leaf=2)
rfr.fit(train[features], train['cnt'])
prediction = rfr.predict(test[features])
mse = mean_squared_error(test['cnt'], prediction)

mse


# The accuracy is improved with random forest algorithm.
