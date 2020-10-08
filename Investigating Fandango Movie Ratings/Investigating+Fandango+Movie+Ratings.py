#!/usr/bin/env python
# coding: utf-8

# # Investigating Fandango Movie Ratings

# In October 2015, a data journalist named Walt Hickey analyzed movie ratings data and found strong evidence to suggest that Fandango's rating system was biased and dishonest (Fandango is an online movie ratings aggregator).
# 
# 1. The actual rating was almost always rounded up to the nearest half-star. For instance, a 4.1 movie would be rounded off to 4.5 stars, not to 4 stars, as you may expect.
# 2. In the case of 8% of the ratings analyzed, the rounding up was done to the nearest whole star. For instance, a 4.5 rating would be rounded off to 5 stars.
# 3. For one movie rating, the rounding off was completely bizarre: from a rating of 4 in the HTML of the page to a displayed rating of 5 stars.
# 
# The goal of this project is to determine whether there has been any changes in Fandango's rating system after Hickey's analysis. In this project we will analyze the system's characteristics previous and after Hickey's analysis using:
# 
# * the data Hickey made in 2015
# * a new dataset with movies released in 2016 and 2017

# ## Understanding the data

# In[1]:


import pandas as pd
previous = pd.read_csv('fandango_score_comparison.csv')
after = pd.read_csv('movie_ratings_16_17.csv')

previous.head(3)


# In[2]:


after.head(3)


# Below we isolate only the columns that offer information about Fandango and make copies to avoid any SettingWithCopyWarning later on.

# In[3]:


fandango_prev = previous[['FILM', 'Fandango_Stars', 'Fandango_Ratingvalue', 'Fandango_votes', 'Fandango_Difference']].copy()
fandango_after= after[['movie','year','fandango']].copy()

fandango_prev.head(3)


# In[4]:


fandango_after.head(3)


# Our original goal to determine if there has been any change in Fandango's rating system after FiveThirtyEight's original analysis in 2015. 
# 
# The population of interest is based on the ratings for all movies on Fandango's website.The data we have was separated into two different tables by different time periods. Hence we need to determine what methodology to use.
# 
# By looking at FiveThirtyEight's github post sharing the data we can see that they looked at movies that:
# 
# - Had at least 30 fan reviews on Fandango's website at time of sampling.
# - The film must have had tickets on sale in 2015.
# - The sampling is not random, since some movies won't be included at all. One thing we have to be careful of is temporal trends. Some years may have a set of better received movies than other years.
# 
# The new sample of movies since the original analysis has the following properties:
# 
# - Released in 2016 and before March 22, 2017.
# - Must have received a 'significat number of votes' although the significance is not clear.

# ##  Changing the goal of our analysis
# 
# These two samples will not be able to tell with great certainty whether or not the full ratings system on Fandango has changed since the original analysis. However, changing the goal a little can still produce some interesting findings.
# 
#     New Goal: Find out whether there's any difference between Fandango's ratings for popular movies in 2015 and Fandango's ratings for popular movies in 2016. 
# 
# ### Define "popularity"
# We need to identify what the term "popular" means before we continue. We'll use Hickey's benchmark of 30 fan ratings and consider a movie as "popular" only if it has 30 fan ratings or more on Fandango's website.
# 
# However, the "after" dataset doesn't provide information about the number of fan ratings. We need to furtherly verified the representativity of the dataset.

# In[5]:


# First, we extract 10 samples from our original rating
fandango_after.sample(10, random_state = 1)


# As of April 2018, the fan ratings are as below:
# 
# |Movie |Fan ratings|
# | :- | -: | :-: |
# |Mechanic: Resurrection | 2247|
# |Warcraft | 7271|
# |Max Steel | 493|
# |Me Before You | 5263|
# |Fantastic Beasts and Where to Find Them | 13400|
# |Cell | 17|
# |Genius | 127|
# |Sully | 11877|
# |A Hologram for the King | 500|
# |Captain America: Civil War | 35057|
# 
# 90% of the movies has more than 30 ratings. We can assume that the "after" dataset may be valid enough for our analysis.

# ### Isolate the movies released in 2015 and 2016
# 
# In the two dataframes, some movies were not released in 2015 and 2016. We need to isolate only the sample points that belong to our populations of interest.

# In[6]:


## deal with fandango_prev first
fandango_prev['Year'] = fandango_prev['FILM'].str[-5:-1]
fandango_prev['Year'].value_counts()


# In[7]:


fandango_2015 = fandango_prev[fandango_prev['Year'] == '2015'].copy()
fandango_2015['Year'].value_counts()


# In[8]:


# deal with fandango_after
fandango_after['year'].value_counts()


# In[9]:


fandango_2016 = fandango_after[fandango_after['year'] == 2016].copy()
fandango_2016['year'].value_counts()


# ## Analyzing the data
# 
# We can now start analyzing the two samples we isolated before. Once again, Our goal is to determine whether there's any difference between Fandango's ratings for popular movies in 2015 and Fandango's ratings for popular movies in 2016.
# 
# ### The shapes of the distributions of moving ratings

# In[10]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('fivethirtyeight')

fandango_2015['Fandango_Stars'].plot.kde(label = '2015', legend = True, figsize = (8,5.5))
fandango_2016['fandango'].plot.kde(label = '2016', legend = True)

plt.title("Comparing distribution shapes for Fandango's ratings\n(2015 vs 2016)",
          y = 1.07) # the `y` parameter pads the title upward
plt.xlabel('Stars')
plt.xlim(0,5) # because ratings start at 0 and end at 5
plt.xticks(np.arange(0,5.1,.5))
plt.show()


# We can see that both distributions are strongly left skewed. However, it shows that ratings were slightly lower in 2016 compared to 2015. This suggests that there was a change indeed between Fandango's ratings for popular movies in 2015 and Fandango's ratings for popular movies in 2016. 

# ### Relative Frequencies

# In[11]:


print('[2015]')
fandango_2015['Fandango_Stars'].value_counts(normalize=True).sort_index()


# In[12]:


print('[2016]')
fandango_2016['fandango'].value_counts(normalize=True).sort_index()


# We can see that in the ratings in 2016 is lower than than that in 2015. The minimum rating in 2016 is 2.5 with 5.0 less than 1%. In 2015, however, the lowest rating is 3.0. 7 percent of the ratings even goes to 5.0.

# ### Determining the Direction of the Change

# In[13]:


mean_2015 = fandango_2015['Fandango_Stars'].mean()
mean_2016 = fandango_2016['fandango'].mean()

median_2015 = fandango_2015['Fandango_Stars'].median()
median_2016 = fandango_2016['fandango'].median()

mode_2015 = fandango_2015['Fandango_Stars'].mode()[0] # the output of Series.mode() is a bit uncommon
mode_2016 = fandango_2016['fandango'].mode()[0]

summary = pd.DataFrame()
summary['2015'] = [mean_2015, median_2015, mode_2015]
summary['2016'] = [mean_2016, median_2016, mode_2016]
summary.index = ['mean', 'median', 'mode']

plt.style.use('fivethirtyeight')
summary['2015'].plot.bar(color = '#0066FF', align = 'center', label = '2015', width = .25)
summary['2016'].plot.bar(color = '#CC0000', align = 'edge', label = '2016', width = .25, rot = 0, figsize = (8,5))

plt.title('Comparing summary statistics: 2015 vs 2016', y = 1.07)
plt.ylim(0,5.5)
plt.yticks(np.arange(0,5.1,.5))
plt.ylabel('Stars')
plt.legend(framealpha = 0, loc = 'upper center')
plt.show()


# ## Conclusion
# 
# Our analysis showed that there's indeed a slight difference between Fandango's ratings for popular movies in 2015 and Fandango's ratings for popular movies in 2016. However it is unsure what caused the change. It may be the results of Fandango fixing the biased rating system after Hickey's analysis, or simply by audience's preference.
