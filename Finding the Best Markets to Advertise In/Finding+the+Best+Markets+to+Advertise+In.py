#!/usr/bin/env python
# coding: utf-8

# # Finding the Best Markets to Advertise In
# 
# In this project, we're working for an an e-learning company that offers courses on programming. Most of the courses are on web and mobile development, but they also cover many other domains, like data science, game development, etc. 
# 
# They want to promote our product and we'd like to invest some money in advertisement. Our goal in this project is to find out the two best markets to advertise our product in.
# 
# To save time, we can try to search existing data that might be relevant for our purpose. One good candidate is the data from freeCodeCamp's 2017 New Coder Survey. freeCodeCamp is a free e-learning platform that offers courses on web development. Because they run a popular Medium publication (over 400,000 followers), their survey attracted new coders with varying interests (not only web development), which is ideal for the purpose of our analysis.

# In[1]:


import pandas as pd
survey = pd.read_csv('2017-fCC-New-Coders-Survey-Data.csv')

print(survey.shape)
survey.columns


# In[2]:


survey.head()


# ## The most popular subjects

# For the purpose of our analysis, we want to answer questions about a population of new coders that are interested in the subjects we teach. We'd like to know:
# 
# * Where are these new coders located.
# * What locations have the greatest densities of new coders.
# * How much money they're willing to spend on learning.
# 
# The "JobRoleInterest" column describes for every participant the role(s) they'd be interested in working. Let's futher investigate this column.

# In[3]:


survey['JobRoleInterest'].value_counts(normalize=True)


# From the frequency table we can see that most people have more than one interest. 

# In[4]:


interests_no_nulls = survey['JobRoleInterest'].dropna()
splitted_interest = interests_no_nulls.str.split(',')

splitted_interest['n_interest'] = splitted_interest.apply(lambda x: len(x))
splitted_interest['n_interest'].value_counts(normalize=True).sort_index() * 100 


# Only 31% of the interviewees has a really clear interest.  But given that we offer courses on various subjects, the fact that new coders have mixed interest might be actually good for us.
# 
# Our focus of the courses is on web and mobile development, so let's find out how many respondents chose at least one of these two options.

# In[5]:


web_or_mobile = interests_no_nulls.str.contains('Web Developer|Mobile Developer') 
web_or_mobile.value_counts(normalize = True) * 100


# 86% of the people are interested in web or mobile development.

# ## New coders' location
# 
# The data set provides information about the location of each participant at a country level. The CountryCitizen variable describes the country of origin for each participant, and the CountryLive variable describes what country each participants lives in (which may be different than the origin country).

# In[6]:


# drop all the rows where participants didn't answer what role they are interested in
survey = survey[survey['JobRoleInterest'].notnull()].copy()

country_no_nulls = survey['CountryLive'].dropna()

absolute_freq = country_no_nulls.value_counts()
relative_freq = country_no_nulls.value_counts(normalize=True) * 100

freq_table = pd.DataFrame(data = {'Absolute frequency': absolute_freq, 
                     'Percentage': relative_freq})
print(freq_table)


# Nearly 46% of the new coders live in the United States. The United States will be our top priority, followed by India, UK, and Canada.
# 
# However, to find out what the countries where new coders live are is not enough. wWe need to go more in depth with our analysis by figuring out how much money new coders are actually willing to spend on learning.

# ## Money spent per month for learning
# 
# The MoneyForLearning column describes in American dollars the amount of money spent by participants from the moment they started coding until the moment they completed the survey. Our company sells subscriptions at a price of $59 per month, and for this reason we're interested in finding out how much money each student spends per month.
# 
# It's better to narrow down our analysis to only four countries: the US, India, the United Kingdom, and Canada.
# 
# 1. These are the countries having the highest absolute frequencies in our sample, which means we have a decent amount of data for each.
# 2. Our courses are written in English, and English is an official language in all these four countries. The more people that know English, the better our chances to target the right people with our ads.

# In[7]:


# Replace 0s with 1s to avoid division by 0
survey['MonthsProgramming'].replace(0,1, inplace = True)

# create new column for the amount of money each student spends each month
survey['money_per_month'] = survey['MoneyForLearning'] / survey['MonthsProgramming']
survey['money_per_month'].notnull().value_counts()


# In[8]:


# drop thee 675 null values
survey['money_per_month'] = survey['money_per_month'].dropna()
survey['money_per_month'].notnull().value_counts()


# In[9]:


# Remove the rows with null values in 'CountryLive'
survey['CountryLive'] = survey['CountryLive'].dropna()

# group by CountryLive
countries_mean = survey.copy().groupby('CountryLive').mean()
countries_mean = countries_mean['money_per_month'][['United States of America',
                            'India', 'United Kingdom',
                            'Canada']]
print(countries_mean)

# visualize the results
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')

countries_mean.plot.bar()
plt.ylabel('Money Per Month', fontsize=10)
plt.xlabel('Country', fontsize=10)
plt.xticks(fontsize=12, rotation=20)
plt.yticks(fontsize=12)
plt.title('Money Spent Per Month by Country', fontsize=12)
plt.show()


# The results for the United Kingdom and Canada are surprisingly low relative to the values we see for India. If we considered a few socio-economical metrics (like GDP per capita), we'd intuitively expect people in the UK and Canada to spend more on learning than people in India.
# 
# The result could be correct. Yet it might also be that we don't have have enough representative data for the United Kingdom, Canada, and India, or we have some outliers (maybe coming from wrong survey answers) making the mean too big for India, or too low for the UK and Canada. 

# ## Deal with extreme outlier

# In[10]:


# extract rows of the four countries
countries = survey.CountryLive.isin(['United States of America','India','United Kingdom','Canada']).values
top_four = survey[countries]
    
# visualize the distribution
import seaborn as sns
sns.boxplot(y = 'money_per_month', x = 'CountryLive',
            data = top_four)
plt.title('Money Spent Per Month Per Country\n(Distributions)',
         fontsize = 16)
plt.ylabel('Money per month (US dollars)')
plt.xlabel('Country')
plt.xticks(range(4), ['US', 'UK', 'India', 'Canada']) # avoids tick labels overlap
plt.show()


# It's hard to see on the plot above if there's anything wrong with the data for the United Kingdom, India, or Canada, but we can see immediately that there's something really off for the US: two persons spend each month \$50000 or more for learning. This is not impossible, but it seems extremely unlikely, so we'll remove every value that goes over \$20,000 per month.

# In[11]:


# revome extreme values
top_four = top_four[top_four.money_per_month < 20000]


# calculate the means again by country
countries_mean_2 = top_four.groupby('CountryLive').mean()
countries_mean_2 = countries_mean_2['money_per_month'][['United States of America',
                            'India', 'United Kingdom',
                            'Canada']]

print(countries_mean_2)

# visualize the distribution
import seaborn as sns
sns.boxplot(y = 'money_per_month', x = 'CountryLive',
            data = top_four)
plt.title('Money Spent Per Month Per Country\n(Distributions)',
         fontsize = 16)
plt.ylabel('Money per month (US dollars)')
plt.xlabel('Country')
plt.xticks(range(4), ['US', 'UK', 'India', 'Canada']) # avoids tick labels overlap
plt.show()


# In this plot, we can see a few extreme outliers for Unitesd State, India, and Canada. We don't know whether they are good data or not. So we need to investigate further.
# 
# #### US's outliers

# In[12]:


us_outliers = top_four[
    (top_four['CountryLive'] == 'United States of America') & 
    (top_four['money_per_month'] >= 6000)]

print(us_outliers)


# Out of these 11 extreme outliers:
# 
# * six people attended bootcamps, which justify the large sums of money spent on learning
# * For the other five, it's hard to figure out from the data where they could have spent that much money on learning. 
# 
# Consequently, we'll remove those rows where participants reported spending \$6000 each month but never attended a bootcamp.
# 
# The data also shows that eight respondents had been programming for no more than three months when they completed the survey. They most likely paid a large sum of money for a bootcamp that was going to last for several months. As a consequence, we'll remove every these eight outliers (because they probably didn't spend anything for the next couple of months after the survey). 

# In[13]:


# remove those who didn't attend bootcamps
# Had been programming for three months or less 
# when at the time they completed the survey

no_bootcamp = top_four[
    (top_four['CountryLive'] == 'United States of America') & 
    (top_four['money_per_month'] >= 6000) &
    (top_four['AttendedBootcamp'] == 0)
]

top_four = top_four.drop(no_bootcamp.index)


less_than_3_months = top_four[
    (top_four['CountryLive'] == 'United States of America') & 
    (top_four['money_per_month'] >= 6000) &
    (top_four['MonthsProgramming'] <= 3)
]

top_four = top_four.drop(less_than_3_months.index)


# #### India's outliers

# In[14]:


india_outliers = top_four[
    (top_four['CountryLive'] == 'India') & 
    (top_four['money_per_month'] >= 2500)]

print(india_outliers)


# It seems that neither participant attended a bootcamp. Overall, it's really hard to figure out from the data whether these persons really spent that much money with learning. They might have misunderstood the questions in the survey. It seems safer to remove these rows.

# In[15]:


top_four = top_four.drop(india_outliers.index)


# #### Canada's outliers

# In[16]:


canada_outliers = top_four[
    (top_four['CountryLive'] == 'Canada') & 
    (top_four['money_per_month'] >= 4000)]

print(canada_outliers)


# The situation is similar to some of the US respondents — this participant had been programming for no more than two months when he completed the survey. It's better that we remove this row.

# In[17]:


top_four = top_four.drop(canada_outliers.index)


# Recompute the mean values and generate the final box plots.

# In[18]:


# calculate the means again by country
countries_mean_final = top_four.groupby('CountryLive').mean()
countries_mean_final = countries_mean_final['money_per_month'][['United States of America',
                            'India', 'United Kingdom',
                            'Canada']]

print(countries_mean_final)

# visualize the distribution
import seaborn as sns
sns.boxplot(y = 'money_per_month', x = 'CountryLive',
            data = top_four)
plt.title('Money Spent Per Month Per Country\n(Distributions)',
         fontsize = 16)
plt.ylabel('Money per month (US dollars)')
plt.xlabel('Country')
plt.xticks(range(4), ['US', 'UK', 'India', 'Canada']) # avoids tick labels overlap
plt.show()


# ## The markets to invest in
# 
# Undoubtedly, USA has a really big potential, so we'll definitely invest in USA. However the question is: do we need to invest in India and Canada? 
# 
# At this point, it seems that we have several options:
# 
# 1. Advertise in the US, India, and Canada by splitting the advertisement budget in various combinations:
# 
#     60% for the US, 25% for India, 15% for Canada.
#     50% for the US, 30% for India, 20% for Canada; etc.
#     
# 2. Advertise only in the US and India, or the US and Canada but split the advertisement budget unequally. 
# 
# 3. Advertise only in the US.
# 
# We might need to resort to market testing or discussion with other business sector to make the decision.
