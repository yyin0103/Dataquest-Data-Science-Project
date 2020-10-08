#!/usr/bin/env python
# coding: utf-8

# # Exploring Hackers News Post
# 
# In this project, we're going to explore the status and two types of posts, Ask HN and Show HN, in Hackers News. 
# 
# Hacker News is a site started by the startup incubator Y Combinator, where user-submitted stories (known as "posts") are voted and commented upon, similar to reddit.
# 
# Users submit "Ask HN" posts to ask the Hacker News community a specific question. Likewise, users submit "Show HN" posts to show the Hacker News community a project, product, or just generally something interesting.
# 
# To maximize the amount of comments, we want to determine:
# 1. which type of post receive more comments on average?
# 2. at what certain time do posts receive more comments on average?
# 
# It should be noted that our dataset has excluded all posts without comments.

# ## Understanding Data

# In[1]:


#Import hacker_news.csv
import pandas as pd
hn = pd.read_csv('hacker_news.csv')

hn.info()


# In[2]:


hn.head()


# ## Calculating the Average Number of Comments for Ask HN and Show HN Posts

# In[3]:


#extract posts that begin with either Ask HN or Show HN
hn['title'] = hn['title'].str.lower()
ask_hn = hn[hn['title'].apply(lambda x: x.startswith('ask hn'))]
show_hn = hn[hn['title'].apply(lambda x: x.startswith('show hn'))]


# In[4]:


#count the mean of "ask hn" and "show hn" comments
avg_ask_comment = ask_hn['num_comments'].mean()
avg_show_comment = show_hn['num_comments'].mean()

print('avg_ask_comment: ', avg_ask_comment)
print('avg_show_comment: ', avg_show_comment)


# We can see that posts starting with "Ask hn" receive more comments that that of "Show hn" on average. 

# ## Finding the Amount of Ask Posts and Comments by Hour Created
# 
# Now we're going to determine if we can maximize the amount of comments an ask post receives by creating it at a certain hour. 

# In[5]:


# create a table that shows average comments in every hour
import datetime as dt
ask_cmt_by_hr = ask_hn.loc[:, ['num_comments','created_at']]

# strift "hour" from the "create_at" column
ask_cmt_by_hr['created_at'] = pd.to_datetime(ask_cmt_by_hr['created_at'], format="%m/%d/%Y %H:%M")
ask_cmt_by_hr['created_at'] = ask_cmt_by_hr['created_at'].apply(lambda x: x.strftime('%H'))

ask_cmt_by_hr = ask_cmt_by_hr.groupby(['created_at']).mean()
ask_cmt_by_hr.sort_values(['num_comments'], ascending=False)


# The top 5 hours for 'Ask HN' comments are:
# 02:00-03:00,
# 15:00-16:00,
# 16:00-17:00,
# 20:00-21:00,
# 21:00-22:00
# 
# The hour that receives the most comments per post on average is 15:00, with an average of 38.59 comments per post. There's about a 60% increase in the number of comments between the hours with the highest and second highest average number of comments.

# ## Finding the Amount of Show Posts and Comments by Hour Created

# In[6]:


# create a table that shows average comments in every hour
import datetime as dt
show_cmt_by_hr = show_hn.loc[:, ['num_comments','created_at']]

# strift "hour" from the "create_at" column
show_cmt_by_hr['created_at'] = pd.to_datetime(show_cmt_by_hr['created_at'], format="%m/%d/%Y %H:%M")
show_cmt_by_hr['created_at'] = show_cmt_by_hr['created_at'].apply(lambda x: x.strftime('%H'))

show_cmt_by_hr = show_cmt_by_hr.groupby(['created_at']).mean()
show_cmt_by_hr.sort_values(['num_comments'], ascending=False)


# The top 5 hours for 'Post HN' comments are:
# 00:00-01:00
# 14:00-15:00
# 18:00-19:00
# 22:00-23:00
# 23:00-00:00

# ## Conclusion
# In this project, we analyzed ask posts and show posts to determine which type of post and time receive the most comments on average. Based on our analysis, to maximize the amount of comments a post receives, we'd recommend the post be categorized as ask post and created between 15:00 and 16:00 (3:00 pm est - 4:00 pm est) to reach the most engagement.
# 
# And while show posts did not receive comments as much as the ask posts did, if we want to release show posts for maximun amount of comments, the best time slot would be between 18:00 and 19:00.
