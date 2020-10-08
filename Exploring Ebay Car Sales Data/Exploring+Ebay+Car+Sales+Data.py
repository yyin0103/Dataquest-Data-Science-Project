#!/usr/bin/env python
# coding: utf-8

# # Exploring Ebay Car Sale Data
# 
# In this project, we'll work with a dataset of used cars from eBay Kleinanzeigen, a classifieds section of the German eBay website.The aim of this project is to clean the data and analyze the included used car listings.
# 
# The data dictionary provided with data is as follows:
# 
# * dateCrawled - When this ad was first crawled. All field-values are taken from this date.
# * name - Name of the car.
# * seller - Whether the seller is private or a dealer.
# * offerType - The type of listing
# * price - The price on the ad to sell the car.
# * abtest - Whether the listing is included in an A/B test.
# * vehicleType - The vehicle Type.
# * yearOfRegistration - The year in which the car was first registered.
# * gearbox - The transmission type.
# * powerPS - The power of the car in PS.
# * model - The car model name.
# * kilometer - How many kilometers the car has driven.
# * monthOfRegistration - The month in which the car was first registered.
# * fuelType - What type of fuel the car uses.
# * brand - The brand of the car.
# * notRepairedDamage - If the car has a damage which is not yet repaired.
# * dateCreated - The date on which the eBay listing was created.
# * nrOfPictures - The number of pictures in the ad.
# * postalCode - The postal code for the location of the vehicle.
# * lastSeenOnline - When the crawler saw this ad last online.

# ## Understanding data

# In[1]:


#Import pandas and Numpy libraries
import pandas as pd
import numpy as np
autos = pd.read_csv('autos.csv', encoding='Latin-1')

autos.info()


# In[2]:


autos.isnull().sum()


# In[3]:


autos.head()


# Our observations:
# 
# 1. The dataset contains 20 columns.
# 2. Some columns have null values, but none have more than ~20% null values.
# 3. The column names use camelcase instead of Python's preferred snakecase, which means we can't just replace spaces with underscores.
# 4. Columns like price and odometer are numerical values store in text.

# In[4]:


autos.describe(include='all')


# Our observations:
# 
# 1. All (or nearly all) of the values in Columns "seller" and "offer_type" are the same.
# 2. The num_photos column looks odd, we'll need to investigate this further.

# ## Cleaning Data
# 
# As our findings above, there are a fews things to cope with before we investigate the dataset.
# 
#     a. Change camelcase to snakecase
#     b. Investigate "num_photos" column
#     c. Drop columns containing only one value: "seller" and "offer_type"
#     d. Change "price" and "odermeter" from string to numerical values

# In[5]:


# a. change camelcase to snakecase
rename_dic = {'dateCrawled':'date_crawled', 'offerType':'offer_type', 'vehicleType':'vehicle_type', 'yearOfRegistration':'registration_year','powerPS':'power_ps', 'monthOfRegistration':'registration_month','fuelType':'fuel_type','notRepairedDamage':'unrepaired_damage','dateCreated':'ad_created','nrOfPictures':'n_pictures','postalCode':'postal_code', 'lastSeenOnline':"last_seen"}
autos = autos.rename(rename_dic, axis=1)
autos.columns = map(str.lower, autos.columns)
print(autos.columns) 


# In[6]:


# b. Investigate "num_photos"
autos['n_pictures'].value_counts()


# Every row of "n_pictures" contains 0 value. We can delete this column.

# In[7]:


# c. Delete columns containing only one value
autos.drop(['seller','offer_type','n_pictures'], axis=1, inplace=True)


# In[8]:


# d. Change "price" and "odermeter" from string to numerical values
autos['price'] = autos['price'].astype(str).str.replace('$',' ').str.replace(',','').astype(int)
autos['odometer'] = autos['odometer'].astype(str).str.replace(',','').str.replace('km','').astype(int)

print(autos['price'].head(2))
print(autos['odometer'].head(2))


# ## Investigating Dataset

# ### Exploring Odometer and Price

# In[9]:


autos['odometer'].value_counts(normalize=True) * 100


# We can see that the values in this field are rounded, which might indicate that sellers had to choose from pre-set options for this field. Additionally, around 75% of the cars are high mileage vehicles with odometer above 125000 km.

# In[10]:


print(autos['price'].unique().shape)


# In[11]:


autos["price"].describe()


# In[12]:


autos['price'].value_counts()


# The prices in this column seem rounded as well. However, given there are 2357 unique values in the column, that may just be people's tendency to round prices on the site.
# 
# There are 1,421 cars listed with $0 price - given that this is only 2% of the of the cars, we might consider removing these rows. The maximum price is one hundred million dollars, which seems a lot, let's look at the highest prices further.

# In[13]:


# check the items with the highest price
autos["price"].value_counts().sort_index(ascending=False).head(20)


# In[14]:


# check the items with the highest price
autos["price"].value_counts().sort_index(ascending=True).head(20)


# There are a number of listings with prices below \$30, including about 1,400 at \$0. There are also a small number of listings with very high values, including 14 at around or over $1 million.
# 
# Given that eBay is an auction site, there could legitimately be items where the opening bid is \$1. We will keep the \$1 items, but remove anything above \$350,000, since it seems that prices increase steadily to that number and then jump up to less realistic numbers.

# In[15]:


# remove vehicles with price at $0 and way above 350,000
autos = autos[autos["price"].between(1,350000)]
autos["price"].describe()


# ### Exploring columns with dates
# 
# There are 5 columns that should represent date values: 
# * `date_crawled`: added by the crawler
# * `last_seen`: added by the crawler
# * `ad_created`: from the website
# * `registration_month`: from the website
# * `registration_year`: from the website
# 
# The date_crawled, ad_created, and last_seen columns are still strings of digit. We need to convert it to numerical values.
# 
# Column registration_month and registration_year are numerical already, we can use Series.describe() to understand the distribution.

# In[16]:


# see how date_crawled, ad_created, and last_seen are formatted
autos[['date_crawled', 'lastseen', 'ad_created']].head(3)


# The first 10 characters represent the day (e.g. 2016-03-12).

# In[17]:


# include missing values in the distribution
# use percentages instead of counts
autos['date_crawled'].str[:10].value_counts(normalize=True, dropna=False).sort_index()


# The date crawled are between March 5th to April 7th in 2016. The distribution of listings crawled on each day is roughly uniform.

# In[18]:


autos['lastseen'].str[:10].value_counts(normalize=True, dropna=False).sort_index()


# The "lastseen" column the day the listing was removed, presumably because the car was sold.
# 
# The last three days contain a disproportionate amount of 'last seen' values. Given that these are 6-10x the values from the previous days, it's unlikely that there was a massive spike in sales, and more likely that these values are to do with the crawling period ending and don't indicate car sales.

# In[19]:


autos['ad_created'].str[:10].value_counts(normalize=True, dropna=False).sort_index()


# There is a large variety of ad created dates. Most fall within 1-2 months of the listing date, but a few are quite old, with the oldest at around 9 months.

# In[20]:


autos['registration_year'].describe()


# The "registration_year" columns contains odd values. The minimum value is 1000, and the maximum value is 9999.
# 
# A car can't be registered after the listing, so value above 2016 are wrong. Determining the earliest valid year is more difficult. Realistically, it could be somewhere in the first few decades of the 1900s.
# 
# We need to check how many rows include registration_year below 1900 and above 2016, and if it's safe to remove them.

# In[21]:


(~autos["registration_year"].between(1900,2016)).sum() / autos.shape[0]


# Given that only 4% of the list is out of the range, we'll remove those columns.

# In[22]:


autos = autos[autos["registration_year"].between(1900,2016)]


# ### Exploring Price by Brand

# In[23]:


autos['brand'].value_counts(normalize=True)


# The top five common brands are Volkswagen, BMW, Opel, Mercedez_Benz, and Audi. Volkswagen is by far the most popular brand, with approximately double the cars for sale of the next two brands combined.
# 
# German manufacturers represent four out of the top five brands, almost 50% of the overall listings. 
# 
# There are lots of brands that don't have a significant percentage of listings, so we will limit our analysis to brands representing more than 5% of total listings.
# 
# Let's going to look at the average price of vehicles by come common brand.

# In[24]:


autos_by_brand = autos[autos['brand'].map(autos['brand'].value_counts()) > 0.05]
avg_price_by_brand = autos_by_brand[['brand','price']].groupby(['brand']).mean()

print(avg_price_by_brand.sort_values(['price'], ascending=False))


# With an average price of 45,643, the most expensive brand of car is Porsche, which is way above the average price of all following brands.

# In[25]:


top_six = ['volkswagen', 'bmw', 'opel', 'mercedes_benz', 'audi', 'ford']
top_six_price = avg_price_by_brand.reindex(top_six)
print(top_six_price)


# Audi, BMW and Mercedes Benz are more expensive. Ford and Opel are relatively low-priced. Volkswagen is in between - this may explain its popularity.

# ### Exploring Mileage by Brand

# In[26]:


avg_mil_by_brand = autos_by_brand[['brand','odometer']].groupby(['brand']).mean()
top_six_mil = avg_mil_by_brand.reindex(top_six)

print(top_six_mil)


# In[27]:


top_six_price_mil = pd.concat([top_six_price, top_six_mil], axis=1)
print(top_six_price_mil.sort_values(['price'], ascending=False))


# The range of car mileages does not vary as much as the prices do by brand.
