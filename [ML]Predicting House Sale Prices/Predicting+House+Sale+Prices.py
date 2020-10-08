#!/usr/bin/env python
# coding: utf-8

# # Predicting House Sale Price
# 
# In this project, we'll work with housing data for the city of Ames, Iowa, United States from 2006 to 2010. We will explore how the linear regression model worked, understood how the two different approaches to model fitting worked, and some techniques for cleaning, transforming, and selecting features.

# In[1]:


import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import linear_model


# In[2]:


house = pd.read_csv('AmesHousing.tsv', delimiter='\t')


# ## Create a pipeline of functions
# 
# We'll start by creating a pipeline of functions that let us quickly iterate on different models.
# 
# 1. transform features
# 2. select features
# 3. train and test

# In[3]:


# return the train data frame
def transform_features(df):
    return df


# In[4]:


def select_features(df):
    return df[['Gr Liv Area','SalePrice']]


# In[5]:


def train_and_test(df):
    train = df[0:1460]
    test = df[1460:]
    
    # select numeric values
    train_num = train.select_dtypes('number')
    test_num = test.select_dtypes('number')
    
    # drop the target column
    features = train_num.columns.drop('SalePrice')
    
    # test model
    lr = linear_model.LinearRegression()
    lr.fit(train[features], train['SalePrice'])
    predictions = lr.predict(test[features])
    mse = mean_squared_error(test['SalePrice'],predictions)
    rmse = np.sqrt(mse)
    
    return rmse


# In[6]:


# train and test the functions
trans_house = transform_features(house)
filtered_house = select_features(trans_house)
rmse = train_and_test(filtered_house)

rmse


# ## Feature engineering
# 
# Before we get started, we need to dive deeper into the dataset, and
# 
# * address missing values or data leakage
# * transform features into proper format:
#     1. numerical to categorical
#     2. scaling numerical
#     3. fill in missing values
# * create new features by combining other features

# #### Cope with missing values

# In[7]:


# drop the columns with more than 5% of missing values
house.dropna(thresh=house.shape[0]*0.05,how='all',axis=1, inplace=True)


# In[8]:


# For the columns containing less than 5% missing values
# fill in the missing values with the most popular value.

null_percentage = house.isnull().sum() / len(house)
less_than_five = house.columns[null_percentage < 0.05]
less_than_five = house.columns[null_percentage != 0]

for col in less_than_five:
    house[col].fillna(house[col].value_counts().idxmax(), inplace=True)
    
house[less_than_five].isnull().sum()


# #### Drop text columns with one or more missing values

# In[9]:


text_mv_counts = house.select_dtypes(include=['object']).isnull().sum().sort_values(ascending=False)

## Filter Series to columns containing *any* missing values
drop_missing_cols_2 = text_mv_counts[text_mv_counts > 0]

house = house.drop(drop_missing_cols_2.index, axis=1)


# #### Create new features

# In[10]:


years_sold = house['Yr Sold'] - house['Year Built']

# if the value is negative, it's wrong
print(years_sold[years_sold < 0])

years_since_remod = house['Yr Sold'] - house['Year Remod/Add']
print(years_since_remod[years_since_remod < 0])


# In[11]:


# create new column & drop the rows contain error
house['year_sold'] = house['Yr Sold'] - house['Year Built']
house['years_since_remod'] = house['Yr Sold'] - house['Year Remod/Add']
house.drop([1702, 2180, 2181], axis=0, inplace=True)

# 'Year Built' and 'Year Remod/Add' are no longer needed
house.drop(["Year Built", "Year Remod/Add"], axis=1)


# #### Drop unnecessary columns

# In[12]:


# Drop columns that aren't useful for ML
house = house.drop(["PID", "Order"], axis=1)

# Drop columns that leak info about the final sale
house = house.drop(["Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"], axis=1)


# #### Update the transform_features() function

# In[13]:


def transform_features(df):
    
    # drop the columns with more than 5% of missing values
    df.dropna(thresh=df.shape[0]*0.05,how='all',axis=1, inplace=True)
    
    text_mv_counts = df.select_dtypes(include=['object']).isnull().sum().sort_values(ascending=False)
    drop_missing_cols_2 = text_mv_counts[text_mv_counts > 0]
    df = df.drop(drop_missing_cols_2.index, axis=1)
    
    # fill in the remain missing values with the most frequent value of that column
    null_percentage = df.isnull().sum() / len(df)
    less_than_five = df.columns[null_percentage < 0.05]
    less_than_five = df.columns[null_percentage != 0]

    for col in less_than_five:
        df[col].fillna(df[col].value_counts().idxmax(), inplace=True)
    
    # create new column & drop the rows contain error
    df['Years Before Sale'] = df['Yr Sold'] - df['Year Built']
    df['Years Since Remod'] = df['Yr Sold'] - df['Year Remod/Add']
    df.drop([1702, 2180, 2181], axis=0, inplace=True)
    
    # Drop unnecessary columns 
    df.drop(['PID', 'Order','Mo Sold', 'Sale Condition', 'Sale Type', 'Year Built', 'Year Remod/Add'], axis=1, inplace=True)

    return df


# In[14]:


house = pd.read_csv("AmesHousing.tsv", delimiter="\t")
transform_house = transform_features(house)
filtered_house = select_features(transform_house)
rmse = train_and_test(filtered_house)

rmse


# ## Feature selection
# 
# Now that we have cleaned and transformed a lot of the features in the data set, it's time to move on to feature selection for numerical features.
# 
# #### Numerical features

# In[15]:


# select numerical columns
house_num = transform_house.select_dtypes('number')


# In[16]:


sns.heatmap(house_num.corr(), linewidths=0.1)
sns.set(font_scale=0.8)


# In[17]:


abs_corr_coeffs = house_num.corr()['SalePrice'].abs().sort_values()
print(abs_corr_coeffs)


# In[18]:


# keep the columns with a correlation coefficient larger than 0.4
abs_corr_coeffs[abs_corr_coeffs > 0.4]


# In[19]:


low_corr = abs_corr_coeffs[abs_corr_coeffs < 0.4].index
transform_house.drop(low_corr, axis=1, inplace=True)
transform_house.head()


# #### Categorical features
# 
# For this step, we need to determine:
# 
# * which columns are currently numerical but need to be encoded as categorical instead?
# * if a categorical column has hundreds of unique values (or categories), should we keep it? 

# In[20]:


# the list that meant to be categorical
categorical = ["PID", "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config", "Neighborhood", 
                "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl", "Exterior 1st", 
                "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air", "Garage Type", 
                "Misc Feature", "Sale Type", "Sale Condition"]

# find out colums that is categorical but numerical 
cat_but_num =[]
for col in categorical:
    if col in transform_house.columns:
        cat_but_num.append(col)
        
# count how many unique values these columns has
unique_values = transform_house[cat_but_num].apply(lambda col: len(col.value_counts())).sort_values()

# remove columns with more than 10 unique values
many_unique = unique_values[unique_values > 10].index
transform_house.drop(many_unique, axis=1, inplace=True)

transform_house.head()


# In[21]:


# Select the remaining text columns and convert to categorical
text_cols = transform_house.select_dtypes(include=['object'])
for col in text_cols:
    transform_house[col] = transform_house[col].astype('category')

# create dummy columns and add back to the data frame
dummies = pd.get_dummies(transform_house.select_dtypes(include=['category']))
transform_house = pd.concat([transform_house, dummies]).drop(text_cols, axis=1)

transform_house.head()


# #### Update the logic for the select_features() function

# In[22]:


def select_features(df, coeff_threshold=0.4, uniq_threshold=10):
    
    # look at numerical columns first then categorical
    # select numerical columns
    df_num = df.select_dtypes('number')
    
    # keep the columns 
    # with a correlation coefficient larger than 0.4
    abs_corr_coeffs = df_num.corr()['SalePrice'].abs().sort_values()
    low_corr = abs_corr_coeffs[abs_corr_coeffs < coeff_threshold].index
    df.drop(low_corr, axis=1, inplace=True)
    
    # the list that meant to be categorical
    categorical = ["PID", "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config", "Neighborhood", 
                "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl", "Exterior 1st", 
                "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air", "Garage Type", 
                "Misc Feature", "Sale Type", "Sale Condition"]

    # find out colums that is categorical but numerical 
    cat_but_num =[]
    for col in categorical:
        if col in df.columns:
            cat_but_num.append(col)
        
    # count how many unique values these columns has
    unique_values = df[cat_but_num].apply(lambda col: len(col.value_counts())).sort_values()

    # remove columns with more than 10 unique values
    many_unique = unique_values[unique_values > uniq_threshold].index
    df.drop(many_unique, axis=1, inplace=True)
    
    # Select the remaining text columns and convert to categorical
    text_cols = df.select_dtypes(include=['object'])
    for col in text_cols:
        df[col] = df[col].astype('category')

    # create dummy columns and add back to the data frame
    dummies = pd.get_dummies(df.select_dtypes(include=['category']))
    df = pd.concat([df, dummies], axis=1).drop(text_cols, axis=1)
    
    return df


# #### Update the train_and_test function()

# In[23]:


def train_and_test(df, k=0):
    numeric_df = df.select_dtypes(include=['integer', 'float'])
    features = numeric_df.columns.drop("SalePrice")
    lr = linear_model.LinearRegression()
    
    if k == 0:
        train = df[:1460]
        test = df[1460:]

        lr.fit(train[features], train["SalePrice"])
        predictions = lr.predict(test[features])
        mse = mean_squared_error(test["SalePrice"], predictions)
        rmse = np.sqrt(mse)

        return rmse
    
    if k == 1:
        # Randomize *all* rows (frac=1) from `df` and return
        shuffled_df = df.sample(frac=1, )
        train = df[:1460]
        test = df[1460:]
        
        lr.fit(train[features], train["SalePrice"])
        predictions_one = lr.predict(test[features])        
        
        mse_one = mean_squared_error(test["SalePrice"], predictions_one)
        rmse_one = np.sqrt(mse_one)
        
        lr.fit(test[features], test["SalePrice"])
        predictions_two = lr.predict(train[features])        
       
        mse_two = mean_squared_error(train["SalePrice"], predictions_two)
        rmse_two = np.sqrt(mse_two)
        
        avg_rmse = np.mean([rmse_one, rmse_two])
        print(rmse_one)
        print(rmse_two)
        return avg_rmse
    else:
        kf = KFold(n_splits=k, shuffle=True)
        rmse_values = []
        for train_index, test_index, in kf.split(df):
            train = df.iloc[train_index]
            test = df.iloc[test_index]
            lr.fit(train[features], train["SalePrice"])
            predictions = lr.predict(test[features])
            mse = mean_squared_error(test["SalePrice"], predictions)
            rmse = np.sqrt(mse)
            rmse_values.append(rmse)
        print(rmse_values)
        avg_rmse = np.mean(rmse_values)
        return avg_rmse


# In[24]:


def train_and_test(df, k=0):
    df_num = df.select_dtypes('number')
    features = df_num.columns.drop('SalePrice')
    lr = linear_model.LinearRegression()
    
    if k == 0:
        train = df_num[:1460]
        test = df_num[1460:]
    
        # test model
        lr.fit(train[features], train['SalePrice'])
        predictions = lr.predict(test[features])
        mse = mean_squared_error(test['SalePrice'],predictions)
        rmse = np.sqrt(mse)
    
        return rmse
    
    if k == 1:
        # shuffle the ordering
        shuffled_df = df.sample(frac=1)
        fold_one = shuffled_df[:1460]
        fold_two = shuffled_df[1460:]
    
        # train on fold_one and test on fold_two
        lr.fit(fold_one[features], fold_one['SalePrice'])
        predictions = lr.predict(fold_two[features])
        mse_1 = mean_squared_error(fold_two['SalePrice'],predictions)
        rmse_1 = np.sqrt(mse_1)
        print(rmse_1)
        
        # train on fold_two and test on fold_one
        lr.fit(fold_two[features], fold_two['SalePrice'])
        predictions = lr.predict(fold_one[features])
        mse_2 = mean_squared_error(fold_one['SalePrice'],predictions)
        rmse_2 = np.sqrt(mse_2)
        print(rmse_2)
        
        return (rmse_1 + rmse_2) / 2
    
    else:
        # When k is greater than 0
        # implement k-fold cross validation using k folds
        kf = KFold(n_splits=k, shuffle=True)
        rmse_values = []
        for train_index, test_index, in kf.split(df):
            train = df.iloc[train_index]
            test = df.iloc[test_index]
            lr.fit(train[features], train["SalePrice"])
            predictions = lr.predict(test[features])
            mse = mean_squared_error(test["SalePrice"], predictions)
            rmse = np.sqrt(mse)
            rmse_values.append(rmse)
        
        print(rmse_values)
            
        return np.mean(rmse_values)


# In[25]:


house = pd.read_csv("AmesHousing.tsv", delimiter="\t")
transform_house = transform_features(house)
filtered_house = select_features(transform_house)
rmse = train_and_test(filtered_house, k=4)

rmse

