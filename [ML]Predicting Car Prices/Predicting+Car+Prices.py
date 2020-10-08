#!/usr/bin/env python
# coding: utf-8

# # Predicting Car Prices 
# 
#  In this guided project, we'll practice the machine learning workflow to predict a car's market price using the k-nearest neighbors algorithm. 

# In[1]:


import pandas as pd
import numpy as np
cars = pd.read_csv('imports-85.data', header=None)
cars.columns = ['symboling','normalized-losses', 'make', 'fuel-type','aspiration','num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price'] 
cars.info()


# In[2]:


cars.head()


# ## Reorganize the data
# 
# For predictive modeling, we need to:
# 
# 1. Select columns with continuous values
# 2. Replace missing values
# 3. Rescale the values in each columns so that they range from 0 to 1

# ### Select only numeric values

# In[3]:


#select only numeric columns
continuous_values_cols = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
cars_num = cars[continuous_values_cols]


# ### Replace missing values

# In[4]:


# check and see how many missing values there are in each columns
cars_num.replace('?', np.nan, inplace=True)

# convert columns to numeric types
cars_num = cars_num.astype('float')

cars_num.isnull().sum()


# Because `price` is the column we want to predict, let's remove any rows with missing `price` values. As for other comluns with missing values, we'll replace them with the average values from that column.

# In[5]:


#deal with missing values
cars_num = cars_num.dropna(subset=['price'])
cars_num = cars_num.fillna(cars_num.mean())
cars_num.isnull().sum()


# ### Normalize the numeric ones so all values range from 0 to 1 except the target column

# In[6]:


price_col = cars_num['price']
cars_num = (cars_num - cars_num.min()) / (cars_num.max() - cars_num.min())
cars_num['price'] = price_col

cars_num.head()


# ## Univariate k-nearest neighbors models

# In[7]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def knn_train_test(train_col, target_col, df):
    knn = KNeighborsRegressor()
    np.random.seed(1)
    
    # randomize the rows of the dataframe
    shuffled_index = np.random.permutation(df.index)
    df = df.reindex(shuffled_index)
    
    # split the dataset into a training and test set
    train_test_split = int(len(df) / 2)
    train_set = df.iloc[0: train_test_split]
    test_set = df.iloc[train_test_split:]
    
    # fit the model on the training set
    knn.fit(train_set[[train_col]], train_set[target_col])
    
    # make predictions on the test set
    prediction = knn.predict(test_set[[train_col]])
    
    # calculate and return RMSE
    mse = mean_squared_error(test_set[target_col], prediction)
    rmse = np.sqrt(mse)
    
    return rmse


# In[8]:


# use the function to train and test univariate models 
# using the different numeric columns in the data set

rmse_results = {}
train_cols = cars_num.columns.drop('price')

for col in train_cols:
    rmse_val = knn_train_test(col, 'price', cars_num)
    rmse_results[col] = rmse_val

rmse_results_series = pd.Series(rmse_results)
rmse_results_series.sort_values()


# The `engine-size` perform the best value using the default k value. Let's modify the knn_train_test() function to accept a parameter for the k value.

# In[9]:


def knn_train_test(train_col, target_col, df):
    np.random.seed(1)
        
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    # k_value
    k_values = [1,3,5,7,9]
    k_rmses = {}
    
    # Fit a KNN model using default k value.
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[[train_col]], train_df[target_col])
    
    # Make predictions using model.
        predicted_labels = knn.predict(test_df[[train_col]])

    # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        k_rmses[k] = rmse
    
    return k_rmses


# In[10]:


k_rmse_results = {}
train_cols = cars_num.columns.drop('price')

for col in train_cols:
    rmse_val = knn_train_test(col, 'price', cars_num)
    k_rmse_results[col] = rmse_val

k_rmse_results


# In[11]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.figure(figsize=[8,4])
for k,v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())
    plt.plot(x,y)

plt.xlabel('k value')
plt.ylabel('RMSE')
plt.legend()


# ## Multivariate Model

# In[12]:


# Compute average RMSE across different `k` values for each feature.
feature_avg_rmse = {}
for k,v in k_rmse_results.items():
    avg_rmse = np.mean(list(v.values()))
    feature_avg_rmse[k] = avg_rmse
series_avg_rmse = pd.Series(feature_avg_rmse)
sorted_series_avg_rmse = series_avg_rmse.sort_values()
print(sorted_series_avg_rmse)

sorted_features = sorted_series_avg_rmse.index


# Modify the knn_train_test() function to accept a list of column names (instead of just a string)

# In[13]:


def knn_train_test(train_cols, target_col, df):
    np.random.seed(1)
        
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    # k_value
    k_values = [5]
    k_rmses = {}
    
    # Fit a KNN model using default k value.
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[train_cols], train_df[target_col])
    
    # Make predictions using model.
        predicted_labels = knn.predict(test_df[train_cols])

    # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        k_rmses[k] = rmse
    
    return k_rmses


# In[14]:


k_rmse_results = {}

for nr_best_feats in range(2,7):
    k_rmse_results['{} best features'.format(nr_best_feats)] = knn_train_test(
        sorted_features[:nr_best_feats],
        'price',
        cars_num
    )

k_rmse_results


# ## Hyperparameter Tuning

# In[15]:


def knn_train_test(train_cols, target_col, df):
    np.random.seed(1)
        
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    # k_value
    k_values = [i for i in range(1, 25)]
    k_rmses = {}
    
    # Fit a KNN model using default k value.
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[train_cols], train_df[target_col])
    
    # Make predictions using model.
        predicted_labels = knn.predict(test_df[train_cols])

    # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        k_rmses[k] = rmse
    
    return k_rmses


# In[16]:


k_rmse_results = {}

for nr_best_feats in range(2,6):
    k_rmse_results['{} best features'.format(nr_best_feats)] = knn_train_test(
        sorted_features[:nr_best_feats],
        'price',
        cars_num
    )

k_rmse_results


# In[17]:


plt.figure(figsize=[8,4])
for k,v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())
    plt.plot(x,y, label="{}".format(k))

plt.xlabel('k value')
plt.ylabel('RMSE')
labels = ['2 best features', '3 best features', '4 best features', '5 best features']
plt.legend(labels, loc=4)

