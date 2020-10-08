#!/usr/bin/env python
# coding: utf-8

# # Create a Kaggle Workflow
# 
# In this guided project, we're going to create a data science workflow.
# 
# Machine learning problems are often caused by:
# 
# * Bugs in implementation
# * Algorithm design
# * Model issues
# * Data quality

# ## Read original train.csv and test.csv files

# In[1]:


import pandas as pd
train = pd.read_csv('train.csv')  
holdout = pd.read_csv('test.csv')
print(holdout.head())  


# ## Preprocessing the Data
# 
# Use %load to import external files.

# In[2]:


# %load functions.py
def process_missing(df):
    """Handle various missing values from the data set

    Usage
    ------

    holdout = process_missing(holdout)
    """
    df["Fare"] = df["Fare"].fillna(train["Fare"].mean())
    df["Embarked"] = df["Embarked"].fillna("S")
    return df

def process_age(df):
    """Process the Age column into pre-defined 'bins' 

    Usage
    ------

    train = process_age(train)
    """
    df["Age"] = df["Age"].fillna(-0.5)
    cut_points = [-1,0,5,12,18,35,60,100]
    label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

def process_fare(df):
    """Process the Fare column into pre-defined 'bins' 

    Usage
    ------

    train = process_fare(train)
    """
    cut_points = [-1,12,50,100,1000]
    label_names = ["0-12","12-50","50-100","100+"]
    df["Fare_categories"] = pd.cut(df["Fare"],cut_points,labels=label_names)
    return df

def process_cabin(df):
    """Process the Cabin column into pre-defined 'bins' 

    Usage
    ------

    train process_cabin(train)
    """
    df["Cabin_type"] = df["Cabin"].str[0]
    df["Cabin_type"] = df["Cabin_type"].fillna("Unknown")
    df = df.drop('Cabin',axis=1)
    return df

def process_titles(df):
    """Extract and categorize the title from the name column 

    Usage
    ------

    train = process_titles(train)
    """
    titles = {
        "Mr" :         "Mr",
        "Mme":         "Mrs",
        "Ms":          "Mrs",
        "Mrs" :        "Mrs",
        "Master" :     "Master",
        "Mlle":        "Miss",
        "Miss" :       "Miss",
        "Capt":        "Officer",
        "Col":         "Officer",
        "Major":       "Officer",
        "Dr":          "Officer",
        "Rev":         "Officer",
        "Jonkheer":    "Royalty",
        "Don":         "Royalty",
        "Sir" :        "Royalty",
        "Countess":    "Royalty",
        "Dona":        "Royalty",
        "Lady" :       "Royalty"
    }
    extracted_titles = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
    df["Title"] = extracted_titles.map(titles)
    return df

def create_dummies(df,column_name):
    """Create Dummy Columns (One Hot Encoding) from a single Column

    Usage
    ------

    train = create_dummies(train,"Age")
    """
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df


# Apply the above function to train and holdout dataframes.

# In[3]:


def preprocess_data(df):
    df = process_missing(df)
    df = process_age(df)
    df = process_fare(df)
    df = process_titles(df)
    df = process_cabin(df)
    
    cols = ["Age_categories", "Fare_categories", "Title", "Cabin_type", "Sex"]
    for col in cols:
        df = create_dummies(df, col)
        
    return df


# In[4]:


train = preprocess_data(train)
holdout = preprocess_data(holdout)


# ## The Four Standard Process
# 
# 1. Data Exploration 
# 2. Feature Engineering
# 3. Feature Selection
# 4. Model Selection/Tuning

# ### Data exploration
# 
# Steps:
# 
# * inspect the "types" of the columns
# * use pivot tables to look at the survival rate for different values of the columns
# * combine the columns and look at the resulting distribution of values and survival rate
# * write a markdown cell to explain your findings

# In[5]:


explore_cols = ["SibSp","Parch","Survived"]
explore = train[explore_cols].copy()
explore.info()


# In[6]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

explore.drop("Survived",axis=1).plot.hist(alpha=0.5,bins=8)
plt.show()


# In[ ]:


explore["familysize"] = explore[["SibSp","Parch"]].sum(axis=1)
explore.drop("Survived",axis=1).plot.hist(alpha=0.5,bins=10)
plt.xticks(range(11))
plt.show()


# In[ ]:


import numpy as np

for col in explore.columns.drop("Survived"):
    pivot = explore.pivot_table(index=col,values="Survived")
    pivot.plot.bar(ylim=(0,1),yticks=np.arange(0,1,.1))
    plt.axhspan(.3, .6, alpha=0.2, color='red')
    plt.show()


# The SibSp column shows the number of siblings and/or spouses each passenger had on board, while the Parch columns shows the number of parents or children each passenger had onboard. Neither column has any missing values.
# 
# The distribution of values in both columns is skewed right, with the majority of values being zero.
# 
# You can sum these two columns to explore the total number of family members each passenger had onboard. The shape of the distribution of values in this case is similar, however there are less values at zero, and the quantity tapers off less rapidly as the values increase.
# 
# Looking at the survival rates of the the combined family members, you can see that few of the over 500 passengers with no family members survived, while greater numbers of passengers with family members survived.

# Engineering new features: isalone
# 1 if the passenger has zero family members onboard
# 0 if the passenger has one or more family members on board

# In[ ]:


def process_isalone(df):
    df['familysize'] = df[["SibSp","Parch"]].sum(axis=1)
    df['isalone'] = 0
    df.loc[(df["familysize"] == 0),"isalone"] = 1
    df = df.drop("familysize",axis=1)
    return df

train = process_isalone(train)
holdout = process_isalone(holdout)


# ### Feature selection

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

def select_features(df):
    # removes non-numeric columns or columns containing null values
    df = df.select_dtypes([np.number]).dropna(axis=1)
    
    all_x = df.drop(['PassengerId','Survived'], axis=1)
    all_y = df['Survived']
    
    clf = RandomForestClassifier(random_state=1)
    selector = RFECV(clf, cv=10)
    selector.fit(all_x, all_y)
    
    best_columns = list(all_x.columns[selector.support_])
    
    print("Best_Columns \n"+ "-"*12+"\n{}\n".format(best_columns))
    
    return best_columns

cols = select_features(train)


# ###  Select and Tune Different Algorithms

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def select_model(df, features):
    
    all_X = df[features]
    all_y = df["Survived"]

    # List of dictionaries, each containing a model name,
    # it's estimator and a dict of hyperparameters
    models = [
        {
            "name": "LogisticRegression",
            "estimator": LogisticRegression(),
            "hyperparameters":
                {
                    "solver": ["newton-cg", "lbfgs", "liblinear"]
                }
        },
        {
            "name": "KNeighborsClassifier",
            "estimator": KNeighborsClassifier(),
            "hyperparameters":
                {
                    "n_neighbors": range(1,20,2),
                    "weights": ["distance", "uniform"],
                    "algorithm": ["ball_tree", "kd_tree", "brute"],
                    "p": [1,2]
                }
        },
        {
            "name": "RandomForestClassifier",
            "estimator": RandomForestClassifier(random_state=1),
            "hyperparameters":
                {
                    "n_estimators": [4, 6, 9],
                    "criterion": ["entropy", "gini"],
                    "max_depth": [2, 5, 10],
                    "max_features": ["log2", "sqrt"],
                    "min_samples_leaf": [1, 5, 8],
                    "min_samples_split": [2, 3, 5]

                }
        }
    ]

    for model in models:
        print(model['name'])
        print('-'*len(model['name']))

        grid = GridSearchCV(model["estimator"],
                            param_grid=model["hyperparameters"],
                            cv=10)
        grid.fit(all_X,all_y)
        model["best_params"] = grid.best_params_
        model["best_score"] = grid.best_score_
        model["best_model"] = grid.best_estimator_

        print("Best Score: {}".format(model["best_score"]))
        print("Best Parameters: {}\n".format(model["best_params"]))

    return models

result = select_model(train, cols)


# #### Submit Result

# In[ ]:


def save_submission_file(model, cols, filename="submission_file"):
    holdout_data = holdout[cols]
    predictions = model.predict(holdout_data)
    
    holdout_ids = hold['PassengerId']
    submission_df = {"PassengerId": holdout_ids,
                 "Survived": predictions}
    submission = pd.DataFrame(submission_df)
    
    submission.to_csv(filename, index=False)
    
best_rf_model = result[2]["best_model"]
save_submission_file(best_rf_model,cols)

