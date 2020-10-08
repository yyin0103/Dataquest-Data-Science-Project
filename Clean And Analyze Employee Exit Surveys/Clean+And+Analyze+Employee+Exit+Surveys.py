#!/usr/bin/env python
# coding: utf-8

# # Clean and Analyze Employee Exit Surveys
# 
# In this project, we'll work with exit surveys from employees of the Department of Education, Training and Employment (DETE) and the Technical and Further Education (TAFE) institute in Queensland, Australia.
# 
# We want to determine:
# 
# Are employees who only worked for the institutes for a short period of time resigning due to some kind of dissatisfaction? What about employees who have been there longer?
# 
# 
# Below is a preview of a couple columns we'll work with from the dete_survey.csv:
# 
# * ID: An id used to identify the participant of the survey
# * SeparationType: The reason why the person's employment ended
# * Cease Date: The year or month the person's employment ended
# * DETE Start Date: The year the person began employment with the DETE
# 
# Below is a preview of a couple columns we'll work with from the tafe_survey.csv:
# 
# * Record ID: An id used to identify the participant of the survey
# * Reason for ceasing employment: The reason why the person's employment ended
# * LengthofServiceOverall. Overall Length of Service at Institute (in years): The length of the person's employment (in years)

# ## Understanding the Data

# In[1]:


#read the data 
import pandas as pd
dete_survey = pd.read_csv('dete_survey.csv')
tafe_survey = pd.read_csv('tafe_survey.csv')


# In[2]:


#explore the two data
dete_survey.info()


# In[3]:


dete_survey.head()


# In[4]:


tafe_survey.info()


# In[5]:


tafe_survey.head()


# In[6]:


print(tafe_survey.info())
print(tafe_survey.head())


# Our findings:
# 
# - The dete_survey dataframe contains "Not Stated" but not "NaN".
# - There are lots of columns which we don't need for our analysis:
#     Ex. dete_survey's column 28-49 & tafe_survey's column 17-66
# - Some columns are in both names but of different names. We'll need to rename the columns.
# 
# dete_survey   | tafe_survey  | Definition
# --------------|:------------:|----------------------------------------
# ID            | Record ID    | An id used to identify the participant of the survey
# SeparationType| Reason for ceasing| The reason why the participant's employment ended
# Cease Date    | CESSATION YEAR | The year or month the participant's employment ended
# DETE Start Date |  | The year the participant began employment with the DETE
#  | LengthofServiceOverall. Overall Length of Service at Institute (in years) | The length of the person's employment (in years)
# Age  | CurrentAge. Current Age | The age of the participant
# Gender | Gender. What is your Gender? | The gender of the participant
# 
# - The reasons why employees resigned are spread to several columns.

# ## Cleaning the Data
# 
# We'll drop unnecessary data and rename the columns.

# In[7]:


#change "Not Stated" to "NaN"
dete_survey = pd.read_csv('dete_survey.csv',na_values='Not Stated')


# In[8]:


#drop columns that we don't need
dete_survey_updated = dete_survey.drop(dete_survey.columns[28:49], axis=1)
tafe_survey_updated = tafe_survey.drop(tafe_survey.columns[17:66], axis=1)


# In[9]:


#change the column names to the same in both charts
dete_survey_updated.columns = dete_survey_updated.columns.str.replace(" ",'_').str.strip().str.lower()

rename_dict = {'Record ID': 'id', 'CESSATION YEAR': 'cease_date', 'Reason for ceasing employment': 'separationtype', 'Gender. What is your Gender?': 'gender', 'CurrentAge. Current Age': 'age',
       'Employment Type. Employment Type': 'employment_status',
       'Classification. Classification': 'position',
       'LengthofServiceOverall. Overall Length of Service at Institute (in years)': 'institute_service',
       'LengthofServiceCurrent. Length of Service at current workplace (in years)': 'role_service'}
tafe_survey_updated.rename(rename_dict, axis=1, inplace=True)

print(dete_survey_updated.columns)
print(tafe_survey_updated.columns)


# ## Filtering the Data
# 
# For this project, we'll analyze survey respondents who resigned, so we'll only keep the separation type that contains the string 'Resignation'.

# In[10]:


#Try to answer this question:Are employees who have only worked for the institutes for a short period of time resigning due to some kind of dissatisfaction? What about employees who have been at the job longer?
print(dete_survey_updated['separationtype'].value_counts(), "\n")
print(tafe_survey_updated['separationtype'].value_counts())


# We also need to note that dete_survey_updated dataframe contains multiple separation types with the string 'Resignation':
# 
# - Resignation-Other reasons
# - Resignation-Other employer
# - Resignation-Move overseas/interstate

# In[11]:


#select those who have a resignation separation type
dete_resign_type = dete_survey_updated['separationtype'].isin(['Resignation-Other reasons','Resignation-Other employer','Resignation-Move overseas/interstate'])
dete_survey_updated = dete_survey_updated[dete_resign_type]

tafe_resign_type = tafe_survey_updated['separationtype'] == 'Resignation'
tafe_survey_updated = tafe_survey_updated[tafe_resign_type]

# Update all separation types in dete_survey to 'Resignation'
dete_survey_updated['separationtype'] = dete_survey_updated['separationtype'].str.slice(stop=11)

# make sure we keep only those who resigned
print(dete_survey_updated['separationtype'].value_counts(), "\n")
print(tafe_survey_updated['separationtype'].value_counts())


# ## Verifying the Data
# 
# In this step, we'll focus on whether the cease_date and dete_start_date columns make sense.
# 
# Since the cease_date is the last year of the person's employment and the dete_start_date is the person's first year of employment, it wouldn't make sense to have years after the current date.
# 
# Given that most people in this field start working in their 20s, it's also unlikely that the dete_start_date was before the year 1940

# In[12]:


#check if dete stat dates are earlier than cease date.
print(dete_survey_updated['dete_start_date'].value_counts().sort_index(), "\n")
print(dete_survey_updated['cease_date'].value_counts().sort_index(), "\n")
print(tafe_survey_updated['cease_date'].value_counts().sort_index())


# In[13]:


# Extract the years and convert them to an int type
dete_survey_updated['cease_date'] = dete_survey_updated['cease_date'].str.split('/').str[-1]
dete_survey_updated['cease_date'] = dete_survey_updated['cease_date'].astype(float)

# Check the values again and look for outliers
dete_survey_updated['cease_date'].value_counts()


# The years in both dataframes don't completely align. The tafe_survey_updated dataframe contains some cease dates in 2009, but the dete_survey_updated dataframe does not. The tafe_survey_updated dataframe also contains many more cease dates in 2010 than the dete_survey_updaed dataframe. Since we aren't concerned with analyzing the results by year, we'll leave them as is.

# ## Investigating the Data: Finding the Answer
# 
# After data cleansing, it's time to investigate the datasets to answer our question:
# 
# Are employees who only worked for the institutes for a short period of time resigning due to some kind of dissatisfaction? What about employees who have been there longer?
# 
# For this question, we'll create a new column "institue_service" for the length of time the employee spent in their workplace.
# 
# ##### Calculate the length of service

# In[14]:


#create a new "institute column" in dete_survey
dete_survey_updated['institute_service'] = dete_survey_updated['cease_date'] - dete_survey_updated['dete_start_date']
dete_survey_updated['institute_service'].describe()


# In[15]:


dete_survey_updated['institute_service'].value_counts().head()


# As we see in the chart, the average length of service is around 10.5 years. A quarter of people had spent more than 16 years with the same company before they resigned. While the shortest length of service is less than a year, while longest is up to 49 years. Most people resigned at the fifth year of their service. 
# 
# Next, we'll identify dissatisfied people and their reasons by looking into the following columns.
# 
# 1. tafe_survey_updated:
#     * Contributing Factors. Dissatisfaction
#     * Contributing Factors. Job Dissatisfaction
# 2. dafe_survey_updated:
#     * job_dissatisfaction
#     * dissatisfaction_with_the_department
#     * physical_work_environment
#     * lack_of_recognition
#     * lack_of_job_security
#     * work_location
#     * employment_conditions
#     * work_life_balance
#     * workload
# 
# If any of the columns is marked "True," it means the person resigned out of dissatisfaction to some degree. Update the values in the new "dissatisfaction" column to be either True, False, or NaN.

# In[16]:


# check the values of the two columns in tafe_survey
tafe_survey_updated['Contributing Factors. Dissatisfaction'].value_counts(dropna=False)


# In[17]:


tafe_survey_updated['Contributing Factors. Job Dissatisfaction'].value_counts(dropna=False)


# In[18]:


import numpy as np

# to align with dete_survey, change the values to boolean
def update_vals(x):
    if pd.isnull(x):
        return np.nan
    elif x == '-':
        return False
    else:
        return True

tafe_survey_updated['dissatisfied'] = tafe_survey_updated[['Contributing Factors. Dissatisfaction', 'Contributing Factors. Job Dissatisfaction']].applymap(update_vals).any(1, skipna=False)
tafe_survey_updated['dissatisfied'].value_counts(dropna=False)


# In[19]:


dete_survey_updated['dissatisfied'] = dete_survey_updated[['job_dissatisfaction',
       'dissatisfaction_with_the_department', 'physical_work_environment',
       'lack_of_recognition', 'lack_of_job_security', 'work_location',
       'employment_conditions', 'work_life_balance',
       'workload']].any(1, skipna=False)

dete_survey_updated['dissatisfied'].value_counts(dropna=False)


# Now we're going to combine the two charts. But beforehand we'll add a new column to each dataframe that will allow us to easily distinguish between the two.

# In[20]:


tafe_survey_updated['institute'] = "TAFE"
dete_survey_updated['institute'] = "DETE"


# ##### Combine the two dataframes
# We still have some columns left in the dataframe that we don't need to complete our analysis. Columns with less than 500 non null values will be dropped.

# In[21]:


combined = pd.concat([dete_survey_updated, tafe_survey_updated], ignore_index=True)

# Verify the number of non null values in each column
combined.notnull().sum().sort_values()


# In[22]:


# drop any columns with less than 500 non null values
combined_updated = combined.dropna(thresh = 500, axis =1).copy()


# In[23]:


combined_updated.head()


# To answer the question, we need "institute_service" and "dissatisfaction." Yet we need to clean up institute_service as it contains values in a couple different forms:

# In[24]:


combined_updated['institute_service'].value_counts()


# We'll categorize the values in the institute_service column as below:
# 
# * New: Less than 3 years in the workplace
# * Experienced: 3-6 years in the workplace
# * Established: 7-10 years in the workplace
# * Veteran: 11 or more years in the workplace

# In[25]:


# change the type to string
combined_updated['institute_service'] = combined_updated['institute_service'].astype(str).str.extract(r'(\d+)')
combined_updated['institute_service'] = combined_updated['institute_service'].astype(float)

# Check the years extracted are correct
combined_updated['institute_service'].value_counts()


# In[26]:


# Convert years of service to categories
def categorize_service(val):
    if val >= 11:
        return "Veteran"
    elif 7 <= val < 11:
        return "Established"
    elif 3 <= val < 7:
        return "Experienced"
    elif pd.isnull(val):
        return np.nan
    else:
        return "New"

# create a new column to store the  categories
combined_updated['service_cat'] = combined_updated['institute_service'].apply(categorize_service)


# ##### Cope with missing values in the combined dataframe

# In[27]:


# confirm the values in the dissatisfied column
combined_updated['dissatisfied'].value_counts(dropna=False)


# In[28]:


import matplotlib
get_ipython().magic('matplotlib inline')
# fill the missing values with the most frequent value: False
combined_updated['dissatisfied'].fillna(False, inplace=True)

# calculate the percentage of dissatisfied employees by service category
dis_by_ser_cat = combined_updated.pivot_table(index='service_cat', values='dissatisfied')

# showcase the results in a plot
dis_by_ser_cat.plot(kind='bar')


# ## Conclusion
# From the initial analysis above, we can tentatively conclude that employees with 7 or more years of service are more likely to resign due to some kind of dissatisfaction with the job than employees with less than 7 years of service. 
# 
# However, we need to handle the rest of the missing data to finalize our analysis.
