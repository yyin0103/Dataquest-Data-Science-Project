#!/usr/bin/env python
# coding: utf-8

# #  Analyzing CIA Factbook Data Using SQL
# 
# In this project, we'll work with data from the CIA World Factbook, a compendium of statistics about all of the countries on Earth. The Factbook contains demographic information like:
# 
# * population - The population as of 2015.
# * population_growth - The annual population growth rate, as a percentage.
# * area - The total land and water area.

# In[1]:


get_ipython().run_cell_magic('capture', '', '%load_ext sql\n%sql sqlite:///factbook.db')


# In[2]:


get_ipython().run_cell_magic('sql', '', 'SELECT * \n  FROM facts \n  LIMIT 5;')


# ## Calculating some summary statistics and look for outlier countries

# In[3]:


get_ipython().run_cell_magic('sql', '', 'SELECT\n    MIN(population) min_pop,\n    MAX(population) max_pop, \n    MIN(population_growth) min_pop_grwth,\n    MAX(population_growth) max_pop_grwth \nFROM facts;')


# There's a country with a population of 0. There's also a country with more than 7.2 billion people. Let's zoom in to these countries.
# 
# Countrie(s) with the minimum population:

# In[4]:


get_ipython().run_cell_magic('sql', '', 'SELECT *\n  FROM facts\n  WHERE population == (SELECT MIN(population) FROM facts);')


# Countrie(s) with the maximum population:

# In[5]:


get_ipython().run_cell_magic('sql', '', 'SELECT *\n  FROM facts\n  WHERE population == (SELECT MAX(population) FROM facts);')


# The table contains a row for the whole world, which explains the population of over 7.2 billion. We should recalculate the summary statistics we calculated earlier, while excluding the row for the whole world.

# In[6]:


get_ipython().run_cell_magic('sql', '', "SELECT MIN(population) AS min_pop,\n       MAX(population) AS max_pop,\n       MIN(population_growth) AS min_pop_growth,\n       MAX(population_growth) AS max_pop_growth \n  FROM facts\n WHERE name <> 'World';")


# ## Exploring Average Population and Area

# In[7]:


get_ipython().run_cell_magic('sql', '', "SELECT AVG(population) AS avg_population, AVG(area) AS avg_area\n  FROM facts\n WHERE name <> 'World';")


# ## Finding Densely Populated Countries

# In[8]:


get_ipython().run_cell_magic('sql', '', 'SELECT *\n  FROM facts\n  WHERE population > (SELECT AVG(population) FROM facts)\n  AND area > (SELECT AVG(area) FROM facts);')

