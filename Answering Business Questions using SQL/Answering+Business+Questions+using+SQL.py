#!/usr/bin/env python
# coding: utf-8

# # Answering Business Questions using SQL

# In[1]:


get_ipython().run_cell_magic('capture', '', '%load_ext sql\n%sql sqlite:///chinook.db')


# ## Getting familiar with the database

# In[2]:


get_ipython().run_cell_magic('sql', '', 'SELECT\n    name,\n    type\nFROM sqlite_master\nWHERE type IN ("table","view");')


# ## Selecting New Albums to Purchase

# 1. Find out how many pieces of track sold in USA.

# In[3]:


get_ipython().run_cell_magic('sql', '', "    SELECT \n        c.country country,\n        i.invoice_id invoice_id,\n        il.track_id track_id\n    FROM invoice i\n    LEFT JOIN customer c ON c.customer_id = i.customer_id\n    LEFT JOIN invoice_line il ON il.invoice_id = i.invoice_id\n    WHERE c.country = 'USA'")


# 2. Find out what genre the track belongs to.

# In[4]:


get_ipython().run_cell_magic('sql', '', "WITH usa_track AS\n    (\n    SELECT \n        c.country country,\n        i.invoice_id invoice_id,\n        il.track_id track_id\n    FROM invoice i\n    LEFT JOIN customer c ON c.customer_id = i.customer_id\n    LEFT JOIN invoice_line il ON il.invoice_id = i.invoice_id\n    WHERE c.country = 'USA'\n    )\nSELECT  \n    g.name genre,\n    COUNT(us.track_id) number_sold,\n    CAST(COUNT(us.track_id) AS FLOAT) / (SELECT COUNT(*) from usa_track) percentage_sold\nFROM usa_track us \nINNER JOIN track t ON us.track_id = t.track_id\nINNER JOIN genre g ON t.genre_id = g.genre_id\nGROUP BY 1\nORDER BY 2 DESC;")


# Rock is the most popular genre, which accounts for 53% of sales, followed by Alternative Punk and Metal.

# ## Analyzing Employee Sales Performance
# 
# Finds the total dollar amount of sales assigned to each sales support agent within the company.

# In[5]:


get_ipython().run_cell_magic('sql', '', 'SELECT DISTINCT(title) FROM employee')


# In[6]:


get_ipython().run_cell_magic('sql', '', 'SELECT \n    e.first_name || " " || e.last_name employee_name,\n    e.country country,\n    e.reports_to supervisor,\n    e.birthdate birth,\n    SUM(i.total) total_amount_of_sales\nFROM employee e \nINNER JOIN customer c ON c.support_rep_id = e.employee_id\nINNER JOIN invoice i ON i.customer_id = c.customer_id\nWHERE e.title = \'Sales Support Agent\'\nGROUP BY e.employee_id\nORDER BY 5;')


# While there is a 20% difference in sales between Jane (the top employee) and Steve (the bottom employee), the difference roughly corresponds with the differences in their hiring dates.

# ## Analyzing purchases from different countries
# 
# We want to determine:
# 
# * Total number of customers
# * Total value of sales
# * Average value of sales per customer
# * Average order value
# 
# Countries that has only one customer will be collected into an "Other" group.

# In[7]:


get_ipython().run_cell_magic('sql', '', '\nWITH country_or_other AS\n    (\n     SELECT\n       CASE\n           WHEN (\n                 SELECT count(*)\n                 FROM customer\n                 where country = c.country\n                ) = 1 THEN "Other"\n           ELSE c.country\n       END AS country,\n       c.customer_id,\n       il.*\n     FROM invoice_line il\n     INNER JOIN invoice i ON i.invoice_id = il.invoice_id\n     INNER JOIN customer c ON c.customer_id = i.customer_id\n    )\n\nSELECT\n    country,\n    customers,\n    total_sales,\n    average_order,\n    customer_lifetime_value\nFROM\n    (\n    SELECT\n        country,\n        count(distinct customer_id) customers,\n        SUM(unit_price) total_sales,\n        SUM(unit_price) / count(distinct customer_id) customer_lifetime_value,\n        SUM(unit_price) / count(distinct invoice_id) average_order,\n        CASE\n            WHEN country = "Other" THEN 1\n            ELSE 0\n        END AS sort\n    FROM country_or_other\n    GROUP BY country\n    ORDER BY sort ASC, total_sales DESC\n    );')


# ## Albums or individual tracks
# 
# The Store allows customer to make purchases in one of the two ways:
# 
# * purchase a whole album
# * purchase a collection of one or more individual tracks.
# 
# Management are currently considering changing their purchasing strategy to save money. The strategy they are considering is to purchase only the most popular tracks from each album from record companies, instead of purchasing every track from an album.
# 
# We have to find out what percentage of purchases are individual tracks vs whole albums.

# In[8]:


get_ipython().run_cell_magic('sql', '', '\nWITH invoice_first_track AS\n    (\n     SELECT\n         il.invoice_id invoice_id,\n         MIN(il.track_id) first_track_id\n     FROM invoice_line il\n     GROUP BY 1\n    )\n\nSELECT\n    album_purchase,\n    COUNT(invoice_id) number_of_invoices,\n    CAST(count(invoice_id) AS FLOAT) / (\n                                         SELECT COUNT(*) FROM invoice\n                                      ) percent\nFROM\n    (\n    SELECT\n        ifs.*,\n        CASE\n            WHEN\n                 (\n                  SELECT t.track_id FROM track t\n                  WHERE t.album_id = (\n                                      SELECT t2.album_id FROM track t2\n                                      WHERE t2.track_id = ifs.first_track_id\n                                     ) \n\n                  EXCEPT \n\n                  SELECT il2.track_id FROM invoice_line il2\n                  WHERE il2.invoice_id = ifs.invoice_id\n                 ) IS NULL\n             AND\n                 (\n                  SELECT il2.track_id FROM invoice_line il2\n                  WHERE il2.invoice_id = ifs.invoice_id\n\n                  EXCEPT \n\n                  SELECT t.track_id FROM track t\n                  WHERE t.album_id = (\n                                      SELECT t2.album_id FROM track t2\n                                      WHERE t2.track_id = ifs.first_track_id\n                                     ) \n                 ) IS NULL\n             THEN "yes"\n             ELSE "no"\n         END AS "album_purchase"\n     FROM invoice_first_track ifs\n    )\nGROUP BY album_purchase;')


# Based on the table, album purchase takes up around. This means that purchasing only individual tracks from albums from record companies might induce to losing one fifth of of the revenue.
