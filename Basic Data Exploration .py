#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


# save filepath to variable for easier access
melbourne_file_path = r'C:\Users\slmnp\Downloads\melb_data.csv\melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
melbourne_data.describe()


# In[ ]:




