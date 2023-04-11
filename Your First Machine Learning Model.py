#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


# save filepath to variable for easier access
melbourne_file_path = r'C:\Users\slmnp\Downloads\melb_data.csv\melb_data.csv'
melbourne_data=pd.read_csv(melbourne_file_path)
melbourne_data.columns


# In[5]:


# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)
# We'll learn to handle missing values in a later tutorial.  
# Your Iowa data doesn't have missing values in the columns you use. 
# So we will take the simplest option for now, and drop houses from our data. 
# Don't worry about this much for now, though the code is:

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)


# In[6]:


y=melbourne_data.Price


# In[7]:


melbourne_features=['Rooms','Bathroom','Landsize','Lattitude','Longtitude']


# In[8]:


X=melbourne_data[melbourne_features]


# In[9]:


X.describe()


# In[10]:


X.head()


# In[11]:


from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)


# In[12]:


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))


# In[ ]:




