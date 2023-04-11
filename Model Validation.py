#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


# save filepath to variable for easier access
melbourne_file_path = r'C:\Users\slmnp\Downloads\melb_data.csv\melb_data.csv'
melbourne_data=pd.read_csv(melbourne_file_path)
melbourne_data.columns


# In[13]:


from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)


# In[14]:


from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# In[ ]:




