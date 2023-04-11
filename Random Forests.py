#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


# save filepath to variable for easier access
melbourne_file_path = r'C:\Users\slmnp\Downloads\melb_data.csv\melb_data.csv'
melbourne_data=pd.read_csv(melbourne_file_path)
melbourne_data.columns


# In[18]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))


# In[ ]:




