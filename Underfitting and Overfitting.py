#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


# save filepath to variable for easier access
melbourne_file_path = r'C:\Users\slmnp\Downloads\melb_data.csv\melb_data.csv'
melbourne_data=pd.read_csv(melbourne_file_path)
melbourne_data.columns


# In[15]:


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# In[16]:


# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# In[ ]:




