#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


ds = pd.read_csv('50-Startups.csv')


# In[5]:


ds.head()


# In[11]:


X = ds.iloc[:,:-1].values

y = ds.iloc[:,4]


# In[12]:



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=0)


# In[38]:


X_train


# In[7]:



le = LabelEncoder()
X[:,3] = le.fit_transform(X[:,3])
X


# In[18]:


X_test


# In[19]:


y_test


# In[9]:


model = LinearRegression()
model.fit(X_train,y_train)


# In[10]:


predictions = model.predict(X_test)


# In[11]:


model.score(X_train,y_train)


# In[46]:


colors= [286.135, 288.556, 286.135, 288.55, 286.13, 288.55627, 286.13, 288.556, 342.713, 333.98, 342.713, 333.9834, 342.713, 333.9834, 342.71, 333.98]
colors = np.array(colors)


# In[47]:



plt.scatter(predictions,y_test,colors)


# In[48]:


plt.xlabel("predictions")


# In[49]:


plt.ylabel("y_test")


# In[50]:


plt.show()


# In[ ]:




