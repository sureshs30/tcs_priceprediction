#!/usr/bin/env python
# coding: utf-8

# # Stock Market📈 Prediction🤔 with Linear Regression

# I fetch Stock data from Quandl website using Quandl Library

# In[1]:


get_ipython().system('pip install Quandl')


# importing libraires for eda

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import quandl


# Now with Quandl API we will fetch TCS stock data of 1 Month for our prediction

# In[11]:


quandl.ApiConfig.api_key = ''## enter your key
stock_data = quandl.get('NSE/TCS', start_date='2018-12-01', end_date='2018-12-31')


# In[12]:


print(stock_data)


# In[14]:


dataset=pd.DataFrame(stock_data)


# In[15]:


dataset.head()


# In[16]:


dataset.info()


# In[17]:


dataset.describe()


# checking for nullvalues

# In[18]:


dataset.isnull().sum()


# our dataset having 0 null values

# In[19]:


dataset.columns


# In[20]:


dataset.skew()


# # visulization

# In[38]:


plt.boxplot(data=dataset,x='Open')
plt.xlabel('open')


# In[40]:


plt.scatter(data=dataset,x='Open',y='Close',c='g')
plt.xlabel('open')
plt.ylabel('close')


# In[49]:


plt.figure(1 , figsize = (17 , 8))
sb.heatmap(dataset.corr(),annot=True)


# In[50]:


sb.pairplot(data=dataset)


# In[54]:


sb.factorplot(x = "Open", y = "Close", hue = "Low",data = dataset)
plt.xticks(rotation=90)


# In[55]:


sb.distplot(dataset['Turnover (Lacs)'],kde = False)


# In[57]:


sb.swarmplot(x = "Total Trade Quantity", y = "Open", data = dataset)
plt.xticks(rotation=90)


# Now we have to divide data in Dependent and Independent variable

# We can see Date column in useul for our prediction but for simplicity we have to remove it because date format is not proper

# Now we have to predict open price so this column is out dependent variable because open price depend on High,Low,Close,Last,Turnover etc..

# In[60]:


x = dataset.loc[:,'High':'Turnover (Lacs)']
y=dataset.loc[:,'Open']


# In[61]:


x.head()


# In[62]:


y.head()


# In[65]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[86]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)


# In[87]:


LR=LinearRegression()


# In[88]:


LR.fit(x_train,y_train)


# In[89]:


LR.score(x_test,y_test)


# ##I given a test data of random day

# In[90]:


Test_data = [[2017.0 ,1979.6 ,1990.00 ,1992.70 ,2321216.0 ,46373.71]]
prediction=LR.predict(Test_data)


# In[91]:


print(prediction)


# # On that day TCS open on 1998.0 price and our model predicted price is 2001.75 so we can near to the prediction
