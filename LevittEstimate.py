#!/usr/bin/env python
# coding: utf-8

# The following code was written by @maestro_rayo following:
# 
# https://www.medrxiv.org/content/10.1101/2020.06.26.20140814v2.full.pdf
# 
# by M. Levitt et al.
# 
# 

# In[47]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log
from datetime import datetime, timedelta


# 
# ### Get data

# In[48]:


covid_data = pd.read_csv("https://covid.ourworldindata.org/data/ecdc/total_deaths.csv", parse_dates=["date"], skiprows=[1])


# In[49]:


location = 'Sweden'
data = covid_data[location].fillna(0).astype(int)
#data = data.rolling(window=7).mean().fillna(0).astype(int)
plt.plot(data)
today = datetime.today()
print("Max value to date %s for %s is %d" % (today.strftime("%m/%d/%Y"),location,max(data)))


# In[50]:


threshold = 0.99

#Assuming one would get a reasonable estimate at least at a third of the final value
max_times = 3

def ML(data, days_ago):
  data = data[:len(data)-days_ago]
  max_search = max_times*max(data)
  coeff = []
  for N in range(max(data)+1,max(data) + max_search):
    y = np.array([-np.log(np.log(N) - np.log(x)) for x in data if x > 0])
    days = np.array(range(len(y)))
    correlation = np.corrcoef(days, y)[0][1]
    coeff.append(correlation)
    if (correlation >= threshold):
      break
  N= max(data) + np.argmax(coeff)
  return (N, max(coeff))


# In[55]:


#What was the prediction X days ago?
days_ago = 60
d = today - timedelta(days=days_ago)
prediction, max_coeff = ML(data,days_ago)
plt.plot(data,label='Accumulated')
plt.rcParams["figure.figsize"] = (12,10)
plt.plot([prediction]*len(data),label='Prediction')
plt.legend(loc="lower right")

print("For %s the prediction on %s was %d.\nMaximum correlation %f" % (location,d.strftime("%m/%d/%Y"),prediction,max_coeff))


# In[ ]:




