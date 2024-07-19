#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df= pd.read_csv("HR_comma_sep.csv")


# In[3]:


df.head(30)


# In[4]:


df.info()


# In[5]:


df.corr()


# In[6]:


plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True, cmap='magma')


# According to the map above, the number of employees leaving have negative relation with their work satisfaction_level.
# i.e., the lower the satisfaction, the great the chances of employee leaving the company. 

# In[7]:


df['salary'].unique()


# In[8]:


df['salary'].value_counts()


# In[9]:


df[['salary','left']]


# In[10]:


salary_wise_left=df.groupby(['left','salary']).size().unstack().fillna(0)
salary_wise_left


# In[13]:


salary_wise_left.plot(figsize=(10,7))


# Therefore, from above table 42% left are of Low salary group, 25% left are of medium salary group and only 0.07% left are from high salary group

# In[14]:


df['role'].unique()


# In[15]:


dummy=pd.get_dummies(df['role'])
dummy


# In[16]:


df=pd.concat([df,dummy], axis=1)


# In[18]:


df.head()


# In[19]:


x=df[['time_spend_company','satisfaction_level']].values
#dependen
y=df[['left']].values


# In[46]:


#splitting the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test= train_test_split(x,y,test_size=0.3, random_state=0)


# In[47]:


#Applying logistic Regression model

from sklearn.linear_model import LogisticRegression

classifier= LogisticRegression(random_state=0)
#fitting to training set
classifier.fit(x_train, y_train)


# In[48]:


print(classifier.predict([[5,0.3]]))


# In[49]:


print(classifier.predict([[5,0.9]]))


# In[50]:


y_pred=classifier.predict(x_test)


# In[51]:


from sklearn.metrics import confusion_matrix, accuracy_score

cm=confusion_matrix(y_test, y_pred)
print(cm)

accuracy=accuracy_score(y_pred,y_test)
print(accuracy)


# In[52]:


#Applying K-Nearest Neighbours method

from sklearn.neighbors import KNeighborsClassifier

classifier=KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
classifier.fit(x_train,y_train)


# In[55]:


print(classifier.predict([[5,0.3]]))


# In[65]:


print(classifier.predict([[3,0.5]]))


# In[63]:


y_pred=classifier.predict(x_test)


# In[64]:


cm=confusion_matrix(y_test, y_pred)
print(cm)

accuracy=accuracy_score(y_pred,y_test)
print(accuracy)

