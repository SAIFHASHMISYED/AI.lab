#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


pip install pandas


# In[3]:





# In[4]:


df=pd.read_csv("diabetes.csv")


# In[5]:


df.head()


# In[7]:


df.shape


# In[8]:


df.isnull().sum()


# In[11]:


x=df.iloc[:,:-1].to_numpy()
y=df.iloc[:,-1].to_numpy()


# In[10]:


df.iloc[:,:-1]


# In[12]:


x


# In[13]:


y


# In[15]:


x=df.iloc[:,:-1].to_numpy()
y=df.iloc[:,-1].to_numpy()


# In[21]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[23]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion="entropy",random_state=0)
clf.fit(x_train,y_train)


# In[24]:


import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))


# In[25]:


pip install matplotlib


# In[27]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))


# In[39]:


plot_tree(clf,feature_names=['Glucose','BMI'],class_names=['No','Yes'])
plt.show()


# In[41]:


clf.set_params(max_depth=2)


# clf.set_params(max_depth=2)

# In[42]:


clf.fit(x_train,y_train)
plt.figure(figsize=(20,10))
plot_tree(clf,feature_names=['Glucose','BMI'],class_names=['No','Yes'])
plt.show()


# In[43]:


predictions=clf.predict(x_test)


# In[44]:


predicitons


# In[45]:


predictions


# In[47]:


clf.predict([[90,20],[200,30]])


# In[48]:


from sklearn import metrics
cf=metrics.confusion_matrix(y_test,predictions)
cf


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




