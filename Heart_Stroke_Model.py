#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling


# In[2]:


pwd


# In[3]:


df = pd.read_csv('healthcaredatasetstrokedata.csv')


# In[4]:


df.head()


# In[5]:


df.columns


# In[7]:


df.shape


# In[9]:


df.isna().sum()


# In[13]:


from pandas_profiling import ProfileReport

Reports = ProfileReport(df)

Reports.to_file('Report.html')

ProfileReport(df).to_notebook_iframe()


# In[18]:


df.dtypes


# In[177]:


plt.figure(figsize=(12,8))

sns.boxplot(x= df['Residence_type'],y= df['avg_glucose_level'],)


# In[176]:


plt.figure(figsize=(12,8))
sns.countplot(data=df,x=df['smoking_status'],hue="stroke")


# In[181]:


plt.figure(figsize=(12,8))
sns.countplot(data=df,x=df['heart_disease'],hue="stroke")


# In[22]:


# dropping id as its irrelevant .
df.drop('id',axis=1,inplace=True)


# In[23]:


df.columns


# In[24]:


from sklearn.impute import SimpleImputer


# In[25]:


imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')


# In[27]:


df.bmi = imputer.fit_transform(df['bmi'].values.reshape(-1,1))


# In[31]:





# In[128]:


data = df.copy() 

pd.get_dummies(data.gender)


# In[131]:


data.columns


# In[86]:


data.head()


# In[47]:


from sklearn.genderpreprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
data = columnTransformer.fit_transform((data))


# In[49]:


pd.DataFrame(data)


# In[129]:


cat_vars = ['gender','ever_married','work_type','Residence_type','smoking_status']

for vars in cat_vars:
  cat_list = pd.get_dummies(data[vars], prefix=vars)
  data = data.join(cat_list)
data


# In[132]:


data.drop(['gender','ever_married','work_type','Residence_type','smoking_status'] , axis=1,inplace=True)


# In[133]:


data


# In[134]:



x = data.drop('stroke',axis =1)
y = data.stroke

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split (x,y,random_state=0 , test_size=0.2)


# In[135]:


x_train


# In[166]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import export_graphviz # display the tree within a Jupyter notebook
from ipywidgets import interactive, IntSlider, FloatSlider, interact


# In[167]:


@interact #To convert any function into an inteactive one just write "@interact" immediately before the function definition

def plot_tree(
    crit = ['gini', 'entropy'],
    split = ['best','random'],
    depth = IntSlider(min = 1, max = 25, value =2, continuous_update = False),
    min_split = IntSlider(min = 1, max = 5, value =2, continuous_update = False),
    #min_split is the minimum number of samples  required to split an internal node in our decision tree
    min_leaf = IntSlider(min = 1, max = 5, value =1, continuous_update = False)):
  
  estimator = DecisionTreeClassifier(criterion=crit,
                                     splitter=split,
                                     max_depth = depth,
                                     min_samples_split = min_split,
                                     min_samples_leaf = min_leaf
                                     )
  estimator.fit(x_train, y_train)
  print('Decision Tree Training Accuracy:', accuracy_score(y_train, estimator.predict(x_train)))
  print('Decision Tree Testing Accuracy:', accuracy_score(y_test, estimator.predict(x_test)))

  a = accuracy_score(y_train, estimator.predict(x_train))
  b = accuracy_score(y_test, estimator.predict(x_test))

  if a > 0.9:
    print('Decision Tree Training Accuracy',a, 'Decision Tree Testing Accuracy', b)
    print('Criterion:',crit,'\n', 'Split:', split,'\n', 'Depth:', depth,'\n', 'Min_split:', min_split,'\n', 'Min_leaf:', min_leaf,'\n')

  


# In[168]:


@interact
def plot_tree_rf(crit= ['gini','entropy'],
                 bootstrap= ['True', 'False'],
                 depth=IntSlider(min= 1 ,max= 20,value=3, continuous_update=False),
                 forests=IntSlider(min= 1,max= 1000,value= 100,continuous_update=False),
                 min_split=IntSlider(min= 2,max= 5,value= 2, continuous_update=False),
                 min_leaf=IntSlider(min= 1,max= 5,value= 1, continuous_update=False)):
  
  estimator = RandomForestClassifier(
      random_state = 1,
      criterion = crit,
      bootstrap = bootstrap,
      n_estimators = forests,
      max_depth = depth, 
      min_samples_split = min_split,
      min_samples_leaf = min_leaf,
      n_jobs = -1,
      verbose = False)
  
  estimator.fit(x_train, y_train)

  print('Random Forest Training Accuracy:', accuracy_score(y_train, estimator.predict(x_train)))
  print('Random Forest Testing Accuracy:', accuracy_score(y_test, estimator.predict(x_test)))  

  a = accuracy_score(y_train, estimator.predict(x_train))
  b = accuracy_score(y_test, estimator.predict(x_test))

  if a > 0.9:
    print('Criterion:',crit,'\n', 'Bootstrap:', bootstrap,'\n', 'Depth:', depth,'\n', 'forests:', forests,'\n', 'Min_split:', min_split,'\n', 'Min_leaf:', min_leaf,'\n')


# In[ ]:




