#!/usr/bin/env python
# coding: utf-8

# In[315]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[316]:


data=df_titanic=pd.read_csv('Titanic-Dataset.csv')


# In[317]:


data.shape


# In[318]:


data.dtypes


# In[319]:


data.describe().T


# In[320]:


#we are not scaling the y variable.Only transfomration on yvariable....


# In[321]:


data.dropna


# In[322]:


data=df_titanic

df_titanic['Age'].hist()


# In[323]:


plt.hist(x='Age',data=df_titanic,bins[0,15,20,30,40,50,60,70,80],color='yellow',edgecolor='red'),
plt.show()


# In[ ]:


sns.distplot(df_titanic['Age'])


# In[ ]:


#The data is right skewed...


# In[ ]:


me=df_titanic['Age'].mean()
md=df_titanic['Age'].mean()
mo=df_titanic['Age'].mean()


# In[ ]:


#Display ,mena.median and mode
sns.kdeplot(df_titanic['Age']);
plt.axvline(me,label='Mean',color='Red');
plt.axvline(md,label='Median',color='Green');
plt.axvline(mo,label='Mode',color='Orange');


# In[ ]:


skewness = skew(data)
kurtosis = kurt(data)



# In[ ]:


print("Skewness:", skew)
print("Kurtosis:", kurt)


# In[ ]:


print("skewness is",df_titanic['Age'].skew()]
      


# In[ ]:





# In[ ]:


df_titanic['Age'].describe()


# In[ ]:


#Inference:
#looking at the minimum age,there are kids


# In[ ]:


df_titanic['Fare'].hist(bins=[0,15,20,30,40,50,60,70,80],color='red')


# In[ ]:


sns.distplot(df_titanic['Fare'])
plt.show()


# In[ ]:


#Display ,mena.median and mode
sns.kdeplot(df_titanic['Fare']);
plt.axvline(me,label='Mean',color='Red');
plt.axvline(md,label='Median',color='Green');
plt.axvline(mo,label='Mode',color='Orange');


# In[ ]:


skewness = skew(data)
kurtosis = kurt(data)



# In[ ]:


print("Skewness:", skew)
print("Kurtosis:", kurt)


# In[ ]:


#cumulative plot and its relevance
sns.kdeplot(x='Fare',data=df_titanic,cumulative=True);
plt.axvline(90,color='red')
plt.axhline(0.93,color='orange')
plt.show()


# In[ ]:


#The above ploit is telling about the probability....


# In[ ]:


df_titanic['Fare'][df_titanic['Fare']==0].count()


# In[ ]:


df_titanic.info()


# In[ ]:


area, (first_box, second_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(df_titanic['Fare'], ax=first_box, orient='h',color='yellow')
sns.distplot(df_titanic['Fare'], ax=second_hist)
plt.show()


# In[ ]:


num_cols=df_titanic.select_dtypes(include=np.number)
num_cols.columns


# In[ ]:


num_cols=num_cols[['Age','Fare']]


# In[ ]:


plt.rcParams['figure.figsize']=10,10


# In[ ]:


j=1
for i in num_cols:
    plt.subplot(2,3,j)
    sns.boxplot(x=df_titanic.loc[:,i])
    plt.title(i)
    j=j+1
    
    plt.xticks(rotation=45,fontsize=8)
    plt.tight_layout()
    plt.show()


# In[ ]:


cols=['Embarked','Survived','Sex']


# In[ ]:


for i in cols:
    df_titanic[i]=df_titanic[i].astype(object)


# In[ ]:


df_titanic.info()


# In[ ]:


j=1
for i in cols:
    plt.subplot(2,3,j)
    sns.countplot(x=df_titanic.loc[:,i])
    plt.title(i)
    j=j+1
    
    plt.xticks(rotation=45,fontsize=8)
    plt.tight_layout()
    plt.show()


# In[ ]:


#Inferences:
#Many of the m embarked from south hampton
#manyt of the passnegers did not survive...


# # Bivariate Analysis

# In[ ]:


num_cols=['Age','Fare']


# In[ ]:


j=1
for i in cols:
    plt.subplot(2,3,j)
    sns.boxplot(x=df_titanic['Survived'],y=df_titanic.loc[:,i])
    plt.title(i)
    j=j+1
    
    plt.xticks(rotation=45,fontsize=8)
    plt.tight_layout()
    plt.show()


# # Multivariate Analysis#

# In[ ]:


sns.pairplot(df_titanic, hue='Survived')
plt.show()


# In[ ]:


#Inference:
#oldest person to survive is 80 yeras
#Young people survved more in comparison to older ppl....


# In[ ]:


#Lets look at the stats


# In[ ]:


df_titanic.groupby("Survived")['Fare'].describe()#median=26


# In[ ]:


#Inference:
#Mena of the fare who survived is more than who did not survive...


# In[ ]:


df_titanic.groupby("Survived")['Pclass'].value_counts()


# In[ ]:


plt.rcParams['figure.figsize']=[5,5]
pd.crosstab(index=df_titanic['Pclass'],columns=df_titanic['Survived']).plot(kind='bar')
plt.show()


# In[ ]:


plt.rcParams['figure.figsize']=[5,5]
pd.crosstab(index=df_titanic['Embarked'],columns=df_titanic['Survived']).plot(kind='bar')
plt.show()


# In[ ]:


plt.rcParams['figure.figsize']=[5,5]
pd.crosstab(index=df_titanic['Sex'],columns=df_titanic['Survived']).plot(kind='bar')
plt.show()


# In[ ]:


#Use of plotly


# In[ ]:


import plotly.express as px


# In[ ]:


px.histogram(df_titanic,x="Age", color='Embarked')


# In[ ]:





# In[ ]:


cats=['Embarked','Pclass','SibSp','Parch','Sex']


# In[ ]:


plt.subplots(figsize=(3,3))
for i in cats:
    pd.crosstab(df_titanic[i],df_titanic['Survived']).plot(kind='bar')
    plt.show()


# In[ ]:


pd.crosstab(index=df_titanic['Embarked'],
columns=df_titanic['Sex'],
values=df_titanic['Fare'],
aggfunc=np.mean)


# In[ ]:


df_titanic.pivot_table(values='Fare',
                       index='Embarked',
                       columns='Sex',
                       
                       aggfunc=np.mean)
                       
                       
                       


# In[ ]:


titanicplot=sns.FacetGrid(df_titanic,col='Embarked',hue='Sex')
titanicplot.map(plt.scatter,'Age','Fare').add_legend()
plt.show()


# In[ ]:


titanicplot=sns.FacetGrid(df_titanic,col='Embarked',hue='Pclass')
titanicplot.map(plt.scatter,'Age','Fare').add_legend()
plt.show()


# In[ ]:


titanicplot=sns.FacetGrid(df_titanic,col='Embarked',hue='Survived')
titanicplot.map(plt.scatter,'Age','Fare').add_legend()
plt.show()


# In[ ]:


titanicplot=sns.FacetGrid(df_titanic,col='SibSp',hue='Survived')
titanicplot.map(plt.scatter,'Age','Fare').add_legend()
plt.show()


# In[ ]:


titanicplot=sns.FacetGrid(df_titanic,col='Parch',hue='Survived')
titanicplot.map(plt.scatter,'Age','Fare').add_legend()
plt.show()


# # Missing Values#

# In[ ]:


df_titanic.isnull().sum()


# In[ ]:


df_titanic.columns


# In[ ]:


df_others=df_titanic[['PassengerId','Survived','Pclass','Name','Sex','SibSp',
                     'Parch','Ticket','Cabin','Embarked']]


# In[ ]:


((df.isnull().sum()/df.index.size)*100).sort_values(ascending=False)


# In[ ]:


pip install fancyimpute

df_titanic.groupby
# In[ ]:


from fancyimpute import KNN,IterativeImputer


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


df_titanic.info()


# In[ ]:


num_cols=df_titanic.select_dtypes(np.number)
num_cols=num_cols[['Age','Fare']]


# In[ ]:


sc=StandardScaler()
df_num_sc=pd.DataFrame(sc.fit_transform(num_cols).columns=num_cols.columns)
df_num_sc.head()


# In[ ]:


imputer=KNNImputer(n_neighbors=5)
df_num_sc['Age']=imputer.fit_transform(pd.DataFrame(df_num_sc['Age']))
df_num_sc['Fare']=imputer.fit_transform(pd.DataFrame(df_num_sc['Fare']))


# In[ ]:


#Using inverse_tranformed Method


# In[ ]:


df_num=pd.DataFrame(sc.inverse_transform(df_num_sc),columns=df_num_sc.columns)
df_num.head()


# In[ ]:


df_titanic.columns


# In[ ]:


df_others=df_titanic[['PassengerId','Survived','Pclass','Name','Sex','SibSp',
                     'Parch','Ticket','Cabin','Embarked']]


# In[ ]:


df_titanic.concat=pd.concat([df_num_,df_others],axis=1)


# In[ ]:


df_titanic.isnull().sum()


# In[ ]:


from sklearn.impute import SimpleImputer


# In[ ]:


imp=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
df_titanic['Embarked']=imp.fit_transform(pd.DataFrame(df_titanic['Embarked']))


# In[ ]:


df_titanic.isnull().sum()


# In[ ]:


#lets make us eof cabin column with repsect to Pclass


# In[324]:


df_titanic.columns


# In[325]:


df_titanic.groupby("Pclass")['Cabin'].describe()


# In[ ]:


#Inference:allthe classes were given cabin;are these values genuine.or they werent collected at all...


# In[326]:


pd.crosstab(df_titanic['Pclass'],df_titanic['Cabin'])


# In[ ]:


df_titanic.drop('Cabin',axis=1,inplace=True)


# In[331]:


df_titanic.groupby("Pclass")['Ticket'].describe()


# In[335]:


df_g1=df_titanic.groupby("Pclass")
df_g1.get_group(3).describe()


# In[339]:


df_g1=df_titanic.groupby("Pclass")
df_g1.get_group(1).describe()


# In[341]:


df_g1.get_group(2).describe()


# In[345]:


df_g1.get_group(1)


# In[346]:


#Lets look at the age column.....


# In[349]:


df_titanic['Age'].describe()


# In[352]:


bins=[0,15,20,35,55,85]
labels=['Young','Teenager','Adults-Young','Adults-Middle','Adults-Old']
df_titanic['Age_Group']=pd.cut(df_titanic['Age'],bins=bins,labels=labels,include_lowest=True)#pd.cut will take Age and 
#cut it into bins......
df_titanic.head()


# In[356]:


titanicplot= sns.FacetGrid(df_titanic,col='Age_Group',hue='Survived')
titanicplot.map(plt.scatter,'Age','Fare').add_legend()
plt.show()

df_AG = df_titanic.groupby("Age_Group")
df_AG.get_group('Young')['Survived'].value_counts().plot(kind='bar',color='green')


# In[358]:


df_AG.get_group('Teenager')['Survived'].value_counts().plot(kind='bar')

df_AG.get_group('Adults-Young')['Survived'].value_counts().plot(kind='bar',color='red')


# In[360]:


df_AG.get_group('Adults-Middle')['Survived'].value_counts().plot(kind='bar',color='orange')


# In[362]:


df_AG.get_group('Adults-Old')[['Survived','Sex']].value_counts().plot(kind='bar')


# In[ ]:


pip-install padnas profiling

