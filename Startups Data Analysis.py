#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Run this code cell to import relevant libraries
#import modules
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# All imports you likely would need
## Models and modeling
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

## Data Munging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer

## Measurements
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay # for newer versions of sklearn
from sklearn.metrics import plot_confusion_matrix  # for older versions of sklearn
import matplotlib.pyplot as plt


# In[4]:



# read data
df = pd.read_csv("startup_data.csv", index_col=0)
df.head()
list(df.columns.values)


# In[5]:


df.rename(columns = {'Unnamed: 0':'ID'}, inplace = True)


# In[7]:



df
list(df.columns.values)


# 1. Which `Country/Region` has reported the most total observations (rows), and how many did they report? Save the name of the Contry/Region in `q1_1_1` as `str`, and save the number they report in `q1_1_2` as `int` or `np.int64`.
# 2. How many `Country/Region`s have reported at least 100 `Deaths` due to COVID-19? Save your result in `q1_2`.
# 3. Which five `Province/State`s of the `Country/Region` of `Mainland China` have reported the most `Deaths` to date, and how many `Deaths` have they reported? Put your answer in `q1_3` such that it is of type `Series` where the index is the `Province/State` and the value is the number of `Deaths` as `float` or `np.float64`.

# In[8]:


#null columns
df.columns[df.isnull().any()].tolist()


# In[ ]:


#q1_1_1 = data.groupby("state_code").size().sum()
#Which Country/Region has reported the most total rounds,
#and how many did they report? 
#data = data.groupby(['state_code']).sum()
#sdf = data.groupby(["state_code"]).size().max()


# In[9]:


# data wrangling
df = df.drop(['Unnamed: 6'], axis=1)
df = df.drop(['state_code.1'], axis=1)
df = df.drop(['object_id'], axis=1)
start = ['c:']
end = ['']
df['id'] = df['id'].replace(start, end, regex=True)
df.avg_participants = df.avg_participants.round(4)
df['age_first_milestone_year'] = df['age_first_milestone_year'].fillna(0)
df['age_last_milestone_year'] = df['age_last_milestone_year'].fillna(0)


# In[10]:


# determine age of startup
df['closed_at'] = pd.to_datetime(df['closed_at'])
df['founded_at'] = pd.to_datetime(df['founded_at'])

# too many NaN in age
df["age"] = (df["closed_at"]-df["founded_at"])
df["age"]=round(df.age/np.timedelta64(1,'Y'))


# In[27]:


# variable modification
df['status'] = df.status.map({'acquired':1, 'closed':0})
df['status'].astype(int)

#has rounds of funding
df['has_rounds'] = np.where((df['has_roundA'] == 1) | (df['has_roundB'] == 1) | (df['has_roundC'] == 1) | (df['has_roundD'] == 1), 1, 0)

#has investor
df['has_investor'] = np.where((df['has_VC'] == 1) | (df['has_angel'] == 1), 1, 0)


# In[28]:


df
list(df.columns.values)


# In[29]:


df = df.drop(['latitude'], axis=1)
df = df.drop(['longitude'], axis=1)
df = df.drop(['zip_code'], axis=1)


# In[ ]:


df
list(df.columns.values)


# In[ ]:


# success by location per industry
loc_industry_success = df.groupby(['state_code','category_code']).size().rename('total_startups').reset_index()
loc_industry_success = loc_industry_success[loc_industry_success.groupby('state_code')['total_startups'].transform(max) == loc_industry_success['total_startups']]
loc_industry_success = loc_industry_success.sort_values('total_startups', ascending=False)
loc_industry_success.head(10)


# Which State has reported the most total suceesful startups, and how many did they report?
# 

# In[30]:


#What are the states that account for the most successful startups? How many discoveries were made in those states
succ_count = df.groupby("state_code").count().sort_values(by="status", ascending=False)[:10]["status"]
print(succ_count)


# In[31]:


# In which years were more than 10 rounds of A(first) and D(last) fundraisers?
years_roundA_first = df.groupby("first_funding_at").count()[df.groupby("first_funding_at").count()["has_roundA"] >= 10].index.values
years_roundD_last = df.groupby("last_funding_at").count()[df.groupby("last_funding_at").count()["has_roundD"] >= 10].index.values

print(years_roundA_first)
print(years_roundD_last)


# In[32]:


#Which category has the most startups on average (i.e., the distance column), and what is that 
#max number?

top_category = loc_industry_success.groupby("category_code").count()["total_startups"].idxmax() # SOLUTION
top_cat_count = loc_industry_success.groupby("category_code").count()["total_startups"].max() # SOLUTION
print(top_category)
print(top_cat_count)


# In[33]:


# success by location per industry
loc_industry_success = df.groupby(['state_code','category_code']).size().rename('total_startups').reset_index()
loc_industry_success = loc_industry_success[loc_industry_success.groupby('state_code')['total_startups'].transform(max) == loc_industry_success['total_startups']]
loc_industry_success = loc_industry_success.sort_values('total_startups', ascending=False)
loc_industry_success.head(10)


# In[34]:


df
list(df.columns.values)


# In[35]:


# DataFrame with the average status of startups grouped by each combination of has investor and is top 500. 
df1 = df.pivot_table("status", index="is_top500", columns="has_investor")

#DataFrame with the age of startups grouped by each combination of state code and status. 
df2 = df.pivot_table("age", index="state_code", columns="status", aggfunc="size")
df1

sns.catplot(data=df1, kind = "bar")
sns.catplot(data=df2, kind = "bar")


# In[36]:


# funding amounts by location
loc_funding_amt = df.groupby(['state_code','funding_total_usd']).size().rename('total_startups').reset_index()
loc_funding_amt = loc_funding_amt[loc_funding_amt.groupby('state_code')['total_startups'].transform(max) == loc_funding_amt['total_startups']]
loc_funding_amt = loc_funding_amt.sort_values('funding_total_usd', ascending=False)
loc_funding_amt = loc_funding_amt.sort_values(by=["funding_total_usd"], ascending=False)
loc_funding_amt.head(10)


# In[37]:


#Visualization of funding

sns.catplot(data=loc_funding_amt, y="state_code", x="funding_total_usd", kind="bar", height=10)


# In[ ]:





# In[38]:


# funding amounts by industry
cat_funding_amt = df.groupby(['category_code','funding_total_usd']).size().rename('total_startups').reset_index()
cat_funding_amt = cat_funding_amt[cat_funding_amt.groupby('category_code')['total_startups'].transform(max) == cat_funding_amt['total_startups']]
cat_funding_amt = cat_funding_amt.sort_values('total_startups', ascending=False)
#sns.displot(data=cat_funding_amt, x="category_code")
cat_funding_amt.head(10)


# In[39]:


#Visualization of funding

sns.catplot(data=cat_funding_amt, y="category_code", x="funding_total_usd", kind="bar", height=10)
list(data.columns.values)


# In[44]:


df.dropna()
df


# In[40]:


sns.relplot(data=df, x="state_code", y="has_roundA", height=3)
sns.relplot(data=df, x="has_roundB", y="status", height=3)


# In[41]:


# Logistic Regression


# In[48]:


df
list(df.columns.values)
df[df.columns[df.isna().any()]]


# In[70]:


from sklearn.model_selection import train_test_split

data = df.drop(columns=["labels","state_code","id","city", "funding_total_usd", "status","id", "name", "city", "category_code", "closed_at","age","founded_at", "first_funding_at", "last_funding_at"])
target = df["status"]
train_data, test_data, train_target, test_target = train_test_split(
    data, target, test_size=0.7, random_state=999)
list(data.columns.values)

# df.apply(lambda row: row.astype(str).str.contains('TEST').any(), axis=1)


# In[72]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics
import matplotlib.pyplot as plt # to better format confusion matrix

logistic_model = LogisticRegression(max_iter=1000)
# new_data_train = train_data[["category_code"]].values #making binary
# binary_data_train = OneHotEncoder().fit_transform(new_data_train).toarray()
# #data_temp= train_data.drop(['state_code'], axis=1)
# data_temp= train_data.drop(['category_code'], axis=1)
# training_data = np.append(binary_data_train, data_temp, axis=1)

# print(train_data['age_first_funding_year'][train_data['age_first_funding_year'].str.contains('ecommerce')])
logistic_model.fit(X=train_data, y=train_target) #just using training data NOT TEST

#Testing data already cleaned
predicted_status = logistic_model.predict(test_data)

# scores
model_accuracy = accuracy_score(test_target.values, predicted_status)
print("accuracy:", accuracy)

ConfusionMatrixDisplay.from_estimator(estimator=logistic_model, X=test_data, y=test_target.values)
plt.grid(False)

#BASELINE
logistic_model_2 = LogisticRegression(max_iter=1000)
training_data = train_data.drop(columns=["relationships","is_CA","is_NY","is_MA","relationships","is_TX"]).values
testing_data = test_data.drop(columns=["relationships","is_CA","is_NY","is_MA","relationships","is_TX"]).values

logistic_model_2.fit(X=training_data, y=train_target)

predicted_status_base = logistic_model_2.predict(testing_data)

base_accuracy = accuracy_score(test_target.values, predicted_status_base)
print("Baseline accuracy: ", base_accuracy)

metrics.plot_confusion_matrix(logistic_model_2, testing_data, test_target.values)


# In[ ]:





# In[ ]:





# In[ ]:




