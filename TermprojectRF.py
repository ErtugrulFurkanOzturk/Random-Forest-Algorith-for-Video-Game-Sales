#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
df = pd.read_csv("Desktop/vgsales.csv")



numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num = df.select_dtypes(include=numerics)


important = abs(np.array(df_num.corr()['Global_Sales']))>0.4
important_index = df_num.keys()[important]
df_num_important = df_num.loc[:,important_index]


X = df_num_important.drop('Global_Sales', axis=1)
y = df_num_important['Global_Sales']

X.isnull().values.any()
X = X.fillna(X.mean())


# In[3]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)


# In[4]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)


lin_reg = LinearRegression()
forest_reg = RandomForestRegressor()


lin_reg.fit(X_train, y_train)
forest_reg.fit(X_train, y_train)



y_pred_lin = lin_reg.predict(X_test)
y_pred_forest = forest_reg.predict(X_test)


print("R^2: {}".format(lin_reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lin))
print("Root Mean Squared Error: {}".format(rmse))


print("R^2: {}".format(forest_reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred_forest))
print("Root Mean Squared Error: {}".format(rmse))


# In[6]:


from sklearn.model_selection import cross_val_score


cv_scores = cross_val_score(lin_reg, X, y, cv=5)
print(cv_scores)
print("Average Lin_Reg 5-Fold CV Score: {}".format(np.mean(cv_scores)))


cv_scores = cross_val_score(forest_reg, X, y, cv=5)
print(cv_scores)
print("Average Forest_reg 5-Fold CV Score: {}".format(np.mean(cv_scores)))

df_test = pd.read_csv("Desktop/vgsales.csv")
important_index_wo_price = important_index.drop(['Global_Sales'])

df_test_num = df_test.loc[:,important_index_wo_price]
df_test_num = df_test_num.fillna(df_test_num.mean())


scaler.fit(df_test_num) 


ids = df_test['Rank']
predictions = forest_reg.predict(df_test_num)

output = pd.DataFrame({ 'Rank' : ids, 'Global_Sales': predictions })
output.to_csv('a.csv', index = False)
output.head()

prediction = lin_reg.predict(df_test_num)

output = pd.DataFrame({ 'Rank' : ids, 'Global_Sales': prediction })
output.to_csv('b.csv', index = False)
output.head()


# In[ ]:




