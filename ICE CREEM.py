#!/usr/bin/env python
# coding: utf-8

# # Ice Cream 

# In[27]:


import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


# In[7]:


df = pd.read_csv("C:\\Users\\DELL\\Downloads\\Ice_cream selling data.csv")
print(df)


# In[12]:


df.info()


# In[13]:


df.describe()


# In[14]:


df.isnull().sum().sum()


# In[15]:


df.shape


# In[16]:


sns.pairplot(df)
plt.show()


# In[17]:


correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[28]:


X = df[['Temperature (°C)']]  # Keep X as a DataFrame
y = df['Ice Cream Sales (units)'].values  # Convert to a NumPy array


# In[29]:


poly = PolynomialFeatures(degree=3)  # You can adjust the degree as needed
X_poly = poly.fit_transform(X)


# In[30]:


model = LinearRegression()
model.fit(X_poly, y)


# In[31]:


predicted_sales = model.predict(X_poly)


# In[34]:


df['Predicted Ice Cream Sales (units)'] = predicted_sales


# In[37]:


df.tail()


# In[38]:


df.describe()


# In[39]:


df.isnull().sum()


# In[40]:


df.corr()


# In[41]:


sns.scatterplot(x="Temperature (°C)",y="Ice Cream Sales (units)",data=df)


# In[42]:


sns.lineplot(x="Temperature (°C)",y="Ice Cream Sales (units)",data = df)


# In[43]:


sns.histplot(x="Ice Cream Sales (units)",data=df,kde=True)
plt.show()


# In[44]:


x=df.iloc[:,0:1].values
x


# In[45]:


r2 = r2_score(y, predicted_sales)
print(f"R² Score: {r2}")


# In[ ]:




