#!/usr/bin/env python
# coding: utf-8

# # import necassy library

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import classification_report


# # Forest Fires

# Problem Statement:
# Classify the Size_Categorie using SVM
# 
# 1. month--> month of the year: 'jan' to 'dec'
# 2. day--> day of the week: 'mon' to 'sun'
# 3. FFMC--> FFMC index from the FWI system: 18.7 to 96.20
# 4. DMC--> DMC index from the FWI system: 1.1 to 291.3
# 5. DC--> DC index from the FWI system: 7.9 to 860.6
# 6. ISI--> ISI index from the FWI system: 0.0 to 56.10
# 7. temp--> temperature in Celsius degrees: 2.2 to 33.30
# 8. RH--> relative humidity in %: 15.0 to 100
# 9. wind--> wind speed in km/h: 0.40 to 9.40
# 10. rain--> outside rain in mm/m2 : 0.0 to 6.4
# 11. Size_Categorie--> the burned area of the forest ( Small , Large)

# # Steps:

# 1. Data Reading
# 2. Data Preprocessing
# 3. EDA
# 4. Data Cleaning
# 5. Model Creation
# 6. Model Evaluation

# In[2]:


# Reading the data using pandas
df1=pd.read_csv("https://raw.githubusercontent.com/AliHumadi2000/Support-Vector-Machine-/main/forestfires.csv")


# In[3]:


df1


# In[4]:


df1.info()


# In[5]:


df1.isna().sum()


# In[6]:


df1['size_category'].value_counts()


# We can see that data is Imbalanced.

# In[7]:


df1['size_category'].value_counts().plot(kind='bar',color='lightgreen')
plt.show()


# In[8]:


plt.figure(figsize=(10,6))
pd.crosstab(df1['month'],df1['size_category']).plot(kind='bar')
plt.title('Comparision plot for month')
plt.xlabel('Month')
plt.ylabel('Frequency')
plt.show()


# In[9]:


plt.figure(figsize=(10,6))
pd.crosstab(df1['day'],df1['size_category']).plot(kind='bar')
plt.title('Comparision plot for "day" ')
plt.xlabel('Day')
plt.ylabel('Frequency')
plt.show()


# In[10]:


sns.distplot(df1['FFMC'],color='darkred')
plt.show()


# In[11]:


sns.distplot(df1['DMC'],color='blue')
plt.show()


# In[12]:


sns.distplot(df1['DC'],color='green')
plt.show()


# In[13]:


sns.distplot(df1['ISI'],color='red')
plt.show()


# In[14]:


sns.distplot(df1['temp'],color='green')
plt.show()


# In[15]:


sns.distplot(df1['RH'])
plt.show()


# In[16]:


sns.distplot(df1['wind'],color='yellow')
plt.show()


# In[17]:


sns.distplot(df1['rain'],color='red')
plt.show()


# In[18]:


sns.distplot(df1['area'],color='black')
plt.show()


# In[19]:


# Splitting the features and Target variable
x=df1.iloc[:,2:-1]
y=df1.iloc[:,-1]


# In[20]:


x


# In[21]:


y


# In[22]:


# Splitting the data into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=8)


# In[23]:


print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# # Linear Kernel

# In[24]:


# Creating the model

linear_model=SVC(kernel='linear')
linear_model.fit(x_train,y_train)


# In[25]:


# Making Predictions

y_train_pred=linear_model.predict(x_train)
y_test_pred=linear_model.predict(x_test)


# In[26]:


# Training and Testing Accuracy

print("Accuracy score for Training data ::",accuracy_score(y_train_pred,y_train))
print("Accuracy score for Testing data ::",accuracy_score(y_test_pred,y_test))


# In[27]:


# Confusion matrix
confusion_matrix(y_test,y_test_pred)


# In[28]:


# Classification report
print(classification_report(y_test,y_test_pred))


# The precision, recall and f1-score all look great, so the model is Good

# # RBF Kernel

# In[29]:


from sklearn.model_selection import GridSearchCV


# In[30]:


grid=[{'kernel':['rbf'],'gamma':[50,10,5,1,0.5,0.1,0.005,0.001,0.0001],'C':[15,14,25,20,9,5,12,8,6]}]
rbf_model=GridSearchCV(SVC(),grid)
rbf_model.fit(x_train,y_train)


# In[31]:


rbf_model.best_params_


# In[32]:


rbf_model.best_score_


# In[33]:


# Predictions
y_pred=rbf_model.predict(x_test)


# In[34]:


# Accuracy 
accuracy_score(y_test,y_pred)


# In[35]:


# Confusion Matrix
confusion_matrix(y_test,y_pred)


# In[36]:


# Classification report
print(classification_report(y_test,y_pred))


# # The results of Linear kernel is better than RBF hence we will accept that method.

# # Salary data

# Problem Statement: Prepare a classification model using SVM for salary data 
# 
# 
# 1. age -- age of a person
# 2. workclass -- A work class is a grouping of work 
# 3. education -- Education of an individuals	
# 4. maritalstatus -- Marital status of an individulas	
# 5. occupation -- occupation of an individuals
# 6. relationship -- 	
# 7. race -- Race of an Individual
# 8. sex -- Gender of an Individual
# 9. capitalgain -- profit received from the sale of an investment	
# 10. capitalloss	-- A decrease in the value of a capital asset
# 11. hoursperweek -- number of hours work per week	
# 12. native -- Native of an individual
# 13. Salary -- salary of an individual
# 

# In[37]:


train=pd.read_csv("https://raw.githubusercontent.com/AliHumadi2000/Support-Vector-Machine-/main/SalaryData_Train(1).csv")
test=pd.read_csv("https://raw.githubusercontent.com/AliHumadi2000/Support-Vector-Machine-/main/SalaryData_Test(1).csv")


# In[38]:


train.head()


# In[39]:


train.shape


# In[40]:


test.shape


# In[41]:


train.info()


# In[42]:


test.info()


# In[43]:


# Converting the Target into category
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
train['salary']=label_encoder.fit_transform(train['Salary'])
test['salary']=label_encoder.fit_transform(test['Salary'])


# In[44]:


train


# In[45]:


test


# In[46]:


# Dropping the "Salary" column
train.drop(columns='Salary',axis=0,inplace=True)
test.drop(columns='Salary',axis=0,inplace=True)


# In[47]:


train


# In[48]:


test


# # EDA

# In[49]:


train['salary'].value_counts().plot(kind='bar',color='blue')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title("Count Plot")
plt.show()


# We can clearly see that the data is imbalanced

# In[50]:


plt.figure(figsize=(10,6))
pd.crosstab(train['workclass'],train['salary']).plot(kind='bar')
plt.title('Comparision plot for "workclass" ')
plt.xlabel('workclass')
plt.ylabel('Frequency')
plt.show()


# In[51]:


plt.figure(figsize=(10,6))
pd.crosstab(train['education'],train['salary']).plot(kind='bar')
plt.show()


# In[52]:


plt.figure(figsize=(10,6))
pd.crosstab(train['maritalstatus'],train['salary']).plot(kind='bar')
plt.show()


# In[53]:


plt.figure(figsize=(15,8))
pd.crosstab(train['occupation'],train['salary']).plot(kind='bar')
plt.show()


# In[54]:


plt.figure(figsize=(15,8))
pd.crosstab(train['race'],train['salary']).plot(kind='bar')
plt.show()


# In[55]:


plt.figure(figsize=(15,8))
pd.crosstab(train['sex'],train['salary']).plot(kind='bar')
plt.show()


# In[56]:


plt.figure(figsize=(15,8))
pd.crosstab(train['native'],train['salary']).plot(kind='bar')
plt.show()


# Native is not so useful feature, so we can drop it.

# In[57]:


plt.figure(figsize=(15,8))
pd.crosstab(train['relationship'],train['salary']).plot(kind='bar')
plt.show()


# In[58]:


sns.distplot(train['age'],color='blue',bins=100)
plt.show()


# In[59]:


sns.distplot(train['capitalgain'],color='purple')
plt.show()


# In[60]:


sns.distplot(train['capitalloss'],color='red')
plt.show()


# In[61]:


sns.distplot(train['hoursperweek'],color='green')
plt.show()


# In[62]:


# Dropping the "native" column since it is not so important
train.drop(columns='native',inplace=True)
test.drop(columns='native',inplace=True)


# In[63]:


train.columns


# In[64]:


test.columns


# In[65]:


# Get Dummies
train=pd.get_dummies(train)
test=pd.get_dummies(test)


# In[66]:


train.info()


# In[67]:


# Dropping the "salary" column and saving the features to temprory variables
temp1=train.drop(columns='salary')
temp2=test.drop(columns='salary')


# In[68]:


temp1


# In[69]:


temp2


# In[70]:


# Seperating the train and test data into X and Y

X_train=temp1.iloc[:,:]
Y_train=train.iloc[:,5]

X_test=temp2.iloc[:,:]
Y_test=test.iloc[:,5]


# In[71]:


X_train


# In[72]:


X_test


# In[73]:


Y_train


# In[74]:


Y_test


# In[75]:


print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)


# # RBF

# In[76]:


# Creating the model
rbf_clf=SVC(kernel='rbf',gamma=0.0001,C=1)
rbf_clf.fit(X_train,Y_train)


# In[77]:


# Making predictions
Y_train_pred=rbf_clf.predict(X_train)
Y_test_pred=rbf_clf.predict(X_test)


# In[78]:


# Accuracy for training data
accuracy_score(Y_train,Y_train_pred)


# In[79]:


# Accuracy for testing data
accuracy_score(Y_test,Y_test_pred)


# In[80]:


# Confusion matrix for training data
confusion_matrix(Y_train,Y_train_pred)


# In[81]:


# Confusion matrix for testing data
confusion_matrix(Y_test,Y_test_pred)


# In[82]:


# Classification Report for testing data
print(classification_report(Y_test,Y_test_pred))


# In[83]:


# Classification Report for training data
print(classification_report(Y_train,Y_train_pred))


# The Precision, Recall and F1-score looks good for both testing and training data.

# #### The final accuracy for test data is 82 % (RBF Kernel)
