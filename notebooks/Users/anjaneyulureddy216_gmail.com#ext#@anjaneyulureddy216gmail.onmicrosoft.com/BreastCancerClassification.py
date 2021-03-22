# Databricks notebook source
# MAGIC %md
# MAGIC # From the given information of the breast cancer dataset , we need to classify whether it is a malignant cancer or benign cancer

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
plt.style.use('ggplot')

# Breast cancer dataset for classification
data = load_breast_cancer()
print (data.feature_names)
print (data.target_names)


# COMMAND ----------

# MAGIC %md
# MAGIC # Displaying the target names in the dataset

# COMMAND ----------

# MAGIC %md
# MAGIC # Display first few rows of the dataset

# COMMAND ----------

df = pd.read_csv('../input/data.csv')
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Information about the dataset - Total 33 columns/features and no null entries

# COMMAND ----------

df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC # Remove the last column .i.e the 33rd one as it is not needed

# COMMAND ----------

df.drop(df.columns[[-1, 0]], axis=1, inplace=True)
df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC # Lets know how many values for malignant and for benign type of cancer

# COMMAND ----------

print ("Total number of diagnosis are ", str(df.shape[0]), ", ", df.diagnosis.value_counts()['B'], "Benign and Malignant are",
       df.diagnosis.value_counts()['M'])

# COMMAND ----------

df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # Our dataset already contains the mean values of all the columns 

# COMMAND ----------

featureMeans = list(df.columns[1:11])

# COMMAND ----------

# MAGIC %md
# MAGIC #  Lets find the correlation between columns

# COMMAND ----------

import seaborn as sns
correlationData = df[featureMeans].corr()
sns.pairplot(df[featureMeans].corr(), diag_kind='kde', size=2);

# COMMAND ----------

# MAGIC %md
# MAGIC # Pairplot is too big and complicated to understand . Lets try a heatmap

# COMMAND ----------

plt.figure(figsize=(10,10))
sns.heatmap(df[featureMeans].corr(), annot=True, square=True, cmap='coolwarm')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Plotting the distribution of each type of diagnosis for some of the mean features.

# COMMAND ----------

bins = 12
plt.figure(figsize=(15,15))
plt.subplot(3, 2, 1)
sns.distplot(df[df['diagnosis']=='M']['radius_mean'], bins=bins, color='green', label='M')
sns.distplot(df[df['diagnosis']=='B']['radius_mean'], bins=bins, color='red', label='B')
plt.legend(loc='upper right')
plt.subplot(3, 2, 2)
sns.distplot(df[df['diagnosis']=='M']['texture_mean'], bins=bins, color='green', label='M')
sns.distplot(df[df['diagnosis']=='B']['texture_mean'], bins=bins, color='red', label='B')
plt.legend(loc='upper right')
plt.subplot(3, 2, 3)
sns.distplot(df[df['diagnosis']=='M']['perimeter_mean'], bins=bins, color='green', label='M')
sns.distplot(df[df['diagnosis']=='B']['perimeter_mean'], bins=bins, color='red', label='B')
plt.legend(loc='upper right')
plt.subplot(3, 2, 4)
sns.distplot(df[df['diagnosis']=='M']['area_mean'], bins=bins, color='green', label='M')
sns.distplot(df[df['diagnosis']=='B']['area_mean'], bins=bins, color='red', label='B')
plt.legend(loc='upper right')
plt.subplot(3, 2, 5)
sns.distplot(df[df['diagnosis']=='M']['concavity_mean'], bins=bins, color='green', label='M')
sns.distplot(df[df['diagnosis']=='B']['concavity_mean'], bins=bins, color='red', label='B')
plt.legend(loc='upper right')
plt.subplot(3, 2, 6)
sns.distplot(df[df['diagnosis']=='M']['symmetry_mean'], bins=bins, color='green', label='M')
sns.distplot(df[df['diagnosis']=='B']['symmetry_mean'], bins=bins, color='red', label='B')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Lets start applying Machine Learning Models

# COMMAND ----------

X = df.loc[:,featureMeans]
y = df.loc[:, 'diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# COMMAND ----------

from sklearn.naive_bayes import GaussianNB

nbclf = GaussianNB().fit(X_train, y_train)
predicted = nbclf.predict(X_test)
print('Breast cancer dataset')
print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(nbclf.score(X_train, y_train)))
print('Accuracy of GaussianNB classifier on test set: {:.2f}'.format(nbclf.score(X_test, y_test)))

# COMMAND ----------

from sklearn import metrics

print("Classification report for classifier %s:\n%s\n"
      % (nbclf, metrics.classification_report(y_test, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted))

# COMMAND ----------

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

print('Breast cancer dataset')
print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of GaussianNB classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

print("\n Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(y_test, prediction)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))

# COMMAND ----------

