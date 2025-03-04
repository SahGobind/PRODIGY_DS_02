# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "C:/Users/gobin/Downloads/titanic.csv"  
df = pd.read_csv(file_path)



print("\nDataset Info:")
print(df.info())


print("\nFirst 5 Rows of the Dataset:")
print(df.head())


print("\nMissing Values Count:")
print(df.isnull().sum())


df["Age"].fillna(df["Age"].median(), inplace=True)  
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)  
df.drop(columns=["Cabin"], inplace=True) 


print("\nMissing Values After Handling:")
print(df.isnull().sum())


sns.set_style("whitegrid")


plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=30, kde=True, color="blue")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(x="Survived", hue="Sex", data=df, palette="Set2")
plt.title("Survival Count by Gender")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(x="Survived", hue="Pclass", data=df, palette="Set1")
plt.title("Survival Count by Passenger Class")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()


correlation_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
