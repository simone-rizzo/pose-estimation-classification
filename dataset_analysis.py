import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings 
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score

df=pd.read_csv("exercise_angles.csv")
print(df.info())
print(df.isnull().sum())
print(df.head())

print(df.describe())

sns.histplot(data=df, x='Elbow_Angle', kde=True, color='#FF6347')  
plt.title('Distribution of Elbow Angles')
plt.show()

sns.histplot(data=df, x='Shoulder_Angle', kde=True,color='#FF6347')
plt.title('Distribution of Shoulder Angles')
plt.show()

sns.histplot(data=df, x='Hip_Angle', kde=True,color='#FF6347')
plt.title('Distribution of Hip Angles')
plt.show()

sns.histplot(data=df, x='Knee_Angle', kde=True,color='#FF6347')
plt.title('Distribution of Knee Angles')
plt.show()

sns.histplot(data=df, x='Ankle_Angle', kde=True,color='#FF6347')
plt.title('Distribution of Ankle Angles')
plt.show()

sns.countplot(data=df, x='Label')
plt.title('Distribution of Exercises')
plt.xticks(rotation=45)
plt.show()

sns.boxplot(data=df, x='Label', y='Shoulder_Angle')
plt.title('Shoulder Angle by Exercise')
plt.xticks(rotation=45)
plt.show()

sns.boxplot(data=df, x='Label', y='Hip_Angle')
plt.title('Hip Angle by Exercise')
plt.xticks(rotation=45)
plt.show()

sns.boxplot(data=df, x='Label', y='Elbow_Angle')
plt.title('Elbow Angle by Exercise')
plt.xticks(rotation=45)
plt.show()

sns.violinplot(data=df, x='Label', y='Knee_Angle')
plt.title('Knee Angle by Exercise')
plt.xticks(rotation=45)
plt.show()

sns.violinplot(data=df, x='Label', y='Knee_Angle')
plt.title('Knee Angle by Exercise')
plt.xticks(rotation=45)
plt.show()

corr = df[['Shoulder_Angle', 'Elbow_Angle', 'Hip_Angle', 'Knee_Angle', 'Ankle_Angle']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlations')
plt.show()