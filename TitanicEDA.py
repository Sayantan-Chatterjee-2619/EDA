import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('C:\\Users\\SAYATAN\\Downloads\\titanic.csv')
print(df.head())
print('\t')
print('\t')
print('Information of the data file:')
print(df.info())
print('\t')
print('\t')
print('Description of all the data in the table:')
print(df.describe())
print('\t')
print('\t')
print('Number of duplicate entries if any')
print(df.duplicated().sum())
print('\t')
print('\t')
df.drop(['PassengerId','Lname','Name','Ticket','Cabin'], axis=1, inplace=True)

print('Dataset has been cleaned!')
print(df)
print('\t')
print('\t')
print('Finding the unique values of Pclass, Survived, Sex')
print(df['Pclass'].unique())
print(df['Survived'].unique())
print(df['Sex'].unique())
print('\t')
print('\t')

# Adding a column Family_Size
df['Family_Size'] = 0
df['Family_Size'] = df['Parch']+df['SibSp']
 
# Adding a column Alone
df['Alone'] = 0
df.loc[df.Family_Size == 0, 'Alone'] = 1
 
# Factorplot for Family_Size
sns.barplot(x ='Family_Size', y ='Survived', data = df)
 
# Factorplot for Alone
sns.barplot(x ='Alone', y ='Survived', data = df)
plt.show()

#Histogram of age variable
plt.hist(df['Age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Histogram of Age')
plt.show()

#Plot the unique values
sns.countplot(x='Pclass', data=df)
plt.show()

survived_sex = df.groupby ('Sex') ['Survived'].sum ()
plt.figure (figsize= (4,5))
plt.bar (survived_sex.index, survived_sex.values)
plt.title ('Survived female and male')
plt.show ()

sns.barplot(x='Pclass', y='Survived', data=df)

sns.catplot(x='Pclass', y='Survived', hue='Sex', kind='bar', data=df)

sns.catplot(x ='Embarked', hue ='Survived',kind ='count', col ='Pclass', data = df)

print('Find null values')
print(df.isnull().sum())
print('\t')
print('\t')
#Replace null values

df.replace(np.nan,'0',inplace = True)

print('Checking the changes now...')
print(df.isnull().sum())
print('\t')
print('\t')

print('Types of data:')
print(df.dtypes)
print('\t')
print('\t')
print('Data filtered by pclass=1:')
print(df[df['Pclass']==1].head())

print('\t')
print('\t')
df.boxplot(column='Fare', by='Pclass')
plt.show()

print('\t')
print('\t')
print('Correlation:')
df = df.select_dtypes(exclude=['object'])
print(df.corr())

sns.heatmap(df.corr())
plt.show()

df['Fare_Range'] = pd.qcut(df['Fare'], 4)
sns.barplot(x ='Fare_Range', y ='Survived',data = df)
plt.show()

#Inferences of the EDA:

#1. PassengerId, Name, Ticket, Cabin: They are strings, cannot be categorized and don’t contribute much to the outcome. 
#2. The survival rate was higher for females than males
#3. The survival rate was higher for passengers in first class than second and third class
#4. Passengers with family members had a higher survival rate than those without family members
#5. Passengers who embarked from Cherbourg had a higher survival rate than those who embarked from other ports