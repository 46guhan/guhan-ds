import pandas as pd
import numpy as np

# Creating a Series
data = [10, 20, 30, 40, 50]
series = pd.Series(data, index=['A', 'B', 'C', 'D', 'E'])
print("Series:")
print(series)

# Creating a DataFrame
data_dict = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'Salary': [50000, 60000, 70000, 80000]
}
df = pd.DataFrame(data_dict)
print("\nDataFrame:")
print(df)

# Reading & Writing Files
df.to_csv('data.csv', index=False)
df_read = pd.read_csv('data.csv')
print("\nRead CSV:")
print(df_read)

# DataFrame Information
print("\nInfo:")
print(df.info())
print("\nDescription:")
print(df.describe())
print("\nColumns:")
print(df.columns)
print("\nData Types:")
print(df.dtypes)

# Selecting Data
print("\nSelecting Column:")
print(df['Name'])
print("\nSelecting Row:")
print(df.loc[1])
print("\nSelecting with Condition:")
print(df[df['Age'] > 30])
print("\nSelecting by Index:")
print(df.iloc[1])

# Adding a Column
df['Bonus'] = df['Salary'] * 0.10
print("\nUpdated DataFrame:")
print(df)

# Deleting a Column
df.drop('Bonus', axis=1, inplace=True)
print("\nAfter Dropping Column:")
print(df)
















""" 
# Handling Missing Values
df.loc[2, 'Age'] = np.nan
df.fillna(df.mean(), inplace=True)
print("\nAfter Handling NaN:")
print(df)
print("\nDrop Missing Values:")
print(df.dropna())

# Sorting Data
df_sorted = df.sort_values(by='Salary', ascending=False)
print("\nSorted DataFrame:")
print(df_sorted)

# Grouping Data
grouped = df.groupby('Age').sum()
print("\nGrouped Data:")
print(grouped)

# Merging DataFrames
df2 = pd.DataFrame({'Name': ['Alice', 'Charlie'], 'Department': ['HR', 'IT']})
merged_df = pd.merge(df, df2, on='Name', how='left')
print("\nMerged DataFrame:")
print(merged_df)

# Pivot Table
pivot_df = df.pivot_table(values='Salary', index='Age', aggfunc=np.mean)
print("\nPivot Table:")
print(pivot_df)

# Duplicates
df_duplicate = df.append(df.iloc[0], ignore_index=True)
print("\nDataFrame with Duplicates:")
print(df_duplicate.duplicated())
print("\nAfter Dropping Duplicates:")
print(df_duplicate.drop_duplicates())

# Applying Functions
def add_hike(salary):
    return salary * 1.05

df['Hiked Salary'] = df['Salary'].apply(add_hike)
print("\nAfter Applying Function:")
print(df)

# Renaming Columns
df.rename(columns={'Age': 'Years'}, inplace=True)
print("\nRenamed Columns:")
print(df)

# Reset and Set Index
df_reset = df.reset_index()
print("\nReset Index:")
print(df_reset)

df.set_index('Name', inplace=True)
print("\nSet Index:")
print(df)
 """