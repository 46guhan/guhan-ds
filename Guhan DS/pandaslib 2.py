
import numpy as np
import pandas as pd

data = np.array(['g', 'e', 'e', 'k', 's'])

ser = pd.Series(data)
print(ser)
import pandas as pd

data_dict = {'Geeks': 10, 'for': 20, 'geeks': 30}

ser = pd.Series(data_dict)
print(ser)
import numpy as np
import pandas as pd

ser = pd.Series(np.linspace(1, 10, 5))
print(ser)
import pandas as pd

ser = pd.Series(range(5, 15))
print(ser)
import pandas as pd
ser=pd.Series(range(1,20,3), index=[x for x in 'abcdefg'])
print(ser)


data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df = pd.DataFrame(data, columns=['A', 'B', 'C'])
print(df)
import pandas as pd

dict = {'name':["aparna", "pankaj", "sudhir", "Geeku"],
        'degree': ["MBA", "BCA", "M.Tech", "MBA"],
        'score':[90, 40, 80, 98]}

df = pd.DataFrame(dict)

print(df)



data = {'Name': ['John', 'Alice', 'Bob', 'Eve', 'Charlie'], 
        'Age': [25, 30, 22, 35, 28], 
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'], 
        'Salary': [50000, 55000, 40000, 70000, 48000]}

df = pd.DataFrame(data)
# Display the entire DataFrame
print(df)
# Access the 'Age' column
age_column = df['Age']
print(age_column)
# Access the row at index 1 (second row)
second_row = df.iloc[1]
print(second_row)

# Access the first three rows and the 'Name' and 'Age' columns
subset = df.loc[0:2, ['Name', 'Age']]
print(subset)
# Access rows where 'Age' is greater than 25
filtered_data = df[df['Age'] > 25]
print(filtered_data)
# Access the 'Salary' of the row with label 2
salary_at_index_2 = df.at[2, 'Salary']
print(salary_at_index_2)

# Import pandas package
import pandas as pd

# Define a dictionary containing employee data
data = {'Name':['Jai', 'Princi', 'Gaurav', 'Anuj'],
		'Age':[27, 24, 22, 32],
		'Address':['Delhi', 'Kanpur', 'Allahabad', 'Kannauj'],
		'Qualification':['Msc', 'MA', 'MCA', 'Phd']}

# Convert the dictionary into DataFrame 
df = pd.DataFrame(data)

# select two columns
print(df[['Name', 'Qualification']])

import pandas as pd

# Define a dictionary containing Students data
data = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj'],
        'Height': [5.1, 6.2, 5.1, 5.2],
        'Qualification': ['Msc', 'MA', 'Msc', 'Msc']}

# Convert the dictionary into DataFrame
df = pd.DataFrame(data)

# Declare a list that is to be converted into a column
address = ['Delhi', 'Bangalore', 'Chennai', 'Patna']

# Using 'Address' as the column name
# and equating it to the list
df['Address'] = address

# Observe the result
print(df)

# making data frame 
df = pd.read_csv(nba.csv, index_col =Name) 

df.head(10)

new_row = pd.DataFrame({'Name':'Geeks', 'Team':'Boston', 'Number':3,
                        'Position':'PG', 'Age':33, 'Height':'6-2',
                        'Weight':189, 'College':'MIT', 'Salary':99999},
                                                            index =[0])
# simply concatenate both dataframes
df = pd.concat([new_row, df]).reset_index(drop = True)
df.head(5)

# importing pandas module
import pandas as pd

# making data frame from csv file
data = pd.read_csv("nba.csv", index_col ="Name" )

# dropping passed values
data.drop(["Avery Bradley", "John Holland", "R.J. Hunter",
							"R.J. Hunter"], inplace = True)

# display
data

## Pandas Extracting rows using .loc[] – Python
import pandas as pd 

# making data frame from csv file 
data = pd.read_csv("nba.csv", index_col ="Name") 

# retrieving row by loc method 
first = data.loc["Avery Bradley"] 
second = data.loc["R.J. Hunter"] 

print(first, "\n\n\n", second) 

import pandas as pd 

# making data frame from csv file 
data = pd.read_csv("nba.csv", index_col ="Name") 

# retrieving rows by loc method 
rows = data.loc[["Avery Bradley", "R.J. Hunter"]] 

print(type(rows)) 

rows 

import pandas as pd 

# making data frame from csv file 
data = pd.read_csv("nba.csv", index_col ="Team") 

# retrieving rows by loc method 
rows = data.loc["Utah Jazz"] 

print(type(rows)) 

rows 

import pandas as pd 

# making data frame from csv file 
data = pd.read_csv("nba.csv", index_col ="Name") 

# retrieving rows by loc method 
rows = data.loc["Avery Bradley":"Isaiah Thomas"] 

print(type(rows)) 

rows 

## Extracting rows using Pandas .iloc[] in Python
# importing pandas package
import pandas as pd

# making data frame from csv file
data = pd.read_csv('nba.csv')

# retrieving rows by loc method
row1 = data.iloc[[4, 5, 6, 7]]

# retrieving rows by loc method
row2 = data.iloc[4:8]

# comparing values
row1 == row2

import pandas as pd

# Creating a sample DataFrame
data = pd.DataFrame({
    'Name': ['Geek1', 'Geek2', 'Geek3', 'Geek4', 'Geek5'],
    'Age': [25, 30, 22, 35, 28],
    'Salary': [50000, 60000, 45000, 70000, 55000]
})

# Setting 'Name' column as the index for clarity
data.set_index('Name', inplace=True)

# Displaying the original DataFrame
print("Original DataFrame:")
print(data)

# Extracting a single row by index
row_alice = data.iloc[0, :]
print("\nExtracted Row (Geek1):")
print(row_alice)

# Extracting multiple rows using a slice
rows_geek2_to_geek3 = data.iloc[1:3, :]
print("\nExtracted Rows (Geek2 to Geek3):")
print(rows_geek2_to_geek3)

df.iloc[:0]

## Indexing and Selecting Data with Pandas
first = data[["Age", "College", "Salary"]]
print("\nMultiple Columns selected from Dataset")
display(first.head(5))  

# importing pandas package
import pandas as pd

# making data frame from csv file
data = pd.read_csv("nba.csv", index_col ="Name")

# retrieving row by loc method
first = data.loc["Avery Bradley"]
second = data.loc["R.J. Hunter"]


print(first, "\n\n\n", second)

# Select multiple rows
first = data.loc[["Avery Bradley", "R.J. Hunter"]]
display(first)

# Select two rows and three columns
first = data.loc[["Avery Bradley", "R.J. Hunter"], ["Team", "Number", "Position"]]
print(first)

# Select all rows and specific columns
first = data.loc[:, ["Team", "Number", "Position"]]
print(first)

import pandas as pd
data = pd.read_csv("nba.csv", index_col="Name")

# Select a single row by position
row2 = data.iloc[3]
print(row2)

# Select multiple rows by position
row2 = data.iloc[[3, 5, 7]]
display(row2)

# Select two rows and two columns by position
row2 = data.iloc[[3, 4], [1, 2]]
print(row2)

# Select all rows and specific columns
row2 = data.iloc[:, [1, 2]]
print(row2)
