import pymysql as mysql
import pandas as pd
import matplotlib.pyplot as plt

connect=mysql.connect(host="localhost",user="root",password="livewire",database="company")
curser=connect.cursor()

curser.execute("select * from employee")
details=curser.fetchall()
# print(details)

columns=[col[0] for col in curser.description]
# print(columns)

employee_df=pd.DataFrame(details,columns=columns)
print(employee_df)

low_sal=employee_df[employee_df["salary"]<25000]
high_sal=employee_df[employee_df["salary"]>25000]

label=["salary below 25000","salary above 25000"]
count=[len(low_sal),len(high_sal)]
plt.bar(label,count,color=['blue','red'])
plt.show()

curser.execute("insert into employee(sname,age,salary,did) values('banu','26','34000','3')")
curser.execute("update employee set salary=29500 where sname='harish'")
curser.execute("select * from employee where sname like 'a%'")
a=curser.fetchall()
curser.execute("select * from employee where sname like 'b%'")
b=curser.fetchall()
print(a,b,sep='\n')

column=[col[0] for col in curser.description]
print(column)

dfa=pd.DataFrame(a,columns=column)
dfb=pd.DataFrame(b,columns=column)

counts=[len(dfa),len(dfb)]
label=["A letter names","B letter names"]
plt.bar(label,counts)
plt.show()