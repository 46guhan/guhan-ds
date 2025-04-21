import random
import matplotlib.pyplot as plt

def toss():
    H_T= random.randint(0,1)
    if H_T == 0:
        return "Heads"
    else:
        return "Tails"

def prob(n):
    result={"Heads":0,"Tails":0}
    for i in range(n):
        result[toss()]+=1
    print("heads:",result["Heads"])
    print("tails:",result["Tails"])

    print("-----probebility-----")
    print("Heads prob=",result["Heads"]/n)
    print("Tails prob=",result["Tails"]/n)

    plt.bar(result.keys(),result.values(),color=['green','red'])
    plt.show()
prob(10)


""" def ballpick():
    balls=['blueball','greenball','redball']
    pick=random.randint(0,2)
    if pick==0:
        return 'blueball'
    elif pick==1:
        return 'greenball'
    else:
        return 'redball'
    
def prob(n):
    result={'blueball':0,'greenball':0,'redball':0}
    for i in range(n):
        result[ballpick()]+=1
    print("blueballs:",result["blueball"])
    print("greenballs:",result['greenball'])
    print("redballs:",result['redball'])

    blue=result["blueball"]/n
    green=result["greenball"]/n
    red=result["redball"]/n

    print("blueball prob:",blue)
    print("greenball prob:",green)
    print("redball prob:",red)

    ball=result.keys()
    counts=result.values()
    plt.bar(ball,counts,color=["blue","green","red"])
    plt.show()

prob(100) """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

adult=pd.read_csv("datascience\dataset/adult.csv")

""" total=adult["sex"].count()
genders=adult.sex
gender_count={"Male":0,"Female":0}
for i in genders:
    if (i=="Male"):
        gender_count["Male"]+=1
    else:
        gender_count["Female"]+=1
print("Male:",gender_count["Male"])
print("Female:",gender_count["Female"])
print("Male prob:",gender_count["Male"]/total)
print("Female prob:",gender_count["Female"]/total)

values=gender_count.keys()
counts=gender_count.values()
plt.bar(values,counts,color=["black","hotpink"])
plt.show() """

""" education=adult.education
unique=education.unique()
total=education.count()
counts=education.value_counts()

education_prob=[]
x=0
for i in counts:
    i=i/total
    education_prob.append(i)
    print(unique[x],"prob=",i)
    x+=1

plt.bar(unique,education_prob)
plt.xticks(rotation=45)
plt.show()

plt.plot(unique,education_prob,color="r",linewidth=2,marker="o",mfc="k")
plt.xticks(rotation=45)
plt.title("education probrbility chart")
plt.ylabel("probebility counts")
plt.show() """

""" print("-------marital_status probebility-------")
marital_status_labels=adult["marital-status"].unique()
marital_status_counts=[]
x=0
for i in adult["marital-status"].value_counts():
    i=i/adult["marital-status"].count()
    marital_status_counts.append(i)
    print(marital_status_labels[x],"prob=",i)
    x+=1

plt.plot(marital_status_labels,marital_status_counts,linewidth=3,color='g',marker='H',ms=7,mec='b',mfc='b')
plt.xticks(rotation=45)
plt.title("marital-status probebility chart")
plt.xlabel("marital-status")
plt.ylabel("probability counts")
plt.show()

print("")
print("-------race probebility--------")
race_labels=adult["race"].unique()
race_counts=[]
x=0
for i in adult["race"].value_counts():
    i=i/adult["race"].count()
    race_counts.append(i)
    print(race_labels[x],"prob=",i)
    x+=1
plt.plot(race_labels,race_counts,color='g',marker='P',ms=5,mec='y',mfc='y')
plt.title("race probebolity chart")
plt.xticks(rotation=10)
plt.ylabel("probebility counts")
plt.show()

print("")
print("-------relationship probebility-------")
relationship_labels=adult["relationship"].unique()
relationship_counts=[]
x=0
for i in adult["relationship"].value_counts():
    i=i/adult["relationship"].count()
    relationship_counts.append(i)
    print(relationship_counts[x],"prob=",i)
    x+=1

plt.plot(relationship_labels,relationship_counts,marker='d',ms=8,mec='m',mfc='m')
plt.title('relationship probebility chart')
plt.xticks(rotation=45)
plt.ylabel("probrbility counts")
plt.show() """

""" adult.dropna(subset=['occupation'],inplace=True)
# adult["occupation"].dropna(inplace=True) 
occupation_labels=adult["occupation"].unique()
print(occupation_labels)
occupation_counts=[]
x=0
for i in adult["occupation"].value_counts():
    i=i/adult["occupation"].count()
    occupation_counts.append(i)
    print(occupation_labels[x],"prob=",i)
    x+=1
plt.plot(occupation_labels,occupation_counts,marker="D",ms=4,mfc='c',mec='c')
plt.title("occupation probebility chart")
plt.ylabel("probebility counts")
plt.xticks(rotation=45)
plt.show() """