""" import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

Bird=pd.read_csv('birds.csv')
# print(Bird.info())

mn=np.mean(Bird.MaxLength)
sigma=np.std(Bird.MaxLength)
num_samples=Bird.MaxLength.count()

data=np.random.normal(mn,sigma,num_samples)
plt.hist(data,bins=10, density=True, alpha=0.7)

xmin,xmax=plt.xlim()
x=np.linspace(xmin,xmax,num_samples)
p=norm.pdf(x,mn,sigma)
plt.plot(x,p,'r')
plt.show() """

""" mlen=Bird['MaxLength']

mu, sigma=norm.fit(mlen)
plt.hist(mlen, bins=30, density=True, alpha=0.9, color='green')

xmin,xmax=plt.xlim()
x=np.linspace(xmin,xmax,100)
p=norm.pdf(x,mu,sigma)
plt.plot(x,p, 'r')
plt.show()

print("meean=",mu)
print("sigma=",sigma)

from scipy.stats import zscore
print("zscore=",zscore(mlen))

from scipy import stats
mnlen=Bird['MinLength']
con_level=0.98
mean=np.mean(mnlen)
std_err=stats.sem(mnlen)
con_intervel=stats.norm.interval(con_level, loc=mean, scale=std_err)

plt.hist(mnlen, bins=10, color='hotpink', alpha=0.7)
plt.axvline(mean, color='red', linewidth=2, label="mean")
plt.axvline(con_intervel[0], color='blue', linestyle='dotted')
plt.axvline(con_intervel[1], color='blue', linestyle='dotted')
plt.legend()
plt.show() """

""" Bird.plot()
plt.show() """

# https://github.com/selva86/datasets/blob/master/Hitters.csv