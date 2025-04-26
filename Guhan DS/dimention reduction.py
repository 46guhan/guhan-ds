import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df=pd.read_csv('dataset/wine.csv')
new_df=df.reset_index(drop=True)
print(new_df)

ss=StandardScaler().fit_transform(new_df)

pca=PCA(n_components=2)
ssn=pca.fit_transform(ss)

plt.scatter(ssn[:,0],ssn[:,1])
plt.show()