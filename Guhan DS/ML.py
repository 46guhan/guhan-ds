""" import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

company=pd.read_csv('dataset\Company_Data.csv')
company_c=company.copy()
le=LabelEncoder()
company_c['ShelveLoc']=le.fit_transform(company_c['ShelveLoc'])
company_c['Urban']=le.fit_transform(company_c['Urban'])
company_c['US']=le.fit_transform(company_c['US'])

x=company_c.drop(['US'],axis=1)
y=company_c['US']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
gnb=GaussianNB()
gnb.fit(x_train,y_train)
print(x_test)
y_preb=gnb.predict(x_test)
print(y_preb)

test={
    "Sales": 7.49,
    "CompPrice": 138.0,
    "Income": 39.0,
    "Advertising": 11.0,
    "Population": 416.0,
    "Price": 120.0,
    "ShelveLoc": 0,
    "Age": 42.0,
    "Education": 17.0,
    "Urban": 1,
  
}

test_df=pd.DataFrame([test])
pred=gnb.predict(test_df)
print(pred[0]) """

""" accuracy=accuracy_score(y_test,y_preb)
print("Accuracy:",accuracy) """

""" import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

college=pd.read_csv('dataset\College.csv')
college_c=college.copy()

le=LabelEncoder()
college['Private']=le.fit_transform(college['Private'])

x=college.drop(['Private'],axis=1)
y=college['Private']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
y_preb=rfc.predict(x_test)
# print(y_preb)
accuracy=accuracy_score(y_test,y_preb)
# print(accuracy)

c={
    "Apps": 1000,
    "Accept": 500,
    "Enroll": 200,
    "Top10perc": 50,
    "Top25perc": 75,
    "F.Undergrad": 1500,
    "P.Undergrad": 200,
    "Outstate": 20000,
    "Room.Board": 5000,
    "Books": 500,
    "Personal": 2000,
    "PhD": 80,
    "Terminal": 70,
    "S.F.Ratio": 10.0,
    "perc.alumni": 20.0,
    "Expend": 10000,
    "Grad.Rate": 90.0
}
c_df=pd.DataFrame([c])
pred=rfc.predict(c_df)
print(pred) """

""" import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

honey=pd.read_csv('dataset/honey.csv')
honey_c=honey.copy()
x=honey_c.drop(['priceperlb','state'],axis=1)
y=honey_c['priceperlb']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

lr=LinearRegression()
lr.fit(x_train,y_train)

y_preb=lr.predict(x_test)
print(y_preb)

print(mean_squared_error(y_test,y_preb)) """

""" import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

ad=pd.read_csv('dataset/Admission.csv')
add=ad.drop(['Serial No.'],axis=1)

scalar=StandardScaler()
scalar.fit_transform(add)

kmean=KMeans(n_clusters=2,random_state=42)
add['k-cluster']=kmean.fit_predict(add)
print(add)

plt.scatter(add['GRE Score'],add['CGPA'],c=add['k-cluster'],cmap='Set1',s=100,edgecolor='k')
plt.show() """
""" cmap--> 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r',
'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Grays_r', 'Greens', 'Greens_r', 'Greys',
'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1',
'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr',
'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r',
'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 
'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 
'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'berlin', 'berlin_r', 'binary',
'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 
'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth',
'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_grey_r', 'gist_heat', 'gist_heat_r', 
'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 
'gist_yarg_r', 'gist_yerg', 'gist_yerg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 
'grey', 'grey_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 
'managua', 'managua_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 
'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 
'summer', 'summer_r', 'tab10', 'tab10_r','tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 
'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 
'twilight_shifted_r', 'vanimo', 'vanimo_r', 'viridis', 'viridis_r', 'winter', 'winter_r' """

""" bc=pd.read_csv('dataset/BreastCancer.csv')
bc.dropna(inplace=True)
bcc=bc.drop(['Id'],axis=1)
scalar=StandardScaler()
scalar.fit_transform(bcc)

kmean=KMeans(n_clusters=2,random_state=42)
bcc['cluster']=kmean.fit_predict(bcc)
print(bcc)

plt.scatter(bcc['Cl.thickness'],bcc['Cell.size'],c=bcc['cluster'],cmap='bone',s=150,edgecolor='k')
plt.show() """

# #Hierarchal clustering =>  unsupervised learning
""" import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram,linkage,fcluster

iris=pd.read_csv('dataset/irisdataset.csv')
iris_c=iris.drop(['Class'],axis=1)

scalar=StandardScaler()
x=scalar.fit_transform(iris_c)

linked=linkage(x,method='ward')

dendrogram(linked,orientation='top',distance_sort='descenting')
plt.show()

salary=pd.read_csv('dataset/Salary_Data.csv')
sclar=StandardScaler()
salarys=sclar.fit_transform(salary)

link=linkage(salary,method='ward')

dendrogram(link,orientation='top',distance_sort='descending')
plt.show()

wide=pd.read_csv('dataset/wide.csv')
wide_c=wide.drop(['ReadingDateTime'],axis=1)

scalar=StandardScaler()
cab=scalar.fit_transform(wide_c)
lin=linkage(cab,method='ward')

dendrogram(lin,orientation='top',distance_sort='descending')
plt.show() """

""" import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster

book=pd.read_csv('dataset/Book_unsupervised.csv')
scaler=StandardScaler()
data=scaler.fit_transform(book)
link=linkage(data,method='ward')

dendrogram(link,orientation='top',distance_sort='descending')
plt.show() """

# text mining
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

""" df=pd.read_csv('dataset\Clothing reviews.csv')
df.dropna(inplace=True)
tf=TfidfVectorizer()
data=tf.fit_transform(df['Review Text'])

tf_data=pd.DataFrame(data.toarray(),columns=tf.get_feature_names_out())
kmean=KMeans(n_clusters=2,random_state=42)
df['cluster']=kmean.fit_predict(data)

cluster=df['cluster'].value_counts().sort_index()
plt.bar(cluster.index,cluster.values)
plt.xticks([0,1])
plt.show() """

""" samsung=pd.read_csv('dataset/Samsung_review.csv')
tf=TfidfVectorizer()
samsung_df=tf.fit_transform(samsung['Overall_Quality'])

data=pd.DataFrame(samsung_df.toarray(),columns=tf.get_feature_names_out())
kmeans=KMeans(n_clusters=2,random_state=42)
samsung['cluster']=kmeans.fit_predict(samsung_df)

cluster=samsung['cluster'].value_counts().sort_index()
plt.bar(cluster.index,cluster.values)
plt.xticks([0,1])
plt.show() """

# nlp & wordcloud
import pandas as pd
import matplotlib.pyplot as plt
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

nltk.download('stopwords')

""" review=pd.read_csv('dataset\Clothing reviews.csv')
review.dropna(inplace=True)
print(len(review))

def word_clean(text):
    text=text.lower()
    text=text.translate(str.maketrans('','',string.punctuation))
    tokens=text.split()
    tokens=[word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

review['clean_text']=review['Review Text'].apply(word_clean)
print(review['clean_text'])

real=" ".join(review[review['Recommended IND']==1]['clean_text'])
fake=" ".join(review[review['Recommended IND']==0]['clean_text'])

realwc=WordCloud(width=800,height=400,background_color="black",colormap="Reds").generate(real)
fakewc=WordCloud(width=800,height=400,background_color="white",colormap="Reds").generate(fake)

plt.subplot(1,2,1)
plt.imshow(realwc)
plt.subplot(1,2,2)
plt.imshow(fakewc)
plt.tight_layout()
# plt.show() """

amazon=pd.read_csv('dataset/amazon_reviews.csv')
amazon.dropna(inplace=True)
print(len(amazon))

def imp_words(text):
    text=text.lower()
    text=text.translate(str.maketrans('','',string.punctuation))
    tokens=text.split()
    tokens=[words for words in tokens if words not in stopwords.words('english')]
    return " ".join(tokens)

amazon['main texts']=amazon['reviewText'].apply(imp_words)
# print(amazon['main texts'])

good=amazon[amazon['overall']>3.0]['main texts'].reset_index()
bad=amazon[amazon['overall']<3.0]['main texts'].reset_index()

print(good)
# print(good['main texts'].tolist())