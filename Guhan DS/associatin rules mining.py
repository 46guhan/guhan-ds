""" import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules

df=pd.read_csv('dataset\supermarket_transactions.csv')
trans=df["Items"].apply(lambda x:x.split(","))

te=TransactionEncoder()
te_data=te.fit_transform(trans)

new_df=pd.DataFrame(te_data,columns=te.columns_)

frequent=apriori(new_df,min_support=0.05,use_colnames=True)
as_rules=association_rules(frequent,metric='lift',min_threshold=1)

print(as_rules[['antecedents','consequents','support','confidence','lift']].sort_values(by="lift",ascending=False)) """

import pandas as pd 
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules

df = pd.read_csv('dataset/supermarket.csv')
df['extract']=df['Gender']+','+df['Product line']

transaction=df['extract'].apply(lambda x: x.split(","))

ts=TransactionEncoder()
te_data=ts.fit_transform(transaction)
df_encoded=pd.DataFrame(te_data,columns=ts.columns_)

apri=apriori(df_encoded,min_support=0.05,use_colnames=True)
rule=association_rules(apri,metric='lift',min_threshold=1)

print(rule[["antecedents","consequents","support","confidence","lift"]])
