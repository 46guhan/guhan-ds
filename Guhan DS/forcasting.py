import pandas as pd
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

df=pd.read_csv('dataset\/forecasting_dataset.csv',parse_dates=["Date"])
df.set_index('Date',inplace=True)


for row in range(1,13):
    df[f'row_{row}']=df["Sales"].shift(row)

df.dropna(inplace=True)


x=df.drop('Sales',axis=1)
y=df['Sales']


xg=XGBRegressor(n_extimators=100)
xg.fit(x,y)

future_pred=[]
new_df=df.copy()

for i in range(12):
    last_row = new_df.iloc[-1]
    input_data = new_df.iloc[-1:][[f"row_{j}" for j in range(1, 13)]].copy()

    for j in range(12, 1, -1):
        input_data[f"row_{j}"] = input_data[f"row_{j-1}"]
    input_data["row_1"] = last_row["Sales"]

    next_pred = xg.predict(input_data)[0]
    future_pred.append(next_pred)

    new_row = pd.DataFrame({
        "Sales": [next_pred],
        **{f"row_{j}": input_data.iloc[0][f"row_{j}"] for j in range(1, 13)}
    }, index=[new_df.index[-1] + pd.DateOffset(months=1)])

    new_df = pd.concat([new_df, new_row])

future_dates=pd.date_range(start=df.index[-1] + pd.DateOffset(months=1),periods=12,freq="M")

plt.plot(df.index,df['Sales'],label='Historical')
plt.plot(future_dates,future_pred,label='forcasting',linestyle='--',color='g')
plt.title('Forcasting')
plt.xlabel('Dates')
plt.ylabel('Sales')
plt.grid(True)
plt.legend()
plt.show()
