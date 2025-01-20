#The k-lines are converted to other periods
import pandas as pd
import numpy as np
year='2022-2023'
period=30/15
file_name='RESSET_INDXSH'+year+'_000300.csv'
Data=pd.read_csv('Data/15m000300/'+file_name)
date_range = pd.date_range(start='2022-01-01', end='2024-12-31')
Data["time"]=pd.to_datetime(Data["time"])
col=list(Data.columns.values)
TransferedData=pd.DataFrame(columns=col)
TransferedData['close']
LenK=8
for date in date_range:
    date_str=date.strftime('%Y-%m-%d')
    filtered_data = Data[Data['time'].dt.strftime('%Y-%m-%d') == date_str].copy()
    filtered_data.reset_index(drop=True,inplace=True)
    if filtered_data.shape[0] == 0: continue
    for i in range(LenK):
        index=(i+1)*period
        code=filtered_data.loc[index-period,'code']
        new_open=filtered_data.loc[index-period,'open']
        new_close=filtered_data.loc[index-1,'close']
        new_high=max(filtered_data.loc[index-period:index-1,'high'])
        new_low = min(filtered_data.loc[index - period :index-1, 'low'])
        new_avg=sum(filtered_data.loc[index-period:index-1,'avg'])/period
        new_vol=sum(filtered_data.loc[index - period :index-1, 'vol'])
        new_amount = sum(filtered_data.loc[index - period :index-1, 'amount'])
        new_time=filtered_data.loc[index-1,'time']

        new_data = {'code': code, 'high': new_high, 'low': new_low, 'open': new_open,'close':new_close,'vol': new_vol,'avg':new_avg,
                    'amount': new_amount, 'time': new_time}
        new_data = pd.DataFrame(new_data, index=[0])
        TransferedData = pd.concat([TransferedData, new_data], ignore_index=True, axis=0)

TransferedData.to_csv('./Data/30m000300/'+file_name,index=False)





