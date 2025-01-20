import pandas as pd
from Setting import arg
import math

#Convert trading data to log returns and volume to relative volume
class TransferData():
    def __init__(self,
                 Data,#Old Data
                 TimeCursor,
                 BeforeN,#The window length over which the model observes the market is counted by the k line
                 BeforeVOLN,#Compared with the average volume of the previous n days
                 Period=1
                 ):
        self.Data=Data
        self.BeforeN=BeforeN
        self.DataBuffer=pd.DataFrame(columns=Data.columns)
        self.BeforeVOLN=BeforeVOLN
        self.ADayTime=arg.ADayTime
        self.Period=Period

        for i in range(self.BeforeN):
            cursor=TimeCursor-i
            code = Data.loc[cursor, 'code']
            time = Data.loc[cursor,'time']
            next_time=Data.loc[cursor+arg.ADayTime,'time']
            new_next_day=int((next_time-time).days)
            new_high=math.log(Data.loc[cursor,'high']/Data.loc[cursor-1,'close'])
            new_low = math.log(Data.loc[cursor, 'low'] / Data.loc[cursor - 1, 'close'])
            new_open = math.log(Data.loc[cursor, 'open'] / Data.loc[cursor- 1, 'close'])
            new_close = math.log(Data.loc[cursor, 'close'] / Data.loc[cursor - 1, 'close'])
            new_avg = math.log(Data.loc[cursor, 'avg'] / Data.loc[cursor - 1, 'close'])
            BeforeSumVol=0
            BeforeSumAmount = 0
            for j in range(self.BeforeVOLN):
                BeforeSumVol=BeforeSumVol+Data.loc[cursor-int(self.ADayTime*(j+1)/self.Period),'vol']
                BeforeSumAmount = BeforeSumAmount+Data.loc[cursor - int(self.ADayTime * (j + 1)/self.Period),'amount']
            new_vol=math.log(Data.loc[cursor,'vol']/(BeforeSumVol/self.BeforeVOLN))
            if BeforeSumAmount==0:
                new_amount=0
            else:
                new_amount = math.log(Data.loc[cursor, 'amount'] / (BeforeSumAmount / self.BeforeVOLN))
            new_data = {'code': code, 'high': new_high, 'low': new_low, 'open': new_open, 'close':new_close,"avg": new_avg, 'vol': new_vol,
                        'amount': new_amount, 'time': time,'PositionMarker':Data.loc[cursor, 'PositionMarker'],'NextDay':new_next_day,'HV':Data.loc[cursor, 'HV']}
            new_data=pd.DataFrame(new_data,index=[0])
            self.DataBuffer=pd.concat([self.DataBuffer,new_data],ignore_index=True,axis=0)

        self.DataBuffer=self.DataBuffer.sort_index(ascending=False)
        self.DataBuffer=self.DataBuffer.reset_index(drop=True)

#processed Data
    def OrdinaryToLog(self,TimeCursor):
        Data=self.Data
        code = Data.loc[TimeCursor, 'code']
        time = Data.loc[TimeCursor, 'time']
        next_time = Data.loc[TimeCursor + arg.ADayTime, 'time']
        new_next_day = int((next_time - time).days)
        new_high = math.log(Data.loc[TimeCursor, 'high'] / Data.loc[TimeCursor - 1, 'close'])
        new_low = math.log(Data.loc[TimeCursor, 'low'] / Data.loc[TimeCursor - 1, 'close'])
        new_open = math.log(Data.loc[TimeCursor, 'open'] / Data.loc[TimeCursor - 1, 'close'])
        new_close = math.log(Data.loc[TimeCursor, 'close'] / Data.loc[TimeCursor - 1, 'close'])
        new_avg = math.log(Data.loc[TimeCursor, 'avg'] / Data.loc[TimeCursor - 1, 'close'])
        BeforeSumVol = 0
        BeforeSumAmount = 0
        for j in range(self.BeforeVOLN):
            BeforeSumVol = BeforeSumVol + Data.loc[TimeCursor - int(self.ADayTime * (j + 1)/self.Period), 'vol']
            BeforeSumAmount = BeforeSumAmount + Data.loc[TimeCursor - int(self.ADayTime * (j + 1)/self.Period), 'amount']
        new_vol = math.log(Data.loc[TimeCursor, 'vol'] / (BeforeSumVol / self.BeforeVOLN))
        if BeforeSumAmount == 0:
            new_amount = 0
        else:
            new_amount = math.log((Data.loc[TimeCursor, 'amount']+10) / (BeforeSumAmount / self.BeforeVOLN))
        new_data = {'code': code, 'high': new_high, 'low': new_low, 'open': new_open, 'close':new_close,"avg": new_avg, 'vol': new_vol,
                    'amount': new_amount, 'time': time,'PositionMarker':Data.loc[TimeCursor, 'PositionMarker'],'NextDay':new_next_day,'HV':Data.loc[TimeCursor, 'HV']}
        new_data = pd.DataFrame(new_data, index=[0])
        self.DataBuffer = pd.concat([self.DataBuffer, new_data], ignore_index=True, axis=0)
        self.DataBuffer.drop(index=0,inplace=True)
        self.DataBuffer=self.DataBuffer.reset_index(drop=True)







if __name__ == '__main__':
    year = '2022'
    file_name = 'RESSET_INDXSH' + year + '_000300.csv'
    Data = pd.read_csv('Data/B000300/' + file_name)
    Data["time"] = pd.to_datetime(Data['time'])
    Data["PositionMarker"] = 0.0
    Data["NextDay"] = 1
    TimeCursor = 30 * arg.ADayTime

    transfer=TransferData(Data=Data,TimeCursor=TimeCursor,BeforeN=10*arg.ADayTime,BeforeVOLN=5)

    transfer.OrdinaryToLog(TimeCursor=TimeCursor+1)

    print(transfer.DataBuffer)

