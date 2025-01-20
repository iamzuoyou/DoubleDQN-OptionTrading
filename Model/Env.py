import statistics
from Preprocess.Indicator import LogHV
from Setting import arg
import pandas as pd

from Preprocess.TransferData import TransferData
from Preprocess.TransferData_KV import TransferData_KV
from Preprocess.SettleAccount import Account
import numpy as np
import math


# Responsible for generating market state data
class Env():
    def __init__(self, data_path15m,data_path30m,data_path60m):
        #read Data
        self.Data = pd.read_csv(data_path15m)
        self.Data15m=pd.read_csv(data_path15m)
        self.Data30m=pd.read_csv(data_path30m)
        self.Data60m = pd.read_csv(data_path60m)
        self.Data["time"] = pd.to_datetime(self.Data['time'])

        self.Data["PositionMarker"] = 0.0  # 新添加持仓浮盈浮亏
        self.Data["NextDay"] = 1  # 下一个交易日间隔
        self.Data['HV'] = 0.16  # 波动率


        self.ModelWindow = arg.history_data_len * arg.ADayTime# Give a cursor a time window of data that the model can see

        #The time starting point used for model test
        self.TimeCursor = 15568
        self.TimeCursor30m=7784
        self.TimeCursor60m = 3892
        
        self.Time = self.Data.loc[self.TimeCursor, 'time']
        self.DataLen = self.Data.shape[0]
        self.ResistanceCursor = 0  # Window cursor to calculate the resistance level
        self.headTtoTailDifference1 = 0
        self.headTtoTailDifference2 = 0
        self.down_flag = 0  # Downward trend marker
        self.up_flag = 0  # Up trend marker
        self.before_action = 0
        self.account = Account()  # Capital account
        self.hold_time = 0
        self.open_point = 0
        self.ResistancePointFlag = 0  # Mark if the price moves to resistance levels

        self.ResistancePoint = [[self.ResistanceCursor, self.Data.loc[self.ResistanceCursor, 'time'],
                                 self.Data.loc[self.ResistanceCursor, 'avg']]]
        self.SupportPoint = [[self.ResistanceCursor, self.Data.loc[self.ResistanceCursor, 'time'],
                              self.Data.loc[self.ResistanceCursor, 'avg']]]

        # volatility
        self.loghv = LogHV(Data=self.Data, TimeCursor=5 * arg.ADayTime + 1, N=5)
        for i in range(5 * arg.ADayTime + 1, self.TimeCursor + 1):
            self.HV = self.loghv.getLogHV(i)
            self.Data.loc[i, 'HV'] = self.HV


        self.transfer = TransferData(Data=self.Data, TimeCursor=self.TimeCursor, BeforeN=self.ModelWindow, BeforeVOLN=5,Period=1)
        self.transfer15m=TransferData_KV(Data=self.Data15m, TimeCursor=self.TimeCursor, BeforeN=self.ModelWindow, BeforeVOLN=5,Period=1)
        self.transfer30m = TransferData_KV(Data=self.Data30m, TimeCursor=self.TimeCursor30m, BeforeN=self.ModelWindow, BeforeVOLN=5,Period=2)
        self.transfer60m = TransferData_KV(Data=self.Data60m, TimeCursor=self.TimeCursor60m, BeforeN=self.ModelWindow,BeforeVOLN=5,Period=4)
        self.Observation = self.transfer.DataBuffer[
            ['high', 'low', 'open', 'close', 'avg', 'vol', 'amount', 'PositionMarker', 'NextDay', 'HV']]
        self.Observation15m=self.transfer15m.DataBuffer[['high', 'low', 'open', 'close', 'avg', 'vol', 'amount']]
        self.Observation30m=self.transfer30m.DataBuffer[['high', 'low', 'open', 'close', 'avg', 'vol', 'amount']]
        self.Observation60m=self.transfer60m.DataBuffer[['high', 'low', 'open', 'close', 'avg', 'vol', 'amount']]

        self.getResistanceSupport()

        self.Order = dict()

    def getResistanceSupport(self):  # Gain support resistance
        windows = arg.window
        while self.ResistanceCursor < self.TimeCursor - windows + 1:
            fragment = self.Data.loc[self.ResistanceCursor:self.ResistanceCursor + windows, 'avg'].values
            self.headTtoTailDifference2 = fragment[-1] - fragment[0]


            if self.headTtoTailDifference2 * self.headTtoTailDifference1 < 0:
                ReturnFragment = self.Data.loc[self.ResistanceCursor - 1:self.ResistanceCursor + windows, :]

                if self.headTtoTailDifference2 > 0:  # Bottom of the decline
                    Index = ReturnFragment.idxmin(axis=0).loc['avg']
                    ReturnPoint = self.Data.loc[Index, "avg"]
                    time = self.Data.loc[Index, "time"]

                    if Index != self.SupportPoint[-1][0]:
                        if self.down_flag == 1:  # Keep falling
                            basePoint = self.SupportPoint[-1][-1]
                            if math.log(ReturnPoint / basePoint) < -0.01:
                                self.SupportPoint.append([Index, time, ReturnPoint])
                        else:  # Fall from the top
                            basePoint = self.ResistancePoint[-1][-1]
                            if math.log(ReturnPoint / basePoint) < -0.015:
                                self.SupportPoint.append([Index, time, ReturnPoint])
                                self.down_flag = 1
                                self.up_flag = 0

                if self.headTtoTailDifference2 < 0:  # The rise has peaked
                    Index = ReturnFragment.idxmax(axis=0).loc['avg']
                    ReturnPoint = self.Data.loc[Index, "avg"]
                    time = self.Data.loc[Index, "time"]

                    if Index != self.ResistancePoint[-1][0]:
                        if self.up_flag == 1:  # Keep rising
                            basePoint = self.ResistancePoint[-1][-1]
                            if math.log(ReturnPoint / basePoint) > 0.01:
                                self.ResistancePoint.append([Index, time, ReturnPoint])
                        else:  # Rise from the bottom
                            basePoint = self.SupportPoint[-1][-1]
                            if math.log(ReturnPoint / basePoint) > 0.015:
                                self.ResistancePoint.append([Index, time, ReturnPoint])
                                self.down_flag = 0
                                self.up_flag = 1
            self.ResistanceCursor = self.ResistanceCursor + 1
            self.headTtoTailDifference1 = self.headTtoTailDifference2


    def Outliers_detection(self):
        win = 3
        ResistancePointFrontAmplitudes = []
        ResistancePointAfterAmplitudes = []
        # Amplitude filtering of the Resistance
        for i in range(1, len(self.ResistancePoint)):
            MaxPoint = max(self.Data.loc[int(self.ResistancePoint[i][0]) - win:int(self.ResistancePoint[i][0]), "avg"])
            MinPoint = min(self.Data.loc[int(self.ResistancePoint[i][0]) - win:int(self.ResistancePoint[i][0]), "avg"])
            FrontAmplitude = MaxPoint - MinPoint
            ResistancePointFrontAmplitudes.append(FrontAmplitude)

            MaxPoint = max(self.Data.loc[int(self.ResistancePoint[i][0]):int(self.ResistancePoint[i][0] + win), "avg"])
            MinPoint = min(self.Data.loc[int(self.ResistancePoint[i][0]):int(self.ResistancePoint[i][0] + win), "avg"])
            FrontAmplitude = MaxPoint - MinPoint
            ResistancePointAfterAmplitudes.append(FrontAmplitude)

        i = 1
        j = 0
        StdFrontAmplitudes = statistics.stdev(ResistancePointFrontAmplitudes)
        MeanFrontAmplitudes = statistics.mean(ResistancePointFrontAmplitudes)
        StdAfterAmplitudes = statistics.stdev(ResistancePointAfterAmplitudes)
        MeanAfterAmplitudes = statistics.mean(ResistancePointAfterAmplitudes)
        while i < len(self.ResistancePoint):
            if ResistancePointFrontAmplitudes[j] > MeanFrontAmplitudes + 3 * StdFrontAmplitudes:
                if ResistancePointAfterAmplitudes[j] > MeanAfterAmplitudes + 3 * StdAfterAmplitudes:
                    del self.ResistancePoint[i]
                    i = i - 1
            i = i + 1
            j = j + 1
        SupportPointFrontAmplitudes = []
        SupportPointAfterAmplitudes = []

        for i in range(1, len(self.SupportPoint)):
            MaxPoint = max(self.Data.loc[int(self.SupportPoint[i][0]) - win:int(self.SupportPoint[i][0]), "avg"])
            MinPoint = min(self.Data.loc[int(self.SupportPoint[i][0]) - win:int(self.SupportPoint[i][0]), "avg"])
            FrontAmplitude = MaxPoint - MinPoint
            SupportPointFrontAmplitudes.append(FrontAmplitude)

            MaxPoint = max(self.Data.loc[int(self.SupportPoint[i][0]):int(self.SupportPoint[i][0] + win), "avg"])
            MinPoint = min(self.Data.loc[int(self.SupportPoint[i][0]):int(self.SupportPoint[i][0] + win), "avg"])
            FrontAmplitude = MaxPoint - MinPoint
            SupportPointAfterAmplitudes.append(FrontAmplitude)

        i = 1
        j = 0
        StdFrontAmplitudes = statistics.stdev(SupportPointFrontAmplitudes)
        MeanFrontAmplitudes = statistics.mean(SupportPointFrontAmplitudes)
        StdAfterAmplitudes = statistics.stdev(SupportPointAfterAmplitudes)
        MeanAfterAmplitudes = statistics.mean(SupportPointAfterAmplitudes)
        while i < len(self.SupportPoint):
            if SupportPointFrontAmplitudes[j] > MeanFrontAmplitudes + 3 * StdFrontAmplitudes:
                if SupportPointAfterAmplitudes[j] > MeanAfterAmplitudes + 3 * StdAfterAmplitudes:
                    del self.SupportPoint[i]

                    i = i - 1
            i = i + 1
            j = j + 1
        return

    def getResistancePointFlag(self, beforeN):  # When the price moves near resistance, it signals 1; otherwise it signals 0
        bar = self.Data.loc[self.TimeCursor, :]
        for j in range(beforeN):
            if j > len(self.ResistancePoint) or j > len(self.SupportPoint): break
            if abs(math.log(bar['avg'] / self.ResistancePoint[-j][-1])) < 0.003:
                self.ResistancePointFlag = 1
                return
            if abs(math.log(bar['avg'] / self.SupportPoint[-j][-1])) < 0.003:
                self.ResistancePointFlag = 1
                return
        self.ResistancePointFlag = 0
        return

    def step(self, action=0.6):  # The next k-bar is played to show the model the next time state, the corresponding reward, and whether the round is over or not
        action = round(action)
        bar = self.Data.loc[self.TimeCursor, :]
        beforeHV = self.HV
        self.TimeCursor = self.TimeCursor + 1
        next_bar = self.Data.loc[self.TimeCursor, :]
        self.HV = self.loghv.getLogHV(self.TimeCursor)
        self.Data.loc[self.TimeCursor, 'HV'] = self.HV
        if self.before_action == 1:
            self.Data.loc[self.TimeCursor, 'PositionMarker'] = math.log(self.account.getMarketValue(price=next_bar['close'], time=next_bar['time'],IV=self.HV) / self.account.OpenMarketValue)

        done = -1  # The transaction did not begin.
        reward = 0
        if action == 1 and self.before_action == 0:  # open
            self.account.OpenPosition(price=bar['close'], time=bar['time'], IV=beforeHV)

            self.Order['OpenTime'] = bar['time']
            self.Order['OpenPoint'] = bar['close']
            self.Order['OpenHV']=beforeHV
            self.Data.loc[self.TimeCursor, 'PositionMarker'] = math.log(self.account.getMarketValue(price=next_bar['close'], time=next_bar['time'],IV=self.HV) / self.account.OpenMarketValue)
            self.hold_time = 1
            self.open_point = bar['close']
            reward = 0
            done = 0
        if action == 0 and self.before_action == 1:  # close
            self.account.ClosePosition(price=bar['close'], time=bar['time'], IV=beforeHV)

            self.Order['CloseTime'] = bar['time']
            self.Order['ClosePoint'] = bar['close']
            self.Order['CloseHV'] = beforeHV
            self.Order['ProfitRate']=bar['PositionMarker']
            self.Order=dict()

            reward = self.reward_fun(action)
            self.Data.loc[:, 'PositionMarker'] = 0.0
            self.transfer.DataBuffer.loc[:, 'PositionMarker'] = 0.0
            done = 1
        if action == 1 and self.before_action == 1:  # hold
            reward = self.reward_fun(action)
            done = 0
        if action == 0 and self.before_action == 0:  # No positions
            reward = 0
            done = 0

        self.before_action = action
        self.getResistancePointFlag(beforeN=5)

        self.transfer.OrdinaryToLog(self.TimeCursor)
        self.Observation = self.transfer.DataBuffer[['high', 'low', 'open', 'close', 'avg', 'vol', 'amount', 'PositionMarker', 'NextDay', 'HV']]

        self.transfer15m.OrdinaryToLog(self.TimeCursor)
        self.Observation15m = self.transfer15m.DataBuffer[['high', 'low', 'open', 'close', 'avg', 'vol', 'amount']]
        if (self.TimeCursor+1)%2==0:
            self.TimeCursor30m=self.TimeCursor30m+1
            self.transfer30m.OrdinaryToLog(self.TimeCursor30m)
            self.Observation30m = self.transfer30m.DataBuffer[['high', 'low', 'open', 'close', 'avg', 'vol', 'amount']]

            if (self.TimeCursor30m+1) % 2 == 0:
                self.TimeCursor60m = self.TimeCursor60m + 1
                self.transfer60m.OrdinaryToLog(self.TimeCursor60m)
                self.Observation60m = self.transfer60m.DataBuffer[['high', 'low', 'open', 'close', 'avg', 'vol', 'amount']]

        if self.TimeCursor % arg.ADayTime == 0:  # At the end of the day, update the resistance level
            self.getResistanceSupport()

        return self.Observation, self.Observation15m,self.Observation30m,self.Observation60m,reward, done

    def reward_fun(self, action):
        reward = 0

        if action == 1 and self.before_action == 1:  # Hold and punish if you hit the stop loss line
            bar = self.Data.loc[self.TimeCursor, :]
            self.hold_time = self.hold_time + 1
            if bar['PositionMarker'] < -0.12:
                reward = math.exp(bar['PositionMarker']) - 1

        if action == 0 and self.before_action == 1:  # close
            bar = self.Data.loc[self.TimeCursor - 1, :]
            if bar['PositionMarker'] < -0.12:  # Hit the stop loss line to close the position, the correct action, give rewards
                reward = 0.01
            else:
                reward = (math.exp(bar['PositionMarker']) - 1)  # Normal close, settlement of profit and loss

            if abs(math.log(bar['close'] / self.open_point)) > 0.015:  # The fluctuation exceeds the standard and the reward is doubled
                reward = 2 * reward

            self.hold_time = 0

        return reward

if __name__ == '__main__':

    env = Env(data_path15m="Data/15m000300/RESSET_INDXSH2018-2021_000300.csv",
              data_path30m="Data/30m000300/RESSET_INDXSH2017-2021_000300.csv",
              data_path60m="Data/60m000300/RESSET_INDXSH2017-2021_000300.csv")
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=1)
    env.step(action=0)

    for i in range(100):
        xxx = env.step()
    buffer = []