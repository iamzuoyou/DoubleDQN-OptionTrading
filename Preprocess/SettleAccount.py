from Preprocess.BSmodel import BSM
import pandas as pd

class Account():
    def __init__(self, rate=0.1, initCash=1000000):  # 限制的开仓比例
        self.Call = 0
        self.Put = 0
        self.rate = rate
        self.initCash = initCash
        self.AllCash = initCash
        self.Cash = 0
        self.OpenMarketValue = 0
        self.Postition = 0
        self.StrikeSpread = 50
        self.IV = 0.16
        self.bsm = BSM()
        self.SettlementDay = pd.to_datetime('1900-01-01 00:00:00')
        self.PutStrikePrice = 0
        self.CallStrikePrice = 0

    def OpenPosition(self, price, time, IV=0.16):
        self.IV = IV
        self.Cash = self.AllCash * self.rate
        self.AllCash = self.AllCash * (1 - self.rate)
        self.OpenMarketValue = self.Cash
        PriceGap = int(price / self.StrikeSpread)
        if price - self.StrikeSpread * PriceGap < self.StrikeSpread / 3:
            self.PutStrikePrice = self.StrikeSpread * PriceGap
            self.CallStrikePrice = self.StrikeSpread * PriceGap
        if price - self.StrikeSpread * PriceGap > self.StrikeSpread / 3 and price - self.StrikeSpread * PriceGap < self.StrikeSpread * 2 / 3:
            self.PutStrikePrice = self.StrikeSpread * PriceGap
            self.CallStrikePrice = self.StrikeSpread * (PriceGap + 1)
        if price - self.StrikeSpread * PriceGap > self.StrikeSpread * 2 / 3:
            self.PutStrikePrice = self.StrikeSpread * (PriceGap + 1)
            self.CallStrikePrice = self.StrikeSpread * (PriceGap + 1)
        self.SettlementDay = self.bsm.get_expiration_time(time)
        Expiration = (self.SettlementDay - time).days
        PutGreeks = self.bsm.greeks(CP='P', S=price, X=self.PutStrikePrice, sigma=self.IV, T=Expiration / 365, r=0.02, b=0.02 - 0.015)
        CallGreeks = self.bsm.greeks(CP='C', S=price, X=self.CallStrikePrice, sigma=self.IV, T=Expiration / 365, r=0.02, b=0.02 - 0.015)

        ACombinationPrice = CallGreeks['option_value'] * abs(PutGreeks['delta']) + PutGreeks['option_value'] * abs(CallGreeks['delta'])
        CombinationCount = self.Cash / ACombinationPrice
        self.Call = CombinationCount * abs(PutGreeks['delta'])
        self.Put = CombinationCount * abs(CallGreeks['delta'])
        self.Cash = 0
        self.AllCash = self.AllCash - 2 * 0.15 * (self.Call + self.Put)

    def ClosePosition(self, price, time, IV=0.16):
        self.IV = IV
        Expiration = (self.SettlementDay - time).days
        PutGreeks = self.bsm.greeks(CP='P', S=price, X=self.PutStrikePrice, sigma=self.IV, T=Expiration / 365, r=0.02, b=0.02 - 0.015)
        CallGreeks = self.bsm.greeks(CP='C', S=price, X=self.CallStrikePrice, sigma=self.IV, T=Expiration / 365, r=0.02, b=0.02 - 0.015)
        self.Cash = self.Call * CallGreeks['option_value'] + self.Put * PutGreeks['option_value']
        self.AllCash = self.AllCash + self.Cash
        self.OpenMarketValue = 0
        self.Cash = 0
        self.Put = 0
        self.Call = 0

    def Opendanbian(self, price, time, type='C'):
        self.SettlementDay = self.bsm.get_expiration_time(time)
        PriceGap = int(price / self.StrikeSpread)
        self.PutStrikePrice = self.StrikeSpread * PriceGap
        self.CallStrikePrice = self.StrikeSpread * (PriceGap + 1)
        Expiration = (self.SettlementDay - time).days
        PutGreeks = self.bsm.greeks(CP='P', S=price, X=self.PutStrikePrice, sigma=self.IV, T=Expiration / 365, r=0.02, b=0.02 - 0.015)
        CallGreeks = self.bsm.greeks(CP='C', S=price, X=self.CallStrikePrice, sigma=self.IV, T=Expiration / 365, r=0.02, b=0.02 - 0.015)

        if type == 'C':
            self.Call = self.Cash / CallGreeks['option_value']
            self.Put = 0

        if type == 'P':
            self.Put = self.Cash / PutGreeks['option_value']
            self.Call = 0

        self.Cash = 0

    def getMarketValue(self, price, time, IV):
        self.IV = IV
        Expiration = (self.SettlementDay - time).days
        if self.Put == 0 or self.Call == 0:
            return 0
        PutGreeks = self.bsm.greeks(CP='P', S=price, X=self.PutStrikePrice, sigma=self.IV, T=Expiration / 365, r=0.02, b=0.02 - 0.015)
        CallGreeks = self.bsm.greeks(CP='C', S=price, X=self.CallStrikePrice, sigma=self.IV, T=Expiration / 365, r=0.02, b=0.02 - 0.015)
        return self.Call * CallGreeks['option_value'] + self.Put * PutGreeks['option_value']

if __name__ == '__main__':
    account = Account()
    account.OpenPosition(43330, time=pd.to_datetime('2022-03-08 09:35:00'))