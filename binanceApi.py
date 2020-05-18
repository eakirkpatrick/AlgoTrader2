import requests
import json
from requests.auth import HTTPBasicAuth

class APIWrapper:
    def __init__(self, coin):
        self.base_url = "https://api.binance.us/"
        self.session = requests.Session()
        self.coin = coin
        self.auth = HTTPBasicAuth('user', 'pass')
        self.day_in_ms = 86400000


    def getRequest(self, url, params=False):

        try:
            if params:
                response = self.session.get(url, timeout=120, params=params)
            else:
                response = self.session.get(url, timeout=120)
            response.raise_for_status()
            content = json.loads(response.content)
            return content

        except Exception as error:
            try:
                content = json.loads(response.content)
                raise ValueError(content)
            except json.JSONDecodeError:
                pass
            raise


    def ping(self):
        url = '{0}api/v3/ping'.format(self.base_url)
        response = self.getRequest(url)
        return response

    def getTime(self):
        url = '{0}api/v3/time'.format(self.base_url)
        response = self.getRequest(url)
        return response

    def getOrderBook(self):
        params = {'symbol': self.coin}
        url = '{0}api/v3/depth'.format(self.base_url)
        response = self.getRequest(url, params=params)
        return response

    def getExchangeInfo(self):
        url = url = '{0}api/v3/exchangeInfo'.format(self.base_url)
        response = self.getRequest(url)
        return response

    def getTickerPrice(self):
        params = {'symbol': self.coin}
        url = '{0}api/v3/ticker/24hr'.format(self.base_url)
        response = self.getRequest(url, params=params)
        return response

    def getCurrentPrice(self):
        params = {'symbol': self.coin}
        url = '{0}api/v3/ticker/price'.format(self.base_url)
        response = self.getRequest(url, params=params)
        return response

    def getCandleStickData(self, interval, startTime, endTime):
        params = {'symbol': self.coin, 'interval': interval, 'startTime': startTime, 'endTime': endTime}
        url = '{0}api/v3/klines'.format(self.base_url)
        response = self.getRequest(url, params=params)
        return response

    def collectData(self, daysBack):
        data = []
        timeBack_ms = daysBack * self.day_in_ms
        current_time = self.getTime()['serverTime']
        start_time = current_time - timeBack_ms
        end_time = start_time + self.day_in_ms

        while end_time <= current_time:
            response = self.getCandleStickData('1h', start_time, end_time)
            data.extend(response)
            start_time = end_time
            end_time += self.day_in_ms

        return data
