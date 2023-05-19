from polygon import RESTClient
import pandas as pd

client = RESTClient(api_key="eSp3s06vNjpV6CMErNOyIcMpATBKvxDT")

ticker = "X:BTCUSD"

bars = client.get_aggs(ticker=ticker, multiplier=1, timespan="hour", from_="2017-05-31", to="2023-05-11", limit=50000)
df = pd.DataFrame(bars)