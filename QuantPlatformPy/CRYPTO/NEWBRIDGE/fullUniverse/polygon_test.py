from polygon import RESTClient
import json
import requests
import base64
import pandas as pd
client = RESTClient(api_key="eSp3s06vNjpV6CMErNOyIcMpATBKvxDT")

def convert_bytes_to_json(byte_array:bytes):
    result = byte_array.decode('utf8').replace("'",'"')
    return json.loads(result)

def prettify_json(json_file:dict):
    return json.dumps(json_file, indent=4, sort_keys=True)

ticker = "X:BTCUSD"

#get_aggs has additional commment with raw=True which retruns raw content and not list content

bars = client.get_aggs(ticker=ticker, multiplier=1, timespan="hour", from_="2017-05-31", to="2023-05-11",raw= True, limit=50000)
json_file = convert_bytes_to_json(bars.data)
pretty_output = prettify_json(json_file)


df = pd.DataFrame(bars)