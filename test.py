import requests

stock = 'AMZN'
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + stock + '&outputsize=full&datatype=csv&apikey=E0WTCPU2OH7QNKY5'     

with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open('output.csv', 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)