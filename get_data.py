import pandas as pd
import io
import requests
import os

url = "https://raw.githubusercontent.com/elleobrien/wine/master/wine_quality.csv"
s = requests.get(url).content
ds = pd.read_csv(io.StringIO(s.decode('utf-8')),sep=',')

# Save it
ds.to_csv('wine_quality.csv',index=False)


