#%%
import pandas as pd

df = pd.read_xml("https://water.weather.gov/ahps2/hydrograph_to_xml.php?gage=vido3&output=xml", xpath="./forecast/datum")

# %%
df
# %%
import xml.etree.ElementTree as ET
import urllib.request

url = 'https://water.weather.gov/ahps2/hydrograph_to_xml.php?gage=vido3&output=xml'
response = urllib.request.urlopen(url).read()
tree = ET.fromstring(response)
# %%
forecast = tree.findall("./forecast/datum")
# %%
