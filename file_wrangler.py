# %%
import os
import json
import pandas as pd
filenames = os.listdir("C:/Users/orion/Downloads/illinois-kerby-20230517T200312Z-001/illinois-kerby")

dfs = {}
for file in filenames:
    path = "C:/Users/orion/Downloads/illinois-kerby-20230517T200312Z-001/illinois-kerby/" + file
    df = pd.read_csv(path)
    dfs[file] = df
# %%
list(dfs.keys())[0]
# %%

names = {"37_42_121908_-123_811815_6214d124b433eb000c322c21.csv": "lon_-123_81_lat_42_12",
         "38_42_06732_-123_735599_6214d124b433eb000c322c21.csv": "lon_-123_74_lat_42_07",
         "39_42_036081_-123_815442_6214d124b433eb000c322c21.csv": "lon_-123_82_lat_42_04",
         "40_41_988114_-123_716582_6214d124b433eb000c322c21.csv": "lon_-123_72_lat_41_99",
         "41_41_951352_-123_608504_6214d124b433eb000c322c21.csv": "lon_-123_61_lat_41_95",
         "42_42_07535_-123_61605_6214d124b433eb000c322c21.csv": "lon_-123_62_lat_42_08",
         "43_42_031164_-123_534899_6214d124b433eb000c322c21.csv": "lon_-123_53_lat_42_03",
         "44_42_068429_-123_449217_6214d124b433eb000c322c21.csv": "lon_-123_45_lat_42_07",
         "45_42_131317_-123_365868_6214d124b433eb000c322c21.csv": "lon_-123_37_lat_42_13",
         "46_42_175018_-123_470912_6214d124b433eb000c322c21.csv": "lon_-123_47_lat_42_18",
         "47_42_121267_-123_67246_6214d124b433eb000c322c21.csv": "lon_-123_67_lat_42_12",
         "48_42_119886_-123_567812_6214d124b433eb000c322c21.csv": "lon_-123_56_lat_42_12", }


for key in dfs.keys():
    cur_dir = "weather-map/data/" + names[key]
    os.makedirs(cur_dir, exist_ok=True)
    meta_data = {}
    with open(cur_dir + "/meta.json", "w") as f:
        json.dump(meta_data, f)
    df = dfs[key]
    df['dt_iso'] = pd.to_datetime(df['dt_iso'].str.replace(' \+0000 UTC', ''), format='%Y-%m-%d %H:%M:%S')
    df.set_index('dt_iso', inplace=True)
    df.to_parquet(cur_dir + "/data.parquet")

# %%

for key in dfs.keys():
    cur_dir = "weather-map/data/" + names[key]
    meta_data = {}
    with open(cur_dir + "/units.json", "w") as f:
        json.dump(meta_data, f)
# %%
with open("sample.json", "w") as outfile:
    json.dump(meta_data, outfile)
# %%
