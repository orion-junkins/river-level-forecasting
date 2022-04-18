import boto3
import pickle 
import os

DEFAULT_LOCAL_PATH = os.path.join("data", "aws_dispatch")
os.makedirs(DEFAULT_LOCAL_PATH, exist_ok=True)

def pickle_to_aws(y, river_gauge_name, file_prefix, local_dir=DEFAULT_LOCAL_PATH):
    out_path = os.path.join(local_dir, river_gauge_name)
    pickle_out = open(out_path, "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()
    s3 = boto3.client('s3')
    s3.upload_file(out_path, f'generated-forecasts', f'{river_gauge_name}/{file_prefix}_dataframe.pickle')
