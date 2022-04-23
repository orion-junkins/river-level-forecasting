import boto3
import pickle 
import os

DEFAULT_LOCAL_PATH = os.path.join("data", "aws_dispatch")
os.makedirs(DEFAULT_LOCAL_PATH, exist_ok=True)

def pickle_to_aws(payload, river_gauge_name, model_name, filename, local_dir=DEFAULT_LOCAL_PATH):
    """
    Pickle the given payload locally, and upload the file to AWS

    Args:
        payload (object): any python object, typically a DataFrame
        river_gauge_name (string): name of the current river gauge. Determines local/S3 directory.
        model_name (strign): name of the current model. Determines local/s3 directory.
        filename (string): desired name for the file. Determines local/s3 file name.
        local_dir (path, optional): Root dir for local storage. Defaults to DEFAULT_LOCAL_PATH.
    """
    out_dir = os.path.join(local_dir, river_gauge_name, model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename + ".pickle")
    pickle_out = open(out_path, "wb")
    pickle.dump(payload, pickle_out)
    pickle_out.close()
    s3 = boto3.client('s3')
    s3.upload_file(out_path, f'generated-forecasts', f'{river_gauge_name}/{model_name}/{filename}.pickle')
