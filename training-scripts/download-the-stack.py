# Download the stack dataset.
# Code based on the work by lucmon / Zhijie

import os
import boto3
import gzip
from botocore import UNSIGNED
from botocore.config import Config
from datasets import load_dataset
from botocore.exceptions import ClientError
from datasets import load_dataset
from huggingface_hub import login
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

login()
# dataset streaming (will only download the data as needed)
ds = load_dataset("bigcode/the-stack-v2-train-smol-ids", streaming=True, split="train")


s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
bucket_name = "softwareheritage"



def download_single_file(file):
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    try:
        key = f"content/{file['blob_id']}"
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        with gzip.GzipFile(fileobj=obj['Body']) as fin:
            file["text"] = fin.read().decode("utf-8", errors="ignore")
    except ClientError as e:
        print(f"Error downloading {file['blob_id']}: {e}")
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"File not found: {key}")
            file["text"] = ""
    return file


def download_contents(files):
    with ThreadPoolExecutor(max_workers=16) as executor:
        files = list(executor.map(download_single_file, files))
    return {"files": files, "download_success": True}


for row_id, row in enumerate(tqdm(ds)):
    # We only want to download repositories with more than one file
    # So we can have cross-file context
    if len(row["files"]) <= 1:
        continue
    print(row["repo_name"])
    repo_name_list = row["repo_name"].split("/")
    REPO_DIR = "/work/nvme/becw/sma2/the-stack-v2-20k/repos/{}_{}".format(repo_name_list[0].replace('_', '-'), repo_name_list[1])
    os.makedirs(REPO_DIR, exist_ok=True)
    files = download_contents(row["files"])['files']
    for i in range(len(files)):
        os.makedirs(os.path.dirname(REPO_DIR + files[i]['path']), exist_ok=True)
        with open(REPO_DIR + files[i]['path'], "w") as wt_fn:
            wt_fn.write(files[i]['text'])
    if row_id > 20000:
        break