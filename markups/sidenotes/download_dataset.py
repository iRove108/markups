import os
from azure.storage.blob import BlobServiceClient,ContainerClient, BlobClient
import datetime

DATASETS_DIR = 'datasets'

# Create dest directory if it doesn't exist
if not os.path.exists(DATASETS_DIR):
    os.makedirs(DATASETS_DIR)

# Setup client for Azure API
blob_service_client = BlobServiceClient("https://handwriting.blob.core.windows.net/")
container_client = blob_service_client.get_container_client("leasedata")

def download_blob(blob_client, destination_file):
    print("[{}]:[INFO] : Downloading {} ...".format(datetime.datetime.utcnow(), destination_file))
    with open(destination_file, "wb") as my_blob:
        blob_data = blob_client.download_blob()
        blob_data.readinto(my_blob)
    print("[{}]:[INFO] : download finished".format(datetime.datetime.utcnow()))

# Download all blobs in the container
blob_list = container_client.list_blobs()
for blob in blob_list:
    print("[{}]:[INFO] : Blob name: {}".format(datetime.datetime.utcnow(), blob.name))
    #check if the path contains a folder structure, create the folder structure
    if "/" in "{}".format(blob.name):
        #extract the folder path and check if that folder exists locally, and if not create it
        head, tail = os.path.split("{}".format(blob.name))
        if not (os.path.isdir(DATASETS_DIR + "/" + head)):
            #create the diretcory and download the file to it
            print("[{}]:[INFO] : {} directory doesn't exist, creating it now".format(datetime.datetime.utcnow(), DATASETS_DIR + "/" + head))
            os.makedirs(DATASETS_DIR + "/" + head, exist_ok=True)
    # Finally, download the blob
    blob_client = container_client.get_blob_client(blob.name)
    download_blob(blob_client, DATASETS_DIR + "/" + blob.name)
