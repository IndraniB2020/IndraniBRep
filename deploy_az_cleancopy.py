###event based batch approach
import os
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

storage_connection_string = 'DefaultEndpointsProtocol=https;AccountName=cctdash0000prd0504blob;AccountKey=T0EqkygFP+KJC4zmhn2OnI+lZX/kfzZHhx+sXqYhIuobl2i9iFGbrm9h6S4dvdRpj0atOs0ZvNdPrVitFXdWjA==;EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)

#print("wastewtaer")
#create container reference/entity
container_id = 'testww'
blob_service_client.get_container_client(container_id)

####overwrite = True/False , depending if data has to be/not to be overwritten
overwrite = False
target_directory = os.path.join(os.getcwd(),'Z:\\LatestTrends') 

for folder in os.walk(target_directory):
    for file in folder[-1]:
        try:
            #blob_path = os.path.join(folder[0].replace(os.getcwd()), file)
            blob_path = os.path.join(file)
            blob_obj = blob_service_client.get_blob_client(container=container_id, blob=blob_path)
            print("blobpath... file ------ "+file)
            with open(os.path.join(folder[0], file), mode='rb') as file_data:
                blob_obj.upload_blob(file_data, overwrite=overwrite)
        except ResourceExistsError:
             print('Blob "{0}" already exists'.format (blob_path))  
             continue





