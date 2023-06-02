###event based batch approach
import os
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

storage_connection_string = 'DefaultEndpointsProtocol=https;AccountName=____BLOB_ACCOUNT_NAME;AccountKey=___BLOB_ACCOUNT_KEY'
blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)

#print("wastewtaer")
#create container reference/entity
container_id = '___NAME_of_the_BLOBCONTAINER____'
blob_service_client.get_container_client(container_id)

####overwrite = True/False , depending if data has to be/not to be overwritten
overwrite = False
target_directory = os.path.join(os.getcwd(),'___NAME_of_FTP_FOLDER___________') 

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





