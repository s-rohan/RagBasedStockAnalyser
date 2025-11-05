from minio import Minio
import os
from minio.error import S3Error
from dotenv import load_dotenv
from typing import Any
# Load environment variables from .env file
load_dotenv()
from RagBasedStockAnalyser.common.logging_config import setup_logging
logger = setup_logging(logger_name=__name__)
class S3Store:
    def __init__(self,bucket_name: str, client: Minio=None, **kargs):
        self.client = client
        if self.client is None:
            # Initialize MinIO client
            self.client = Minio(
            os.getenv("S3_HOST"),
            access_key=os.getenv("S3_ACESS_KEY"),
            secret_key=os.getenv("S3_SECRET_KEY"),
            region="none",
            secure=False)

            if bucket_name is not None:
                self.bucket_name = bucket_name
            else:
                self.bucket_name = os.getenv("S3_BUCKET")
            self.check_bucket_exists()

    def check_bucket_exists(self):
        try:
            exists = self.client.bucket_exists(self.bucket_name)
            logger.info(f"Bucket exists: {exists}")
        except S3Error as err:
            logger.error(f"Bucket check failed: {err}")
            raise err

    def upload_file(self, file_path: str, object_name: str = None, content_type: str = "application/octet-stream")->bool:
        try:
            if not os.path.isfile(file_path):
                logger.error(f"File not found: {file_path}")
                return False

            file_size = os.path.getsize(file_path)
            with open(file_path, "rb") as file_data:
                self.client.put_object(
                    bucket_name=self.bucket_name,
                    object_name=object_name or os.path.basename(file_path),
                    data=file_data,
                    length=file_size,
                    content_type=content_type
                )
            logger.info(f"Upload succeeded: {object_name or os.path.basename(file_path)}")
            return True
        except S3Error as err:
            logger.error(f"Upload failed: {err}")
        return False
        


    def download_File(self, object_name: str, file_path: str):
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            with open(file_path, "wb") as f:
                for chunk in response.stream(32 * 1024):
                    f.write(chunk)
            logger.info(f"Download succeeded: {object_name} → {file_path}")
            response.close()
            response.release_conn()
            return True
        except S3Error as err:
            logger.error(f"Download failed: {err}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False
        return False
    def list_objects(self)->list[Any]:
        try:
            # list objects in the configured bucket
            objects = list(self.client.list_objects(self.bucket_name))
            return objects
        except S3Error as err:
            logger.error(f"List failed: {err}")
        return []
    
    def delete_object(self,object_name: str)->bool:
        try:
            # remove object from the configured bucket
            self.client.remove_object(self.bucket_name, object_name)
            logger.info("Delete succeeded")
            return True
        except S3Error as err:
            logger.error(f"Delete failed: {err}")
        return False
    



