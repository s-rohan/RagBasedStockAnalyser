import unittest
from minio.error import S3Error
import tempfile

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from RagBasedStockAnalyser.equity.storeData.S3Store import S3Store

class TestS3Store(unittest.TestCase):


    def test_upload_and_download_calls(self):
        store = S3Store(bucket_name="earnings")
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "example.txt")
            with open(file_path, "w") as f:
                f.write("MinIO diagnostic test")
            res=store.upload_file(object_name="object.txt", file_path=str(file_path))
            self.assertTrue(res)
 




        # test download

        with tempfile.TemporaryDirectory() as temp_dir:
            download_path = os.path.join(temp_dir, "downloaded.txt")
            res=store.download_File("object.txt", str(download_path))
            self.assertTrue(res)
            with open(download_path, "r") as f:
                content = f.read()      
                self.assertEqual(content, "MinIO diagnostic test")

    def test_list_and_delete(self):
        store = S3Store(bucket_name="earnings")
        # upload a new object
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "todelete.txt")
            with open(file_path, "w") as f:
                f.write("to be deleted")
            res = store.upload_file(object_name="todelete.txt", file_path=str(file_path))
            self.assertTrue(res)

        # list objects and ensure our object appears
        objs = store.list_objects()
        names = [getattr(o, 'object_name', None) for o in objs]
        self.assertIn("todelete.txt", names)

        # delete the object
        res = store.delete_object("todelete.txt")
        self.assertTrue(res)

        # list again and ensure it's gone
        objs_after = store.list_objects()
        names_after = [getattr(o, 'object_name', None) for o in objs_after]
        self.assertNotIn("todelete.txt", names_after)


    
if __name__ == "__main__":
    unittest.main()