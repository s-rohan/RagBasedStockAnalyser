import os
import json
from typing import Any
class DataStore:
    def __init__(self,baseFolder:str='../../data',**kargs):
        self.baseDataFolder = os.path.join(os.path.dirname(__file__), baseFolder)
    def store_json_to_file(self,objs: Any, folder: str, filename: str) -> str:
        """
        Stores a list of JSON-serializable objects to a file under the data/<folder>/<filename>.jsonl.
        Each object is written as a separate line (JSONL format).
        Creates the folder if it does not exist.
        Returns the path to the stored file.
        """
        base_dir=os.path.join( self.baseDataFolder,folder)
        os.makedirs(base_dir, exist_ok=True)
        file_path = os.path.join(base_dir, f"{filename}.json1")
        with open(file_path, 'w', encoding='utf-8') as f:
            if isinstance(objs, list):
                for obj in objs:
                    f.write(json.dumps(obj, ensure_ascii=False) + '\n')
            else:
                f.write(json.dumps(objs, ensure_ascii=False) + '\n')
        return file_path
