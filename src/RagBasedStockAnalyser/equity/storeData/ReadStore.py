import os
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReadStore:
    def __init__(self, data_dir=None):
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        else:
            self.data_dir = data_dir
        logger.info(f"ReadStore initialized with data_dir: {self.data_dir}")

    def list_files_in_folder(self, folder: str, extension: str = None) -> List[str]:
        """
        List all files in the given data/<folder> directory, optionally filtering by extension (e.g., '.jsonl').
        Returns a list of file paths.
        """
        base_dir = os.path.abspath(os.path.join(self.data_dir, folder))
        logger.info(f"Listing files in folder: {base_dir} with extension: {extension}")
        if not os.path.exists(base_dir):
            logger.warning(f"Folder does not exist: {base_dir}")
            return []
        files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
        if extension:
            files = [f for f in files if f.upper().endswith(extension.upper())]
        logger.info(f"Found {len(files)} files.")
        return [os.path.join(base_dir, f) for f in files]

    def read_file(self, filepath: str,processor:callable) -> str:
        """
        Read the contents of a file and return as a string.
        """
        logger.info(f"Reading file: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
           content=[processor(line) for line in f if line.strip()]
        logger.info(f"Read {len(content)} characters from file.")
        return content

    def list_folders(self) -> List[str]:
        """
        List all folders under the data directory.
        """
        logger.info(f"Listing folders in data directory: {self.data_dir}")
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return []
        folders = [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))]
        logger.info(f"Found {len(folders)} folders.")
        return [os.path.join(self.data_dir, f) for f in folders]
