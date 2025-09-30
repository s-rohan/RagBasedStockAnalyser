import unittest
import os
import json
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from RagBasedStockAnalyser.equity.storeData.ReadStore import ReadStore

def dummy_processor(line):
    return json.loads(line)

class TestReadStore(unittest.TestCase):
    def setUp(self):
        self.read_store = ReadStore()
        self.test_folder = 'earningcall'
        self.test_filename = 'AAPL_2025_q3.jsonl'


    def test_list_files_in_folder(self):
        self.files = self.read_store.list_files_in_folder(self.test_folder, extension='.jsonl')
        self.assertTrue(len(self.files)>0)
        content = self.read_store.read_file(self.files[0], processor=dummy_processor)
        self.assertEqual(len(content)>0)

if __name__ == "__main__":
    unittest.main()
