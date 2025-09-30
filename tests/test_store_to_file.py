import unittest
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from RagBasedStockAnalyser.equity.storeData.StoreToFile import DataStore

class TestStoreToFile(unittest.TestCase):
    def setUp(self):
        self.test_folder = 'testfolder'
        self.test_filename = 'testfile'
        self.test_data = [
            {"a": 1, "b": "foo"},
            {"a": 2, "b": "bar"}
        ]
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data', self.test_folder))
        self.file_path = os.path.join(self.data_dir, f'{self.test_filename}.jsonl')
        # Clean up before test
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        if os.path.exists(self.data_dir) and not os.listdir(self.data_dir):
            os.rmdir(self.data_dir)

    def test_store_json_to_file(self):
        ds=DataStore()
        result_path = ds.store_json_to_file(self.test_data, self.test_folder, self.test_filename)
        self.assertTrue(os.path.exists(result_path))
        with open(result_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), len(self.test_data))
        for line, obj in zip(lines, self.test_data):
            self.assertEqual(json.loads(line), obj)

    def tearDown(self):
        # Clean up after test
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        if os.path.exists(self.data_dir) and not os.listdir(self.data_dir):
            os.rmdir(self.data_dir)

if __name__ == "__main__":
    unittest.main()
