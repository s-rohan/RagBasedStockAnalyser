import unittest
import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from RagBasedStockAnalyser.equity.pipeline.PdfReportPipeline import PDFSemanticChunker

class TestPDFSemanticChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = PDFSemanticChunker()
        # You may want to set up a sample PDF for testing
        self.sample_pdf = 'sample.pdf'  # Replace with a real test PDF path
        self.file_id = 'TSLA_2025_Q2'
        self.folder_path = 'test_chunks_folder'
        os.makedirs(self.folder_path, exist_ok=True)


    
    def test_chunk_pdf_empty(self):
        # Should return empty list for non-existent PDF
        result = self.chunker.chunk_pdf('nonexistent.pdf', self.file_id)
        self.assertEqual(result, [])

    def test_parseExtractBlock(self):
        block = {
            "content": "Heading:\nSome text here.",
            "type": "text",
            "page_number": 1
        }
        chunks = self.chunker.parseExtractBlock(self.file_id, block)
        self.assertIsInstance(chunks, list)
        for chunk in chunks:
            self.assertIn("chunk", chunk)
            self.assertIn("heading", chunk)
            self.assertEqual(chunk["doc_name"], self.file_id)

    def test_store_chunks(self):
        chunks = [{"chunk": "text", "chunk_index": 0, "page_number": 1, "block_type": "text", "heading": "Heading", "doc_name": self.file_id}]
        path= self.chunker.store_chunks(chunks, self.file_id, self.folder_path)
       
        self.assertTrue(os.path.exists(path))
        os.remove(path)
    
    def test_storePdfChunks_and_loadPdfChunkstoRedis(self):
        pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/RagBasedStockAnalyser/data/TSLA_2025_Q2/TSLA_2025_Q2.pdf'))
        file_id = 'TSLA_2025_Q2'
        folder_path = self.folder_path
        # Chunk and store PDF
        result = self.chunker.storePdfChunks(pdf_path, file_id, folder_path)
        self.assertTrue(os.path.exists(result))
    
    def test_loadPdfChunkstoRedis(self):
        file_id = 'TSLA_2025_Q2'
        folder_path = self.folder_path
        # Load chunks to Redis
        loaded = self.chunker.loadPdfChunkstoRedis(folder_path, file_id)
        self.assertTrue(loaded)
    
if __name__ == "__main__":
    unittest.main()
