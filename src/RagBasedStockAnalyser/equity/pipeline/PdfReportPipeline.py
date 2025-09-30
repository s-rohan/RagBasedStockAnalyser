
import pdfplumber
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict
from RagBasedStockAnalyser.equity.fetch.ReportsEmbeddingOrganizer import ReportsEmbeddingOrganizer as embOrganizer
from RagBasedStockAnalyser.equity.storeData.ReadStore import ReadStore
import os
import json
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFSemanticChunker:
    def __init__(self, breakpoint_threshold_amount=0.75):
        logger.info(f"Initializing PDFSemanticChunker with breakpoint_threshold_amount={breakpoint_threshold_amount}")
        self.embeddings = OpenAIEmbeddings()
        self.chunker = SemanticChunker(
            self.embeddings,
            add_start_index=True,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=breakpoint_threshold_amount
        )

    def extract(self, page, page_number: int) -> List[Dict]:
        logger.info(f"Extracting page {page_number}")
        data_blocks = []

        # Extract tables
        tables = page.extract_tables()
        for i, table in enumerate(tables):
            flat_table = "\n".join([
                " | ".join(str(cell) if cell is not None else "" for cell in row) for row in table])
            data_blocks.append({
                "type": "table",
                "content": flat_table,
                "table_index": i,
                "page_number": page_number
            })

        # Extract text and detect figures
        text = page.extract_text() or ""
        if "Figure" in text or "Chart" in text:
            block_type = "figure_caption"
        else:
            block_type = "text"

        data_blocks.append({
            "type": block_type,
            "content": text,
            "page_number": page_number
        })

        logger.info(f"Extracted {len(data_blocks)} blocks from page {page_number}")
        return data_blocks

    def chunk_pdf(self, pdf_path: str, file_id: str) -> List[Dict]:
        logger.info(f"Chunking PDF: {pdf_path} with file_id: {file_id}")
        chunks_with_meta = []
        if os.path.isfile(pdf_path):
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        blocks = self.extract(page, page_number=i + 1)
                        for block in blocks:
                            chunks_with_meta.extend(self.parseExtractBlock(file_id, block))
                logger.info(f"Chunked {len(chunks_with_meta)} blocks from PDF: {pdf_path}")
            except Exception as e:
                logger.error(f"Error chunking PDF {pdf_path}: {e}")
                return []
        else:
            logger.warning(f"File not found: {pdf_path}")
        return chunks_with_meta

    def parseExtractBlock(self, file_id, block) -> List[Dict]:
        logger.info(f"Parsing block for file_id: {file_id}, page: {block.get('page_number')}")
        chunks_with_meta = []
        heading = self._extract_heading(block["content"])
        chunks = self.chunker.split_text(block["content"])
        for idx, chunk in enumerate(chunks):
            chunks_with_meta.append({
                "chunk": chunk,
                "chunk_index": idx,
                "page_number": block["page_number"],
                "block_type": block["type"],
                "heading": heading,
                "doc_name": file_id
            })
        logger.info(f"Parsed {len(chunks_with_meta)} chunks from block on page {block.get('page_number')}")
        return chunks_with_meta

    def _extract_heading(self, text: str) -> str:
        for line in text.split("\n"):
            if line.strip() and line.strip()[0].isupper():
                return line.strip()
        return "Untitled"
    def store_chunks(self, chunks: List[Dict], file_id: str, folder_path: str):
        logger.info(f"Storing {len(chunks)} chunks for file_id: {file_id} in folder: {folder_path}")
        from RagBasedStockAnalyser.equity.storeData.StoreToFile import DataStore
        dataStore = DataStore()
        path=dataStore.store_json_to_file(chunks, folder=folder_path, file_name=f"{file_id}_chunks.json")
        logger.info(f"Stored {len(chunks)} chunks for {file_id}.")
        return path
    
    def storePdfChunks(self, pdf_path: str, file_id: str, folder_path: str):
        logger.info(f"Storing PDF chunks for file_id: {file_id} from PDF: {pdf_path}")
        chunks = self.chunk_pdf(pdf_path, file_id)
        path =self.store_chunks(chunks, file_id, folder_path)
        logger.info(f"PDF chunks stored for file_id: {file_id}")
        return path
    def loadPdfChunkstoRedis(self, folder_path: str, file_id: str) -> bool:
        logger.info(f"Loading PDF chunks to Redis for file_id: {file_id} from folder: {folder_path}")
        readStore = ReadStore()
        org = embOrganizer()
        files = [
            f for f in readStore.list_files_in_folder(folder_path)
            if file_id in os.path.basename(f)
        ]
        if not files:
            logger.warning(f"No chunk files found for {file_id} in {folder_path}")
            return False
        for file_path in files:
            logger.info(f"Processing chunk file: {file_path}")
            blocks = readStore.read_file(file_path, processor=lambda x: json.loads(x))
            parts = re.split(r'[_.]', file_id)
            boo = org.storeReportsData(blocks, parts[0], int(parts[1]), parts[2])

            logger.info(f"Stored chunks from {file_path} to Redis: {boo}")
        logger.info(f"All chunks loaded to Redis for file_id: {file_id}")
        return True