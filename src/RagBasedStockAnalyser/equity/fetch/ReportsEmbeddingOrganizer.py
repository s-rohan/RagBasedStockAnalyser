from RagBasedStockAnalyser.redis.ReportVectorStore import ReportDoc,ReportVectorStore
from .EmbeddingOrganizer import EmbeddingOrganizer
import asyncio
class ReportsEmbeddingOrganizer(EmbeddingOrganizer):
    def __init__(self,**kargs):
        self.lexical_prefix =kargs.get("lexical_prefix","lexical_report")
        self.semantic_prefix =kargs.get("semantic_prefix","report")
        super().__init__(lexical_prefix=self.lexical_prefix)
        # Any additional initialization for ReportsEmbeddingOrganizer can go here
    
    async def storeReportsData(self,chunks:list[dict],ticker:str,year:int,quater:str,skip_length:int=5)->bool:
        '''Stores report chunks into ReportVectorStore and their lexical data.
        Each chunk in `chunks` should be a dict with keys like 'chunk', 'block_type', 'chunk_index', 'page_number', 'heading'.
        '''
        logging=EmbeddingOrganizer.logger
        logging.debug(f"Storing {len(chunks)} report chunks into ReportVectorStore.")
        try:
            
            cleaned_content=[]
            doc_name=self.formatDefault(quater, year, ticker)
            docs=[]
            for i,d in enumerate(chunks):    
                d["id"] = f"{self.semantic_prefix}_{ticker}_{year}_{quater}_{i}"
                d["doc_name"] = f"{self.semantic_prefix}_{ticker}_{year}_{quater}"
                d["year"] = year
                d["content_type"]=d.get("block_type","")
                d["chunk_id"] = d.get("chunk_index",i)
                d["page_no"]=d.get("page_number",0)
                d["heading"]=d.get("heading","") 
                d["content"]=d.get("chunk")   
                doc=ReportDoc(**d)
                docs.append(doc)
                cleaned_content.append(self.preprocess(f'{d.get("chunk","")}'))
            results =await asyncio.gather(
            self._storeVectors(docs),
            self._storeLexicalData(ticker, year, quater, cleaned_content, doc_name)
                )
            final_result= False if any(isinstance(e,Exception) for e in results) else True
            logging.info("Report chunks stored successfully.{final_result}")
            
            return  final_result
        except Exception as e:
            logging.error(f"Error storing report chunks. :{e}",exc_info=True)
            return False

    async def _storeLexicalData(self, ticker, year, quater, cleaned_content, doc_name):
        return self.storeLexicalData(cleaned_content, quater=quater, year=year, ticker=ticker,doc_name=doc_name)

    async def _storeVectors(self, docs):
        rvs=ReportVectorStore()
        return rvs.storeReports(docs)