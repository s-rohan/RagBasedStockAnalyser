from RagBasedStockAnalyser.redis.ReportVectorStore import ReportDoc,ReportVectorStore
from .EmbeddingOrganizer import EmbeddingOrganizer
class ReportsEmbeddingOrganizer(EmbeddingOrganizer):
    def __init__(self):
        super().__init__()
        # Any additional initialization for ReportsEmbeddingOrganizer can go here
    
    def storeReportsData(self,chunks:list[dict],ticker:str,year:int,quater:str,skip_length:int=5)->bool:
        '''Stores report chunks into ReportVectorStore and their lexical data.
        Each chunk in `chunks` should be a dict with keys like 'chunk', 'block_type', 'chunk_index', 'page_number', 'heading'.
        '''
        logging=EmbeddingOrganizer.logger
        logging.debug(f"Storing {len(chunks)} report chunks into ReportVectorStore.")
        try:
            rvs=ReportVectorStore()
            cleaned_content=[]
            for i,d in enumerate(chunks):    
                if len(set(d.get("chunk")))<=skip_length:
                    logging.debug(f"Skipping empty or too short distinct chunk at index {i}.")
                    continue    
                d["id"] = f"report_{ticker}_{year}_{quater}_{i}"
                d["doc_name"] = f"report_{ticker}_{year}_{quater}"
                d["year"] = year
                d["content_type"]=d.get("block_type","")
                d["chunk_id"] = d.get("chunk_index",i)
                d["page_no"]=d.get("page_number",0)
                d["heading"]=d.get("heading","") 
                d["content"]=d.get("chunk")   
                doc=ReportDoc(**d)
                #rvs.storeReports([doc])
                cleaned_content.append(self.preprocess(f'{d.get("chunk","")}'))
            lexical_id_docs=f"lexical_{ticker}_{year}_{quater}_report"
            doc_name=self.formatDefault(quater, year, ticker,doc_type="report")
            self.storeLexicalData(cleaned_content, quater=quater, year=year, ticker=ticker,lexical_id_docs=[lexical_id_docs],doc_name=doc_name)
            logging.info("Report chunks stored successfully.")
            
            return True
        except Exception as e:
            logging.error(f"Error storing report chunks. :{e}")
            return False