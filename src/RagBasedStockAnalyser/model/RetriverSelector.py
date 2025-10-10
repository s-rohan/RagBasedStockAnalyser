from .BasicQueryAgent import BasicQueryAgent
from .BaseRedisMemoryAgent import BaseRedisMemoryAgent
from RagBasedStockAnalyser.equity.fetch.QueryWithIDF import TranscriptQueryWithIDF,ReportQueryWithIDF,QueryWithIDF
from langchain_core.prompts import ChatPromptTemplate
import re
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class RetriverSelector(BaseRedisMemoryAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 

        self.retriver_information=self.get_retriver_info()
        self.prompt=f"""
Evaluate the query and identify the company its about ,year and quarter if present.
return TICKER year and [quarter ] for the ticker or company being queried .
If more than one entries are present separate them by comma
example1:
How is APPLE doing in 2025
Output AAPL 2025
example2 :
How is AAPL doing in 2025
Output AAPL 2025
"""
        self.system_template = self.escape_braces(self.prompt)
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("human", "{system_template} \n Query: {query} \n Output:")
            ]
        )
        self.chain = self.prompt_template | self.model

    async def call_llm(self, query: str, session_id: str = None)->dict:
        result = await self.chain.ainvoke(
            {"query": query,"system_template":self.system_template},
            config={"configurable": {"session_id": session_id},"callbacks": [self.tracer]}
        )
        # Ensure the result is serializable

        logger.info(f"result: {result}")

        if hasattr(result, "content"):
            return {"response": result.content}
        return {"response": str(result)}



        


    def escape_braces(self,text: str) -> str:
        if isinstance(text,str):
            return text.replace("{", "{{").replace("}", "}}")
        else:
            return str(text).replace("{", "{{").replace("}", "}}")

    def normalize_metadata(self,metadata):
        for retriever, entries in metadata.items():
            for entry in entries:
                entry["year"] = entry.pop("Year", entry.get("year"))
                entry["quarter"] = entry.pop("Quater", entry.get("quarter"))
        return metadata

    def get_retriver_info(self)->dict:
        return {
                'report':  [{'ticker': 'TSLA', 'year': 2025, 'quarter': 'q2'},{'ticker': 'MSFT', 'Year': 2025, 'quarter': 'q2'}]
                ,
                'transcript':[{'ticker': 'AAPL', 'year': 2025, 'quarter': 'q3'}]
                }
            

    
    def match_retrievers(self,query, metadata):
        ticker = re.search(r"\b[A-Z]{2,5}\b", query)
        if ticker is not None:
            ticker=ticker.group()
        else:
            return []
        year = re.search(r"\b20\d{2}\b", query)
        if year is not None:
            year = year.group()
        quarter= re.search(r"q[1-4]", query, re.IGNORECASE)
        if quarter is not None:
            quarter = quarter.group().lower()

        matches = []
        for name, entries in metadata.items():
            for entry in entries:
                if (
                    entry.get("ticker", "").lower() == ticker.lower() and
                    (str(entry.get("year")) == year or year is None) and
                    (entry.get("quarter", "").lower() == quarter or quarter is None)
                ):
                    matches.append(name)
                    break
        return matches or []
    async def call(self, query: str, session_id: str = None)->dict:
        retriver_info=self.normalize_metadata(self.get_retriver_info())
        retrivers=self.match_retrievers(query,retriver_info)
        if len(retrivers)==0:
            ret=[]
            response =await self.call_llm(query=query,session_id=session_id)
            if response:
                values=response["response"].split(",")
                for q in values:
                    ret.extend(self.match_retrievers(q, retriver_info))
                retrivers = set(ret)
                if len(retrivers)==0:
                     return {"response":"No retriever found."}
                else:
                    return {"response":",".join(retrivers)}

           
        else:
            return {"response":",".join(retrivers)}




