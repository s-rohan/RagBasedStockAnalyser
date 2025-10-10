from .BaseRedisMemoryAgent import BaseRedisMemoryAgent
from .RetriverSelector import RetriverSelector
from RagBasedStockAnalyser.equity.fetch.QueryWithIDF import TranscriptQueryWithIDF,ReportQueryWithIDF,QueryWithIDF
from langgraph.graph import StateGraph
import RagBasedStockAnalyser.redis.VectorStore as vs_module
VectorStore=vs_module.VectorStore
from RagBasedStockAnalyser.redis.ReportVectorStore import ReportVectorStore
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
import logging
from typing import  Annotated
import asyncio
from langgraph.graph.message import MessagesState, add_messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class EquityAgent(BaseRedisMemoryAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_transcript_query_engine()
        self.retriver_information={
            'report':{'tickers':['TSLA']},
            'transcript':{'tickers':['AAPL']}

            }
        self.set_report_query_engine()
        # Define prompt components
        self.prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert assistant tasked with advising the user based on two sources:\n"
               "1. Retrieved Context (RAG)\n"
               "2. Chat History\n"
               "Instructions:\n"
               "- Use the retrieved context as your primary source of truth.\n"
               "- Refer to chat history only to understand the user's intent.\n"
               "- If the context is insufficient, say: 'I cannot answer confidently based on the current information.'\n"
               "- Ask for clarification if needed.\n"
               "- Be concise, factual, and helpful."),
    ("placeholder", "{history}"),
    ("human", "{query}) \n\n Context:\n{retrieved_documents}")
])
        self.graph=self.createGraph()

    def set_report_query_engine(self):
        reportVS=ReportVectorStore()
        self.query_reports= ReportQueryWithIDF(reportVS)
    def get_report_engine(self)->QueryWithIDF:
        return  self.query_reports
    def get_transcript_engine(self)->QueryWithIDF:
        return self.query_engine

    def set_transcript_query_engine(self):
        vs= VectorStore()
        self.query_engine = TranscriptQueryWithIDF(vs)
       
    def retrive_transcript_node(self, state: dict) -> dict:
        ''' This retrive node should be a seperate microservice to scale'''
        query = state["query"]
        query_engine:QueryWithIDF=self.get_transcript_engine()
        documents = query_engine.fetch_and_lexical(query)
        #flatten docs:
        flattened_messages = self.get_text_data(documents)

        return {**state, "transcript_retrieved_documents": flattened_messages}
           
    def retrive_reports_node(self, state: dict) -> dict:
        ''' This retrive node should be a seperate microservice to scale'''
        query = state["query"]
        query_reports:QueryWithIDF=self.get_report_engine(self)
        documents = query_reports.fetch_and_lexical(query)
        #flatten docs:
        flattened_messages = self.get_text_data(documents)

        return {**state, "report_retrieved_documents": flattened_messages}

    def get_text_data(self, documents)->list:
        flattened_messages=[]
        for entry in documents:
            tag = f"{entry['ticker']}:{entry['year']}_{entry['quater']}"
            formatted_docs = "\n".join([f"{tag}_{i}: {doc}" for i, doc in enumerate(entry['results'])])
            flattened_messages.append(formatted_docs)
        return flattened_messages
        
    def generateResponse(self,state:dict) -> dict:  
        runnable_with_history = RunnableWithMessageHistory(
            
        self.prompt| self.model,
            get_session_history=self.get_merged_messages,
            input_messages_key="query",
            history_messages_key="history"
        )
        result=runnable_with_history.invoke(state,config={"configurable": {"session_id": state.get("session_id","default_session")},"callbacks": [self.tracer]})
        return {**state,"answer":result.content}
    def retrive_invoke_node(self,state:dict) -> dict:
        query = state["query"]
        retrive_selector=RetriverSelector()
        response = asyncio.run(retrive_selector.call(session_id=state.get("session_id","default_session"),query=query))
        retriver=response["response"]
        if "transcript" in retriver:
            state["invoke_transcript"]=True
        if "report" in retriver:
            state["invoke_report"]=True
        return {**state}
    
    
    def merge_retrived_docs(self, state)->dict:
        retrived_docs=[]
        transcript_docs=state.get("transcript_retrieved_documents",None)
        report_docs= state.get("report_retrieved_documents")

        if  transcript_docs:
            retrived_docs.extend(transcript_docs)
        if report_docs:
             retrived_docs.extend(report_docs)
             #retrieved_documents
        return {**state,"retrieved_documents":retrived_docs}
    
    def route_retrievers(self,state):
        paths = []
        if state.get("invoke_transcript"):
            paths.append("retrieve_transcript")
        if state.get("invoke_report"):
            paths.append("retrieve_report")
        if len(paths)==0:
            raise ValueError("No retriver found for this Query")
        return paths






        
        

    

    def createGraph(self):
        rag_graph = StateGraph(state_schema=EquityState)
        rag_graph.add_node("retrieve_nodes", self.retrive_invoke_node)
        rag_graph.add_node("retrieve_transcript", self.retrive_transcript_node)
        rag_graph.add_node("retrieve_report", self.retrive_transcript_node)
        rag_graph.add_node("generate", self.generateResponse)  # your advisory agent
        rag_graph.add_edge("retrieve", "generate")
        rag_graph.add_node("retrieve", self.merge_retrived_docs)
        rag_graph.add_edge("retrieve_report", "retrieve")
        rag_graph.add_edge("retrieve_transcript", "retrieve")
        rag_graph.add_conditional_edges(
            "retrieve_nodes",
            self.route_retrievers
        )
       
        rag_graph.set_entry_point("retrieve_nodes")

        graph = rag_graph.compile()
        return graph
    def query(self, query_text: str, session_id: str = "default_session") -> str:
        """
        Executes a query using RAG retrieval and merged Redis-backed history.
        """
        graph=self.graph
        logger.info(f"Executing query: {query_text} with session_id: {session_id}")
        response={}
        try:
            response = graph.invoke({"query": query_text, "session_id": session_id},config={"configurable": {"session_id": session_id},"callbacks": [self.tracer]})
            logger.info(response)
            
        except ValueError:
            logger.warning("Insufficient information to answer query")
            response={"answer":"Insufficient information to answer query"}
        return response


class EquityState(MessagesState):
    retrieved_documents: str
    session_id: str
    query: str
    answer: Annotated[str, add_messages("generate", "answer")]