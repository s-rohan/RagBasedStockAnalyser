from .BaseRedisMemoryAgent import BaseRedisMemoryAgent
from RagBasedStockAnalyser.equity.fetch.QueryWithIDF import QueryWithIDF
from langgraph.graph import StateGraph
import RagBasedStockAnalyser.redis.VectorStore as vs_module
VectorStore=vs_module.VectorStore
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
import logging
from typing import  Annotated
from langgraph.graph.message import MessagesState, add_messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class EquityAgent(BaseRedisMemoryAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vs= kwargs.get("VectorStore",VectorStore())
        self.query_engine = QueryWithIDF(vs)
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
       
    def retrieve_node(self,state: dict) -> dict:
        query = state["query"]
        documents = self.query_engine.fetch_and_lexical(query)
        #flatten docs:
        flattened_messages = []

        for entry in documents:
            tag = f"{entry['ticker']}:{entry['year']}_{entry['quater']}"
            for result in entry['results']:

                formatted_docs = "\n".join([f"{tag}_{i}: {doc}" for i, doc in enumerate(entry['results'])])
                flattened_messages.append(formatted_docs)

        return {**state, "retrieved_documents": flattened_messages}
        
    def generateResponse(self,state:dict) -> dict:  
        runnable_with_history = RunnableWithMessageHistory(
        self.prompt| self.model,
            get_session_history=self.get_merged_messages,
            input_messages_key="query",
            history_messages_key="history"
        )
        result=runnable_with_history.invoke(state,config={"configurable": {"session_id": state.get("session_id","default_session")},"callbacks": [self.tracer]})
        return {**state,"answer":result.content}

    def createGraph(self,query:str):
        rag_graph = StateGraph(state_schema=EquityState)
        rag_graph.add_node("retrieve", self.retrieve_node)
        rag_graph.add_node("generate", self.generateResponse)  # your advisory agent
        rag_graph.add_edge("retrieve", "generate")
        rag_graph.set_entry_point("retrieve")

        graph = rag_graph.compile()
        return graph
    def query(self, query_text: str, session_id: str = "default_session") -> str:
        """
        Executes a query using RAG retrieval and merged Redis-backed history.
        """
        graph=self.createGraph(query_text)
        logger.info(f"Executing query: {query_text} with session_id: {session_id}")
        response = graph.invoke({"query": query_text, "session_id": session_id},config={"configurable": {"session_id": session_id},"callbacks": [self.tracer]})
        
        logger.info(response)

        return response


class EquityState(MessagesState):
    retrieved_documents: str
    session_id: str
    query: str
    answer: Annotated[str, add_messages("generate", "answer")]