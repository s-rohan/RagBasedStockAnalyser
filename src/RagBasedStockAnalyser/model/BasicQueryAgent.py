import os
from dotenv import load_dotenv
import redis
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import uvicorn
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from .BaseRedisMemoryAgent import BaseRedisMemoryAgent
# Load environment variables from .env file
load_dotenv()

class BasicQueryAgent(BaseRedisMemoryAgent):
    def __init__(self,systemPrompt:ChatPromptTemplate|str|dict,**kwargs):
        # Load config from environment variables
        super().__init__(**kwargs)
        
        self.human_template = f"{{query}}"
        self.system_template = systemPrompt
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_template),
                MessagesPlaceholder(variable_name="history"),
                ("human", self.human_template)
            ]
        )
        self.chain = self.prompt_template | self.model

        self.runnableWithHistory = RunnableWithMessageHistory(
            self.chain,
            get_session_history=self.get_merged_messages,
            input_messages_key="query",
            history_messages_key="history",
        )
    async def call(self, query: str, session_id: str = "user1")->dict:
        result = await self.runnableWithHistory.ainvoke(
            {"query": query},
            config={"configurable": {"session_id": session_id},"callbacks": [self.tracer]}
        )
        # Ensure the result is serializable
        if hasattr(result, "content"):
            return {"response": result.content}
        return {"response": str(result)}


