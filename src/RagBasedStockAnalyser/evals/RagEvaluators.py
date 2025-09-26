from langsmith.evaluation import RunEvaluator,evaluate,evaluate_existing
from  langchain_openai  import ChatOpenAI
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError
from datetime import datetime
from langsmith.evaluation.llm_evaluator import LLMEvaluator,CategoricalScoreConfig

import os
import abc
from openevals.prompts import RAG_GROUNDEDNESS_PROMPT,CONCISENESS_PROMPT,CORRECTNESS_PROMPT,RAG_RETRIEVAL_RELEVANCE_PROMPT
# Load environment variables from .env file

load_dotenv()

class RagEvaluators(abc.ABC):
    def __init__(self, **kwargs):
        self.llm_model = kwargs.get("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4-0613"))
        self.temperature = kwargs.get("TEMPERATURE", 0)
        self.llm = ChatOpenAI(model=self.llm_model, temperature=self.temperature)
        self.client = Client()
        self.project_name = kwargs.get("LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT", "RagStockAnalyser"))

    @abc.abstractmethod
    def eval(self, run)->dict[str, any]:
        pass
    
    def evaluate_all_runs(self,start_time:datetime=None,end_time:datetime=None,runDate:datetime=None):

        runs = self.get_runs(start_time, end_time, runDate)
        results={}
        for run in runs:
            if run.status == "success" and "answer" in run.outputs:
                eval_result = self.eval(run)
                if eval_result: # Ensure eval_result is not None
                    print(f"Run ID: {run.id} - Eval Result: {eval_result}")
                    results[run.id]=eval_result
        
        return results

    def get_runs(self, start_time, end_time, runDate):
        if start_time is  None:
            if runDate is not None:
                start_time = runDate.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
            else:
                start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        if end_time is None:
            end_time = start_time + timedelta(days=1)

        # Fetch runs
        runs = self.client.list_runs(
            project_name=self.project_name,
            start_time=start_time,
            end_time=end_time,
            is_root=True  
        )
        
        return runs

class GroundednessEval(RagEvaluators):
    
    def __init__(self, model_name="gpt-4-0613", temperature=0):
        super().__init__(LLM_MODEL=model_name, TEMPERATURE=temperature)    
        self.prompt = """You are a helpful evaluator. Given a query ,context and an answer, determine whether the output to the query is fully supported by the context.

                    Query:
                    {query}
                    Context:
                    {context}

                    Output:
                    {output}

                    Respond with "Yes" if the answer is fully supported by the context, otherwise respond with "No".
                    """
       

    def eval(self, run):
        groundedness_judge = self.getGroundnessJudge(prompt=self.prompt)
        result = groundedness_judge.evaluate_run(run, example=None)
        return result.dict()
    
    def getGroundnessJudge(self,score_config=None,prompt=None):
        if score_config is None:
            score_config = CategoricalScoreConfig(
            key="groundedness_score",
            description="Is the answer fully supported by the retrieved context?",
            choices=["Yes", "No"],
            scoring_map={"Yes": True, "No": False},
            include_explanation=True
        )
            if prompt is None:
                prompt = RAG_GROUNDEDNESS_PROMPT
        groundedness_judge = LLMEvaluator.from_model(
        model=self.llm,
        prompt_template=prompt,
        score_config=score_config,
        map_variables=RagEvaluators.map_variables
        )
        return groundedness_judge

    @staticmethod
    def map_variables(run, example=None):
            return {
                "query": run.outputs.get("query",""),
                "context": run.outputs.get("retrieved_documents", ""),
                "outputs": run.outputs.get("answer", "")
            }

   
         
