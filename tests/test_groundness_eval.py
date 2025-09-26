import unittest
from datetime import datetime
import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from RagBasedStockAnalyser.evals.RagEvaluators import GroundednessEval

class DummyRun:
    def __init__(self, context, answer,query):
        self.inputs = {"retrieved_documents": context}
        self.outputs = {"answer": answer}
        self.query = {"query": query}

class TestGroundednessEvaluator(unittest.TestCase):


    def test_eval_with_missing_answer(self):
        evaluator = GroundednessEval()
        run = DummyRun(context="This is the supporting context.", answer="",query="What is the context about?")
        result = evaluator.eval(run)
        self.assertTrue(result["value"] in ["No"])
        

    def test_execute_run_with_rundate(self):
        rundate = datetime.strptime("2025-09-23", "%Y-%m-%d")
        evaluator = GroundednessEval()
        try:
            results = evaluator.evaluate_all_runs(runDate=rundate)
            print(f"executeRun results: {results}")
            for value in results.values():
                self.assertTrue(value["value"] in ["Yes"])

        except Exception as e:
            self.fail(f"executeRun failed: {e}")

if __name__ == "__main__":
    unittest.main()
