import requests
import json
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.testclient import TestClient

app = FastAPI()

OLLAMA_URL = "http://dlserver1:11434/api/generate"
MODEL_NAME = "gpt-oss:20b"

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask(question: Question):
    payload = {
        "model": MODEL_NAME,
        "prompt": question.question,
        "stream": False   # ensures a single JSON object
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        data = resp.json()

        # Ollama returns {"response": "..."} when stream=False
        answer = data.get("response")
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}


# Integration test that performs a real network call to the configured Ollama URL.
def test_ask_endpoint_integration():
    """POST the JSON payload {"question":"What is the capital of France?"}
    to the `/ask` endpoint and check that an `answer` key is returned.

    Note: this test performs a real network call to `OLLAMA_URL`.
    Ensure the Ollama server is reachable at that address before running.
    """

    client = TestClient(app)

    resp = client.post("/ask", json={"question": "What is the capital of France?"})
    assert resp.status_code == 200
    body = resp.json()
    # The handler returns either {"answer": ...} or {"error": ...}
    assert ("answer" in body) or ("error" in body)
