from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend_model import ModelHandler
from typing import Literal

app = FastAPI()

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    text: str

class ModelChoice(BaseModel):
    model_name_: Literal['fine_tuned_model', 'base_model']

modelhandler = None

@app.post("/choose-model/")
def choose_model(model_choice: ModelChoice):
    global modelhandler
    modelhandler = ModelHandler()
    modelhandler.loading_model(model_chosen=model_choice.model_name_)
    return None

@app.post("/ask-question/")
def ask_question(question: Question):
    if modelhandler is None:
        raise HTTPException(status_code=400, detail="No model has been selected")

    triple = {'Instruction': 'Answer the following question:',
              'Input': question.question,
              'Output': ''}

    modelhandler.ask_question(triple)

    model_generated_answer = modelhandler.parse_output()   

    return {"question": question.question, "answer": model_generated_answer}



if __name__ == "__main__":
    import uvicorn
    import os

    # Ensure the port is not in use
    port = 8009
    try:
        os.system(f"fuser -k {port}/tcp")  # Kill any process using the port
    except Exception as e:
        print(f"Failed to release port {port}: {e}")

    uvicorn.run(app, host="127.0.0.1", port=port)
