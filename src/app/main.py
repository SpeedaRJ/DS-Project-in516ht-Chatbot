from typing import Union

from fastapi import FastAPI

from model import *

from os.path import exists
from haystack.utils import print_answers

app = FastAPI()

if not exists("document_store.faiss"):
    build_document_store()

document_store = read_document_store()
retriever = build_retriever(document_store)
reader = build_reader()
generator = build_generator()
pipeline_extractive = build_pipeline(reader, retriever)
pipeline_generative = build_pipeline(generator, retriever)


@app.get("/")
def read_root():
    return "Hello"


@app.get("/{model}/{question}")
def read_item(question: str, model: str):
    if model == "generative":
        model = pipeline_generative
    elif model == "extractive":
        model = pipeline_extractive
    else:
        model = None
    if not model:
        return {"error": "Model not found, try 'generative' or 'extractive'"}
    answers = run_query(model, question)
    return {
        "top_3_answers": [
            answers["answers"][i].to_dict()["answer"]
            for i in range(len(answers["answers"]))
        ]
    }
