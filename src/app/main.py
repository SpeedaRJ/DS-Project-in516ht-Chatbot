from typing import Union

from fastapi import FastAPI

from model import *
import torch

from os.path import exists
from haystack.utils import print_answers

app = FastAPI()

year = 2042
dpr = False
db = "full_combined"

if not exists("document_store.faiss"):
    build_document_store(year)

document_store = read_document_store()
retriever = build_retriever(document_store, year, dpr)
reader = build_reader(year, db)
generator = build_generator(year, db)
pipeline_extractive = build_pipeline(reader, retriever)
pipeline_generative = build_pipeline(generator, retriever)


@app.get("/")
def read_root():
    return "Hello"


@app.get("/db/{db}")
def set_database(db: str):
    db = db
    if db == "handwritten":
        year = 2022
    if "combined" in db:
        year = 2042
    if db not in ["full", "smaller", "handwritten", "full_combined", "smaller_combined"]:
        return {"error": "Database not found, try: 'full', 'smaller', 'handwritten', 'full_combined', 'smaller_combined'"}
    del retriever
    del pipeline_extractive
    del pipeline_generative
    del generator
    del reader
    torch.cuda.empty_cache()
    reader = build_reader(year, db)
    generator = build_generator(year, db)
    retriever = build_retriever(document_store, year, dpr)
    pipeline_extractive = build_pipeline(reader, retriever)
    pipeline_generative = build_pipeline(generator, retriever)


@app.get("/year/{year}")
def set_model_year(year: str):
    year = year
    if year not in ["2020", "2022", "2042"]:
        return {"error": "Year not found, try: '2020', '2022', '2042'"}
    del generator
    del reader
    del retriever
    del pipeline_generative
    del pipeline_extractive
    torch.cuda.empty_cache()
    reader = build_reader(year, db)
    generator = build_generator(year, db)
    retriever = build_retriever(document_store, year, dpr)
    pipeline_extractive = build_pipeline(reader, retriever)
    pipeline_generative = build_pipeline(generator, retriever)


@app.get("/dpr/{finetunning}")
def set_dpr_finetunning(finetunning: bool):
    dpr = finetunning
    del retriever
    del pipeline_extractive
    del pipeline_generative
    torch.cuda.empty_cache()
    retriever = build_retriever(document_store, year, dpr)
    pipeline_extractive = build_pipeline(reader, retriever)
    pipeline_generative = build_pipeline(generator, retriever)


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
