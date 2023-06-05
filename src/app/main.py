from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, staticfiles, Body, responses
from typing import List
from fastapi.templating import Jinja2Templates
import json

import model

import warnings
warnings.filterwarnings('ignore', category=UserWarning,
                        message='TypedStorage is deprecated')

document_store = model.read_document_store()
retriever = model.build_retriever(document_store)
generative = model.build_pipeline(model.build_generator(), retriever)
extractive = model.build_pipeline(model.build_reader(), retriever)

app = FastAPI()

app.mount("/static", staticfiles.StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
def get_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/chat")
def get_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/api/current_user")
def get_user(request: Request):
    return request.cookies.get("X-Authorization")


@app.post("/api/register")
def register_user(response: Response, user: str = Body()):
    user = json.loads(user)
    response.set_cookie(key="X-Authorization",
                        value=user["username"], httponly=True)


@app.get("/api/chat")
def chat(request: Request):
    question = request.query_params["message"]
    print("Getting response for question: " + question)
    # response_data = {
    #     "answer": model.run_query(generative, question)["answers"][0].to_dict()["answer"]
    # }
    response_data = {
        "answer": model.run_query(extractive, question)["answers"][0].to_dict()["answer"]
    }
    print("Got Response...")
    html = """
        <div class='user-message align-self-end d-flex flex-row align-items-center justify-content-end mb-4 rounded d-flex align-items-center border border-dark w-75 ml-2'>
            <div class="chat-message-text d-flex align-items-center ml-2">
                <p class='m-0'>""" + response_data["answer"] + """</p>
            </div>
            <div class='chat-message-user d-flex align-items-center ml-3 mr-2'><img src='./static/logo.jpg' alt='...' class='chat-message-icon'></div>
        </div>
    """
    return responses.HTMLResponse(content=html)
