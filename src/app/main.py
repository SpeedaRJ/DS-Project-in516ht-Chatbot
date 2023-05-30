import uvicorn
from typing import Optional
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from model import *
from html_utils import build_html_chat
import numpy as np
from os.path import exists

app = FastAPI()

# Mounts the static folder that contains the css file
app.mount("/static", StaticFiles(directory="static"), name="static")

# Locates the template files that will be modified at run time
# With the dialog form the user and bot
templates = Jinja2Templates(directory="templates")

if not exists("document_store.faiss"):
    build_document_store()

document_store = read_document_store()
retriever = build_retriever(document_store)
reader = build_reader()
generator = build_generator()
pipeline_extractive = build_pipeline(reader, retriever)
pipeline_generative = build_pipeline(generator, retriever)

chatbot = ChatBot(pipeline_extractive)


@app.post("/", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
async def root(request: Request, message: Optional[str] = Form(None)):

    # If the Form is not None, then get a reply from the bot
    if message is not None:

        # Gets a response of the AI bot
        _ = chatbot.get_reply(message)

        # Converts the chat history into an HTML dialog
        array = np.array([
            build_html_chat(is_me=i %
                            2 == 0, text=msg['text'], time=msg['time'])
            for i, msg in enumerate(chatbot.chat_history)
        ])
        array[::2], array[1::2] = array[1::2], array[::2].copy()
        chat_html = '\n'.join(array[::-1])

    else:
        chat_html = ''

    message_dict = {
        "request": request,
        "chat": chat_html
    }

    # Returns the final HTML
    return templates.TemplateResponse("index.html", message_dict)
