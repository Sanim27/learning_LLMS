from fastapi import FastAPI 
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

llm=Ollama(model="llama3")

prompt=ChatPromptTemplate.from_template({})

add_routes(
    app,
    prompt|llm,
    path="/search"
)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)