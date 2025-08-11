from fastapi import FastAPI
from langserve import add_routes
from dotenv import load_dotenv
import uvicorn
import os

# New import location for OpenAI chat model
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

# os.environ['OPEN_API_KEY'] = os.getenv('OPEN_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app=FastAPI(
    title='Langchain Server',
    version ='1.0',
    description = 'A simple API Server'

)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)

model=ChatOpenAI()
# llm=Ollama(model='llama2')
prompt1 = ChatPromptTemplate.from_template("Write me an Essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me an Poem about {topic} with 100 words")

add_routes(
    app,
    prompt1 | model,
    path='/essay'
)

add_routes(
    app,
    prompt2 | model,
    path='/poem'
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)