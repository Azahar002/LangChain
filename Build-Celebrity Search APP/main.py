#Integrate our code OpenAI API

import os
# from constants import openai_key
from langchain.llms import OpenAI
from dotenv import load_dotenv

import streamlit as st
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
#streamlit framework

st.title("Langchain Demo with OpenAI API")
input_text = st.text_input("Search the topic u want")

#OpenAI LLMs
llm = OpenAI(temperature=0.8)

if input_text:
    st.write(llm(input_text))
