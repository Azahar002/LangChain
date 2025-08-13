# #Integrate our code OpenAI API

# import os
# # from constants import openai_key
# from langchain.llms import OpenAI
# from dotenv import load_dotenv
# from langchain import PromptTemplate
# from langchain.chains import LLMChain

# from langchain.memory import ConversationBufferMemory
# from langchain.chains import SequentialChain

# import streamlit as st
# load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# #streamlit framework

# st.title("Celebrity Search Results")
# input_text = st.text_input("Search the topic u want")


# ##Prompt Templates
# first_input_prompt = PromptTemplate(
#     input_variables = ['name'],
#     template = "Tell me about celebrity {name}"
# )


# #OpenAI LLMs
# llm = OpenAI(temperature=0.8)
# chain = LLMChain(llm = llm, prompt=first_input_prompt, verbose=True, output_key='person' )

# ##Prompt Templates
# second_input_prompt = PromptTemplate(
#     input_variables = ['person'],
#     template = "when was {person} born? Return ONLY the date in YYYY-MM-DD. If unknown, return Unknown."
# )

# chain2 = LLMChain(llm = llm, prompt=second_input_prompt, verbose=True, output_key='dob' )
# parent_chain = SequentialChain(chains=[chain, chain2],
#                 input_variables=["name"],
#                 output_variables=["person","dob"], 
#                 verbose=True)

# if input_text:
#     st.write(parent_chain({'name':input_text}))

import os
from dotenv import load_dotenv
import streamlit as st

from langchain.llms import OpenAI              # (newer: from langchain_openai import ChatOpenAI)
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.title("Celebrity Search Results")
name_input = st.text_input("Search the topic u want")

# 1) Return ONLY the canonical name (no bios!)
clean_name_prompt = PromptTemplate(
    input_variables=["name"],
    template="Given '{name}', output ONLY the celebrity's canonical full name. No extra words."
)

llm = OpenAI(temperature=0)  # deterministic

clean_name_chain = LLMChain(
    llm=llm,
    prompt=clean_name_prompt,
    output_key="person",
    verbose=False
)

# 2) Ask for DOB with strict format
dob_prompt = PromptTemplate(
    input_variables=["person"],
    template="When was {person} born? Return ONLY the date in YYYY-MM-DD. If unknown, return Unknown."
)

dob_chain = LLMChain(
    llm=llm,
    prompt=dob_prompt,
    output_key="dob",
    verbose=False
)

# Wire outputs -> inputs, and define pipeline I/O
pipeline = SequentialChain(
    chains=[clean_name_chain, dob_chain],
    input_variables=["name"],
    output_variables=["dob"],
    verbose=False
)

if name_input:
    try:
        result = pipeline({"name": name_input})
        st.write(result["dob"])          # show only the DOB in UI
    except Exception as e:
        st.error(f"Error: {e}")
