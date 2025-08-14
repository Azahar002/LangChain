from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage  # <-- needed for message format
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


st.header('Research Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need",  
    "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models areFew-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )  

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical",  
    "Code-Oriented", "Mathematical"] )  

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium  (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('template.json')

#fill the Placeholders


# user_input = st.text_input("Enter your Prompt")

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})
    # Use .content instead of .context
    st.write(result.content)
    # st.write("Hello")
