import requests
import streamlit as st

def get_openai_response(input_text):
    response=requests.post("http://localhost:8000/essay/invoke",
    json={'input':{'topic':input_text}})

    return response.json()['output']['content']

def get_gemini_response(input_text):
    response=requests.post(
    "http://localhost:8000/poem/invoke",
    json={'input':{'topic':input_text}})

    return response.json()['output']['content']

    ## streamlit framework

st.title('Langchain Demo With GGemini vs GPT 5')

col1,col2 = st.columns(2)
with col1:
    input_text=st.text_input("Write an essay on")
    if input_text:
        st.write(get_openai_response(input_text))
with col2:
    input_text1=st.text_input("Write a poem on")
    if input_text1:
        st.write(get_gemini_response(input_text1))



