import requests
import streamlit as st

def get_ollama_response(input_text):
    response=requests.post(
        "http://localhost:8000/poem/invoke",
        json={'input':{'topic':input_text}})

    # Check if response status is OK (200) before trying to access json data
    if response.status_code == 200:
        return response.json()['output']
    else:
        return f"Error: API returned status code {response.status_code}"

st.title("Langchain demo with LLama3 API")

input_text=st.text_input("Write a poem on")

if input_text:
    st.write(get_ollama_response(input_text))
