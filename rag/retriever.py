# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_community.llms import Ollama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# import streamlit as st


# loader=PyPDFLoader("qlora.pdf")
# docs=loader.load()



# text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
# text_splitter.split_documents(docs)[:5]
# documents=text_splitter.split_documents(docs)


# db=Chroma.from_documents(documents[:30],OllamaEmbeddings(model="llama3"))


# llm=Ollama(model="llama3")
# #design chatprompt template


# prompt=ChatPromptTemplate.from_template("""
# Answer the following question based only on the provided context.
# Think step by step before providing a detailed answer.
# I will tip you $1000 if user finds the answer helpful.
# <context>
# {context}
# </context>
# Question:{input}
# """
# )


# document_chain=create_stuff_documents_chain(llm,prompt)
# retriever=db.as_retriever()
# retrieval_chain=create_retrieval_chain(retriever,document_chain)

# st.title("Rag to read Qlora")
# input_text=st.text_input("Ask question about Qlora paper")

# if input_text:
#     response=(retrieval_chain.invoke({"input":input_text}))
#     st.write(response)

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load PDF and split into documents
loader = PyPDFLoader("qlora.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(docs)

# Initialize vector store with embeddings
db = Chroma.from_documents(documents, OllamaEmbeddings(model="llama3"))

# Initialize language model
llm = Ollama(model="llama3")

# Design chat prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
I will tip you $1000 if the user finds the answer helpful.
<context>
{context}
</context>
Question: {input}
""")

# Create document and retrieval chains
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Streamlit interface
st.title("RAG to Read Qlora")
input_text = st.text_input("Ask a question about the Qlora paper")

if input_text:
    response = retrieval_chain.invoke({"input": input_text})
    # Extract and display the text content from the response
    if isinstance(response, list):
        for doc in response:
            st.write(doc['page_content'])
    else:
        st.write(response)
