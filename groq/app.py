import os
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

st.title('ChatGroq Demo')
#Loading the GROQ API Key
groq_api_key = os.environ['GROQ_API_KEY']

#Get Website link
link = ''
with st.sidebar:
    website = st.text_input("Paste website link here")

    if website:
        link = website
    else:
        link = "https://docs.smith.langchain.com/"


#Session States
if "vector" not in st.session_state:
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.loader = WebBaseLoader(link)
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.db = FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)



llm=ChatGroq(groq_api_key=groq_api_key,model='gemma-7b-it')

prompt = ChatPromptTemplate.from_template(

"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}


"""
)
#Creating the document chain
document_chain = create_stuff_documents_chain(llm,prompt)
retriever = st.session_state.db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever,document_chain)

question = st.text_input("Please enter your question here")

if question:
    start = time.process_time()
    response = retrieval_chain.invoke({"input":question})
    print("Response Time:",time.process_time()-start)
    st.write(response['answer'])

    #Using Streamlit Expander
    with st.expander("Document Similarity Search"):
        #Find relevant chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("---------------------------")