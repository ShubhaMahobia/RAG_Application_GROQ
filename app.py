import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain #IMP FOR RAG
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain #IMP FOR RAG

from dotenv import load_dotenv
load_dotenv()

##LOAD THE GROQ API
os.environ['GRQ_API_KEY'] = os.getenv('GRQ_API_KEY')

st.title("RAG Document Q&A with Gemma")

groq_api_key = os.getenv('GRQ_API_KEY')

llm = ChatGroq(
    groq_api_key=groq_api_key,model_name = "Gemma-7b-It")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question: {input}

    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader('data') ## DATA INGESTION
        st.session_state.docs = st.session_state.loader.load()## COMPLETE DOC LOADING
        st.session_state.pages = st.session_state.loader.load_and_split()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
    
user_prompt = st.text_input("Enter your query from documents")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm=llm,prompt=prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever,document_chain)

    start = time.process_time()
    response = retriever_chain.invoke({'input':user_prompt})
    print(f"RESPONSE TIME =  {time.process_time() - start}")
    st.write(response['answer'])

    ##With a streamlit Expander
    with st.expander("Document Similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('---------------------------------')
