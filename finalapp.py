import streamlit as st
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

import time

from dotenv import load_dotenv
load_dotenv()

#load the Nvidia API key
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

llm = ChatNVIDIA(model ="nvidia/nemotron-3.5-lightning-30b-a3b")

def vector_embedding():
    if "vectors" not in st.session_state:
        # Update this line to use the active Nemotron embedding model:
        st.session_state.embeddings = NVIDIAEmbeddings(model="nvidia/nemotron-3-embed-1b")
        
        st.session_state.loader = PyPDFDirectoryLoader("./f1_data")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("NVIDIA NIM Demo")

prompt =  ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question: {input}
    """
)


prompt1 =  st.text_input("Enter your question from Documents")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("FAISS Vector Store DB is ready Using NvidiaEmbedding")

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain =create_retrieval_chain(retriever,document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt1})
    print("Response time : ", time.process_time() - start)
    st.write(response["answer"])


    ## With a streamlit expander to show the context used for the answer
    with st.expander("Context used for the answer"):
        #Find the revelant chunks used for the answer

        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("------------------------------")