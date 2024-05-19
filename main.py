import os
import streamlit as st
import pickle
import time
import tempfile
from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
OPENAI_API_KEY = 'Your API key'
llm = OpenAI(temperature=0.9, max_tokens=5000)

st.title("Article Research tool")
st.sidebar.title("Article URLs")

uploaded_file = st.sidebar.file_uploader("Upload a document (PDF or .doc file)", type=['pdf', 'doc'])
readText_click = st.sidebar.button("Read Text")
urls = []
for i in range(3):
    urls_link = st.sidebar.text_input(f"URL {i+1}")
    urls.append(urls_link)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()


# Process uploaded file if available
if uploaded_file:
    st.text("Processing uploaded PDF file...")
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.text("Loading data from PDF file...")
    loader = UnstructuredPDFLoader(temp_file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        keep_separator=['\n', '.', ',', '\n\n'],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)
    st.text("Performing Embeddings")
    embeddings = OpenAIEmbeddings(openai_api_key="OPENAI_API_KEY ")
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local("faiss_store")

    #with open("pdfvectorfile.pkl", 'wb') as f:
     #   pickle.dump(vector_store, f)
    query = st.text_input("Questions: ")
    if query:
        st.text("Answersing....")
        vector_store = FAISS.load_local("faiss_store", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=OpenAI(openai_api_key=OPENAI_API_KEY),
                                                     retriever=vector_store.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        st.header("Answer")
        st.write(result["answer"])
        # Displaying Source of the Answer
        source = result.get("source", "")
        if source:
            st.subheader("Sources: ")
            source_list = source.split("\n")
            for source in source_list:
                st.write(source)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading Data ...")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Splitting Text...Started...")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    main_placeholder.text("Embedding Vectors ...")
    embeddings = OpenAIEmbeddings(openai_api_key="OPENAI_API_KEY ")
    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    # Save the FAISS index to a pickle file
    vectorstore_openai.save_local("vectorstore")
    query = st.text_input("Questions: ")
    if query:
        st.text("Answersing....")
        vectors = FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        chains = RetrievalQAWithSourcesChain.from_llm(llm=OpenAI(openai_api_key=OPENAI_API_KEY),
                                                     retriever=vectors.as_retriever())
        results = chains({"question": query}, return_only_outputs=True)
        st.header(results["Answer"])
        st.write(results["answer"])
        # Displaying Source of the Answer
        source = results.get("source", "")
        if source:
            st.subheader("Sources: ")
            source_list = source.split("\n")
            for source in source_list:
                st.write(source)

#streamlit run C:\Users\matee\PycharmProjects\NLP_project\main.py

