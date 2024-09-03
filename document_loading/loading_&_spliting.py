from langchain.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


loader = PyPDFLoader('../Documents/.....')
documents = loader.load() 



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)



embeddings = OpenAIEmbeddings()


# Create a vector store using Chroma
vectordb = Chroma.from_documents(
    documents=pages,
    embeddings= ,
    persist_directory="../vector_db")

# Persist the vector store
vectordb.persist()
