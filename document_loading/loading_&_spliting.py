# from langchain.document_loaders import PyPDFLoader

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.text_splitter import CharacterTextSplitter

# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma


# loader = PyPDFLoader('../Documents/.....')
# documents = loader.load() 



# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
# docs = text_splitter.split_documents(documents)



# embeddings = OpenAIEmbeddings()


# # Create a vector store using Chroma
# vectordb = Chroma.from_documents(docs, embeddings, persist_directory="../vector_db")

# # Persist the vector store
# vectordb.persist()



import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']


from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("Documents/MachineLearning-Lecture01.pdf")
pages = loader.load()



from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
# # Assuming 'pages' is a list of strings, which each represent a page.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150, 
    chunk_overlap=0, 
    separators=["\n\n", "\n", r"(?<=\.)", " "],
)



splits = text_splitter.split_documents(pages)




# text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)
# docs = text_splitter.split_documents(pages)




# from langchain.embeddings.openai import OpenAIEmbeddings

# embedding = OpenAIEmbeddings()


from sentence_transformers import SentenceTransformer

from langchain_community.embeddings import SentenceTransformerEmbeddings


embedding = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


from langchain_community.vectorstores import Chroma
persist_directory = 'Documents/chroma/'

# !rm -rf ./docs/chroma  # remove old database files if any

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

