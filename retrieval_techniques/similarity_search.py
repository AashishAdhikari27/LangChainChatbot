from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms."""
    """ It is tasty """,
    """Some say poisonous id tasty but it kills """,
]

meta_data = [{"source": "document 1", "page": 1},
             {"source": "document 2", "page": 2},
             {"source": "document 3", "page": 3},
             {"source": "document 4", "page": 4},
             {"source": "document 5", "page": 5},]


persist_directory = 'Documents/chroma/'

embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"

)

vector_db = Chroma.from_texts(
    texts=texts,
    embedding=embedding_function,
    metadatas=meta_data,
)

response = vector_db.similarity_search(
    query="Tell me about all-white mushrooms with large fruiting bodies", k=2)

print(response)

