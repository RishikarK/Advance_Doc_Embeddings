import os
from pymongo import MongoClient
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS  # Updated import
from langchain.chains import RetrievalQA
import fitz  # PyMuPDF
import numpy as np

# Configuration
MONGO_URI = (
    "mongodb://enspirit:Enspirit123@localhost:27017"  # Update with your MongoDB URI
)
DB_NAME = "Embeddings"
COLLECTION_NAME = "document_embeddings"

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Initialize Langchain Ollama LLM and Embeddings model
llm = OllamaLLM(model="llama3:latest")  # Updated import
embeddings_model = OllamaEmbeddings(model="nomic-embed-text:latest")  # Updated import


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def embed_document(document: str):
    """Embed a document and return the embeddings as a list."""
    return embeddings_model.embed_documents([document])[0]  # Keep it as a list


def store_embeddings(key: str, embeddings):
    """Store embeddings in MongoDB."""
    # Store directly without converting to a NumPy array
    collection.update_one(
        {"key": key}, {"$set": {"embeddings": embeddings}}, upsert=True
    )


def load_embeddings(key: str):
    """Load embeddings from MongoDB."""
    doc = collection.find_one({"key": key})
    if doc:
        return np.array(doc["embeddings"])  # Convert to NumPy array when loading
    return None


def answer_query(query: str, document: str, embeddings: np.ndarray):
    """Use Langchain LLM to answer a query based on the document embeddings."""

    # Create a list of tuples (text, embedding) for FAISS
    text_embeddings = [(document, embeddings)]

    # Create a FAISS vector store from the embeddings
    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        # embedding_function=embeddings_model.embed_query,  # Use the embedding function here
        embedding=embeddings_model,
    )

    # Initialize the RetrievalQA chain with the LLM and vector store
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vector_store.as_retriever()
    )

    # Get the answer from the chain using the invoke method
    answer = qa_chain.invoke(query)
    return answer


def process_pdf(pdf_path: str, query: str, key: str):
    """Process the PDF file, extract text, store/load embeddings, and answer the query."""
    document = extract_text_from_pdf(pdf_path)

    embeddings = load_embeddings(key)

    if embeddings is None:
        print("Embeddings not found in database. Creating new embeddings...")
        embeddings = embed_document(document)
        store_embeddings(key, embeddings)
    else:
        print("Loaded embeddings from database.")

    # Answer the query
    answer = answer_query(
        query, document, embeddings
    )  # Pass the document and embeddings
    return answer


# Example usage
if __name__ == "__main__":
    pdf_path = "pdfs/Tendulkar.pdf"  # Replace with the actual PDF file path
    query = "What is the main topic of the document?"
    key = "document_key"  # Use a unique key for the document

    answer = process_pdf(pdf_path, query, key)
    print("Answer:", answer)
