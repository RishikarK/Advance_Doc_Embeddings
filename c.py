import os
import json
import faiss
import pandas as pd
import pickle
from PyPDF2 import PdfReader
from docx import Document
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from pymongo import MongoClient
import redis
import numpy as np

app = Flask(__name__)

# Load environment variables
load_dotenv()
ollama_model = os.getenv("OLLAMA_MODEL")
embed_model = os.getenv("OLLAMA_EMBED_MODEL")
mongo_uri = os.getenv("MONGO_URI")
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = os.getenv("REDIS_PORT", 6379)

# Use Ollama for embeddings
embeddings_model = OllamaEmbeddings(model=embed_model)

# Memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# MongoDB setup
client = MongoClient(mongo_uri)
db = client["Embeddings"]
collection = db["embeddings_store_new"]

# Redis setup
cache = redis.StrictRedis(host=redis_host, port=redis_port, db=0)

retrieval_chain = None

prompt_template = PromptTemplate.from_template(
    """
You are a helpful assistant. Answer the following question based on the provided documents
and return only answer no extra matter:

Answer:
Question: {user_question}
"""
)


# Function to extract text from files
def extract_text_from_files(file_paths):
    """Extract text from various file types given their file paths."""
    text = ""
    for file_path in file_paths:
        try:
            if file_path.endswith(".pdf"):
                with open(file_path, "rb") as file:
                    pdf_reader = PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
            elif file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as file:
                    text += file.read()
            elif file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
                text += df.to_string(index=False)
            elif file_path.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_path)
                text += df.to_string(index=False)
            elif file_path.endswith(".docx"):
                doc = Document(file_path)
                for para in doc.paragraphs:
                    text += para.text + "\n"
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return text


# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks for embedding."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)


# Function to create or load FAISS vectorstore without PCA
def create_or_load_vectorstore(text_chunks, index_type="HNSW"):
    """Create or load the FAISS vector store from MongoDB."""
    try:
        # Step 1: Check if FAISS index exists in MongoDB
        if collection.count_documents({}) > 0:
            print("Loading FAISS vector store from MongoDB...")

            # Step 2: Retrieve FAISS data from MongoDB
            embeddings = collection.find_one({})
            faiss_data = embeddings["faiss_index"]
            docstore_data = embeddings.get("docstore")
            index_to_docstore_id_data = embeddings.get("index_to_docstore_id")

            # Step 3: Deserialize FAISS index and docstore
            faiss_index = pickle.loads(faiss_data)
            docstore = pickle.loads(docstore_data)
            index_to_docstore_id = pickle.loads(index_to_docstore_id_data)

            # Step 4: Create FAISS vector store from deserialized data
            vectorstore = FAISS(
                index=faiss_index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id,
                embedding_function=embeddings_model,
                # embeddings=embeddings_model,
            )
            print("FAISS vector store loaded successfully from MongoDB.")
            return vectorstore

        else:
            print("No embeddings found in MongoDB. Creating new FAISS index...")

            # Step 5: Create FAISS index dynamically based on the index type
            if not text_chunks:
                print("No text chunks available to create embeddings.")
                return None

            first_chunk = text_chunks[0]
            sample_embedding = embeddings_model.embed_query(first_chunk)
            embedding_size = len(sample_embedding)
            print(f"Determined embedding size: {embedding_size}")

            if index_type == "Flat":
                index = faiss.IndexFlatL2(embedding_size)
            elif index_type == "IVF":
                nlist = 100
                quantizer = faiss.IndexFlatL2(embedding_size)
                index = faiss.IndexIVFFlat(quantizer, embedding_size, nlist)
            elif index_type == "HNSW":
                index = faiss.IndexHNSWFlat(embedding_size, 32)

            # Step 6: Convert text to embeddings and add to index
            print("Converting text to embeddings and adding to index...")
            vectorstore = FAISS.from_texts(
                texts=text_chunks, embedding=embeddings_model
            )

            # Step 7: Serialize FAISS index and docstore
            faiss_index_binary = pickle.dumps(vectorstore.index)
            docstore_binary = pickle.dumps(vectorstore.docstore)
            index_to_docstore_id_binary = pickle.dumps(vectorstore.index_to_docstore_id)

            # Step 8: Store serialized data in MongoDB
            embeddings = {
                "faiss_index": faiss_index_binary,
                "docstore": docstore_binary,
                "index_to_docstore_id": index_to_docstore_id_binary,
            }
            collection.insert_one(embeddings)
            print("FAISS vector store created and saved to MongoDB.")
            return vectorstore

    except Exception as e:
        print(f"Error in create_or_load_vectorstore: {e}")
        return None


@app.route("/save-embeddings", methods=["POST"])
def save_embeddings():
    """Route for saving embeddings from files."""
    file_folder_path = "pdfs"  # Path to your files
    file_paths = [
        os.path.join(file_folder_path, file)
        for file in os.listdir(file_folder_path)
        if file.endswith((".pdf", ".txt", ".csv", ".xlsx", ".xls", ".docx"))
    ]

    if not file_paths:
        return jsonify({"error": "No supported files found in the directory."}), 400

    extracted_text = extract_text_from_files(file_paths)
    if not extracted_text:
        return jsonify({"error": "No text extracted from files."}), 400

    text_chunks = split_text_into_chunks(extracted_text)
    vectorstore = create_or_load_vectorstore(text_chunks, index_type="HNSW")

    if vectorstore is None:
        return jsonify({"error": "Failed to create or load vector store."}), 500

    return jsonify({"message": "Embeddings saved successfully in MongoDB!"})


def create_retrieval_chain_with_ollama(vectorstore):
    """Create a retrieval chain using Ollama models.

    Args:
        vectorstore (FAISS): The vector store to use for retrieval.

    Returns:
        ConversationalRetrievalChain: The conversational retrieval chain instance.
    """
    llm = OllamaLLM(model=ollama_model)

    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )


@app.route("/chat", methods=["POST"])
def chat():
    """Handle user questions and retrieve responses."""
    global retrieval_chain

    data = request.get_json()
    user_question = data.get("user_question")

    if not user_question:
        return jsonify({"error": "Missing user_question parameter."}), 400

    # Load existing embeddings
    vectorstore = create_or_load_vectorstore([], index_type="HNSW")

    # Check if vectorstore was successfully created or loaded
    if vectorstore is None:
        return jsonify({"error": "Failed to create or load vector store."}), 500

    # Format the user question with the prompt template
    prompt = prompt_template.format(user_question=user_question)

    retrieval_chain = create_retrieval_chain_with_ollama(vectorstore)

    response = retrieval_chain.invoke({"question": prompt})
    answer = response.get("answer", "Sorry, I couldn't find an answer.")

    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    save_chat_history(chat_history)

    return jsonify({"question": user_question, "answer": answer})


def save_chat_history(chat_history):
    """Save chat history to Redis or MongoDB as needed."""
    # Implementation for saving chat history
    pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8089, debug=False)
