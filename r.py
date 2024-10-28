import os
import json
import pandas as pd
import pickle
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from pymongo import MongoClient
import redis
import asyncio

app = Flask(__name__)

# Paths and constants
FILE_FOLDER = "pdfs"  # Folder for all document types
HISTORY_FILE_PATH = "history.json"

# Load environment variables
load_dotenv()
ollama_model = os.getenv("OLLAMA_MODEL")
embed_model = os.getenv("OLLAMA_EMBED_MODEL")
mongo_uri = os.getenv("MONGO_URI")
redis_host = os.getenv("REDIS_HOST", "localhost")  # Optional Redis host
redis_port = os.getenv("REDIS_PORT", 6379)  # Optional Redis port

# Use Ollama for embeddings
embeddings_model = OllamaEmbeddings(model=embed_model)

# Memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# MongoDB setup
client = MongoClient(mongo_uri)
db = client["Embeddings"]
collection = db["embeddings_store"]

# Redis setup
cache = redis.StrictRedis(host=redis_host, port=redis_port, db=0)

retrieval_chain = None

# Define the prompt template
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


# Function to create or load FAISS vectorstore
def create_or_load_vectorstore(text_chunks):
    """Create or load the FAISS vector store from MongoDB."""
    try:
        # Check if embeddings exist in MongoDB
        if collection.count_documents({}) > 0:
            print("Loading FAISS index from MongoDB...")
            embeddings = collection.find_one({})
            faiss_data = embeddings["faiss_index"]
            docstore_data = embeddings.get("docstore")
            index_to_docstore_id = embeddings.get("index_to_docstore_id")

            # Load FAISS index from binary data
            faiss_index = pickle.loads(faiss_data)
            docstore = pickle.loads(docstore_data)
            index_to_docstore_id = pickle.loads(index_to_docstore_id)

            # Recreate the FAISS object with the embedding function
            vectorstore = FAISS(
                index=faiss_index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id,
                embedding_function=embeddings_model,
            )
            print("FAISS index loaded successfully from MongoDB.")
        else:
            print("No embeddings found in MongoDB. Creating new FAISS index...")
            vectorstore = FAISS.from_texts(
                texts=text_chunks, embedding=embeddings_model
            )

            # Serialize FAISS index, docstore, and index_to_docstore_id
            faiss_index_binary = pickle.dumps(vectorstore.index)
            docstore_binary = pickle.dumps(vectorstore.docstore)
            index_to_docstore_id_binary = pickle.dumps(vectorstore.index_to_docstore_id)

            embeddings = {
                "faiss_index": faiss_index_binary,
                "docstore": docstore_binary,
                "index_to_docstore_id": index_to_docstore_id_binary,
            }
            collection.insert_one(embeddings)
            print("FAISS index created and saved to MongoDB.")

        return vectorstore

    except Exception as e:
        print(f"Error in create_or_load_vectorstore: {e}")
        return None


# Function to create a retrieval chain using Ollama models
def create_retrieval_chain_with_ollama(vectorstore):
    """Create a retrieval chain using Ollama models."""
    llm = OllamaLLM(model=ollama_model)

    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )


# Function to save chat history
def save_chat_history(history):
    """Save chat history to a JSON file."""
    formatted_history = [
        {"content": message.content, "role": "user" if i % 2 == 0 else "bot"}
        for i, message in enumerate(history)
    ]
    with open(HISTORY_FILE_PATH, "w", encoding="utf-8") as file:
        json.dump(formatted_history, file, ensure_ascii=False, indent=4)


@app.route("/", methods=["GET"])
def index():
    """Default route to confirm the server is running."""
    return jsonify({"message": "Server is running. Welcome to the chat application!"})


@app.route("/save-embeddings", methods=["POST"])
def save_embeddings():
    """Route for saving embeddings from files."""
    file_folder_path = FILE_FOLDER
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
    vectorstore = create_or_load_vectorstore(text_chunks)

    if vectorstore is None:
        return jsonify({"error": "Failed to create or load vector store."}), 500

    return jsonify({"message": "Embeddings saved successfully in MongoDB!"})


async def async_chat(user_question):
    """Asynchronous processing for user questions."""
    global retrieval_chain

    # Check Redis cache for the answer
    cached_answer = cache.get(user_question)
    if cached_answer:
        return cached_answer.decode("utf-8")

    # Load existing embeddings
    vectorstore = create_or_load_vectorstore([])

    if vectorstore is None:
        return "Failed to create or load vector store."

    # Create the retrieval chain if not already created
    if retrieval_chain is None:
        print("Creating a new retrieval chain")
        retrieval_chain = create_retrieval_chain_with_ollama(vectorstore)
        if retrieval_chain is None:
            return "Failed to create retrieval chain."
    else:
        print("Using the existing retrieval chain...")

    # Format the user question with the prompt template
    prompt = prompt_template.format(user_question=user_question)

    # Invoke the retrieval chain
    response = retrieval_chain.invoke({"question": prompt})
    answer = response.get("answer", "Sorry, I couldn't find an answer.")

    # Cache the response
    cache.set(user_question, answer, ex=3600)  # Cache for 1 hour

    return answer


@app.route("/chat", methods=["POST"])
async def chat():
    """Handle user questions and retrieve responses."""
    data = request.get_json()
    user_question = data.get("user_question")

    if not user_question:
        return jsonify({"error": "Missing user_question parameter."}), 400

    answer = await async_chat(user_question)

    # Save chat history
    saved_chat_history = memory.load_memory_variables({}).get("chat_history", [])
    save_chat_history(saved_chat_history)

    return jsonify({"question": user_question, "answer": answer})


@app.route("/chat-history", methods=["GET"])
def chat_history():
    """Retrieve chat history."""
    if not os.path.exists(HISTORY_FILE_PATH):
        return jsonify({"error": "No chat history found."}), 404

    with open(HISTORY_FILE_PATH, "r", encoding="utf-8") as file:
        history = json.load(file)

    return jsonify({"chat_history": history})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8989, debug=True)
