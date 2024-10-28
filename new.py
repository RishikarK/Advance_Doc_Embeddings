"""File uses advance embedding-techniques"""

from datetime import datetime, timezone
import os
import pickle
import zlib
import logging
import faiss

from pymongo import MongoClient
from flask import Flask, request, jsonify
from flask_cors import CORS
from redis import StrictRedis
from PyPDF2 import PdfReader

from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain  # pylint: disable=E0611
from langchain_community.vectorstores import FAISS
from langchain_core.messages.ai import AIMessage
from langchain_core.messages import HumanMessage

from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
load_dotenv()

# Loading variables from environment
ollama_model = os.getenv("OLLAMA_MODEL")
embed_model = os.getenv("OLLAMA_EMBED_MODEL")
mongo_uri = os.getenv("MONGO_URI")
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = os.getenv("REDIS_PORT")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use Ollama for embeddings
embeddings_model = OllamaEmbeddings(model=embed_model)

# Memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# MongoDB setup
client = MongoClient(mongo_uri)
db = client["Embeddings"]
collection = db["ksm_compress_embeddings"]
chat_history_collection = db["ksm_chat_history"]

# Redis setup
cache = StrictRedis(host=redis_host, port=redis_port, db=0)

retrieval_chain = None
prompt_template = PromptTemplate.from_template(
    """
You are a helpful assistant. Answer the following question based on the provided documents
and return only answer no extra matter:

Answer:
Question: {user_question}
"""
)


def extract_text_from_files(file_paths):
    """Function to extract text from PDF files."""
    text_contents = []

    for file_path in file_paths:
        with open(file_path, "rb") as file:
            pdf_reader = PdfReader(file)

            # Extract text from each page
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:  # Only append non-empty text
                    text_contents.append(text)

    return text_contents


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


# Function to compress data
def compress_data(data):
    """Compress data using zlib."""
    return zlib.compress(data)


# Function to decompress data
def decompress_data(compressed_data):
    """Decompress data using zlib."""
    return zlib.decompress(compressed_data)


# Function to create or load FAISS vectorstore without PCA
async def create_or_load_vectorstore(text_chunks, index_type="HNSW"):
    """Create or load the FAISS vector store from MongoDB."""
    index = None
    try:
        # Step 1: Check if FAISS index exists in MongoDB
        if collection.count_documents({}) > 0:
            print("Loading FAISS vector store from MongoDB...")

            # Step 2: Retrieve FAISS data from MongoDB
            embeddings = collection.find_one({})
            faiss_data = embeddings["faiss_index"]
            docstore_data = embeddings.get("docstore")
            index_to_docstore_id_data = embeddings.get("index_to_docstore_id")

            # Step 3: Deserialize and decompress FAISS index and docstore
            faiss_index = pickle.loads(decompress_data(faiss_data))
            docstore = pickle.loads(decompress_data(docstore_data))
            index_to_docstore_id = pickle.loads(
                decompress_data(index_to_docstore_id_data)
            )

            # Step 4: Create FAISS vector store from deserialized data
            vectorstore = FAISS(
                index=faiss_index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id,
                embedding_function=embeddings_model,
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

            # Create the appropriate FAISS index based on the specified index type
            if index_type == "Flat":
                index = faiss.IndexFlatL2(embedding_size)
            elif index_type == "IVF":
                nlist = 100
                quantizer = faiss.IndexFlatL2(embedding_size)
                index = faiss.IndexIVFFlat(quantizer, embedding_size, nlist)
            elif index_type == "HNSW":
                index = faiss.IndexHNSWFlat(embedding_size, 32)
            print(index)
            # Step 6: Convert text to embeddings and add to index
            print("Converting text to embeddings and adding to index...")
            vectorstore = FAISS.from_texts(
                texts=text_chunks, embedding=embeddings_model
            )

            # Step 7: Serialize and compress FAISS index and docstore
            faiss_index_binary = compress_data(pickle.dumps(vectorstore.index))
            docstore_binary = compress_data(pickle.dumps(vectorstore.docstore))
            index_to_docstore_id_binary = compress_data(
                pickle.dumps(vectorstore.index_to_docstore_id)
            )

            # Step 8: Store serialized and compressed data in MongoDB
            embeddings = {
                "faiss_index": faiss_index_binary,
                "docstore": docstore_binary,
                "index_to_docstore_id": index_to_docstore_id_binary,
            }
            collection.insert_one(embeddings)
            print("FAISS vector store created and saved to MongoDB.")
            return vectorstore

    except ValueError as e:
        print("Error in create_or_load_vectorstore: %s", e)
        return None


@app.route("/save-embeddings", methods=["POST"])
async def save_embeddings():
    """Route for saving embeddings from files."""
    file_folder_path = "pdfs"  # Path to your folder containing PDF files

    # Check if the folder exists
    if not os.path.exists(file_folder_path):
        return jsonify({"error": "Folder not found."}), 404

    # Get all PDF files in the folder
    pdf_files = [
        os.path.join(file_folder_path, f)
        for f in os.listdir(file_folder_path)
        if f.endswith(".pdf")
    ]

    # Ensure there are PDF files in the directory
    if not pdf_files:
        return jsonify({"error": "No supported files found in the directory."}), 400

    # Extract text from all PDF files
    extracted_text = extract_text_from_files(pdf_files)
    if not extracted_text:
        return jsonify({"error": "No text extracted from files."}), 400

    # Join the extracted text into a single string
    combined_text = "\n".join(extracted_text)

    # Split the combined text into chunks
    text_chunks = split_text_into_chunks(combined_text)

    # Continue with creating/loading the vector store
    vectorstore = await create_or_load_vectorstore(text_chunks, index_type="HNSW")

    if vectorstore is None:
        return jsonify({"error": "Failed to create or load vector store."}), 500

    return jsonify({"message": "Embeddings saved successfully in MongoDB!"})


def create_retrieval_chain_with_ollama(vectorstore):
    """Create a retrieval chain using Ollama models."""
    llm = OllamaLLM(model=ollama_model)

    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )


def save_chat_entry(user_id, user_question, response, chat_history):
    """Convert messages to a simple dictionary format and save chat history."""
    # Convert the new messages to a simple dictionary format
    chat_history_simple = []
    for message in chat_history:
        if isinstance(message, HumanMessage):
            chat_history_simple.append(
                {
                    "role": "user",
                    "content": user_question,
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                }
            )
        elif isinstance(message, AIMessage):
            chat_history_simple.append(
                {
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                }
            )

    # Retrieve existing chat history
    existing_chat_entry = chat_history_collection.find_one({"user_id": user_id})

    if existing_chat_entry:
        # If chat history exists, append new messages to it
        existing_chat_history = existing_chat_entry.get("chat_history", [])
        existing_chat_history.extend(chat_history_simple)
        chat_history_collection.update_one(
            {"user_id": user_id}, {"$set": {"chat_history": existing_chat_history}}
        )
    else:
        # If no chat history exists, create a new entry
        chat_history_collection.insert_one(
            {"user_id": user_id, "chat_history": chat_history_simple}
        )


@app.route("/chat", methods=["POST"])
async def chat():
    """Handle user questions and retrieve responses."""
    data = request.get_json()
    user_question = data.get("user_question")
    user_id = "1"
    if not user_question:
        return jsonify({"error": "Missing user_question parameter."}), 400

    # Load existing embeddings
    vectorstore = await create_or_load_vectorstore([], index_type="HNSW")

    if vectorstore is None:
        return jsonify({"error": "Failed to create or load vector store."}), 500

    prompt = prompt_template.format(user_question=user_question)
    global retrieval_chain  # pylint: disable= W0603
    if retrieval_chain is None:
        retrieval_chain = create_retrieval_chain_with_ollama(vectorstore)
        print("Creating a new retrieval chain.")
    else:
        print("Using an existing retrieval chain.")

    response = retrieval_chain.invoke({"question": prompt})
    answer = response.get("answer", "Sorry, I couldn't find an answer.")

    # Ensure chat_history_collection is not None
    chat_history = [HumanMessage(content=user_question), AIMessage(content=answer)]
    # Save the chat entry
    save_chat_entry(user_id, user_question, answer, chat_history)

    response = {"question": user_question, "answer": answer}

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8089, debug=True)
