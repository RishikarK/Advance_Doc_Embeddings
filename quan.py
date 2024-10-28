"""File uses advanced embedding techniques with model optimizations."""

from datetime import datetime, timezone
import os
import pickle
import zlib
import logging
import faiss
from torch.nn.utils import prune

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
import torch
from transformers import DistilBertModel

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


# Optimization functions
def optimize_torch_model(model):
    """Only quantize model if it is a torch.nn.Module."""
    if isinstance(model, torch.nn.Module):
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    return model


# Load and prune the DistilBERT model
# Load and prune the DistilBERT model
def load_and_prune_model():
    """Load and prune the model."""
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    # Optimize the model by applying dynamic quantization
    model = optimize_torch_model(model)

    # Apply pruning to the model
    for name, module in model.named_modules():
        print(name)
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(
                module, name="weight", amount=0.4
            )  # prune 40% of the weights

    # Remove pruning (optional, to finalize the sparse weights)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.remove(module, "weight")

    return model


def apply_dynamic_quantization(model):
    """Apply dynamic quantization to the model."""
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,  # Only quantize Linear layers to int8
    )
    return quantized_model


# Load the pruned and quantized model
pruned_model = load_and_prune_model()
llm_model = apply_dynamic_quantization(pruned_model)

# Use Ollama for embeddings without applying torch quantization directly
embeddings_model = OllamaEmbeddings(
    model=embed_model
)  # Ollama embeddings do not require quantization

# Memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# MongoDB setup
client = MongoClient(mongo_uri)
db = client["Embeddings"]
collection = db["ksm_quant_embeddings"]
chat_history_collection = db["ksm_chat_history"]

# Redis setup
cache = StrictRedis(host=redis_host, port=redis_port, db=0)

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
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
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
        if collection.count_documents({}) > 0:
            print(index)
            print("Loading FAISS vector store from MongoDB...")
            embeddings = collection.find_one({})
            faiss_data = embeddings["faiss_index"]
            docstore_data = embeddings.get("docstore")
            index_to_docstore_id_data = embeddings.get("index_to_docstore_id")
            faiss_index = pickle.loads(decompress_data(faiss_data))
            docstore = pickle.loads(decompress_data(docstore_data))
            index_to_docstore_id = pickle.loads(
                decompress_data(index_to_docstore_id_data)
            )
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
            if not text_chunks:
                print("No text chunks available to create embeddings.")
                return None
            first_chunk = text_chunks[0]
            sample_embedding = embeddings_model.embed_query(first_chunk)
            embedding_size = len(sample_embedding)
            if index_type == "Flat":
                index = faiss.IndexFlatL2(embedding_size)
            elif index_type == "IVF":
                nlist = 100
                quantizer = faiss.IndexFlatL2(embedding_size)
                index = faiss.IndexIVFFlat(quantizer, embedding_size, nlist)
            elif index_type == "HNSW":
                index = faiss.IndexHNSWFlat(embedding_size, 32)
            vectorstore = FAISS.from_texts(
                texts=text_chunks, embedding=embeddings_model
            )
            faiss_index_binary = compress_data(pickle.dumps(vectorstore.index))
            docstore_binary = compress_data(pickle.dumps(vectorstore.docstore))
            index_to_docstore_id_binary = compress_data(
                pickle.dumps(vectorstore.index_to_docstore_id)
            )
            embeddings = {
                "faiss_index": faiss_index_binary,
                "docstore": docstore_binary,
                "index_to_docstore_id": index_to_docstore_id_binary,
            }
            collection.insert_one(embeddings)
            print("FAISS vector store created and saved to MongoDB.")
            return vectorstore
    except ValueError as e:
        print("Error in create_or_load_vectorstore:", e)
        return None


@app.route("/save-embeddings", methods=["POST"])
async def save_embeddings():
    """Route for saving embeddings from files."""
    file_folder_path = "pdfs"
    if not os.path.exists(file_folder_path):
        return jsonify({"error": "Folder not found."}), 404
    pdf_files = [
        os.path.join(file_folder_path, f)
        for f in os.listdir(file_folder_path)
        if f.endswith(".pdf")
    ]
    if not pdf_files:
        return jsonify({"error": "No supported files found in the directory."}), 400
    extracted_text = extract_text_from_files(pdf_files)
    if not extracted_text:
        return jsonify({"error": "No text extracted from files."}), 400
    combined_text = "\n".join(extracted_text)
    text_chunks = split_text_into_chunks(combined_text)
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
    """Save chat history in MongoDB."""
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
    chat_history_collection.update_one(
        {"user_id": user_id},
        {
            "$set": {"chat_history": chat_history_simple},
            "$setOnInsert": {"user_id": user_id},
        },
        upsert=True,
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
    retrieval_chain = create_retrieval_chain_with_ollama(vectorstore)

    response = retrieval_chain.invoke({"question": prompt})
    answer = response.get("answer", "Sorry, I couldn't find an answer.")

    # Ensure chat_history_collection is not None
    chat_history = [HumanMessage(content=user_question), AIMessage(content=answer)]
    # Save the chat entry
    save_chat_entry(user_id, user_question, answer, chat_history)

    response = {"question": user_question, "answer": answer}

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=False)
