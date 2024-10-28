import os
import subprocess
import json
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS  # pylint: disable=E0611
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings  # Import Ollama embeddings if available

app = Flask(__name__)
CORS(app)

# Load environment variables from .env file
load_dotenv()

EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL")

# Path to save or load the FAISS index
TEXT_FILE_PATH = "ancient_history.txt"
FILE_INDEX_PATH = "llama_faiss"

# Path to Ollama's local model
OLLAMA_MODEL = (
    "llama3:latest"  # Change this to the appropriate model you are using with Ollama
)


@app.route("/")
def home():
    """Cors Enable"""
    return "CORS Enabled"


@app.route("/chat", methods=["POST"])
@cross_origin()
def chat():
    """Q&A based on FAISS"""
    data = request.json
    if not data or "ques" not in data:
        return jsonify({"message": "No question provided"}), 400

    query = data["ques"]

    # Check if FAISS index exists and load it or create it
    if os.path.exists(FILE_INDEX_PATH):
        # Load existing FAISS index
        retriever = load_faiss_index(FILE_INDEX_PATH)

        # Create a QA system using the retriever and Ollama's model
        qa = RetrievalQA.from_chain_type(
            llm=OllamaLLM(), chain_type="stuff", retriever=retriever
        )

        # Run the query and return the answer
        answer = qa.invoke(query)
        return jsonify({"answer": answer})

    # If FAISS index doesn't exist, create it
    load_faiss_index(FILE_INDEX_PATH)
    return jsonify({"message": "FAISS index created. Please ask your question again."})


def load_faiss_index(index_path):
    """Load or create FAISS index"""
    if os.path.exists(index_path):
        print("Loading existing FAISS index...")
        # Initialize Ollama embeddings model
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)

        # Load the FAISS index
        library = FAISS.load_local(
            index_path,
            embeddings=embeddings,  # Pass the embeddings model here
            allow_dangerous_deserialization=True,
        )
    else:
        print("Creating new FAISS index...")

        # Load document and split it into chunks
        loader = TextLoader(TEXT_FILE_PATH)
        doc = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, length_function=len
        )
        chunks = text_splitter.split_documents(doc)

        # Generate embeddings and store in FAISS
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        library = FAISS.from_documents(chunks, embedding=embeddings)

        # Save the FAISS index for future use
        library.save_local(index_path)

    # Return a retriever based on the FAISS index
    return library.as_retriever()


class OllamaLLM(RunnableLambda):
    """Class to interact with Ollama's local models"""

    def __init__(self, model_name=OLLAMA_MODEL):
        self.model_name = model_name

    def _call(self, query):
        """Call Ollama model for inference"""
        response = subprocess.run(
            [
                "ollama",
                "chat",
                "--model",
                self.model_name,
                "--messages",
                json.dumps(
                    [
                        {"role": "user", "content": query},
                    ]
                ),
            ],
            capture_output=True,
            text=True,
        )

        if response.returncode != 0:
            return "Error with Ollama model invocation."

        # Parse the response from Ollama
        response_json = json.loads(response.stdout)
        return response_json.get("message", "No response from model.")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7272)
