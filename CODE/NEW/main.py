"""It gives teh retrieval q&a based on faiss local folder"""

import os
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from langchain_community.vectorstores import FAISS  # pylint: disable=E0611
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings

app = Flask(__name__)
CORS(app)

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL")
# OPENAI_MODEL=os.getenv("OPENAI_MODEL")
# TEMP = os.getenv("TEMPERATURE")

# Ensure the API key is set in the environment
os.environ["OPENAI_API_KEY"] = API_KEY

# Use the OpenAI embeddings model
embeddings_model = OpenAIEmbeddings(model=EMBED_MODEL)

# Path to save or load the FAISS index
TEXT_FILE_PATH = "ancient_history.txt"
FILE_INDEX_PATH = "faiss_history_doc"


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

        # Create a QA system using the retriever
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(), chain_type="stuff", retriever=retriever
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
        library = FAISS.load_local(
            index_path, embeddings_model, allow_dangerous_deserialization=True
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
        library = FAISS.from_documents(chunks, embeddings_model)

        # Save the FAISS index for future use
        library.save_local(index_path)

    # Return a retriever based on the FAISS index
    return library.as_retriever()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7272)
