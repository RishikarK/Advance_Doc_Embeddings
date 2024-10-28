"""The file deals with new methods of RAG"""

import os
import json
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from dotenv import load_dotenv

from flask import Flask, request, jsonify

from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain  # pylint: disable=E0611

from langchain_community.vectorstores import FAISS

from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

app = Flask(__name__)

# Paths and constants
FILE_FOLDER = "pdfs"  # Folder for all document types
HISTORY_FILE_PATH = "history.json"
FAISS_FOLDER_PATH = "ollama_faiss3"  # Folder to save the FAISS index

# Load environment variables
load_dotenv()
ollama_model = os.getenv("OLLAMA_MODEL")
embed_model = os.getenv("OLLAMA_EMBED_MODEL")

# Use Ollama for embeddings
embeddings_model = OllamaEmbeddings(model=embed_model)

# Memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the prompt template
prompt_template = PromptTemplate.from_template(
    """
You are a helpful assistant. Answer the following question based on the provided documents
and return only answer no extra matter:

Answer:
Question: {user_question}
"""
)


def extract_text_from_files(file_paths):
    """Extract text from various file types given their file paths.

    Args:
        file_paths (list): List of file paths.

    Returns:
        str: Concatenated text extracted from all specified files.
    """
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
            elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                df = pd.read_excel(file_path)
                text += df.to_string(index=False)
            elif file_path.endswith(".docx"):
                doc = Document(file_path)
                for para in doc.paragraphs:
                    text += para.text + "\n"
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return text


def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks for embedding.

    Args:
        text (str): The text to be split.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)


def create_or_load_vectorstore(text_chunks):
    """Create or load the FAISS vector store.

    Args:
        text_chunks (list): List of text chunks to store.

    Returns:
        FAISS: A FAISS vector store instance.
    """
    if os.path.exists(FAISS_FOLDER_PATH):
        try:
            vectorstore = FAISS.load_local(
                FAISS_FOLDER_PATH,
                embeddings_model,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            print(f"Failed to load vector store: {e}")
            return None
    else:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings_model)
        os.makedirs(FAISS_FOLDER_PATH, exist_ok=True)
        vectorstore.save_local(FAISS_FOLDER_PATH)

    return vectorstore


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


def save_chat_history(history):
    """Save chat history to a JSON file.

    Args:
        history (list): List of messages exchanged in the chat.
    """
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
    """Route for saving embeddings from files.

    Returns:
        JSON response indicating success or error.
    """
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

    return jsonify({"message": "Embeddings saved successfully in VectorStore!"})


@app.route("/chat", methods=["POST"])
def chat():
    """Handle user questions and retrieve responses.

    Returns:
        JSON response with the question and answer.
    """
    data = request.get_json()
    user_question = data.get("user_question")

    if not user_question:
        return jsonify({"error": "Missing user_question parameter."}), 400

    vectorstore = create_or_load_vectorstore([])  # Load existing embeddings
    prompt = prompt_template.format(user_question=user_question)
    retrieval_chain = create_retrieval_chain_with_ollama(vectorstore)

    if retrieval_chain is None:
        return jsonify({"error": "Failed to create retrieval chain."}), 500

    response = retrieval_chain.invoke({"question": prompt})
    answer = response.get("answer", "Sorry, I couldn't find an answer.")

    saved_chat_history = memory.load_memory_variables({}).get("chat_history", [])
    save_chat_history(saved_chat_history)

    return jsonify({"question": user_question, "answer": answer})


@app.route("/chat-history", methods=["GET"])
def chat_history():
    """Retrieve chat history.

    Returns:
        JSON response containing the chat history or an error.
    """
    if not os.path.exists(HISTORY_FILE_PATH):
        return jsonify({"error": "No chat history found."}), 404

    with open(HISTORY_FILE_PATH, "r", encoding="utf-8") as file:
        history = json.load(file)

    return jsonify(history)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
