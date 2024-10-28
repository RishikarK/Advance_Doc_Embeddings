"""Code responsible from loading embeddings from folder ,
if not there it will create and load them"""

import os
import json
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import (  # pylint: disable=import-error
    ConversationalRetrievalChain,
)

from langchain_ollama.embeddings import ( # type: ignore
    OllamaEmbeddings,
)  # pylint: disable=import-error
from langchain_ollama.llms import OllamaLLM  # type: ignore # pylint: disable=E0401

app = Flask(__name__)

# Path to store chat history
HISTORY_FILE_PATH = "history.json"
FAISS_FOLDER_PATH = "ollama_faiss"  # Folder to save the FAISS index
embeddings_model = OllamaEmbeddings(model="nomic-embed-text:latest")
def extract_text_from_pdfs(pdf_file_paths):
    """Function to extract text from PDFs given their file paths"""
    text = ""
    for pdf_file_path in pdf_file_paths:
        with open(pdf_file_path, "rb") as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text


def split_text_into_chunks(text):
    """Function to split text into chunks"""
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def create_or_load_vectorstore(text_chunks):
    """# Function to create or load the vector store ,
    Check if the FAISS index folder exists"""
    if os.path.exists(FAISS_FOLDER_PATH):
        vectorstore = FAISS.load_local(FAISS_FOLDER_PATH , embeddings_model, allow_dangerous_deserialization=True)
    else:
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        os.makedirs(FAISS_FOLDER_PATH, exist_ok=True)
        vectorstore.save_local(FAISS_FOLDER_PATH)

    return vectorstore


def create_conversation_chain(vectorstore):
    """Function to create conversation chain"""
    llm = OllamaLLM(model="llama3:latest")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def save_chat_history(history):
    """Function to save chat history to a JSON file"""
    with open(HISTORY_FILE_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file)


@app.route("/chat", methods=["POST"])
def chat():
    """function for passing question and getting answer"""
    data = request.get_json()
    user_question = data.get("user_question")

    if not user_question:
        return jsonify({"error": "Missing user_question parameter"}), 400

    pdf_folder_path = "pdfs"
    pdf_file_paths = [
        os.path.join(pdf_folder_path, file)
        for file in os.listdir(pdf_folder_path)
        if file.endswith(".pdf")
    ]

    if not pdf_file_paths:
        return jsonify({"error": "No PDF files found in directory"}), 400

    # Extract text from PDFs
    pdf_text = extract_text_from_pdfs(pdf_file_paths)

    # Split text into chunks
    text_chunks = split_text_into_chunks(pdf_text)

    # Create or load the vector store
    vectorstore = create_or_load_vectorstore(text_chunks)

    # Create conversation chain
    conversation_chain = create_conversation_chain(vectorstore)

    # Handle user question and retrieve responses
    response = conversation_chain.invoke({"question": user_question})

    # Extract chat history
    chat_history = [
        (message.content, "user") if i % 2 == 0 else (message.content, "bot")
        for i, message in enumerate(response["chat_history"])
    ]

    # Append chat history to the history file
    save_chat_history(chat_history)

    # Prepare separate lists for questions and responses
    questions = [message[0] for message in chat_history if message[1] == "user"]
    responses = [message[0] for message in chat_history if message[1] == "bot"]

    # Prepare the response JSON
    response_data = {"questions": questions, "responses": responses}

    return jsonify(response_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
