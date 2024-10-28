import os
from dotenv import load_dotenv
from reflex import App, route, Response
from langchain_community.vectorstores import FAISS  # pylint: disable=E0611
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL")

# Ensure the API key is set in the environment
os.environ["OPENAI_API_KEY"] = API_KEY

# Use the OpenAI embeddings model
embeddings_model = OpenAIEmbeddings(model=EMBED_MODEL)

# Path to save or load the FAISS index
TEXT_FILE_PATH = "ancient_history.txt"
FILE_INDEX_PATH = "faiss_history_doc"


# Main home route
async def home():
    """Cors Enable"""
    return Response("CORS Enabled", status=200)


async def chat(req):
    """Q&A based on FAISS"""
    data = await req.json()
    if not data or "ques" not in data:
        return Response({"message": "No question provided"}, status=400)

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
        return Response({"answer": answer}, status=200)

    # If FAISS index doesn't exist, create it
    load_faiss_index(FILE_INDEX_PATH)
    return Response(
        {"message": "FAISS index created. Please ask your question again."}, status=200
    )


# Function to load or create FAISS index
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


# Initialize and run the Reflex app
app = App()
app.run(debug=True, host="0.0.0.0", port=7272)
