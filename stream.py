"""This is the """

import os
import time
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS  # pylint: disable=E0611
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL")
LOGO = os.getenv("LOGO")

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = API_KEY

# Initialize OpenAI embeddings model
embeddings_model = OpenAIEmbeddings(model=EMBED_MODEL)

# Path to save or load the FAISS index
TEXT_FILE_PATH = "ancient_history.txt"
FILE_INDEX_PATH = "faiss_history_doc2"
st.set_page_config(
    page_title="Q&A",
    page_icon=LOGO,
    layout="centered",
    initial_sidebar_state="auto",
)


def load_faiss_index(index_path):
    """Load or create FAISS index"""
    if os.path.exists(index_path):
        st.toast("Loading existing FAISS index...")
        library = FAISS.load_local(
            index_path, embeddings_model, allow_dangerous_deserialization=True
        )
    else:
        st.toast("Creating new FAISS index...")
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


def main():
    """Main function to run the Streamlit application"""
    st.header("Q&A using FAISS", divider=True)
    st.subheader("Ask Your Questions Below")

    retriever = None  # Initialize retriever variable

    # Create a QA system instance
    qa = None

    # Input for the user's question
    question = st.chat_input("Ask question:")

    if question:
        # Check if FAISS index exists and load it or create it
        if os.path.exists(FILE_INDEX_PATH):
            retriever = load_faiss_index(FILE_INDEX_PATH)
        else:
            st.toast("FAISS index not found. Creating a new index...")
            # Create the FAISS index since it doesn't exist
            retriever = load_faiss_index(FILE_INDEX_PATH)

        # Create a QA system using the retriever
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(), chain_type="stuff", retriever=retriever
        )

        with st.spinner("Wait for it..."):
            time.sleep(2)  # Simulate a delay

            # Run the query and return the answer
            try:
                answer = qa.invoke(question)
                query = answer["query"]
                result = answer["result"]

                # Display the question and the result
                st.markdown(f"**Question:** {query}")
                st.markdown(f"**Answer:** {result}")
            except TimeoutError as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
