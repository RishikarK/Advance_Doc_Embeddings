"""Import necessary libraries"""

import os
import pickle
import faiss
from langchain_community.vectorstores import FAISS  # pylint: disable=E0611
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


os.environ["OPENAI_API_KEY"] = (
    "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"
)


# Initialize the embedding function
def load_documents_from_pickle(file_path):
    """load teh Document form the pickle"""
    with open(file_path, "rb") as file:
        documents = pickle.load(file)
    return documents


# Path to your pickle file
PICKLE_FILE_PATH = "document_embedding.pkl"  # Change this to your actual file path

# Load documents
loaded_documents = load_documents_from_pickle(PICKLE_FILE_PATH)

# Inspect the loaded data
print("Loaded data structure:", type(loaded_documents))
if isinstance(loaded_documents, list):
    print(f"Number of items loaded: {len(loaded_documents)}")
    # Inspect the first item
    if len(loaded_documents) > 0:
        print("First item:", loaded_documents[0])
else:
    print("Loaded data is not a list.")

# Convert to Document objects if necessary
if isinstance(loaded_documents, list) and all(
    isinstance(doc, str) for doc in loaded_documents
):
    # Assuming loaded_documents is a list of strings (text)
    loaded_documents = [
        Document(page_content=doc, metadata={}) for doc in loaded_documents
    ]
elif isinstance(loaded_documents, list) and all(
    isinstance(doc, dict) for doc in loaded_documents
):
    # Assuming loaded_documents is a list of dictionaries with 'content' and 'metadata' keys
    loaded_documents = [
        Document(page_content=doc["content"], metadata=doc.get("metadata", {}))
        for doc in loaded_documents
    ]

# Now we can proceed if the documents are in the correct format
if isinstance(loaded_documents, list) and all(
    isinstance(doc, Document) for doc in loaded_documents
):
    print(f"Loaded {len(loaded_documents)} documents in the expected format.")
else:
    print("Loaded documents are still not in the expected format.")
    raise ValueError("Loaded documents are not of type 'Document'.")

# Initialize the embedding function
embedding_function = OpenAIEmbeddings()

# Create FAISS index
index = faiss.IndexFlatL2(len(embedding_function.embed_query("hello world")))

# Create the In-Memory Docstore
docstore = InMemoryDocstore()

# Create the vector store
vector_store = FAISS(
    embedding_function=embedding_function,
    index=index,
    docstore=docstore,
    index_to_docstore_id={},
)

# Add Documents
# Ensure to provide unique IDs for each document
ids = [str(i) for i in range(len(loaded_documents))]
vector_store.add_documents(documents=loaded_documents, ids=ids)

# (Optional) Print added document contents for verification
print("Added Documents:")
for doc in loaded_documents:
    print(f"* {doc.page_content} [{doc.metadata}]")


# Function to ask a question
def ask_question(query):
    """Ask question based on teh document"""
    results = vector_store.similarity_search(query=query, k=1)  # Adjust k as needed
    if results:
        for docu in results:
            print(f"\nAnswer: {docu.page_content} [{docu.metadata}]")
    else:
        print("No relevant documents found.")


# Example Questions
questions = [
    "What is this document about?",
    "How can I reuse embeddings?",
    "What are the performance optimization techniques mentioned?",
]

for question in questions:
    print(f"\nQuestion: {question}")
    ask_question(question)
