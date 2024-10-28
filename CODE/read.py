import os
import pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

os.environ["OPENAI_API_KEY"] = (
    "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"
)
# Load the stored document embedding
with open("document_embedding.pkl", "rb") as f:
    document_embedding = pickle.load(f)

# Create an instance of OpenAI embeddings again for question embedding
embeddings = OpenAIEmbeddings()


# Function to answer a question
def answer_question(question):
    # Embed the question
    question_embedding = embeddings.embed_query(question)

    # Use FAISS for similarity search (You might want to save/load FAISS index)
    faiss_index = FAISS.from_embeddings(document_embedding)

    # Find the most similar document
    similar_doc_index = faiss_index.similarity_search(question_embedding)

    # Return the most relevant part or the answer
    return similar_doc_index[0].page_content


# Example of asking a question
question = "What is the main topic of the document?"
answer = answer_question(question)
print(answer)
