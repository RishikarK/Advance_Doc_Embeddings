import os
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore  # type=ignore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load OpenAI API key
os.environ["OPENAI_API_KEY"] = (
    "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"
)

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# embedded_query = embeddings_model.embed_query(
#     "Is Venu Participated in Paris olympic events?"
# )
# print(len(embedded_query))

loader = TextLoader("sample_document.txt")

doc = loader.load()

# print("Document------------------------------------->", doc)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50, chunk_overlap=0, length_function=len
)

chunks = text_splitter.split_documents(doc)

all_embeddings = []

for chunk in chunks:
    chunk_embeddings = embeddings_model.embed_documents(chunk.page_content)
    all_embeddings.append(chunk_embeddings)

library = FAISS.from_documents(chunks, embeddings_model)

# answer = library.similarity_search(QUERY)

# print(answer)
retriever = library.as_retriever()

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

RETRIEVER_QUERY = "What is this document about"

ANSWER = qa.invoke(RETRIEVER_QUERY)

print(ANSWER)

library.save_local("faiss_doc")
# saved = FAISS.load_local("faiss_doc", embeddings_model)
