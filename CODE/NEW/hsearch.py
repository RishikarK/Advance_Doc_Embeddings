"""The files is responsible for embedding the syntactic search"""

# import os
import time
from uuid import uuid4
from pinecone import Pinecone, ServerlessSpec

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore  # pylint:disable =E0401

PINECONE_API_KEY = "2fd4dcce-426e-4814-b50a-39690b7eea82"

pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "demo"

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(INDEX_NAME)


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# document_1 = Document(
#     page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
#     metadata={"source": "tweet"},
# )

# document_2 = Document(
#     page_content="The weather forecast for tomorrow is cloudy and overcast, with high of 62 deg.",
#     metadata={"source": "news"},
# )

# document_3 = Document(
#     page_content="Building an exciting new project with LangChain - come check it out!",
#     metadata={"source": "tweet"},
# )

# document_4 = Document(
#     page_content="Robbers broke into the city bank and stole $1 million in cash.",
#     metadata={"source": "news"},
# )

# document_5 = Document(
#     page_content="Wow! That was an amazing movie. I can't wait to see it again.",
#     metadata={"source": "tweet"},
# )

# document_6 = Document(
#     page_content="Is the new iPhone worth the price? Read this review to find out.",
#     metadata={"source": "website"},
# )

# document_7 = Document(
#     page_content="The top 10 soccer players in the world right now.",
#     metadata={"source": "website"},
# )

# document_8 = Document(
#     page_content="LangGraph is the best framework for building stateful, agentic applications!",
#     metadata={"source": "tweet"},
# )

# document_9 = Document(
#     page_content="The stock market is down 500 points today due to fears of a recession.",
#     metadata={"source": "news"},
# )

# document_10 = Document(
#     page_content="I have a bad feeling I am going to get deleted :(",
#     metadata={"source": "tweet"},
# )

# documents = [
#     document_1,
#     document_2,
#     document_3,
#     document_4,
#     document_5,
#     document_6,
#     document_7,
#     document_8,
#     document_9,
#     document_10,
# ]
# uuids = [str(uuid4()) for _ in range(len(documents))]

# vector_store.add_documents(documents=documents, ids=uuids)


# Similarity Search
# results = vector_store.similarity_search(
#     "LangChain provides abstractions to make working with LLMs easy",
#     k=2,
#     filter={"source": "tweet"},
# )
# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")

# Similarity Search With Score
results = vector_store.similarity_search_with_score(
    "Will it be hot tomorrow?", k=1, filter={"source": "news"}
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

# Similarity Search With RAG acting as retriever
# retriever = vector_store.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 2, "score_threshold": 0.5},
# )
# res = retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})
# print(res)
