from langchain import hub


retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
print(retrieval_qa_chat_prompt)