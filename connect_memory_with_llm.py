import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS


HF_TOKEN = os.getenv("HF_TOKEN") 
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    endpoint = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        model_kwargs={"max_length": 512}
    )
    return ChatHuggingFace(llm=endpoint)  # wrap endpoint in ChatHuggingFace



CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    return ChatPromptTemplate.from_messages([
        ("system", custom_prompt_template),
        ("human", "{question}")
    ])


DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)



qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)


user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

print("\nRESULT: ", response["result"])
print("\nSOURCE DOCUMENTS: ")
for doc in response["source_documents"]:
    print(" -", doc.metadata.get("source", "unknown"))

print("\nREFERENCE TEXTS:")
for i, doc in enumerate(response["source_documents"], start=1):
    print(f"\n--- Reference {i} ---")
    print(doc.page_content.strip())
