import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# Load environment variables
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="Telecom RAG Chatbot",
    page_icon="📱"
)

st.title("📱 Telecom RAG Chatbot")
st.write("Ask questions about telecom services and network issues.")


# Load telecom document
with open("data/telecom_knowledge_base.txt", "r", encoding="utf-8") as file:
    text = file.read()


# Create Q&A chunks
chunks = [
    Document(page_content=chunk)
    for chunk in text.split("\n\n")
]


# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ChromaDB
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)


# Retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# Format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Prompt
SYSTEM_PROMPT = """
You are a helpful telecom assistant.

Answer the question using ONLY the context provided below.

If the context does not contain enough information,
say clearly that the information is not available.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])


# LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# RAG chain
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# Chat input
question = st.chat_input("Ask a telecom question...")

if question:
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        response = chain.invoke(question)
        st.write(response)