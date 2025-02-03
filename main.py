import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory   
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv 
import os
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['AZURE_OPENAI_API_KEY'] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv("AZURE_OPENAI_ENDPOINT")

from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment = "gpt-4o",
    api_version = "2024-08-01-preview",
    temperature = 0,
    max_tokens = None,
    timeout = None,
    max_retries = 2,
)

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    # dimensions: Optional[int] = None, # Can specify dimensions with new text-embedding-3 models
    azure_endpoint='https://ai-aimlcoop563171993848.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-05-15'
    # api_key=... # Can provide an API key directly. If missing read env variable AZURE_OPENAI_API_KEY
    # openai_api_version=..., # If not provided, will read env variable AZURE_OPENAI_API_VERSION
)

db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever()

st.title("CCHMC ironman chatbot")

session_id = 'default_session'

if 'store' not in st.session_state:
    st.session_state.store = {}

contextualize_q_system_prompt = """Given a chat history and the latest user question
which might reference context in the chat history, formulate a standalone question
which can be understood without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

system_prompt = (
    "You are an assistant for question answering tasks."
    "You will be provided with a question and a context."
    "You will answer the question based on the context."
    "If the context is not relevant, just say that you don't know and you are not made to answer such qustions."
    "If you don't know the answer, just say that you don't know, don't try to make up an answer."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

user_input = st.text_input("your question:")

if user_input:
    session_history = get_session_history(session_id)
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": 
                {
                    "session_id": session_id
                }
            })
    st.write(st.session_state.store)
    st.success(f"Assistant: {response['answer']}")
    st.write("Chat history:", session_history.messages)
