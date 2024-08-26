import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import requests
import time

# Load environment variables
load_dotenv()

# Set API keys from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]
os.environ["PERPLEXITY_API_KEY"] = st.secrets["perplexity_api_key"]

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = "us-east-1-aws"
index_name = "conference"
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

# Initialize embeddings
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

# Set up vector store
vectorstore = PineconeVectorStore(index=index, embedding=embedding, text_key="text")

def search_with_pinecone(query, top_k=10):
    query_vector = embedding.embed_text(query)
    results = vectorstore.similarity_search(query_vector, k=top_k)
    return results

def query_perplexity(query, context):
    url = 'https://api.perplexity.ai/chat/completions'
    headers = {
        'Authorization': f'Bearer {os.environ["PERPLEXITY_API_KEY"]}',
        'Content-Type': 'application/json'
    }
    prompt = f"Query: {query}\n\nContext:\n" + "\n".join([doc.page_content for doc in context])
    data = {
        "model": "mistral-7b-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

# Streamlit UI setup
st.title("🤖 AI Chatbot")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    st.text(f"{message['role']}: {message['content']}")

# React to user input
prompt = st.text_input("What would you like to know?")
if st.button("Send"):
    if prompt:
        # Display user message
        st.text(f"user: {prompt}")
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            # Query Pinecone
            context = search_with_pinecone(prompt)
            # Query Perplexity
            response = query_perplexity(prompt, context)
        
        # Display assistant response
        st.text(f"assistant: {response}")
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar for additional information or controls
with st.sidebar:
    st.subheader("About this Chatbot")
    st.write("This AI chatbot uses Pinecone for vector search and Perplexity for natural language processing.")
    st.write("It's connected to a knowledge base about various topics.")
    st.write("Feel free to ask any question!")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()
