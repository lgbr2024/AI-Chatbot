import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
import requests

# Streamlit page config
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# Access secrets
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
perplexity_api_key = st.secrets["PERPLEXITY_API_KEY"]

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment="your-environment")  # Replace with your actual environment
index = pinecone.Index("conference")

# Initialize SentenceTransformer for encoding
model = SentenceTransformer('all-MiniLM-L6-v2')

def adjust_vector_dimension(vector, target_dim=1536):
    """Adjust vector dimension to match Pinecone index"""
    current_dim = len(vector)
    if current_dim < target_dim:
        # Pad with zeros
        return vector + [0] * (target_dim - current_dim)
    elif current_dim > target_dim:
        # Truncate
        return vector[:target_dim]
    return vector

def search_vectors(query, top_k=5):
    query_vector = model.encode(query).tolist()
    adjusted_vector = adjust_vector_dimension(query_vector)
    results = index.query(vector=adjusted_vector, top_k=top_k, include_metadata=True)
    return [match['metadata'].get('text', 'No text available') for match in results['matches']]

def send_query_to_perplexity(query, context):
    url = 'https://api.perplexity.ai/chat/completions'
    headers = {
        'Authorization': f'Bearer {perplexity_api_key}',
        'Content-Type': 'application/json'
    }
    prompt = f"Query: {query}\n\nContext:\n" + "\n".join(context)
    data = {
        "model": "mistral-7b-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

def chatbot(query):
    context = search_vectors(query)
    response = send_query_to_perplexity(query, context)
    return response

# Streamlit UI
st.title("🤖 AI Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        response = chatbot(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
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

if __name__ == '__main__':
    # The Streamlit app is already defined in the code above,
    # so we don't need an explicit main() function call here.
    pass
