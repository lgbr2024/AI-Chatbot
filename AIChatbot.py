import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time

# Streamlit page config
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# Access secrets
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
perplexity_api_key = st.secrets["PERPLEXITY_API_KEY"]
pinecone_environment = st.secrets["PINECONE_ENVIRONMENT"]
pinecone_index_name = st.secrets["PINECONE_INDEX_NAME"]
pinecone_host = st.secrets["PINECONE_HOST"]

# Configure retry strategy
retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
    backoff_factor=1
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

# Initialize Pinecone
for attempt in range(3):
    try:
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
        index = pinecone.Index(pinecone_index_name, host=pinecone_host)
        break
    except Exception as e:
        if attempt == 2:
            st.error(f"Failed to initialize Pinecone after 3 attempts: {str(e)}")
            st.stop()
        time.sleep(2 ** attempt)  # Exponential backoff

# Initialize SentenceTransformer for encoding
model = SentenceTransformer('all-MiniLM-L6-v2')

def adjust_vector_dimension(vector, target_dim=1536):
    """Adjust vector dimension to match Pinecone index"""
    current_dim = len(vector)
    if current_dim < target_dim:
        return vector + [0] * (target_dim - current_dim)
    elif current_dim > target_dim:
        return vector[:target_dim]
    return vector

def search_vectors(query, top_k=10):
    for attempt in range(3):
        try:
            query_vector = model.encode(query).tolist()
            adjusted_vector = adjust_vector_dimension(query_vector)
            results = index.query(vector=adjusted_vector, top_k=top_k, include_metadata=True)
            return [match['metadata'].get('text', 'No text available') for match in results['matches']]
        except Exception as e:
            if attempt == 2:
                st.error(f"Error during vector search after 3 attempts: {str(e)}")
                return []
            time.sleep(2 ** attempt)  # Exponential backoff

def send_query_to_perplexity(query, context):
    for attempt in range(3):
        try:
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
            response = http.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            if attempt == 2:
                st.error(f"Error querying Perplexity API after 3 attempts: {str(e)}")
                return "Sorry, I couldn't generate a response at this time."
            time.sleep(2 ** attempt)  # Exponential backoff

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
            response = chatbot(prompt)
        
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

if __name__ == '__main__':
    # The Streamlit app is already defined in the code above,
    # so we don't need an explicit main() function call here.
    pass
