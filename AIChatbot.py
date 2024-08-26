import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
import perplexipy

# Streamlit page config
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# Access secrets
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
perplexity_api_key = st.secrets["PERPLEXITY_API_KEY"]

# Initialize Pinecone
pinecone = pinecone.Pinecone(api_key=pinecone_api_key)
index = pinecone.Index("conference")

# Initialize SentenceTransformer for encoding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Perplexity
perplexity = perplexipy.Client(api_key=perplexity_api_key)

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

def search_knowledge_base(query, top_k=5):
    query_vector = model.encode(query).tolist()
    adjusted_vector = adjust_vector_dimension(query_vector)
    results = index.query(vector=adjusted_vector, top_k=top_k, include_metadata=True)
    return [match['metadata'].get('text', 'No text available') for match in results['matches']]

def generate_response(query, context):
    prompt = f"Query: {query}\n\nContext:\n" + "\n".join(context)
    response = perplexity.chat(messages=[{"role": "user", "content": prompt}])
    return response.content

def chatbot(query):
    context = search_knowledge_base(query)
    response = generate_response(query, context)
    return response

# Streamlit UI (나머지 UI 코드는 그대로 유지)
# ...
