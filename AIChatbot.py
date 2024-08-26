import streamlit as st
import os
import time
import numpy as np
import requests
from operator import itemgetter
from typing import List, Dict, Any
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from sklearn.metrics.pairwise import cosine_similarity

# Initialize API keys
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
    os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]
    os.environ["PERPLEXITY_API_KEY"] = st.secrets["perplexity_api_key"]
except KeyError as e:
    st.error(f"필요한 API 키가 설정되지 않았습니다: {e}")
    st.stop()

# Define the Modified Pinecone Vector Store class
class ModifiedPineconeVectorStore(PineconeVectorStore):
    def __init__(self, index, embedding, text_key: str = "text", namespace: str = ""):
        super().__init__(index, embedding, text_key, namespace)
        self.index = index
        self._embedding = embedding
        self._text_key = text_key
        self._namespace = namespace

    def similarity_search_with_score_by_vector(self, embedding: List[float], k: int = 8, filter: Dict[str, Any] = None, namespace: str = None):
        namespace = namespace or self._namespace
        results = self.index.query(vector=embedding, top_k=k, include_metadata=True, include_values=True, filter=filter, namespace=namespace)
        return [(Document(page_content=result["metadata"].get(self._text_key, ""), metadata={k: v for k, v in result["metadata"].items() if k != self._text_key}), result["score"]) for result in results["matches"]]

    def max_marginal_relevance_search_by_vector(self, embedding: List[float], k: int = 8, fetch_k: int = 30, lambda_mult: float = 0.7, filter: Dict[str, Any] = None, namespace: str = None):
        namespace = namespace or self._namespace
        results = self.index.query(vector=embedding, top_k=fetch_k, include_metadata=True, include_values=True, filter=filter, namespace=namespace)
        if not results['matches']:
            return []
        embeddings = [match['values'] for match in results['matches']]
        mmr_selected = maximal_marginal_relevance(np.array(embedding, dtype=np.float32), embeddings, k=min(k, len(results['matches'])), lambda_mult=lambda_mult)
        return [Document(page_content=results['matches'][i]['metadata'].get(self._text_key, ""), metadata={'source': results['matches'][i]['metadata'].get('source', '').replace('C:\\Users\\minje\\data2\\', '') if 'source' in results['matches'][i]['metadata'] else 'Unknown'}) for i in mmr_selected]

# Define maximal marginal relevance function
def maximal_marginal_relevance(query_embedding: np.ndarray, embedding_list: List[np.ndarray], k: int = 4, lambda_mult: float = 0.5):
    similarity_scores = cosine_similarity([query_embedding], embedding_list)[0]
    selected_indices = []
    candidate_indices = list(range(len(embedding_list)))
    for _ in range(k):
        if not candidate_indices:
            break
        mmr_scores = [lambda_mult * similarity_scores[i] - (1 - lambda_mult) * max([cosine_similarity([embedding_list[i]], [embedding_list[s]])[0][0] for s in selected_indices] or [0]) for i in candidate_indices]
        max_index = candidate_indices[np.argmax(mmr_scores)]
        selected_indices.append(max_index)
        candidate_indices.remove(max_index)
    return selected_indices

# Define function to get perplexity results
def get_perplexity_results(query: str, max_results: int = 5):
    url = "https://api.perplexity.ai/chat/completions"
    payload = {"model": "llama-3.1-sonar-small-128k-online", "messages": [{"role": "system", "content": "You are a helpful assistant that provides concise summaries of web search results."}, {"role": "user", "content": f"Provide a brief summary of web search results for: {query}"}]}
    headers = {"accept": "application/json", "content-type": "application/json", "authorization": f"Bearer {st.secrets['perplexity_api_key']}"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        results = content.split('\n\n')[:max_results]
        return [{"content": result.strip()} for result in results if result.strip()]
    except requests.exceptions.RequestException as e:
        st.error(f"Perplexity API 요청 오류: {str(e)}")
        return [{"content": f"Perplexity 결과 가져오기 오류: {str(e)}"}]
    except Exception as e:
        st.error(f"예상치 못한 오류 발생: {str(e)}")
        return [{"content": f"예상치 못한 오류 발생: {str(e)}"}]

# Define formatting functions
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown source')}\nContent: {doc.page_content}" for doc in docs])

def format_perplexity_results(results: List[Dict[str, str]]) -> str:
    return "\n\n".join([f"Perplexity Result: {result['content']}" for result in results])

def get_response(question, perplexity_results, vectorstore, llm, embedding_handler):
    # 질문에 대한 임베딩 생성
    question_embedding = embedding_handler.encode([question])[0].tolist()

    # VectorStore에서 MMR 검색 실행
    docs_results = vectorstore.max_marginal_relevance_search_by_vector(
        embedding=question_embedding,
        k=10,  # 최대 10개 문서 반환
        fetch_k=20  # 상위 20개 후보에서 MMR 실행
    )

    # ChatOpenAI 모델을 사용하여 질문에 대한 답변을 생성합니다.
    template = f"""Please provide an answer to the following question based on the conference materials and relevant information: \n\n{question}\n\nRelevant Information:\n{format_docs(docs_results)}\n\nWeb Search Results:\n{format_perplexity_results(perplexity_results)}"""
    llm_response = llm.query(template)

    # 필요한 모든 정보를 조합하여 최종 응답을 생성합니다.
    return {
        "answer": llm_response,
        "docs": docs_results,
        "perplexity_results": perplexity_results
    }


# Main function definition
def main():
    st.title("Robot Conference Q&A System")

    # Check and setup required API keys
    required_keys = ["openai_api_key", "pinecone_api_key", "perplexity_api_key"]
    missing_keys = [key for key in required_keys if key not in st.secrets]
    if missing_keys:
        st.error(f"Missing required API keys: {', '.join(missing_keys)}")
        st.stop()

    # Session state setup
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "gpt_model" not in st.session_state:
        st.session_state.gpt_model = "gpt-4o"

    # User interface for model selection
    model_options = ("gpt-4o", "gpt-4o-mini")
    st.session_state.gpt_model = st.selectbox("Select GPT model:", model_options, index=model_options.index(st.session_state.gpt_model))
    llm = ChatOpenAI(model=st.session_state.gpt_model)

    # Pinecone Vector Store setup
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "conference"
    index = pc.Index(index_name)
    vectorstore = ModifiedPineconeVectorStore(index=index, embedding=OpenAIEmbeddings(model="text-embedding-ada-002"), text_key="source")
    retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.7})

    # User interface for questions
    question = st.chat_input("Please ask a question about the conference:")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        handle_question(question, retriever, llm)

def handle_question(question, retriever, llm):
    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    try:
        status_placeholder.text("Processing query...")
        progress_bar.progress(20)
        time.sleep(1)

        status_placeholder.text("Searching database...")
        progress_bar.progress(40)
        time.sleep(1)

        status_placeholder.text("Fetching additional web search results...")
        progress_bar.progress(60)
        perplexity_results = get_perplexity_results(question)
        time.sleep(1)

        status_placeholder.text("Generating answer...")
        progress_bar.progress(80)
        response = get_response(question, perplexity_results, retriever, llm)
        time.sleep(1)

        status_placeholder.text("Finalizing response...")
        progress_bar.progress(100)
        time.sleep(0.5)

        display_response(response)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        status_placeholder.empty()
        progress_bar.empty()

if __name__ == "__main__":
    main()
