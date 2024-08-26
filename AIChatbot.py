import streamlit as st
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
import requests

# 환경 변수 설정
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]

# Perplexity API 호출 함수
def fetch_perplexity_results(query: str) -> List[Dict[str, Any]]:
    api_url = "https://api.perplexity.ai/search"
    headers = {"Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"}
    response = requests.get(api_url, headers=headers, params={"query": query})
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        return []

# 검색 결과 포맷팅 함수
def format_search_results(results: List[Dict[str, Any]]) -> str:
    formatted_results = []
    for result in results:
        source = result.get('source', 'Unknown source')
        content = result.get('content', 'No content available')
        formatted_results.append(f"Source: {source}\nContent: {content}")
    return "\n\n".join(formatted_results)

def main():
    st.title("Conference Q&A System with Perplexity Search")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 사용자 질문 입력
    question = st.text_input("Please ask a question about the conference:")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Searching Perplexity..."):
            # Perplexity 검색 결과 가져오기
            search_results = fetch_perplexity_results(question)
            formatted_results = format_search_results(search_results)

        with st.expander("Perplexity Search Results"):
            st.markdown(formatted_results)

        # ChatGPT를 통한 답변 생성
        # (기존의 ChatGPT 답변 생성 코드 유지)

if __name__ == "__main__":
    main()
