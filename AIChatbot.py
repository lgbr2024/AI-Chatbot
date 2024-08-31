import streamlit as st
import os
from dotenv import load_dotenv
from operator import itemgetter
from typing import List, Tuple, Dict, Any
from pinecone import Pinecone
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings  # OpenAI 임베딩은 그대로 사용
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# API 키 설정 (Streamlit secrets에서 가져옴)
os.environ["ANTHROPIC_API_KEY"] = st.secrets["anthropic_api_key"]
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]

# ModifiedPineconeVectorStore 클래스는 그대로 유지

def main():
    st.title("🤞Conference Q&A System")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mode" not in st.session_state:
        st.session_state.mode = "Report Mode"

    # Pinecone 초기화
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "conference"
    index = pc.Index(index_name)

    # Claude 모델 선택
    if "claude_model" not in st.session_state:
        st.session_state.claude_model = "claude-3-opus-20240229"

    st.session_state.claude_model = st.selectbox(
        "Select Claude model:",
        ("claude-3-opus-20240229", "claude-3-sonnet-20240229"),
        index=("claude-3-opus-20240229", "claude-3-sonnet-20240229").index(st.session_state.claude_model)
    )
    llm = ChatAnthropic(model=st.session_state.claude_model)

    # Pinecone 벡터 스토어 설정
    vectorstore = ModifiedPineconeVectorStore(
        index=index,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        text_key="source"
    )

    # 검색기 설정
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.7}
    )

    # 리포트 모드용 프롬프트 템플릿 설정
    report_template = """
    Human: Please provide a comprehensive report based on the following question and context. Use the style of Harvard Business Review (HBR) and follow these guidelines:

    Question: {question}
    Context: {context}

    Guidelines:
    1. Start with a conference overview, explaining the context and main themes.
    2. Analyze key content from the conference, categorizing it into topics, facts, and your opinions.
    3. Conclude with new trends, insights, and 3 follow-up questions with brief answers.
    4. Use clear and concise business writing targeted at executives.
    5. Answer in Korean and provide rich sentences to enhance the quality of the answer.
    6. Adhere to these length constraints: Conference Overview (약 4000 단어), Contents (약 7000 단어), Conclusion (약 4000 단어).

    Assistant: 네, 주어진 지침에 따라 컨퍼런스에 대한 종합적인 보고서를 작성하겠습니다.

    Human: 이제 위의 지침에 따라 보고서를 작성해 주세요.

    Assistant: [보고서 내용]

    Human: 감사합니다. 이제 챗봇 모드를 위한 간단한 응답을 생성해 주세요. 질문과 컨텍스트는 다음과 같습니다:

    Question: {question}
    Context: {context}

    Assistant: 네, 주어진 질문과 컨텍스트를 바탕으로 간단한 응답을 생성하겠습니다.

    [챗봇 응답 내용]
    """
    report_prompt = ChatPromptTemplate.from_template(report_template)

    # 챗봇 모드용 프롬프트 템플릿 설정
    chatbot_template = """
    Human: 다음 질문에 대해 주어진 컨텍스트를 바탕으로 약 4,000자로 대화체로 답변해 주세요. 한국어로 답변해 주세요.

    Question: {question}
    Context: {context}

    Assistant: 네, 주어진 질문에 대해 컨텍스트를 바탕으로 약 4,000자 분량의 대화체 답변을 한국어로 작성하겠습니다.

    [챗봇 응답 내용]
    """
    chatbot_prompt = ChatPromptTemplate.from_template(chatbot_template)

    # format_docs 함수는 그대로 유지

    def get_report_chain(prompt):
        answer = prompt | llm | StrOutputParser()
        return (
            RunnableParallel(question=RunnablePassthrough(), docs=retriever)
            .assign(context=format)
            .assign(answer=answer)
            .pick(["answer", "docs"])
        )

    def get_chatbot_chain(prompt):
        answer = prompt | llm | StrOutputParser()
        return (
            RunnableParallel(question=RunnablePassthrough(), docs=retriever)
            .assign(context=format)
            .assign(answer=answer)
            .pick("answer")
        )

    report_chain = get_report_chain(report_prompt)
    chatbot_chain = get_chatbot_chain(chatbot_prompt)

    # 모드 선택 및 대화 기록 표시 부분은 그대로 유지

    # 사용자 입력 및 응답 생성 부분
    if question := st.chat_input("컨퍼런스에 대해 질문해 주세요:"):
        # 리셋 키워드 확인 및 대화 초기화 로직은 그대로 유지

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            # 상태 업데이트를 위한 플레이스홀더 생성
            status_placeholder = st.empty()
            progress_bar = st.progress(0)

            try:
                # 쿼리 처리, 데이터베이스 검색, 답변 생성, 응답 마무리 단계는 그대로 유지
                # 단, chain 선택 부분만 수정
                chain = report_chain if st.session_state.mode == "Report Mode" else chatbot_chain
                response = chain.invoke(question)

                # 답변 표시
                answer = response['answer'] if st.session_state.mode == "Report Mode" else response
                st.markdown(answer)

                # 소스 표시 (리포트 모드만)
                if st.session_state.mode == "Report Mode":
                    with st.expander("Sources"):
                        for doc in response['docs']:
                            st.write(f"- {doc.metadata['source']}")

                # 대화 기록에 도우미 응답 추가
                st.session_state.messages.append({"role": "assistant", "content": answer})

            finally:
                # 상태 표시 제거
                status_placeholder.empty()
                progress_bar.empty()

    # 대화 초기화 버튼 추가
    if st.button("대화 초기화"):
        reset_conversation()

if __name__ == "__main__":
    main()
