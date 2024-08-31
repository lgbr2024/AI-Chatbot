import streamlit as st
import os
from typing import List, Tuple, Dict, Any
from pinecone import Pinecone
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# API 키 설정
os.environ["ANTHROPIC_API_KEY"] = st.secrets["anthropic_api_key"]
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]

class ModifiedPineconeVectorStore(PineconeVectorStore):
    def __init__(self, index, embedding, text_key: str = "text", namespace: str = ""):
        super().__init__(index, embedding, text_key, namespace)
        self.index = index
        self._embedding = embedding
        self._text_key = text_key
        self._namespace = namespace

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 8, filter: Dict[str, Any] = None, namespace: str = None
    ) -> List[Tuple[Document, float]]:
        namespace = namespace or self._namespace
        results = self.index.query(
            vector=embedding,
            top_k=k,
            include_metadata=True,
            include_values=True,
            filter=filter,
            namespace=namespace,
        )
        return [
            (
                Document(
                    page_content=result["metadata"].get(self._text_key, ""),
                    metadata={k: v for k, v in result["metadata"].items() if k != self._text_key}
                ),
                result["score"],
            )
            for result in results["matches"]
        ]

    def max_marginal_relevance_search_by_vector(
        self, embedding: List[float], k: int = 8, fetch_k: int = 30,
        lambda_mult: float = 0.7, filter: Dict[str, Any] = None, namespace: str = None
    ) -> List[Document]:
        namespace = namespace or self._namespace
        results = self.index.query(
            vector=embedding,
            top_k=fetch_k,
            include_metadata=True,
            include_values=True,
            filter=filter,
            namespace=namespace,
        )
        if not results['matches']:
            return []

        embeddings = [match['values'] for match in results['matches']]
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embeddings,
            k=min(k, len(results['matches'])),
            lambda_mult=lambda_mult
        )

        return [
            Document(
                page_content=results['matches'][i]['metadata'].get(self._text_key, ""),
                metadata={
                    'source': results['matches'][i]['metadata'].get('source', '').replace('C:\\Users\\minje\\data\\', '') if 'source' in results['matches'][i]['metadata'] else 'Unknown'
                }
            )
            for i in mmr_selected
        ]

def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: List[np.ndarray],
    k: int = 4,
    lambda_mult: float = 0.5
) -> List[int]:
    similarity_scores = cosine_similarity([query_embedding], embedding_list)[0]
    selected_indices = []
    candidate_indices = list(range(len(embedding_list)))
    for _ in range(k):
        if not candidate_indices:
            break

        mmr_scores = [
            lambda_mult * similarity_scores[i] - (1 - lambda_mult) * max(
                [cosine_similarity([embedding_list[i]], [embedding_list[s]])[0][0] for s in selected_indices] or [0]
            )
            for i in candidate_indices
        ]
        max_index = candidate_indices[np.argmax(mmr_scores)]
        selected_indices.append(max_index)
        candidate_indices.remove(max_index)
    return selected_indices

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

    # 프롬프트 템플릿 설정 (리포트 모드와 챗봇 모드)
    report_template = """
    Human: Please write a comprehensive and informative report based on the following question and context. 
    The report should be approximately 4000 words in total, following this structure:

    1. Conference Overview (about 1000 words)
       - Explain the overall context of the conference
       - List the main topics, providing a brief 1-2 sentence explanation for each
       - Mention speakers or company names where possible
       - Mention the scale of the conference (number of attendees, number of presentation sessions, etc.) and its importance
       - Summarize in 1-2 sentences the impact or significance of this conference on the industry

    2. Analysis of Key Content (about 2000 words)
       - Analyze the key content discussed at the conference and reference sources
       - For each key session or topic:
         Topic:
         Fact: {{1. Provide a detailed description of approximately 5 sentences. 2. Include specific examples, numerical data, or case studies mentioned in the session}}
         Your opinion: {{Provide a detailed description of approximately 3 sentences}}
         Source: {{Show 2~3 data sources for each key topic}}

    3. Conclusion and Insights (about 1000 words)
       - Summarize new trends based on the conference content
       - Present derived insights from the conference that relate to the user's question. Focus on forward-looking perspectives and industry developments
       - Suggest 3 follow-up questions that the LG Group representative might ask, and provide brief answers to each (3~4 sentences)

    WRITING GUIDELINES:
    - USE THE PROVIDED CONTEXT TO ANSWER THE QUESTION
    - IF YOU DON'T KNOW THE ANSWER, ADMIT IT HONESTLY
    - WRITE IN KOREAN
    - ADHERE TO THE LENGTH CONSTRAINTS FOR EACH SECTION
    - USE THE STYLE OF HARVARD BUSINESS REVIEW (HBR): CLEAR, CONCISE, AND ANALYTICAL WRITING TARGETED AT BUSINESS EXECUTIVES
    - EMPLOY A PROFESSIONAL TONE WHILE MAINTAINING READABILITY
    - USE RELEVANT BUSINESS TERMINOLOGY AND CONCEPTS WHERE APPROPRIATE
    - INCLUDE DATA-DRIVEN INSIGHTS AND ACTIONABLE RECOMMENDATIONS

    Question: {{question}}
    Context: {{context}}

    Assistant: 네, 주어진 지침에 따라 Harvard Business Review 스타일로 종합적이고 정보가 풍부한 보고서를 한국어로 작성하겠습니다.

    Human: 이제 위의 지침에 따라 약 4000단어 분량의 보고서를 작성해 주세요.

    Assistant: [보고서 내용]
    """
    report_prompt = ChatPromptTemplate.from_template(report_template)

    chatbot_template = """
    Human: 다음 질문에 대해 주어진 컨텍스트를 바탕으로 약 1,000자로 대화체로 답변해 주세요. 한국어로 답변해 주세요.

    Question: {{question}}
    Context: {{context}}

    Assistant: 네, 주어진 질문에 대해 컨텍스트를 바탕으로 약 1,000자 분량의 대화체 답변을 한국어로 작성하겠습니다.

    [챗봇 응답 내용]
    """
    chatbot_prompt = ChatPromptTemplate.from_template(chatbot_template)

    def format_docs(docs: List[Document]) -> str:
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown source')
            formatted.append(f"Source: {source}")
        return "\n\n" + "\n\n".join(formatted)

    def get_report_chain(prompt):
        return (
            RunnableParallel(
                {"question": RunnablePassthrough(), "docs": retriever}
            )
            .assign(context=lambda x: format_docs(x["docs"]))
            .assign(answer=prompt | llm | StrOutputParser())
            .pick(["answer", "docs"])
        )

    def get_chatbot_chain(prompt):
        return (
            RunnableParallel(
                {"question": RunnablePassthrough(), "docs": retriever}
            )
            .assign(context=lambda x: format_docs(x["docs"]))
            .assign(answer=prompt | llm | StrOutputParser())
            .pick("answer")
        )

    report_chain = get_report_chain(report_prompt)
    chatbot_chain = get_chatbot_chain(chatbot_prompt)

    # 모드 선택
    new_mode = st.radio("모드 선택:", ("Report Mode", "Chatbot Mode"), key="mode_selection")
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.session_state.messages = []  # 모드 변경 시 대화 기록 초기화
        st.rerun()

    # 현재 모드 표시
    st.write(f"현재 모드: {st.session_state.mode}")

    # 대화 초기화 함수
    def reset_conversation():
        st.session_state.messages = []
        st.rerun()

    # 리셋 키워드 확인
    reset_keywords = ["처음으로", "초기화", "다시", "안녕"]

    # 대화 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    if question := st.chat_input("컨퍼런스에 대해 질문해 주세요:"):
        # 리셋 키워드 확인
        if any(keyword in question for keyword in reset_keywords):
            reset_conversation()
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                status_placeholder = st.empty()
                progress_bar = st.progress(0)

                try:
                    status_placeholder.text("쿼리 처리 중...")
                    progress_bar.progress(25)
                    time.sleep(1)

                    status_placeholder.text("데이터베이스 검색 중...")
                    progress_bar.progress(50)
                    chain = report_chain if st.session_state.mode == "Report Mode" else chatbot_chain
                    response = chain.invoke({"question": question})
                    time.sleep(1)

                    status_placeholder.text("답변 생성 중...")
                    progress_bar.progress(75)
                    if st.session_state.mode == "Report Mode":
                        answer = response['answer']
                        sources = response['docs']
                    else:
                        answer = response
                        sources = []
                    time.sleep(1)

                    status_placeholder.text("응답 마무리 중...")
                    progress_bar.progress(100)
                    time.sleep(0.5)

                except Exception as e:
                    st.error(f"오류가 발생했습니다: {str(e)}")
                    answer = "죄송합니다. 응답 생성 중 오류가 발생했습니다. 다시 시도해 주세요."
                    sources = []

                finally:
                    status_placeholder.empty()
                    progress_bar.empty()

                st.markdown(answer)

                if st.session_state.mode == "Report Mode" and sources:
                    with st.expander("Sources"):
                        for doc in sources:
                            st.write(f"- {doc.metadata['source']}")

                st.session_state.messages.append({"role": "assistant", "content": answer})

    # 대화 초기화 버튼 추가
    if st.button("대화 초기화"):
        reset_conversation()

if __name__ == "__main__":
    main()
