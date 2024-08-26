import streamlit as st
import os
from operator import itemgetter
from typing import List, Tuple, Dict, Any
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import requests

try:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
    os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]
    os.environ["PERPLEXITY_API_KEY"] = st.secrets["perplexity_api_key"]
except KeyError as e:
    st.error(f"필요한 API 키가 설정되지 않았습니다: {e}")
    st.stop()

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
                    'source': results['matches'][i]['metadata'].get('source', '').replace('C:\\Users\\minje\\data2\\', '') if 'source' in results['matches'][i]['metadata'] else 'Unknown'
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

def get_perplexity_results(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides concise summaries of web search results."
            },
            {
                "role": "user",
                "content": f"Provide a brief summary of web search results for: {query}"
            }
        ]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {st.secrets['perplexity_api_key']}"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
        content = response.json()['choices'][0]['message']['content']
        # Split the content into separate results if needed
        results = content.split('\n\n')[:max_results]
        return [{"content": result.strip()} for result in results if result.strip()]
    except requests.exceptions.RequestException as e:
        st.error(f"Perplexity API 요청 오류: {str(e)}")
        return [{"content": f"Perplexity 결과 가져오기 오류: {str(e)}"}]
    except Exception as e:
        st.error(f"예상치 못한 오류 발생: {str(e)}")
        return [{"content": f"예상치 못한 오류 발생: {str(e)}"}]

def main():
    st.title("Robot Conference Q&A System")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "conference"
    index = pc.Index(index_name)
    
    if "gpt_model" not in st.session_state:
        st.session_state.gpt_model = "gpt-4o"
    
    st.session_state.gpt_model = st.selectbox("Select GPT model:", ("gpt-4o", "gpt-4o-mini"), index=("gpt-4o", "gpt-4o-mini").index(st.session_state.gpt_model))
    llm = ChatOpenAI(model=st.session_state.gpt_model)
    
    vectorstore = ModifiedPineconeVectorStore(
        index=index,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        text_key="source"
    )
    
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.7}
    )
    
    template = """
    <prompt>
    Question: {question} 
    Conference Context: {conference_context} 
    Web Search Results: {web_search_results}
    Answer:

    <context>
    <role>Strategic consultant for LG Group, tasked with uncovering new trends and insights based on various conference trends and additional web search results.</role>
    <audience>
      -LG Group individual business executives
      -LG Group representative
    </audience>
    <knowledge_base>Conference file saved in vector database and additional web search results from Perplexity</knowledge_base>
    <goal>Find and provide organized content related to the conference that matches the questioner's inquiry, along with sources, to help derive project insights. Incorporate relevant information from web search results when appropriate.</goal>
    </context>

    <task>
    <description>
     Provide two separate answers: one based on the conference context, and another based on web search results. Each answer should cover industrial changes, issues, and response strategies related to the question. Explicitly reflect and incorporate the [research principles] throughout your analysis and recommendations.
    </description>

    <format>
     [Conference Answer]
      [Conference Overview]
        - Explain the overall context of the conference related to the question
        - Introduce the main points or topics
                   
      [Conference Contents]
        - Analyze the key content discussed at the conference and reference.
        - For each key session or topic:
          - Gather the details as thoroughly as possible, then categorize them according to the following format: 
            - Topic : 
            - Fact : {{1. Provide a detailed description of approximately 5 sentences. 2. Include specific examples, data points, or case studies mentioned in the session. }}
            - Your opinion : {{Provide a detailed description of approximately 3 sentences.}}
            - Source : {{Show 2~3 data sources for each key topic}}
          
      [Conference Conclusion]
        - Summarize new trends based on the conference content
        - Present derived insights
        - Suggest 2 follow-up questions that the LG Group representative might ask, and provide brief answers to each (2~3 sentences)

     [Web Search Answer]
      [Web Search Overview]
        - Provide an overview of the web search results related to the question
        - Highlight the main points or topics found in the web search

      [Web Search Contents]
        - Analyze the key information found in the web search results
        - For each main point or topic:
          - Topic:
          - Key Findings: {{Provide a detailed description of approximately 3-4 sentences}}
          - Relevance to LG: {{Explain how this information is relevant to LG Group in 2-3 sentences}}
          - Source: {{Indicate the Perplexity search result this information came from}}

      [Web Search Conclusion]
        - Summarize the main insights from the web search results
        - Compare and contrast with the conference information if applicable
        - Suggest 1 follow-up question based on the web search results, and provide a brief answer (2-3 sentences)
    </format>

    <style>Business writing with clear and concise sentences targeted at executives</style>

    <constraints>
        - USE THE PROVIDED CONFERENCE CONTEXT AND WEB SEARCH RESULTS TO ANSWER THE QUESTION
        - IF YOU DON'T KNOW THE ANSWER, ADMIT IT HONESTLY
        - ANSWER IN KOREAN AND PROVIDE RICH SENTENCES TO ENHANCE THE QUALITY OF THE ANSWER
        - ADHERE TO THE LENGTH CONSTRAINTS FOR EACH SECTION
        - SEPARATE YOUR ANSWER INTO TWO MAIN PARTS: [Conference Answer] AND [Web Search Answer], EACH WITH THEIR OWN SUBSECTIONS AS SPECIFIED IN THE FORMAT
        - USE THE EXACT SECTION HEADERS PROVIDED
    </constraints>
    </task>
    </prompt>
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs: List[Document]) -> str:
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown source')
            formatted.append(f"Source: {source}\nContent: {doc.page_content}")
        return "\n\n".join(formatted)

    def format_perplexity_results(results: List[Dict[str, str]]) -> str:
        return "\n\n".join([f"Perplexity Result: {result['content']}" for result in results])

    format_conference = itemgetter("docs") | RunnableLambda(format_docs)
    format_web_search = itemgetter("perplexity_results") | RunnableLambda(format_perplexity_results)
    answer = prompt | llm | StrOutputParser()
    chain = (
        RunnableParallel(question=RunnablePassthrough(), docs=retriever)
        .assign(conference_context=format_conference)
        .assign(web_search_results=format_web_search)
        .assign(answer=answer)
        .pick(["answer", "docs", "perplexity_results"])
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Please ask a question about the conference:"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
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
                response = chain.invoke({"question": question, "perplexity_results": perplexity_results})
                time.sleep(1)
                
                status_placeholder.text("Finalizing response...")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                if isinstance(response, dict) and 'answer' in response:
                    answer_content = response['answer']
                    if isinstance(answer_content, str):
                        # 답변을 Conference Answer와 Web Search Answer로 분리
                        main_sections = answer_content.split('[Conference Answer]')[1].split('[Web Search Answer]')
                        conference_answer = main_sections[0].strip()
                        web_search_answer = main_sections[1].strip() if len(main_sections) > 1 else "Web search answer not provided."

                        # Conference Answer 표시
                        with st.expander("[Conference Answer]"):
                            sections = conference_answer.split('[')
                            for section in sections[1:]:
                                section_title = '[' + section.split(']')[0] + ']'
                                section_content = ']'.join(section.split(']')[1:]).strip()
                                st.subheader(section_title)
                                st.markdown(section_content)

                        # Web Search Answer 표시
                        with st.expander("[Web Search Answer]"):
                            sections = web_search_answer.split('[')
                            for section in sections[1:]:
                                section_title = '[' + section.split(']')[0] + ']'
                                section_content = ']'.join(section.split(']')[1:]).strip()
                                st.subheader(section_title)
                                st.markdown(section_content)
                                
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
            finally:
                status_placeholder.empty()
                progress_bar.empty()

if __name__ == "__main__":
    main()
