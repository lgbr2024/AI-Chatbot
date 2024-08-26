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
        "model": "pplx-7b-online",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides information based on web searches."
            },
            {
                "role": "user",
                "content": f"Provide a summary of web search results for: {query}"
            }
        ],
        "max_tokens": 1024
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        results = content.split('\n\n')[:max_results]
        return [{"content": result} for result in results]
    except requests.exceptions.RequestException as e:
        return [{"content": f"Perplexity 결과 가져오기 오류: {str(e)}"}]
    except Exception as e:
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
    Context: {context} 
    Perplexity Results: {perplexity_results}
    Answer:

    <context>
    <role>Strategic consultant for LG Group, tasked with uncovering new trends and insights based on various conference trends and additional web search results.</role>
    <audience>
      -LG Group individual business executives
      -LG Group representative
    </audience>
    <knowledge_base>Conference file saved in vector database and additional web search results from Perplexity</knowledge_base>
    <goal>Find and provide organized content related to the conference that matches the questioner's inquiry, along with sources, to help derive project insights. Incorporate relevant information from Perplexity web search results when appropriate.</goal>
    </context>

    <task>
    <description>
     Describe about 15,000+ words for covering industrial changes, issues, and response strategies related to the conference. Explicitly reflect and incorporate the [research principles] throughout your analysis and recommendations. Integrate relevant information from the Perplexity web search results to provide a more comprehensive answer.
    </description>

    <format>
     [Conference Overview]
        - Explain the overall context of the conference related to the question
        - Introduce the main points or topics
                   
     [Contents]
        - Analyze the key content discussed at the conference and reference.
        - For each key session or topic:
          - Gather the details as thoroughly as possible, then categorize them according to the following format: 
            - Topic : 
            - Fact : {{1. Provide a detailed description of approximately 5 sentences. 2. Include specific examples, data points, or case studies mentioned in the session. }}
            - Your opinion : {{Provide a detailed description of approximately 3 sentences.}}
            - Source : {{Show 2~3 data sources for each key topic, including Perplexity results when relevant}}
          
      [Conclusion]
        - Summarize new trends based on the conference content and additional web search results
        - Present derived insights, incorporating both conference information and web search data
        - Suggest 3 follow-up questions that the LG Group representative might ask, and provide brief answers to each (3~4 sentences)
    </format>

    <style>Business writing with clear and concise sentences targeted at executives</style>

    <constraints>
        - USE THE PROVIDED CONTEXT AND PERPLEXITY RESULTS TO ANSWER THE QUESTION
        - IF YOU DON'T KNOW THE ANSWER, ADMIT IT HONESTLY
        - ANSWER IN KOREAN AND PROVIDE RICH SENTENCES TO ENHANCE THE QUALITY OF THE ANSWER
        - ADHERE TO THE LENGTH CONSTRAINTS FOR EACH SECTION. [CONFERENCE OVERVIEW] ABOUT 4000 WORDS / [CONTENTS] ABOUT 7000 WORDS / [CONCLUSION] ABOUT 4000 WORDS
    </constraints>
    </task>
    </prompt>
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs: List[Document]) -> str:
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown source')
            formatted.append(f"Source: {source}")
        return "\n\n" + "\n\n".join(formatted)

    def format_perplexity_results(results: List[Dict[str, str]]) -> str:
        return "\n\n".join([f"Perplexity Result: {result['content']}" for result in results])

    format = itemgetter("docs") | RunnableLambda(format_docs)
    format_perplexity = itemgetter("perplexity_results") | RunnableLambda(format_perplexity_results)
    answer = prompt | llm | StrOutputParser()
    chain = (
        RunnableParallel(question=RunnablePassthrough(), docs=retriever)
        .assign(context=format)
        .assign(perplexity_results=lambda x: get_perplexity_results(x["question"]))
        .assign(perplexity_formatted=format_perplexity)
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
                time.sleep(1)
                
                status_placeholder.text("Generating answer...")
                progress_bar.progress(80)
                response = chain.invoke(question)
                time.sleep(1)
                
                status_placeholder.text("Finalizing response...")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Perplexity 결과 확인 및 오류 표시
                for result in response['perplexity_results']:
                    if "오류" in result['content']:
                        st.error(result['content'])
                
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
            finally:
                status_placeholder.empty()
                progress_bar.empty()
            
            st.markdown(response['answer'])
            
            with st.expander("Sources"):
                st.write("Conference Sources:")
                for doc in response['docs']:
                    st.write(f"- {doc.metadata['source']}")
                
                st.write("\nPerplexity Search Results:")
                for result in response['perplexity_results']:
                    st.write(f"- {result['content']}")
            
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})

if __name__ == "__main__":
    main()
