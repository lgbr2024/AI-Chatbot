import streamlit as st
import os
from dotenv import load_dotenv
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

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]
os.environ["PERPLEXITY_API_KEY"] = st.secrets["perplexity_api_key"]

class ModifiedPineconeVectorStore(PineconeVectorStore):
    # ... (이전 코드와 동일)

def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: List[np.ndarray],
    k: int = 4,
    lambda_mult: float = 0.5
) -> List[int]:
    # ... (이전 코드와 동일)

def fetch_perplexity_results(query: str) -> List[Dict[str, Any]]:
    api_url = "https://api.perplexity.ai/search"
    headers = {"Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"}
    response = requests.get(api_url, headers=headers, params={"query": query})
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        return []

def format_search_results(results: List[Dict[str, Any]]) -> str:
    formatted_results = []
    for result in results:
        source = result.get('source', 'Unknown source')
        content = result.get('content', 'No content available')
        formatted_results.append(f"Source: {source}\nContent: {content}")
    return "\n\n".join(formatted_results)

def main():
    st.title("Conference Q&A System with Pinecone and Perplexity")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "conference"
    index = pc.Index(index_name)
    
    # Select GPT model
    if "gpt_model" not in st.session_state:
        st.session_state.gpt_model = "gpt-4o"
    
    st.session_state.gpt_model = st.selectbox("Select GPT model:", ("gpt-4o", "gpt-4o-mini"), index=("gpt-4o", "gpt-4o-mini").index(st.session_state.gpt_model))
    llm = ChatOpenAI(model=st.session_state.gpt_model)
    
    # Set up Pinecone vector store
    vectorstore = ModifiedPineconeVectorStore(
        index=index,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        text_key="source"
    )
    
    # Set up retriever
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.7}
    )
    
    # Set up prompt template and chain for Pinecone results
    pinecone_template = """
    <prompt>
    Question: {question} 
    Context from Pinecone: {context} 
    Answer:

    <context>
    <role>Strategic consultant for LG Group, tasked with uncovering new trends and insights based on various conference trends.</role>
    <audience>
      -LG Group individual business executives
      -LG Group representative
    </audience>
    <knowledge_base>Conference file saved in vector database</knowledge_base>
    <goal>Find and provide organized content related to the conference that matches the questioner's inquiry, along with sources, to help derive project insights.</goal>
    </context>

    <task>
    <description>
     Describe about 10,000+ words for covering industrial changes, issues, and response strategies related to the conference. Explicitly reflect and incorporate the [research principles] throughout your analysis and recommendations. 
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
            - Source : {{Show 2~3 data sources for each key topic}}
          
      [Conclusion]
        - Summarize new trends based on the conference content
        - Present derived insights
        - Suggest 3 follow-up questions that the LG Group representative might ask, and provide brief answers to each (3~4 sentences)
    </format>

    <style>Business writing with clear and concise sentences targeted at executives</style>

    <constraints>
        - USE THE PROVIDED CONTEXT TO ANSWER THE QUESTION
        - IF YOU DON'T KNOW THE ANSWER, ADMIT IT HONESTLY
        - ANSWER IN KOREAN AND PROVIDE RICH SENTENCES TO ENHANCE THE QUALITY OF THE ANSWER
        - ADHERE TO THE LENGTH CONSTRAINTS FOR EACH SECTION. [CONFERENCE OVERVIEW] ABOUT 3000 WORDS / [CONTENTS] ABOUT 5000 WORDS / [CONCLUSION] ABOUT 2000 WORDS
    </constraints>
    </task>
    </prompt>
    """
    pinecone_prompt = ChatPromptTemplate.from_template(pinecone_template)

    def format_docs(docs: List[Document]) -> str:
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown source')
            formatted.append(f"Source: {source}")
        return "\n\n" + "\n\n".join(formatted)

    format = itemgetter("docs") | RunnableLambda(format_docs)
    pinecone_answer = pinecone_prompt | llm | StrOutputParser()
    pinecone_chain = (
        RunnableParallel(question=RunnablePassthrough(), docs=retriever)
        .assign(context=format)
        .assign(answer=pinecone_answer)
        .pick(["answer", "docs"])
    )

    # Set up prompt template for Perplexity results
    perplexity_template = """
    <prompt>
    Question: {question} 
    Additional Context from Perplexity: {perplexity_context}
    Answer:

    <context>
    <role>Strategic consultant for LG Group, providing additional insights based on Perplexity search results.</role>
    <audience>
      -LG Group individual business executives
      -LG Group representative
    </audience>
    <knowledge_base>Additional information from Perplexity search</knowledge_base>
    <goal>Provide supplementary insights and information related to the conference question.</goal>
    </context>

    <task>
    <description>
     Describe about 5,000+ words for providing additional context and insights based on the Perplexity search results. 
    </description>

    <format>
     [Additional Insights]
        - Summarize key points from the Perplexity search results
        - Highlight any new or different perspectives compared to the Pinecone results
        - Provide additional examples or case studies if available
                   
     [Complementary Analysis]
        - Analyze how the Perplexity results complement or contrast with the Pinecone results
        - Suggest any new trends or insights that emerge from this additional information
          
      [Final Thoughts]
        - Summarize how the Perplexity results enhance the overall understanding of the topic
        - Suggest any additional follow-up questions or areas for further research
    </format>

    <style>Business writing with clear and concise sentences targeted at executives</style>

    <constraints>
        - USE THE PROVIDED PERPLEXITY CONTEXT TO ANSWER THE QUESTION
        - IF YOU DON'T KNOW THE ANSWER, ADMIT IT HONESTLY
        - ANSWER IN KOREAN AND PROVIDE RICH SENTENCES TO ENHANCE THE QUALITY OF THE ANSWER
        - ADHERE TO THE LENGTH CONSTRAINTS FOR EACH SECTION. [ADDITIONAL INSIGHTS] ABOUT 2000 WORDS / [COMPLEMENTARY ANALYSIS] ABOUT 2000 WORDS / [FINAL THOUGHTS] ABOUT 1000 WORDS
    </constraints>
    </task>
    </prompt>
    """
    perplexity_prompt = ChatPromptTemplate.from_template(perplexity_template)
    perplexity_answer = perplexity_prompt | llm | StrOutputParser()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if question := st.chat_input("Please ask a question about the conference:"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            # Create placeholders for status updates
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            try:
                # Step 1: Query Processing
                status_placeholder.text("Processing query...")
                progress_bar.progress(20)
                time.sleep(1)  # Simulate processing time
                
                # Step 2: Searching Pinecone Database
                status_placeholder.text("Searching Pinecone database...")
                progress_bar.progress(40)
                pinecone_response = pinecone_chain.invoke(question)
                time.sleep(1)  # Simulate search time
                
                # Step 3: Displaying Pinecone Results
                status_placeholder.text("Displaying Pinecone results...")
                progress_bar.progress(60)
                st.subheader("Pinecone Search Results")
                st.markdown(pinecone_response['answer'])
                with st.expander("Pinecone Sources"):
                    for doc in pinecone_response['docs']:
                        st.write(f"- {doc.metadata['source']}")
                
                # Step 4: Fetching Perplexity Results
                status_placeholder.text("Fetching Perplexity results...")
                progress_bar.progress(80)
                perplexity_results = fetch_perplexity_results(question)
                perplexity_context = format_search_results(perplexity_results)
                time.sleep(1)  # Simulate API call time
                
                # Step 5: Generating Perplexity Answer
                status_placeholder.text("Generating additional insights from Perplexity...")
                progress_bar.progress(90)
                perplexity_response = perplexity_answer.invoke({"question": question, "perplexity_context": perplexity_context})
                time.sleep(1)  # Simulate generation time
                
                # Step 6: Displaying Perplexity Results
                status_placeholder.text("Displaying Perplexity results...")
                progress_bar.progress(100)
                st.subheader("Additional Insights from Perplexity")
                st.markdown(perplexity_response)
                with st.expander("Perplexity Sources"):
                    for result in perplexity_results:
                        st.write(f"- {result.get('source', 'Unknown source')}")
                
            finally:
                # Clear status displays
                status_placeholder.empty()
                progress_bar.empty()
            
            # Add assistant's response to chat history
            combined_response = f"Pinecone Results:\n\n{pinecone_response['answer']}\n\nAdditional Insights from Perplexity:\n\n{perplexity_response}"
            st.session_state.messages.append({"role": "assistant", "content": combined_response})

if __name__ == "__main__":
    main()
