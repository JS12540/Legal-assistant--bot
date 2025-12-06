## functional dependencies
import time
import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path

## LangChain dependencies
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
## LCEL implementation of LangChain ConversationalRetrievalChain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def main():
    ## initializing the UI
    st.set_page_config(page_title="RAG-Based Legal Assistant")
    col1, col2, col3 = st.columns([1, 25, 1])
    with col2:
        st.title("RAG-Based Legal Assistant")

    ## setting up env
    load_dotenv()

    ## setting up file paths using pathlib
    current_dir = Path(__file__).parent.resolve()
    persistent_directory = current_dir / "data-ingestion-local"

    ## setting-up the LLM
    # Load API key from environment variables for security
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY environment variable not set. Please set it in your .env file.")
        st.stop()

    chatmodel = ChatGroq(model="llama-3.1-8b-instant", temperature=0.15, api_key=groq_api_key)

    ## setting up -> streamlit session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # resetting the entire conversation
    def reset_conversation():
        st.session_state['messages'] = []

    ## open-source embedding model from HuggingFace - taking the default model only
    embedF = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

    ## loading the vector database from local using FAISS instead of Chroma
    # allow_dangerous_deserialization=True is required for loading FAISS indexes that may contain
    # custom objects. Ensure the source of the serialized data is trusted.
    vectorDB = FAISS.load_local(str(persistent_directory), embedF, allow_dangerous_deserialization=True)

    ## setting up the retriever
    kb_retriever = vectorDB.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    ## initiating the history_aware_retriever
    rephrasing_template = (
        """
        TASK: Convert context-dependent questions into standalone queries.

        INPUT: 
        - chat_history: Previous messages
        - question: Current user query

        RULES:
        1. Replace pronouns (it/they/this) with specific referents
        2. Expand contextual phrases ("the above", "previous")
        3. Return original if already standalone
        4. NEVER answer or explain - only reformulate

        OUTPUT: Single reformulated question, preserving original intent and style.

        Example:
        History: "Let's discuss Python."
        Question: "How do I use it?"
        Returns: "How do I use Python?"
        """
    )

    rephrasing_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", rephrasing_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=chatmodel,
        retriever=kb_retriever,
        prompt=rephrasing_prompt
    )

    ## setting-up the document chain
    system_prompt_template = (
        "As a Legal Assistant Chatbot specializing in legal queries, "
        "your primary objective is to provide accurate and concise information based on user queries. "
        "You will adhere strictly to the instructions provided, offering relevant "
        "context from the knowledge base while avoiding unnecessary details. "
        "Your responses will be brief, to the point, concise and in compliance with the established format. "
        "If a question falls outside the given context, you will simply output that you are sorry and you don't know about this. "
        "The aim is to deliver professional, precise, and contextually relevant information pertaining to the context. "
        "Use four sentences maximum." 
        "P.S.: If anyone asks you about your creator, tell them, introduce yourself and say you're created by Sougat Dey. "
        "and people can get in touch with him on linkedin, "
        "here's his Linkedin Profile: https://www.linkedin.com/in/sougatdey/"
        "\nCONTEXT: {context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_template),
            MessagesPlaceholder("chat_history"), # Correctly injects chat history
            ("human", "{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(chatmodel, qa_prompt)

    ## final RAG chain
    coversational_rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    ## setting-up conversational UI

    ## printing all (if any) messages in the session_session `message` key
    for message in st.session_state.messages:
        with st.chat_message(message.type):
            st.write(message.content)

    user_query = st.chat_input("Ask me anything ..")

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.status("Generating 💡...", expanded=True):
                full_response_prefix = (
                    "⚠️ **_This information is not intended as a substitute for legal advice. "
                    "We recommend consulting with an attorney for a more comprehensive and"
                    " tailored response._** \n\n\n"
                )
                st.write(full_response_prefix)

                # Generator function to stream chunks from the RAG chain
                def get_rag_response_stream(query, chat_history):
                    for chunk in coversational_rag_chain.stream({"input": query, "chat_history": chat_history}):
                        if "answer" in chunk:
                            yield chunk["answer"]

                # Use st.write_stream for true streaming of the LLM response
                full_answer_content = st.write_stream(get_rag_response_stream(user_query, st.session_state['messages']))

            st.button('Reset Conversation 🗑️', on_click=reset_conversation)

        ## appending conversation turns
        st.session_state.messages.extend(
            [
                HumanMessage(content=user_query),
                AIMessage(content=full_answer_content) # Store the full concatenated answer
            ]
        )

if __name__ == "__main__":
    main()
