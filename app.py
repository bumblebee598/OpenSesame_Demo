import streamlit as st
from main import rag_chain, calculate_similarity, determine_correctness, determine_relevance
import pandas as pd
import warnings

question_list = ["What is the most common type of cancer found in elderly males ?",
                 "What is the most common type of cancer found in elderly females ?",
                 "What is the most common cancer death in the US ?",
                 "How does age affect cancer mortality rates ?",
                 "Which types of Cancer affect young people more than older people?"]

def display_eval(sim, correctness, relevance) :
        data = {"similarity" : [sim],
                "Response Correctness" : correctness,
                "information relevance": relevance}
        df = pd.DataFrame(data)
        st.table(df)

def summary_stats() :
    for i in range(len(st.session_state.query)):
        with st.chat_message("user"):
            st.markdown(st.session_state.query[i])
        with st.chat_message("assistant"):
            with st.container(height=200) :
                st.markdown("context :")
                st.markdown(st.session_state.text[i])
            st.markdown(st.session_state.response[i])
            sim = st.session_state.sim[i]
            correctness = st.session_state.correctness[i]
            relevance = st.session_state.relevance[i]
            display_eval(sim, correctness, relevance)

def main_app() :
    st.markdown("LLM Evaluation - OpenSesame Project")
    with st.sidebar :
        st.write("You have been presented with a medical chatbot which answers questions on cancer research from medical literature")
        st.write("For each question you ask, the context chunk, LLM (command r) response and performance evaluation is presented")
        with st.expander("List of Questions you can ask") :
            for question in question_list :
                st.markdown(question)
        button = st.button(key="button", label="Summarize LLM performance")
        st.write("click this to produce a summary of the LLM's performance based on all the questions you have asked")
        if button :
            st.write("button clicked")
        
    if "query" not in st.session_state:
        st.session_state.query = []
    if "response" not in st.session_state:
        st.session_state.response = []
    if "text" not in st.session_state:
        st.session_state.text = []
    if "sim" not in st.session_state:
        st.session_state.sim = []
    if "correctness" not in st.session_state:
        st.session_state.correctness = []
    if "relevance" not in st.session_state:
        st.session_state.relevance = []

    for i in range(len(st.session_state.query)):
        with st.chat_message("user"):
            st.markdown(st.session_state.query[i])
        with st.chat_message("assistant"):
            with st.container(height=200) :
                st.markdown("context :")
                st.markdown(st.session_state.text[i])
            st.markdown(st.session_state.response[i])
            sim = st.session_state.sim[i]
            correctness = st.session_state.correctness[i]
            relevance = st.session_state.relevance[i]
            display_eval(sim, correctness, relevance)
            
    
    if query := st.chat_input("Ask question") :
        with st.chat_message("user") :
            st.markdown(query)
        st.session_state.query.append(query)
        with st.spinner("searching...") :
            text, response = rag_chain(query)
        with st.chat_message("assistant") :
            with st.container(height=200) :
                st.markdown("context :")
                st.markdown(text)
            st.markdown("response :")
            st.markdown(response)
            with st.spinner("evaluating") :
                sim = calculate_similarity(text, response)
                correctness = determine_correctness(text, response)
                relevance = determine_relevance(text, query)
            display_eval(sim, correctness, relevance)
            st.session_state.text.append(text)
            st.session_state.response.append(response)
            st.session_state.sim.append(sim)
            st.session_state.correctness.append(correctness)
            st.session_state.relevance.append(relevance)

main_app()

        
            