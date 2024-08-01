from pinecone import Pinecone
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import OpenAI
from langchain_cohere import ChatCohere
from data import generate_embeddings
import numpy as np
import os

pinecone_api_key = os.environ["PINECONE_API_KEY"]
openai_api_key = os.environ["OPENAI_API_KEY"]
cohere_api_key = os.environ["COHERE_API_KEY"]

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("demo-index")

def embed_question(query) :
    embedding_model = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    embeddings = embedding_model.embed_query(query)
    return embeddings

def rag_chain(query) :
    vector = embed_question(query)
    data = index.query(namespace="sample",vector=vector, top_k=2, include_metadata=True)
    text = ""
    for match in data["matches"]:
        text += match["metadata"]["text"]
    prompt = f"""You are an honest and helpful medical research assistant.
                 You will be provided with medical information from research papers.
                 Please do your best to answer the user's research questions using the information provided

                 information : {text}

                 question : {query}
    
             """
    llm = ChatCohere(cohere_api_key=cohere_api_key, model="command-r")
    response = llm.invoke(prompt)
    print(text)
    print(response.content)
    return text, response.content


def calculate_similarity(text, response) :
    vec1 = generate_embeddings([text])
    vec2 = generate_embeddings([response])
    similarity = np.dot(vec1[0], vec2[0])
    print(similarity)
    return similarity

def determine_correctness(text, response) :
    prompt = f"""
    You will be given two text chunks of information. 
    One will be information from a medical research paper and the other is an AI response based on the information.

    Please verify if the AI response is correct and consistent with respect to information provided.
    You should output either correct or incorrect.

    medical information : {text}

    AI response : {response}
    
    """
    llm = OpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-instruct")
    response = llm.invoke(prompt)
    print(response)
    return response


def determine_relevance(text, query) :
    prompt = f"""
    You will be given a medical research question asked by a user.
    You will also be given a piece of information from a medical research paper
    You need to determine how relevant that information is to the answer the user's question precisely.
    Please output strong, neutral or weak depending on how relevant the information is to the user's question.

    question : {query}

    information : {text}
    """
    llm = OpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-instruct")
    response = llm.invoke(prompt)
    print(response)
    return response


    