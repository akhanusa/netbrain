from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import random
import re

import os
import streamlit as st

# from dotenv import load_dotenv
# load_dotenv()  # take environment variables from .env (especially openai api key)

# Access the secret
gemini_api_key = st.secrets["GEMINI_API_KEY"]


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key = gemini_api_key)


model_name = "hkunlp/instructor-large"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

instructor_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectordb_file_path = "faiss_index"


def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='codebasics_faq.csv', source_column="prompt")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization = True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

def generate_random_question_from_vectordb():
    # Load the vector database
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)
    
    # Extract raw documents
    documents = vectordb.docstore._dict.values()
    
    # Filter out only those with content
    documents_with_content = [doc for doc in documents if doc.page_content]
    
    # Randomly select a document
    random_doc = random.choice(documents_with_content)
    
    # Extract the question between 'prompt:' and 'response:'
    match = re.search(r'prompt:\s*(.*?)\s*response:', random_doc.page_content)

    if match:
        question = match.group(1)
    else:
        question = "No question found."

    # Use the document's text to generate a synthetic question
    # return f"{random_doc.page_content.strip()[:100]}?"
    return question 


if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    
    # Generate a random question
    random_question = generate_random_question_from_vectordb()
    print("Random Question:", random_question)

    # Get answer from the chain
    result = chain(random_question)
    print("Answer:", result['result'])
