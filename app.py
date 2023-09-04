import re
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
from langchain.llms import VertexAI
from google.cloud import aiplatform

import streamlit as st

from google.cloud import aiplatform





def main():

    load_dotenv()

    path = "temp/HANBIT_Google_Cloud_Platform.pdf"
    pdf_reader = PdfReader(path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)    

    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko-multilingual@latest")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    llm = VertexAI(temperature=0, model_name="text-bison-32k", max_output_tokens=512)
    prompt_template = """Answer the question as precise as possible using the provided context. If the answer is
                    not contained in the context, say "answer not available in context" \n\n
                    Context: \n {context}?\n
                    Question: \n {question} \n
                    Answer:
                """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    st.set_page_config(page_title="Google Cloud Platform",
                    page_icon="👩‍⚕️", 
                    initial_sidebar_state="expanded")   
         

    # 버거 메뉴 + footer 수정
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden; }
                footer {visibility: hidden;}
                footer:after {visibility: visible; content:"2023 VNTG Corp.";}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.header("Google Cloud Platform PDF Q/A 👩‍⚕️")
    st.write("")
    st.write("")
    user_question = st.text_input("검색하고 싶은 내용을 입력하세요:")
    st.write("")

    # aiplatform.init(project="vntg-mlops", location="us-west1")

    



    if(user_question):
        with st.spinner("Searching ..."):
            # result = get_query(vector_retriever, user_question)    
            # st.write(result)

            #user_question="gcp console에서 개인 키 생성하는 방법은?"
            docs = knowledge_base.similarity_search(user_question)
            chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
            response = chain.run(input_documents=docs, question=user_question)
            st.write(response)



if __name__ == "__main__":
    main()    