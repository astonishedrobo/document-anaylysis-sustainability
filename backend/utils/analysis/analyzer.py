from PyPDF2 import PdfReader
import os
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
import requests
import json

    
def read_pdf(file_path: str, return_list: bool = False):
    pdf = PdfReader(file_path)
    corpus = []
    for page in pdf.pages:
        text = page.extract_text()
        corpus.append(text)

    if return_list:
        return corpus
    else:
        return '\n\n'.join(corpus)


def extract_hyperlinks(pdf_reader):
    links = set()
    for page in pdf_reader.pages:
        if '/Annots' in page:
            for annot in page['/Annots']:
                annot_obj = annot.get_object()

                if '/URI' in annot_obj.get('/A', {}):
                    uri = annot_obj['/A']['/URI']
                    links.add(uri)
    return links

def augment_link_content(file_path: str):
    """
    Aguments the contet with the content from the links.
    """
    pdf = PdfReader(file_path)

    # Find all the links
    links = extract_hyperlinks(pdf)

    # Extract Hyperlink Contents
    augmentation_text = "Extra Details: \n\n"
    for link in links:
        if link.endswith(".pdf"):
            response = requests.get(link)
            if response.status_code != 200:
                continue

            os.makedirs("file_cache", exist_ok=True)
            with open("file_cache/cache.pdf", "wb") as file:
                file.write(response.content)

            try:
                augmentation_text += read_pdf("file_cache/cache.pdf") + '\n'
            except:
                pass
            shutil.rmtree("file_cache")

    return augmentation_text

def split_and_store_db(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.create_documents([text])

    vector_db = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings())
    return vector_db

def join_context(docs):
    return '\n\n'.join([doc.page_content for doc in docs])

def get_llm(model_name: str = 'gpt-3.5-turbo'):
    return ChatOpenAI(model_name=model_name)


def analyze_doc_rag(file_path: str, question: str, augment_link: str = False, previous_context: str = None, model_name: str = 'gpt-3.5-turbo'):
    text = read_pdf(file_path)
    if augment_link:
        print("Augmenting Text With Content From Hyperlinks")
        text += augment_link_content(file_path)

    vector_db = split_and_store_db(text)
    
    # Retrieve the most similar chunks
    retriever = vector_db.as_retriever(search_type='similarity')

    # Define llm
    llm = get_llm(model_name=model_name)

    # Define prompt
    template = '''Use the following pieces of context to answer the question at the end in JSON format.
        If you don't know the answer, just retun None as value, don't try to make up an answer.
        

        Context: {context}

        Question: {question}

        Answer: 
    '''
    prompt = PromptTemplate.from_template(template)

    # Rag Chain
    rag_chain = (
        {"context": retriever | join_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | JsonOutputParser()
    )

    # Run the chain
    if previous_context:
        question = previous_context + '\n\n' + question
    answer = rag_chain.invoke(question)

    return answer


def analyze_doc(text: str, question: str, augment_link: str = False, model_name: str = 'gpt-3.5-turbo'):
    # Define llm
    llm = get_llm(model_name=model_name)

    # Define prompt
    template = '''Use the following pieces of context to answer the question at the end in JSON format.
        If you don't know the answer, just return null, don't try to make up an answer.
        

        Context: {context}

        Question: {question}

        Answer: 
    '''
    prompt = PromptTemplate.from_template(template)

    # llm Chain
    llm_chain = (
        {"context": lambda x: text, "question": RunnablePassthrough()}
        | prompt
        | llm
        | JsonOutputParser()
    )

    # Run the chain
    answer = llm_chain.invoke(question)

    return answer

def get_news_corpus(path: str, return_db: bool = False):
    corpus = []
    with open(path, 'r') as file:
        text = json.load(file)
        for news in text:
            corpus.append(news['article'])

    if return_db:
        return corpus, split_and_store_db('\n\n'.join(corpus))
    else:
        return '\n\n'.join(corpus)

def analyze_news_corpus(question: str, text: str = None, vector_db = None, model_name: str = 'gpt-3.5-turbo'):
    if not vector_db:
        vector_db = split_and_store_db(text)
    
    # Retrieve the most similar chunks
    retriever = vector_db.as_retriever(search_type='similarity')

    # Define llm
    llm = get_llm(model_name=model_name)

    # Define prompt
    template = '''Use the following pieces of context to answer the question at the end in JSON format.
        If you don't know the answer, just retun None as value, don't try to make up an answer.
        

        Context: {context}

        Question: {question}

        Answer: 
    '''
    prompt = PromptTemplate.from_template(template)

    # Rag Chain
    rag_chain = (
        {"context": retriever | join_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | JsonOutputParser()
    )

    # Run the chain
    answer = rag_chain.invoke(question)

    return answer
    