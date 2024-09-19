# rag_pipeline.py

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
import json
from dotenv import load_dotenv
from langchain.schema import Document


def read_pdf(file_path: str):
    pdf = PdfReader(file_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def text_to_chunks(text: str, chunk_size: int = 1000, overlap: int = 100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # Split the text into chunks
    chunks = splitter.split_text(text)
    
    # Create documents with chunk number as metadata
    documents = []
    for i, chunk in enumerate(chunks):
        metadata = {"chunk_number": i}
        documents.append(Document(page_content=chunk, metadata=metadata))
    
    return documents

def get_llm(model_name: str = 'gpt-3.5-turbo'):
    return ChatOpenAI(model_name=model_name, temperature=0)

def ensure_json(input_data):
    if isinstance(input_data, AIMessage):
        text = input_data.content
    elif isinstance(input_data, str):
        text = input_data
    else:
        raise TypeError(f"Expected str or AIMessage, got {type(input_data)}")

    if not text:
        return json.dumps({"answer": "No response from AI."})
        
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        try:
            text = text.replace("'", "\"").replace("None", "null").replace("True", "true").replace("False", "false")
            json.loads(text)
            return text
        except json.JSONDecodeError:
            return json.dumps({"answer": text})


def rag_chain_qa(context, chat_history, question, llm, **kwargs):
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the following pieces of context to answer the user's question at the end in JSON format. If you don't know the answer, just return 'null' as value, don't try to make up an answer. Always return your response in a valid JSON format."),
        ("human", "Here's some context that might be helpful: {context}"),
        ("human", "{question}")
    ])
    
    rag_chain = (
        {
            "context": lambda x: "\n\n".join([doc.page_content for doc in context]),
            "question": RunnablePassthrough()
        }
        | template
        | llm
        | (lambda x: ensure_json(x))
        | JsonOutputParser()
    )
    
    result = rag_chain.invoke(question)
    
    # Add this line to debug the response
    print(f"Raw LLM response: {result}")
    
    if result is None:
        if kwargs.get('rerun', False):
            result = rag_chain_qa(context, chat_history, question, llm, rerun=False)
        else:
            result = {"answer": "None"}
            print(f"RAG chain failed to return an answer for: {question}.")
    return result

def analyze_doc_bm25(question: str, retriever, chat_history: list = None, llm: str = 'gpt-3.5-turbo', **kwargs):
    # Define LLM
    llm = get_llm(model_name='gpt-3.5-turbo')
    
    # Get Context
    context = retriever.invoke(question)
    
    # Initialize chat history if not provided
    if chat_history is None:
        chat_history = []
    
    # Get Answer using the conversational RAG chain
    answer = rag_chain_qa(context, chat_history, question, llm, rerun=True)
    
    # Extract the answer string from the JSON response
    answer_str = answer.get('answer', str(answer))
    
    # Update chat history
    chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=answer_str)
    ])
    
    response = {
        "answer": answer,
        "context": context,
        "chat_history": chat_history
    }
    
    return response


class RAGPipeline:
    def __init__(self):
        self.document_chunks = None
        self.retriever = None
        load_dotenv(".env")
        print("RAG Pipeline initialized")

    def process_document(self, file_path, **kwargs):
        text = read_pdf(file_path)
        chunks = text_to_chunks(text)
        self.retriever = BM25Retriever.from_documents(documents=chunks, k=kwargs.get('k', 20))
        print("Document processed successfully")

    def get_answer(self, question, chat_history=None):
        if not self.retriever:
            raise ValueError("No document processed yet")
        print("Chat History:", chat_history)
        try:
            response = analyze_doc_bm25(question, retriever=self.retriever, chat_history=chat_history if chat_history else [])
            # print("Response:", response)
        except Exception as e:
            response = {"error": str(e)}
        
        return response