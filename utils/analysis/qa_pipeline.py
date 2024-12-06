# rag_pipeline.py

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
import json
from dotenv import load_dotenv
from .analyzer import read_pdf, text_to_chunks, get_llm
from langchain.chains import load_summarize_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever


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


def get_history_aware_retriever(llm, retriever):
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever

def rag_chain_qa(question, memory, llm, retriever, summarizer, **kwargs):
    # Manage the chat history for prompt
    k = kwargs.get("k", 5)
    recent_history = memory.chat_memory.messages[-k:]
    older_history = memory.chat_memory.messages[:-k]

    # if older_history:
    #     summarized_history = summarizer({"input_documents": older_history})
    #     combined_history = summarized_history["output_text"] + recent_history
    # else:
    #     combined_history = recent_history
    combined_history = recent_history

    # Load the QA chain
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the following pieces of context to answer the user's question. If you don't know the answer, just return 'null' as value, don't try to make up an answer. {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = get_history_aware_retriever(llm, retriever)
    qa_chain = create_stuff_documents_chain(llm, prompt=template)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    result = rag_chain.invoke({"input": question, "chat_history": combined_history})
    
    if result is None:
        if kwargs.get('rerun', False):
            result = rag_chain_qa(question, memory, llm, retriever, summarizer, rerun=False)
        else:
            result = {"answer": "None"}
            print(f"RAG chain failed to return an answer for: {question}.")

    memory.save_context({"question": question}, {"answer": result.get("answer")})
    print("Context Nums: ", len(result.get('context', [])))

    return result


class RAGPipeline:
    def __init__(self, model_name='gpt-3.5-turbo'):
        load_dotenv(".env")
        self.document_chunks = None
        self.retriever = None
        self.llm = get_llm(model_name)
        self.memory = ConversationBufferMemory(return_messages=True) 
        self.summarizer = load_summarize_chain(self.llm, chain_type="stuff")
        print("RAG Pipeline initialized")
        print(f"Using model: {model_name}")

    def process_document(self, file_path, **kwargs):
        text = read_pdf(file_path)
        chunks = text_to_chunks(text)
        self.retriever = BM25Retriever.from_documents(documents=chunks, k=kwargs.get('k', 20))
        print("Document processed successfully")

    def get_answer(self, question):
        if not self.retriever:
            raise ValueError("No document processed yet")
        try:
            response = rag_chain_qa(question, self.memory, self.llm, self.retriever, self.summarizer)
        except Exception as e:
            print(e)
            response = {"error": str(e)}
        
        return response