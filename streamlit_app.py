import streamlit as st
from streamlit_chat import message
from werkzeug.utils import secure_filename
import os
from utils.analysis.qa_pipeline import RAGPipeline

def format_context(context):
    formatted_text = []
    for doc in context:
        if isinstance(doc, dict) and "page_content" in doc:
            content = doc["page_content"].replace('\n', ' ')  # Remove newline characters
            formatted_text.append({
                "content": content,
                "metadata": doc["metadata"]
            })
        elif isinstance(doc, str):
            formatted_text.append(doc.replace('\n', ' ')) # Remove newline characters

        else:  # If it's a LangChain document
            try:
                content = doc.page_content.replace('\n', ' ') # remove newline
                metadata = {k: v for k, v in doc.metadata.items() if k != "chunk_number"}
                metadata_string = f"({', '.join(f'{k}: {v}' for k, v in metadata.items())})" if metadata else ""
                # formatted_text.append(f"{content} {metadata_string}") # add metadata
                formatted_text.append({
                "content": content,
                "metadata": doc.metadata
            })

            except AttributeError:
                formatted_text.append(f"Could not parse document: {doc}")

    return formatted_text

# Model Selection (add this at the beginning of your script)
available_models = ["gpt-3.5-turbo", "chatgpt-4o-latest"]  # Add all the models you want to support
selected_model = st.sidebar.selectbox("Select Model", available_models) 

# Initialize RAGPipeline with the selected model (ONLY if it's not already in session state or the model has changed)
if "rag_pipeline" not in st.session_state or st.session_state.get("selected_model") != selected_model: 
    st.session_state.rag_pipeline = RAGPipeline(selected_model)
    st.session_state.selected_model = selected_model  # Store the selected model

# Setup session state to store chat history if not already present (rest of your code) ...
if "messages" not in st.session_state:
    st.session_state.messages = []

# Track the ID of the previously uploaded file (initialize to None)
if "last_uploaded_file_id" not in st.session_state:
    st.session_state.last_uploaded_file_id = None

# Streamlit UI
st.title("Chat with your PDF 💬")

# File Uploader (use a key to uniquely identify it)
uploaded_file = st.file_uploader("Upload a PDF", type="pdf", key="pdf_uploader")

if uploaded_file is not None:
    # Get the ID of the currently uploaded file (using its name as a proxy)
    current_file_id = uploaded_file.name

    # Check if the file ID has changed (i.e., a new file has been uploaded)
    if current_file_id != st.session_state.last_uploaded_file_id:
        # --- Clear session state to start fresh with the new PDF ---
        st.session_state.messages = []  # Clear chat history
        st.session_state.rag_pipeline = RAGPipeline(selected_model)  # Re-initialize RAG pipeline with the selected model
        st.session_state.last_uploaded_file_id = current_file_id  # Update the last uploaded file ID

        # Save uploaded file to a temporary location
        filename = secure_filename(uploaded_file.name)
        filepath = os.path.join("uploads", filename)  # Ensure "uploads" folder exists
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the document (you might want to add a status message here)
        with st.spinner("Processing PDF..."):
            st.session_state.rag_pipeline.process_document(filepath)  # Use the re-initialized instance
        st.success("PDF processed successfully!")
    
# Chat interface (only show chat if a file has been uploaded and processed)
if uploaded_file:
    chat_container = st.container()

    if prompt := st.chat_input("Ask a question about the PDF"):
        st.session_state.messages.append({"user": prompt, "bot": "", "context": ""})

        try:
            response = st.session_state.rag_pipeline.get_answer(prompt)
            answer = str(response.get("answer", "No answer found."))
            context = response.get("context", "")
            st.session_state.messages[-1]["context"] = context
        except Exception as e:
            answer = f"An error occurred: {str(e)}"
            context = ""

        st.session_state.messages[-1]["bot"] = answer

    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            col1, _ = st.columns([9, 1])

            with col1:
                st.markdown("""<style>.stChatMessageBox, .stChatMessage {width: 100% !important;}</style>""", unsafe_allow_html=True)
                message(msg["user"], is_user=True, key=str(i) + "_user")
                if msg["bot"]:
                    message(msg["bot"], key=str(i))

                if msg.get("bot"):
                    show_context_key = f"show_context_{i}"
                    show_context = st.session_state.get(show_context_key, False)

                    if st.button(f"{'Hide' if show_context else 'Show'} Context", key=f"show_context_button_{i}"):
                        st.session_state[show_context_key] = not show_context

                    if show_context:
                        st.write("**Context:**")
                        st.write(format_context(msg["context"]))