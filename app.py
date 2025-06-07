import streamlit as st
from document_processor import DocumentProcessor
from chat_engine import ChatEngine
import os

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("The AI powered Chatbot")
st.write("Upload or select documents to chat with")

document_dir = "documents"
available_docs = [f for f in os.listdir(document_dir) if f.endswith('.txt')]
selected_docs = st.multiselect("Select documents", available_docs)

if st.button("Load Documents") and selected_docs:
    with st.spinner("Processing documents..."):
        doc_paths = [os.path.join(document_dir, doc) for doc in selected_docs]
        processor = DocumentProcessor()
        st.session_state.vectorstore = processor.process_documents(doc_paths)
        if st.session_state.vectorstore:
            st.session_state.chat_engine = ChatEngine(st.session_state.vectorstore)
            st.session_state.messages = []
            st.success("Documents loaded successfully! You can now chat.")
        else:
            st.error("Failed to load documents or no documents found.")

if st.session_state.chat_engine:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
else:
    st.warning("Please load documents first to start chatting.")