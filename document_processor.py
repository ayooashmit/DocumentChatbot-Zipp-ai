
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import gc
import pickle
import torch

class DocumentProcessor:
    def __init__(self):
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        print(f"DocumentProcessor using device for embeddings: {device}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len
        )
        self.FAISS_INDEX_DIR = "faiss_index_store"

    def read_file_content(self, file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def process_documents(self, document_paths):
        if not document_paths:
            print("No document paths provided.")
            return None

        doc_names_hash = "_".join(sorted([os.path.basename(p) for p in document_paths])).replace('.', '_').replace(' ', '_')
        current_index_path = os.path.join(self.FAISS_INDEX_DIR, doc_names_hash)

        if os.path.exists(current_index_path):
            print(f"Loading FAISS index from {current_index_path}...")
            try:
                vectorstore = FAISS.load_local(current_index_path, self.embeddings, allow_dangerous_deserialization=True)
                print("FAISS index loaded successfully from disk.")
                return vectorstore
            except Exception as e:
                print(f"Error loading FAISS index: {e}. Rebuilding index...")

        print("No existing FAISS index found for this selection or loading failed. Processing documents from scratch...")
        all_docs = []
        for path in document_paths:
            if not os.path.exists(path):
                print(f"Warning: Document not found at {path}. Skipping.")
                continue
            print(f"Processing {os.path.basename(path)}...")
            file_content = self.read_file_content(path)
            docs = self.text_splitter.create_documents([file_content])
            for doc in docs:
                doc.metadata = {"source": os.path.basename(path)}
            all_docs.extend(docs)
            gc.collect()

        if not all_docs:
            print("No documents to process. Returning None.")
            return None

        vectorstore = FAISS.from_documents(all_docs[:100], self.embeddings)
        for i in range(100, len(all_docs), 100):
            batch = all_docs[i:i + 100]
            if batch:
                vectorstore.add_documents(batch)
            gc.collect()

        print(f"Saving FAISS index to {current_index_path}...")
        os.makedirs(current_index_path, exist_ok=True)
        vectorstore.save_local(current_index_path)
        print("FAISS index saved successfully.")

        return vectorstore