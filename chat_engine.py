from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import torch
import warnings
import traceback
from langchain.prompts import PromptTemplate
import os

warnings.filterwarnings("ignore", category=UserWarning, module='transformers')

class ChatEngine:
    def __init__(self, vectorstore):
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"ChatEngine will attempt to use device for LLM: {self.device} via LlamaCpp")

        self.llm_model_path = os.path.join("models", "phi-3-mini-4k-instruct-q4.gguf")

        if not os.path.exists(self.llm_model_path):
            raise FileNotFoundError(f"GGUF model not found at {self.llm_model_path}. Please download it and place it in the 'models' folder.")

        print(f"Loading GGUF model from: {self.llm_model_path}")

        self.llm = LlamaCpp(
            model_path=self.llm_model_path,
            n_ctx=4096,
            n_gpu_layers=-1 if self.device == "mps" else 0,
            n_batch=512,
            f16_kv=True,
            verbose=False,
            temperature=0.7,
            max_tokens=256,
            top_p=0.9,
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        self.vectorstore = vectorstore

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": self._get_rag_prompt_template()}
        )

    def _get_rag_prompt_template(self):
        template = """<|system|>
You are an AI assistant. Use only the provided context to answer the question. If the answer is not in the context, keep thinking for some time and then state that you don't know. Keep your answers concise and directly relevant to the question.
<|end|>
<|user|>
Here is the context:
{context}

Here is the conversation history:
{chat_history}

Question: {question}<|end|>
<|assistant|>
"""
        return PromptTemplate.from_template(template)

    def chat(self, question):
        try:
            result = self.qa_chain.invoke({"question": question})
            return result["answer"].strip()
        except Exception as e:
            traceback.print_exc()
            print(f"Error during chat: {e}")
            return "An error occurred while generating the response. Please try again."