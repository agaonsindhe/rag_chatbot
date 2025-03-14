import time

import streamlit as st
import yaml

from data_processor import FinancialDataProcessor
from embedding_model import EmbeddingModel
from vector_store import FAISSVectorStore
from chunk_merger import ChunkMerger
from rag_chatbot import RAGChatbot
from guardrails import Guardrails
import os
import pickle
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
import sys

print("Current Working Directory:", os.getcwd())
print("Python Path:", sys.path)

@st.cache_resource
def load_chatbot(model_choice):
    return RAGChatbot(model_name=model_choice)

# Load Configurations
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()

# Streamlit UI for model selection
st.sidebar.title("⚙️ Model Selection")
model_choice = st.sidebar.selectbox("Choose a model:", list(config["slm_models"].keys()), index=list(config["slm_models"].keys()).index(config["default_model"]))

# Initialize Components
processor = FinancialDataProcessor(config["data_path"])
chunks = processor.preprocess_data()
embedding_model = EmbeddingModel(config["embedding_model"])
embedding_cache_path = "data/embeddings.pkl"

if os.path.exists(embedding_cache_path):
    print("✅ Loading cached embeddings...")
    with open(embedding_cache_path, "rb") as f:
        chunk_embeddings = pickle.load(f)
else:
    print("Computing embeddings (first time only)...")
    chunk_embeddings = embedding_model.encode(chunks)
    with open(embedding_cache_path, "wb") as f:
        pickle.dump(chunk_embeddings, f)


vector_store = FAISSVectorStore(chunk_embeddings.shape[1])
vector_store.add_embeddings(chunk_embeddings)
vector_store.save_index("faiss_index.bin")
vector_store.load_index("faiss_index.bin")


# Initialize Retrieval, Chatbot & Guardrails
chunk_merger = ChunkMerger(chunks, embedding_model, vector_store)
chatbot = load_chatbot(model_choice)
guardrails = Guardrails()


# Streamlit UI Implementation
st.title("Financial RAG Chatbot")

# Add dataset preview option
if st.sidebar.button("Preview Dataset"):
    st.write("### Financial Dataset Preview")
    st.dataframe(processor.data.head(10))

query = st.text_input("Enter your financial question:")
if st.button("Get Answer"):
    start_time = time.time()  # Track response time
    if guardrails.validate_input(query):
        retrieved_chunks = chunk_merger.retrieve_chunks(query, top_k=3)

        print(f"🔍 FAISS Retrieved Chunks (Count: {len(retrieved_chunks)}):", retrieved_chunks)  # Debug log
        if not retrieved_chunks:
            print("❌ No relevant chunks retrieved. FAISS might not be returning correct results.")

        if not retrieved_chunks:
            st.write("⚠️ No relevant information found in the dataset.")
        else:
            merged_text = "\n".join(retrieved_chunks)  # Merge retrieved chunks
            response = chatbot.get_response(merged_text, query)
            # response = guardrails.filter_output(response)  # Ensure output filtering
            if response:
                st.write("### Response:")
                formatted_response = response.replace("\n", "  \n")  # Ensure newlines are preserved in Markdown
                st.markdown(f"```\n{formatted_response}\n```")  # Wrap in code block to preserve formatting
            else:
                st.warning("⚠️ No response generated.")

    else:
        st.write("Invalid query. Please enter a financial-related question.")

    end_time = time.time()
    response_time = round(end_time - start_time, 2)
    st.sidebar.write(f"⏳ Response Time: {response_time} seconds")