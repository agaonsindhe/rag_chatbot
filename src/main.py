import streamlit as st
import yaml
import os
import pickle
import time

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
# Get list of model keys
model_keys = list(config["slm_models"].keys())

# Ensure default model key exists
default_model_key = config["default_model"]
if default_model_key not in model_keys:
    raise ValueError(f"Error: Default model '{default_model_key}' not found in slm_models!")

# Select a model using keys
selected_model_key = st.sidebar.selectbox(
    "Choose a model:",
    model_keys,
    index=model_keys.index(default_model_key)
)

# Resolve selected model name
model_choice = config["slm_models"][selected_model_key]  # Convert key to full model path
print("Model choice",model_choice)

# Streamlit UI for selecting RAG mode
rag_mode = st.sidebar.radio("Choose Retrieval Mode:", ["Basic RAG", "Advanced RAG"])

# Initialize Components
processor = FinancialDataProcessor(config["data_path"])
chunks = processor.preprocess_data()
embedding_model = EmbeddingModel(config["embedding_model"])
embedding_cache_path = config["embedding_cache_path"]
# Add dataset preview option
if st.sidebar.button("Preview Dataset"):
    st.write("### Financial Dataset Preview")
    st.dataframe(processor.data.head(10))
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
vector_store.save_index(config["index_path"])
vector_store.load_index(config["index_path"])

# Initialize Retrieval, Chatbot & Guardrails
chunk_merger = ChunkMerger(chunks, processor.data,embedding_model, vector_store)
chatbot = load_chatbot(model_choice)
guardrails = Guardrails()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Layout with two columns and a vertical divider
col1, col2 = st.columns([2, 1])  # Middle column for spacing

with col1:
    st.markdown("<h2 style='text-align: center;'>💬 Financial RAG Chatbot</h2>", unsafe_allow_html=True)
    query = st.text_input("Enter your financial question:")
    if st.button("Get Answer"):
        start_time = time.time()  # Track response time
        response = "⚠️ No relevant information found in the dataset."
        if guardrails.validate_input(query):
            retrieved_chunks = chunk_merger.retrieve_chunks(query, top_k=config["retrieval"]["top_k"])
            print(f"🔍 FAISS Retrieved Chunks (Count: {len(retrieved_chunks)}):", retrieved_chunks)

            if rag_mode == "Basic RAG":
                merged_text = "\n".join(retrieved_chunks)  # Simple merging
            else:
                merged_text = chunk_merger.adaptive_merge_chunks(query, retrieved_chunks)  # Advanced merging
            if not retrieved_chunks:
                st.write(response)
            else:
                response = chatbot.get_response(merged_text, query)
                print("response ", response)
                response = guardrails.filter_output(response)  # Ensure output filtering
                if response:
                    st.write("### Response:")
                    formatted_response = response.replace("\n", "  \n")  # Ensure newlines are preserved in Markdown
                    st.markdown(f"```\n{formatted_response}\n```")  # Wrap in code block to preserve formatting
                else:
                    st.warning("⚠️ No response generated.")
        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        st.sidebar.write(f"⏳ Response Time: {response_time} seconds")
        # Store query, response, and RAG mode in session state chat history
        st.session_state.chat_history.append((query, response, rag_mode))

st.divider()

# Scrollable chat history
with col2:
    st.markdown("""
        <style>
        .chat-history {
            max-height: 400px;
            overflow-y: auto;
            padding-right: 10px;
        }
        </style>
        <div class="chat-history">
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center;'>📜 Chat History</h2>", unsafe_allow_html=True)

    for q, r, mode in reversed(st.session_state.chat_history):
        rag_color = "#ff5733" if mode == "Basic RAG" else "#33b5e5"  # Different color for each mode
        st.markdown(f"<p style='color:{rag_color}; font-weight: bold;'>{mode}</p>", unsafe_allow_html=True)
        st.write(f"**Q:** {q}")
        st.write(f"**A:** {r}")
        st.write("---")

st.markdown("</div>", unsafe_allow_html=True)


