import streamlit as st
from chatbot import RAGChatbot

# Initialize chatbot (toggle between Basic & Advanced RAG)
chatbot = RAGChatbot(use_advanced=True)  # Set to False for Basic RAG

# Streamlit UI
st.title("📊 Financial RAG Chatbot")
st.write("Ask questions about financial statements!")

# User Input
query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query.strip():
        response = chatbot.generate_response(query)
        st.write(f"💬 **Chatbot:** {response}")
    else:
        st.warning("Please enter a valid question!")

# Footer
st.markdown("---")
st.markdown("🔍 **Powered by Retrieval-Augmented Generation (RAG)**")
