from retrieval_basic import BasicRetriever
from retrieval_advanced import AdvancedRetriever
from guardrails import Guardrails

class RAGChatbot:
    def __init__(self, use_advanced=False):
        """
        Initializes the chatbot with retrieval and guardrail mechanisms.

        Args:
            use_advanced (bool): If True, uses advanced retrieval (BM25 + FAISS).
        """
        self.use_advanced = use_advanced
        self.retriever = AdvancedRetriever() if use_advanced else BasicRetriever()
        self.guardrails = Guardrails()

    def generate_response(self, query):
        """
        Generates a response for a given query using retrieval.

        Args:
            query (str): User query.

        Returns:
            str: Chatbot response.
        """
        # Validate input query
        if not self.guardrails.validate_input(query):
            return "[SYSTEM]: Your query is not allowed."

        # Retrieve relevant chunks
        results = self.retriever.hybrid_retrieve(query, top_k=3) if self.use_advanced else self.retriever.retrieve(query, top_k=3)

        if not results:
            return "[SYSTEM]: No relevant information found."

        # Construct response
        response = " ".join(results)

        # Filter based on confidence score (mocked as 0.7 for now)
        return self.guardrails.filter_response(response, confidence_score=0.7)

# Example Usage:
if __name__ == "__main__":
    chatbot = RAGChatbot(use_advanced=True)  # Set to False for Basic RAG

    query = "What was the company's revenue growth last year?"
    response = chatbot.generate_response(query)

    print(f"User: {query}")
    print(f"Chatbot: {response}")
