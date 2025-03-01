from embeddings import EmbeddingModel, VectorDatabase

class BasicRetriever:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2", index_path="models/vector_index.pkl"):
        """
        Initializes the Basic RAG retriever using FAISS.

        Args:
            embedding_model (str): Pre-trained embedding model.
            index_path (str): Path to FAISS vector index.
        """
        self.embedder = EmbeddingModel(model_name=embedding_model)
        self.vector_db = VectorDatabase(index_path=index_path)

    def retrieve(self, query, top_k=5):
        """
        Retrieves the most relevant text chunks for a given query.

        Args:
            query (str): The user query.
            top_k (int): Number of top results to retrieve.

        Returns:
            list: Top matching text chunks.
        """
        query_embedding = self.embedder.generate_embeddings([query]).reshape(1, -1)
        return self.vector_db.search(query_embedding, top_k=top_k)

# Example Usage:
if __name__ == "__main__":
    retriever = BasicRetriever()

    query = "What were the revenue trends?"
    results = retriever.retrieve(query, top_k=3)

    print("Top Matching Results:")
    for result in results:
        print(f"- {result}")
