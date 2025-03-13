class ChunkMerger:
    def __init__(self, chunks, embedding_model, vector_store):
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve_chunks(self, query, top_k=3):
        query_embedding = self.embedding_model.encode([query])
        indices = self.vector_store.search(query_embedding, top_k)
        return [self.chunks[i] for i in indices[0]]

    def merge_chunks(self, query, top_k=3):
        retrieved_chunks = self.retrieve_chunks(query, top_k)
        return "\n".join(retrieved_chunks)
