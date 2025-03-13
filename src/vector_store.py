import faiss


class FAISSVectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)

    def add_embeddings(self, embeddings):
        self.index.add(embeddings)
        print(f"✅ FAISS Index Size: {self.index.ntotal} vectors stored")

    def save_index(self, file_name):
        faiss.write_index(self.index, file_name)

    def load_index(self, file_name):
        self.index = faiss.read_index(file_name)

    def search(self, query_embedding, top_k=3):
        distances, indices = self.index.search(query_embedding, top_k)
        return indices
