from sentence_transformers import SentenceTransformer
import numpy as np
import re
import pandas as pd

class ChunkMerger:
    def __init__(self, chunks, df,embedding_model, vector_store):
        """
                Initialize ChunkMerger with dataset, embedding model, and FAISS vector store.
                """
        self.df = df  # Store dataset DataFrame
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight model for similarity scoring

    def retrieve_chunks(self, query, top_k=3):
        """
        Retrieve top-k relevant chunks based on query embedding and apply structured filtering.
        """
        print(f"🔍 New Query: {query}")
        query_embedding = self.embedding_model.encode([query])
        indices = self.vector_store.search(query_embedding, top_k)
        retrieved_chunks = [self.chunks[i] for i in indices[0]]
        print(f"🔍 FAISS Returned Indices: {indices}")
        # Log retrieved chunks before filtering
        print(f"✅ Retrieved Chunks Before Filtering: {retrieved_chunks}")
        # Convert retrieved chunks into structured DataFrame format
        column_names = list(self.df.columns)  # Dynamically extract column names
        data_rows = []

        for chunk in retrieved_chunks:
            fields = chunk.split(", ")  # Split each chunk properly
            if len(fields) == len(column_names):  # Ensure proper alignment
                data_rows.append(fields)

        df_filtered = pd.DataFrame(data_rows, columns=column_names)

        # **Extract numerical constraints from the query**
        year_match = re.search(r"\b(19|20)\d{2}\b", query)  # Extract year
        income_match = re.search(r"\b\d+\b", query)  # Extract numeric threshold

        # **Apply year filtering**
        if "Year" in column_names and year_match:
            target_year = year_match.group(0)
            df_filtered = df_filtered[df_filtered["Year"] == target_year]
            print(f"📅 Year Filter Applied: {target_year}")

        # **Apply income threshold filtering**
        if "Value" in column_names and income_match:
            income_threshold = float(income_match.group(0))
            df_filtered = df_filtered[df_filtered["Value"].astype(float) > income_threshold]
            print(f"💰 Income Filter Applied: > {income_threshold}")

        print(f"✅ Final Filtered Chunks: {df_filtered}")
        # Return structured text instead of raw chunks
        result = df_filtered.to_string(index=False) if not df_filtered.empty else "No relevant information found."
        print(f"📤 Final Retrieval Output: {result}")
        return result

    def merge_chunks(self, query, top_k=3):
        retrieved_chunks = self.retrieve_chunks(query, top_k)
        return "\n".join(retrieved_chunks)

    def adaptive_merge_chunks(self, query, retrieved_chunks):
        """
        Advanced RAG: Merge retrieved chunks based on semantic similarity.
        """
        chunk_embeddings = self.similarity_model.encode(retrieved_chunks)
        query_embedding = self.similarity_model.encode([query])[0]

        # Compute cosine similarity
        similarities = np.dot(chunk_embeddings, query_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Sort chunks by similarity
        sorted_chunks = [chunk for _, chunk in sorted(zip(similarities, retrieved_chunks), reverse=True)]

        # Merge top 2 most similar chunks
        return " ".join(sorted_chunks[:2])  # Use only top 2 most relevant chunks
