import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter


class FinancialDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()
        self.column_names = []  # Store column names dynamically

    def load_data(self):
        return pd.read_csv(self.file_path)

    def preprocess_data(self, chunk_size=300, chunk_overlap=50):
        """Load, preprocess, and chunk financial data dynamically."""
        self.column_names = list(self.data.columns)  # Extract column names dynamically

        # Convert dataframe into structured text format (row-wise)
        structured_data = "\n".join(self.data.astype(str).apply(lambda row: ", ".join(row), axis=1))

        # Use a text splitter to chunk row-wise structured text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n"])
        return text_splitter.split_text(structured_data)

    def preview_data(self, num_rows=5):
        """Display the first few rows of the dataset."""
        print("\nPreview of Financial Dataset:")
        print(self.data.head(num_rows))


if __name__ == "__main__":

    # Initialize Components
    processor = FinancialDataProcessor("data/BPF1_17032024142500687.csv")
    chunks = processor.preprocess_data()
    processor.preview_data(num_rows=len(chunks))
