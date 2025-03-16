import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter


class FinancialDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()
        print("init ",self.data.columns)
        self.column_names = ["Company Name", "Sector", "Market Cap", "Stock P/E", "Year", "Revenue", "Net Profit"]

    def load_data(self):
        """Load and clean the CSV file into a Pandas DataFrame."""
        df = pd.read_csv(self.file_path)

        # Standardize column names (trim spaces)
        df.columns = df.columns.str.strip()
        print("columns: ",df.columns)

        # Handle missing values (replace with NaN and fill where needed)
        df.replace(["-", "N/A", "NA", ""], pd.NA, inplace=True)

        # Convert numerical columns to float after removing any text artifacts
        for col in ["Market Cap", "Stock P/E", "Revenue", "Net Profit"]:
            df[col] = df[col].astype(str).str.replace("Cr", "").str.strip()  # Remove 'Cr' notation
            df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to numeric
        print("after loop : ", df.columns)
        # Sort by company and year
        df.sort_values(by=["Company Name", "Year"], ascending=[True, True], inplace=True)
        print("after sort : ", df.columns)
        # Set index for efficient querying
        # df.set_index(["Company Name", "Year"], inplace=True)
        # print("before return : ", df.columns)
        return df

    def preprocess_data(self, chunk_size=300, chunk_overlap=50):
        """Format financial data as structured text and split into chunks."""

        # Convert dataframe into structured text format for better context
        structured_data = "\n".join(
            self.data.reset_index().apply(
                lambda row: f"Company: {row['Company Name']}, Year: {row['Year']}, "
                            f"Sector: {row['Sector']}, Market Cap: {row['Market Cap']} Cr, "
                            f"Stock P/E: {row['Stock P/E']}, Revenue: {row['Revenue']} Cr, "
                            f"Net Profit: {row['Net Profit']} Cr",
                axis=1
            )
        )

        # Use a text splitter to chunk structured financial text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                       separators=["\n"])
        return text_splitter.split_text(structured_data)

    def preview_data(self, num_rows=5):
        """Display structured preview of the cleaned dataset."""
        print("\n🔍 Preview of Processed Financial Data:")
        print(self.data.reset_index().head(num_rows).to_string(index=False))


if __name__ == "__main__":
    # Initialize and process data
    processor = FinancialDataProcessor("merged_financial_data_yearly.csv")  # Use new cleaned CSV
    chunks = processor.preprocess_data()
    processor.preview_data(num_rows=5)
