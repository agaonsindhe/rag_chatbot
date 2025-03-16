import os
import pandas as pd

# Define Paths
BASE_DIR = "data/companies"  # Update with actual dataset path
OUTPUT_FILE = "data/merged_financial_data_yearly.csv"  # Output CSV storing year-wise data

# List to store all data
merged_data = []

# Process each company folder
for company_folder in os.listdir(BASE_DIR):
    company_path = os.path.join(BASE_DIR, company_folder)

    if os.path.isdir(company_path):  # Ensure it's a folder
        company_data_list = []  # Store multiple years for each company

        # Define file paths
        files = {
            "info": os.path.join(company_path, f"{company_folder}_Basic_Info.csv"),
            "yearly_profit_loss": os.path.join(company_path, "Yearly_Profit_Loss.csv")
            # "balance_sheet": os.path.join(company_path, "Yearly_Balance_Sheet.csv"),
            # "cash_flow": os.path.join(company_path, "Yearly_Cash_flow.csv"),
            # "ratios": os.path.join(company_path, "Ratios.csv"),
            # "shareholding": os.path.join(company_path, "Yearly_Shareholding_Pattern.csv"),
        }

        # Extract Company Info (Static Fields)
        company_info = {
            "Company Name": company_folder,
            "Sector": None,
            "Market Cap": None,
            "Stock P/E": None
        }

        if os.path.exists(files["info"]):
            df_info = pd.read_csv(files["info"])
            company_info.update({
                "Sector": df_info.iloc[0].get("Sector", None),
                "Market Cap": df_info.iloc[0].get("Market Cap", None),
                "Stock P/E": df_info.iloc[0].get("Stock P/E", None),
            })

        # Extract Yearly Profit/Loss Data
        if os.path.exists(files["yearly_profit_loss"]):
            df = pd.read_csv(files["yearly_profit_loss"], index_col=0)  # Set first column as index
            df = df.T  # Transpose to get years as rows

            for year in df.index:  # Loop through each year
                year_data = company_info.copy()
                year_data["Year"] = year

                # Extract financial metrics
                year_data["Revenue"] = df.loc[year, "Sales"] if "Sales" in df.columns else None
                year_data["Net Profit"] = df.loc[year, "Net Profit"] if "Net Profit" in df.columns else None

                company_data_list.append(year_data)

        # # Extract Yearly Balance Sheet Data
        # if os.path.exists(files["balance_sheet"]):
        #     df = pd.read_csv(files["balance_sheet"], index_col=0).T
        #     for data in company_data_list:
        #         year = data["Year"]
        #         if year in df.index:
        #             data["Total Assets"] = df.loc[year, "Total Assets"] if "Total Assets" in df.columns else None
        #             data["Total Liabilities"] = df.loc[
        #                 year, "Total Liabilities"] if "Total Liabilities" in df.columns else None
        #
        # # Extract Yearly Cash Flow Data
        # if os.path.exists(files["cash_flow"]):
        #     df = pd.read_csv(files["cash_flow"], index_col=0).T
        #     for data in company_data_list:
        #         year = data["Year"]
        #         if year in df.index:
        #             data["Cash Flow"] = df.loc[year, "Net Cash Flow"] if "Net Cash Flow" in df.columns else None
        #
        # # Extract Promoter Holding from Yearly Shareholding Pattern
        # if os.path.exists(files["shareholding"]):
        #     df = pd.read_csv(files["shareholding"], index_col=0).T
        #     for data in company_data_list:
        #         year = data["Year"]
        #         if year in df.index:
        #             data["Promoter Holding (%)"] = df.loc[year, "Promoters"] if "Promoters" in df.columns else None

        merged_data.extend(company_data_list)

# Convert list to DataFrame
merged_df = pd.DataFrame(merged_data)

# Save to CSV
merged_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Merged data saved to {OUTPUT_FILE}")
