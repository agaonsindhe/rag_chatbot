import re

class Guardrails:
    def __init__(self):
        self.blocked_keywords = ["political", "illegal", "personal", "hack", "scam"]  # Example banned words
        self.allowed_topics = ["finance", "stock", "revenue", "profit", "loss", "earnings"]

    def validate_input(self, query):
        print(f"🔍 Validating Query: {query}")  # Log the raw query

        financial_terms = ["interest income", "bank", "revenue", "income", "profit", "loss", "financial","Staff costs","Number of Employees"]
        is_valid = any(term.lower() in query.lower() for term in financial_terms)

        print(f"✅ Query Valid: {is_valid}")  # Log whether it passed validation
        return is_valid

    def filter_output(self, response: str) -> str:
        """Remove misleading or harmful content from response."""
        print("🔍 Before Filtering:", response)  # Log original response
        # Example: Removing sensitive financial speculation
        response = re.sub(r'predicts.*\d+%', 'REDACTED', response, flags=re.IGNORECASE)
        print("✅ After Filtering:", response)  # Log filtered response
        return response
