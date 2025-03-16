import re

class Guardrails:
    def __init__(self):
        """Define restricted keywords and allowed financial topics."""
        self.blocked_keywords = ["political", "illegal", "personal", "hack", "scam", "violence", "hate"]
        self.allowed_topics = ["finance", "stock", "revenue", "profit", "loss", "earnings", "investment", "market","P/E"]

        # Expand financial terms dynamically for validation
        self.financial_terms = [
            "interest income", "bank", "revenue", "income", "profit", "loss", "financial",
            "staff costs", "number of employees", "stock price", "balance sheet", "dividend",
            "market cap", "earnings per share", "PE ratio"
        ]

    def sanitize_query(self, query: str) -> str:
        """Sanitize query by removing excessive symbols & potential SQL injection attempts."""
        query = re.sub(r"[^\w\s?.!,-]", "", query)  # Allow only words, spaces, and basic punctuation
        query = query.strip()  # Remove leading/trailing spaces
        print(f"🔍 Sanitized Query: {query}")  # Debugging log
        return query

    def validate_input(self, query: str) -> bool:
        """Validate if the query is relevant to financial topics."""
        query = self.sanitize_query(query)  # Sanitize input before checking
        print(f"🔍 Validating Query: {query}")  # Log raw query

        is_valid = any(term.lower() in query.lower() for term in self.financial_terms)

        # Block query if it contains restricted keywords
        for blocked in self.blocked_keywords:
            if blocked in query.lower():
                print(f"🚫 Query Blocked (Contains '{blocked}')")
                return False

        print(f"✅ Query Valid: {is_valid}")  # Log validation result
        return is_valid

    def filter_output(self, response: str) -> str:
        """Filter misleading, speculative, or non-financial statements."""
        print("🔍 Before Filtering:", response)  # Log original response

        # Remove speculative financial predictions (e.g., "Company X will grow by 100%")
        response = re.sub(r"predicts.*\d+%", "REDACTED", response, flags=re.IGNORECASE)

        # Remove non-financial investment advice (e.g., "I recommend investing in X")
        response = re.sub(r"I recommend investing in.*", "REDACTED", response, flags=re.IGNORECASE)

        # Ensure no fake financial claims (e.g., "XYZ has infinite revenue")
        response = re.sub(r"(infinite|unlimited) revenue", "REDACTED", response, flags=re.IGNORECASE)

        print("✅ After Filtering:", response)  # Log filtered response
        return response
