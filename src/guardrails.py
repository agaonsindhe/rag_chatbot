import re

class Guardrails:
    def __init__(self, blocked_keywords=None, confidence_threshold=0.3):
        """
        Initializes guardrails for query validation and response filtering.

        Args:
            blocked_keywords (list): List of disallowed words/phrases.
            confidence_threshold (float): Minimum confidence score for response acceptance.
        """
        self.blocked_keywords = blocked_keywords or [
            "hack", "attack", "malware", "fraud", "phishing", "scam", "exploit"
        ]
        self.confidence_threshold = confidence_threshold

    def validate_input(self, query):
        """
        Validates user input by checking for inappropriate content.

        Args:
            query (str): User query.

        Returns:
            bool: True if query is valid, False otherwise.
        """
        query_lower = query.lower()

        # Check for blocked keywords
        for keyword in self.blocked_keywords:
            if keyword in query_lower:
                print(f"[WARNING] Blocked query detected: {query}")
                return False

        # Basic format validation (only alphanumeric and common punctuation allowed)
        if not re.match(r"^[a-zA-Z0-9\s,.!?'-]+$", query):
            print("[WARNING] Query contains invalid characters.")
            return False

        return True

    def filter_response(self, response, confidence_score):
        """
        Filters the chatbot response based on confidence score.

        Args:
            response (str): Generated response.
            confidence_score (float): Confidence score of the response.

        Returns:
            str: Filtered response.
        """
        if confidence_score < self.confidence_threshold:
            return "[SYSTEM]: I'm not confident about this answer. Please verify the information."
        return response

# Example Usage:
if __name__ == "__main__":
    guardrails = Guardrails()

    # Test input validation
    test_queries = ["How to hack a bank?", "Tell me about revenue growth"]
    for query in test_queries:
        is_valid = guardrails.validate_input(query)
        print(f"Query: {query} | Valid: {is_valid}")

    # Test output filtering
    test_response = "The company's revenue grew by 10% last quarter."
    filtered_response = guardrails.filter_response(test_response, confidence_score=0.2)
    print(f"Filtered Response: {filtered_response}")
