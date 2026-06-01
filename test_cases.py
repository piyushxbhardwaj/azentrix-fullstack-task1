import os
from dotenv import load_dotenv
from utils.chunking import get_text_chunks
from utils.embeddings import get_vector_store
from utils.retrieval import generate_answer

# Load env variables
load_dotenv()

def test_cases():
    print("--- Running Test Cases ---")
    
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY is missing in .env file.")
        return

    # Mock Text for Case 1 and 2 (Text Paste mode & Missing Info)
    text = "The Indian Premier League (IPL) is a professional men's Twenty20 cricket league, contested by ten city-based franchise teams. Chennai Super Kings won the IPL in 2023."
    
    print("1. Building Knowledge Base from Sample Text...")
    chunks = get_text_chunks(text)
    vector_store = get_vector_store(chunks)
    print("Knowledge Base built successfully.\n")

    # Case 1: Valid Question
    print("Case 1: Asking a valid question...")
    q1 = "Who won the IPL in 2023?"
    ans1 = generate_answer(q1, vector_store)
    print(f"Q: {q1}\nA: {ans1}")
    if "Chennai Super Kings" in ans1:
        print("[PASS] Case 1 Passed\n")
    else:
        print("[FAIL] Case 1 Failed\n")

    # Case 2: Missing Info
    print("Case 2: Asking for missing info...")
    q2 = "Who won IPL 2025?"
    ans2 = generate_answer(q2, vector_store)
    print(f"Q: {q2}\nA: {ans2}")
    if "This information is not available in the document" in ans2:
        print("[PASS] Case 2 Passed\n")
    else:
        print("[FAIL] Case 2 Failed\n")

    print("--- Testing Completed ---")

if __name__ == "__main__":
    test_cases()
