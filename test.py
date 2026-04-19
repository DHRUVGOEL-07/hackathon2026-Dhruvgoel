import os
from dotenv import load_dotenv
load_dotenv()
print("GROQ KEY:", os.getenv("GROQ_API_KEY")[:8] if os.getenv("GROQ_API_KEY") else "NOT FOUND")