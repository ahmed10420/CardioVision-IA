from dotenv import load_dotenv

from agents.groq_client import create_groq_llm

load_dotenv()

llm = create_groq_llm(
    max_tokens=100,
)

response = llm.invoke("Réponds seulement par OK")

print(response.content)
