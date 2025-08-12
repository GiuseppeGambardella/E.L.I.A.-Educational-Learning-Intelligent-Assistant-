from elia.config import Config
from openai import OpenAI

client = OpenAI(
    api_key=Config.GEMMA_API_KEY,
    base_url=Config.GEMMA_API_URL
    )

def ask_llm(prompt, context):
    messages = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=messages
    )
    return response.choices[0].message.content



