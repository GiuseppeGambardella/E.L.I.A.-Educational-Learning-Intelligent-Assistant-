from elia.config import Config
from openai import OpenAI
import logging

client = OpenAI(
    api_key=Config.GEMMA_API_KEY,
    base_url=Config.GEMMA_API_URL
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ask_llm(prompt, context):
    logger.info("🤖 LLM request in progress...")

    messages = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=messages
    )

    output = response.choices[0].message.content
    logger.info(f"✅ LLM response received")

    return output


