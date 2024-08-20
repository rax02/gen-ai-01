import json
from typing import List, Dict
import time
import re
from ratelimit import limits, sleep_and_retry
from collections import Counter

from openai import OpenAI

# Initialize the AI/ML API client with your API key and base URL
client = OpenAI(
    api_key="my-key",
    base_url="https://api.aimlapi.com",
)

def rate_limited_api_call(messages):
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=messages,
        temperature=0.7,
    )
    return response

def get_model_responses(messages: List[Dict[str, str]], max_retries: int = 3, timeout: int = 60) -> str:
    for attempt in range(max_retries):
        try:
            response = rate_limited_api_call(messages)
            return Markdown(response.choices[0].message.content.strip())
        except Exception as e:
            console.print(f"[bold red] Error in the model response (Attempt {attempt + 1} / {max_retries}): {e} [/bold red]")
            if attempt < max_retries - 1:
                time.sleep(5)
    raise Exception("Failed to get the response after repeated errors")


