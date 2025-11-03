import os
from groq import Groq
from typing import List, Dict, Any, Generator

import dotenv
dotenv.load_dotenv()
class GroqClient:
    def __init__(self, model: str = "llama-3.3-70b-versatile", temperature: float = 0.1, max_tokens: int = 1024):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def chat_completion(self, messages: List[Dict[str, str]], stream: bool = False) -> Any:
        """Basic chat completion with Groq"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream
            )
            return completion
        except Exception as e:
            print(f"Groq API error: {e}")
            raise
    
    def get_response(self, prompt: str, system_message: str = None) -> str:
        """Get a single response from Groq"""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        completion = self.chat_completion(messages)
        return completion.choices[0].message.content