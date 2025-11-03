import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict, Any, Generator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalClient:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", 
                 device: str = "cuda", temperature: float = 0.1, 
                 max_tokens: int = 2048):
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the local model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            
            logger.info(f"Model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _format_chat_template(self, prompt: str, system_message: str = None) -> str:
        """Format messages using the model's chat template"""
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted_prompt
        except Exception as e:
            logger.warning(f"Chat template not available, using fallback: {e}")
            if system_message:
                return f"<s>[INST] {system_message}\n\n{prompt} [/INST]"
            else:
                return f"<s>[INST] {prompt} [/INST]"
    
    def generate_response(self, prompt: str, system_message: str = None, 
                         max_new_tokens: int = None) -> str:
        """Generate response using local model with proper chat formatting"""
        if not self.pipeline:
            raise ValueError("Model not loaded properly")
        
        formatted_prompt = self._format_chat_template(prompt, system_message)
        
        try:
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_new_tokens or self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            response = outputs[0]['generated_text'].strip()
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"