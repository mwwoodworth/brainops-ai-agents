#!/usr/bin/env python3
"""
Hugging Face AI Alternative
Using Inference API for when OpenAI/Anthropic have issues
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

# Hugging Face configuration
HF_API_URL = "https://api-inference.huggingface.co/models/"
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

class HuggingFaceAI:
    """Hugging Face AI implementation"""

    def __init__(self):
        self.headers = {}
        if HF_TOKEN:
            self.headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            logger.info("Hugging Face API configured")
        else:
            logger.warning("No Hugging Face token found")

    def generate(self, prompt: str, model: str = "microsoft/DialoGPT-medium", max_tokens: int = 200):
        """Generate text using Hugging Face"""

        # Model mapping
        models = {
            "default": "microsoft/DialoGPT-medium",
            "gpt2": "gpt2-large",
            "falcon": "tiiuae/falcon-7b-instruct",
            "llama": "meta-llama/Llama-2-7b-chat-hf",
            "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
            "zephyr": "HuggingFaceH4/zephyr-7b-beta"
        }

        # Select model
        if model in models:
            model_id = models[model]
        else:
            model_id = models["default"]

        try:
            # Call Hugging Face Inference API
            response = requests.post(
                f"{HF_API_URL}{model_id}",
                headers=self.headers,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "do_sample": True
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "No response")
                return str(result)
            else:
                logger.error(f"HF API error: {response.status_code} - {response.text}")
                return f"Error: {response.status_code}"

        except Exception as e:
            logger.error(f"Hugging Face error: {e}")
            return f"Error: {str(e)}"

    def generate_embeddings(self, text: str):
        """Generate embeddings using Hugging Face"""

        try:
            response = requests.post(
                f"{HF_API_URL}sentence-transformers/all-MiniLM-L6-v2",
                headers=self.headers,
                json={"inputs": text},
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            return None

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None

# Global instance
hf_ai = HuggingFaceAI()

# Test function
def test_huggingface():
    """Test Hugging Face generation"""
    print("Testing Hugging Face AI...")

    # Set token for testing
    os.environ["HUGGINGFACE_API_TOKEN"] = "hf_YOUR_TOKEN_HERE"

    ai = HuggingFaceAI()
    result = ai.generate("What is the capital of France?", model="default")
    print(f"Result: {result}")

if __name__ == "__main__":
    test_huggingface()
