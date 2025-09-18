# chatbot/ml/llm_backends.py
from dataclasses import dataclass
from typing import Optional
import requests

@dataclass
class GenerationConfig:
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 512
    stop: Optional[list[str]] = None

class OllamaBackend:
    def __init__(self, model: str = "phi4-mini:latest", host: str = "http://localhost:11434"):
        self.model = model
        self.url = f"{host}/api/generate"

    def generate(self, prompt: str, cfg: GenerationConfig) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # desliga streaming: retorna um único JSON
            "options": {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "num_predict": cfg.max_tokens,
            }
        }
        if cfg.stop:
            payload["stop"] = cfg.stop
        r = requests.post(self.url, json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
