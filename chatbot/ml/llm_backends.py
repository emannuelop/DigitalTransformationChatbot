from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import time
import requests

@dataclass
class GenerationConfig:
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 600
    stop: Optional[List[str]] = None
    timeout_s: int = 240
    retries: int = 2
    backoff_s: float = 1.5

class LMStudioBackend:
    """
    Cliente mínimo para /v1/chat/completions do LM Studio.
    """
    def __init__(self, model: str, host: str):
        self.model = model
        self.base = host.rstrip("/")
        self.url = f"{self.base}/v1/chat/completions"
        self.models_url = f"{self.base}/v1/models"

    def health(self) -> Dict[str, Any]:
        try:
            r = requests.get(self.models_url, timeout=8)
            if not r.ok:
                return {"error": f"HTTP {r.status_code}", "detail": r.text}
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def warm_up(self, timeout_s: int = 40) -> None:
        """
        Dispara um prompt curtíssimo só para carregar o modelo (cold start).
        Ignora erros silenciosamente para não travar o fluxo.
        """
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "ok"}],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 1,
                "stream": False,
            }
            requests.post(self.url, json=payload, timeout=timeout_s)
        except Exception:
            pass

    def generate(self, prompt: str, system: Optional[str] = None,
                 cfg: GenerationConfig = GenerationConfig()) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_tokens,
            "stream": False,
        }
        if cfg.stop:
            payload["stop"] = cfg.stop

        last_err = None
        for _ in range(max(1, cfg.retries)):
            try:
                r = requests.post(self.url, json=payload, timeout=cfg.timeout_s)
                if r.ok:
                    data = r.json()
                    msg = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not msg:
                        msg = data.get("choices", [{}])[0].get("text", "")
                    return (msg or "").strip()
                else:
                    try:
                        detail = r.json()
                    except Exception:
                        detail = r.text
                    last_err = RuntimeError(f"LM Studio error {r.status_code}: {detail}")
            except requests.RequestException as e:
                last_err = RuntimeError(f"LM Studio connection error: {e}")
            time.sleep(cfg.backoff_s)

        if last_err:
            raise last_err
        return ""
