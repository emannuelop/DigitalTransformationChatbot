from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import requests
from . import settings

@dataclass
class GenerationConfig:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None        # AUMENTA p/ caber resposta
    stop: Optional[List[str]] = None
    timeout_s: Optional[int] = None
    retries: Optional[int] = None
    backoff_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.temperature is None:
            self.temperature = settings.GEN_TEMPERATURE
        if self.top_p is None:
            self.top_p = settings.GEN_TOP_P
        if self.max_tokens is None:
            self.max_tokens = settings.GEN_MAX_TOKENS
        if self.timeout_s is None:
            self.timeout_s = settings.GEN_TIMEOUT_S
        if self.retries is None:
            self.retries = settings.GEN_RETRIES
        if self.backoff_s is None:
            self.backoff_s = settings.GEN_BACKOFF_S
        if self.stop is None and settings.GEN_STOP:
            self.stop = list(settings.GEN_STOP)

class LMStudioBackend:
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
        }
        if cfg.stop:
            payload["stop"] = cfg.stop  # <-- garante que stop vai

        last_err = None
        attempts = max(1, int(cfg.retries or 1))
        connect_timeout = min(8, cfg.timeout_s) if cfg.timeout_s else 8
        request_timeout = max(1, cfg.timeout_s or settings.GEN_TIMEOUT_S)

        for _ in range(attempts):
            try:
                r = requests.post(
                    self.url,
                    json=payload,
                    timeout=(connect_timeout, request_timeout),
                )
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
            # pequeno backoff
            import time
            time.sleep(max(0.0, cfg.backoff_s or 0.0))

        if last_err:
            raise last_err
        return ""
