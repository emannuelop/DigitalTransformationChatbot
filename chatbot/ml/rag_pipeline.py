from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
import faiss
import joblib, scipy.sparse as sp

from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from .settings import (
    FAISS_INDEX, MAPPING_PARQUET,
    TOP_K, MAX_CONTEXT_TOKENS, OLLAMA_MODEL,
    TFIDF_VECT, TFIDF_MAT
)
from .llm_backends import OllamaBackend, GenerationConfig

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def _approx_token_len(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))

def load_index_and_mapping():
    assert Path(FAISS_INDEX).exists(), "FAISS index não encontrado."
    assert Path(MAPPING_PARQUET).exists(), "Mapping não encontrado."
    index = faiss.read_index(str(FAISS_INDEX))
    mapping = pd.read_parquet(MAPPING_PARQUET)
    # garantias de tipos
    mapping["text"] = mapping["text"].fillna("").astype(str)
    if "lang" in mapping.columns:
        mapping["lang"] = mapping["lang"].fillna("").astype(str)
    return index, mapping

# cache global do encoder de consulta
_SBERT_Q = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def encode_query(query: str) -> np.ndarray:
    v = _SBERT_Q.encode([query], normalize_embeddings=True)
    return v.astype(np.float32)

def _looks_like_english(s: str) -> bool:
    """Heurística leve caso mapping não tenha 'lang'."""
    s2 = s.lower()
    hits = sum(1 for w in ("the ", "and ", "of ", "in ", "to ", "for ", "with ", "on ", "from ") if w in s2)
    pt_markers = sum(1 for w in (" de ", " em ", " para ", " com ", " por ", " que ", " os ", " as ") if w in s2)
    return hits > pt_markers

# ------------------------------------------------------------
# TF-IDF Reranker
# ------------------------------------------------------------
class TfidfReranker:
    def __init__(self, vec_path, mat_path):
        self.vectorizer = joblib.load(vec_path)
        self.matrix = sp.load_npz(mat_path)
    def rerank(self, query, idxs):
        q = self.vectorizer.transform([query])
        sub = self.matrix[idxs, :]
        scores = (sub @ q.T).toarray().ravel()
        order = np.argsort(-scores)
        return [idxs[i] for i in order]

# ------------------------------------------------------------
# RAG
# ------------------------------------------------------------
class RAGChatbot:
    def __init__(self):
        self.index, self.mapping = load_index_and_mapping()
        self.backend = OllamaBackend(model=OLLAMA_MODEL)

        # Controle de tradução
        self.translate_non_pt = True
        self.max_translate_chars = 4000  # segurança para prompts

        # Reranker com checagem de alinhamento
        self.reranker = None
        if TFIDF_VECT.exists() and TFIDF_MAT.exists():
            try:
                tmp = TfidfReranker(TFIDF_VECT, TFIDF_MAT)
                if tmp.matrix.shape[0] == len(self.mapping):
                    self.reranker = tmp
                    print("[DEBUG] Reranker TF-IDF ativado.")
                else:
                    print(f"[WARN] TF-IDF desalinhado (matrix rows={tmp.matrix.shape[0]} vs mapping={len(self.mapping)}). Desativando reranker.")
            except Exception as e:
                print(f"[WARN] Falha ao carregar TF-IDF: {e}. Desativando reranker.")

        # DEBUG de distribuição de tamanhos
        lens = self.mapping["text"].str.len()
        print(f"[DEBUG] mapping rows={len(self.mapping)} | text.len min/median/mean/max = "
              f"{lens.min()}/{lens.median()}/{lens.mean():.1f}/{lens.max()}")

    # --------------------- Tradução ---------------------
    def translate_to_pt(self, text: str) -> str:
        """Traduz para PT-BR usando o backend LLM, sem resumir."""
        if not text:
            return text
        # Limitar tamanho para o prompt de tradução
        chunk = text[: self.max_translate_chars]
        sys_msg = (
            "Você é um tradutor profissional PT-BR. "
            "Traduza fielmente o texto para português do Brasil, mantendo estrutura, listas, títulos e termos técnicos. "
            "Não resuma, não explique; responda apenas com o texto traduzido."
        )
        user_msg = f"Texto a traduzir:\n{chunk}"
        prompt = f"<|system|>\n{sys_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"
        cfg = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=2048)
        return self.backend.generate(prompt, cfg).strip()

    def _needs_translation(self, row: Dict) -> bool:
        if not self.translate_non_pt:
            return False
        lang = (row.get("lang") or "").lower()
        if lang:
            return lang not in ("pt", "pt-br", "pt_br")
        # fallback heurístico se 'lang' não existir
        return _looks_like_english(row.get("text", ""))

    # --------------------- Retrieve ---------------------
    def retrieve(self, query: str, top_k: int = TOP_K) -> List[int]:
        q = encode_query(query)
        D, I = self.index.search(q, top_k)
        ids = I[0].tolist()
        if self.reranker:
            ids = self.reranker.rerank(query, ids)
        return ids

    # --------------------- Contexto ---------------------
    def _select_docs_by_ids(self, doc_ids: List[int], limit_tokens: int) -> Tuple[str, List[Dict]]:
        selected = []
        used = 0
        max_chars = limit_tokens * 4

        for idx in doc_ids:
            if idx < 0 or idx >= len(self.mapping):
                continue
            row = self.mapping.iloc[idx]
            text = (row.get("text", "") or "").strip()
            if not text:
                continue

            tlen = _approx_token_len(text)
            if tlen > limit_tokens:
                # cabe só um pedaço; se nada selecionado ainda, pegue fatia
                if not selected:
                    slice_text = text[:max_chars]
                    item = {**row.to_dict(), "text": slice_text, "_translated": False}
                    # traduz se necessário
                    if self._needs_translation(item):
                        try:
                            t = self.translate_to_pt(slice_text)
                            item["text"] = t
                            item["_translated"] = True
                        except Exception as e:
                            print(f"[WARN] Falha ao traduzir (id={idx}): {e}")
                    selected.append(item)
                    used = limit_tokens
                continue

            if used + tlen <= limit_tokens:
                item = row.to_dict()
                item["_translated"] = False
                # traduz se necessário
                if self._needs_translation(item):
                    try:
                        t = self.translate_to_pt(item["text"])
                        item["text"] = t
                        item["_translated"] = True
                    except Exception as e:
                        print(f"[WARN] Falha ao traduzir (id={idx}): {e}")
                selected.append(item)
                used += tlen
            else:
                remaining = (limit_tokens - used) * 4
                if remaining > 0 and not selected:
                    slice_text = text[:remaining]
                    item = {**row.to_dict(), "text": slice_text, "_translated": False}
                    if self._needs_translation(item):
                        try:
                            t = self.translate_to_pt(slice_text)
                            item["text"] = t
                            item["_translated"] = True
                        except Exception as e:
                            print(f"[WARN] Falha ao traduzir (id={idx}): {e}")
                    selected.append(item)
                    used = limit_tokens
                continue

        parts = []
        for i, it in enumerate(selected, 1):
            translated_tag = " [traduzido]" if it.get("_translated") else ""
            head = f"[Fonte {i}] {it.get('title') or '(sem título)'}{translated_tag} | {it.get('url') or 'sem URL'}"
            parts.append(f"{head}\n{it['text']}\n")
        return "\n\n".join(parts), selected

    def build_context(self, doc_ids: List[int], limit_tokens: int = MAX_CONTEXT_TOKENS) -> Tuple[str, List[Dict]]:
        context, selected = self._select_docs_by_ids(doc_ids, limit_tokens)
        if selected:
            return context, selected

        # Fallback 1: maiores textos
        print("[WARN] Nenhum texto válido. Fallback por comprimento.")
        lens = self.mapping["text"].str.len()
        fallback_ids = lens.sort_values(ascending=False).index[:min(10, len(self.mapping))].tolist()
        context, selected = self._select_docs_by_ids(fallback_ids, limit_tokens)
        if selected:
            return context, selected

        # Fallback 2: primeiro texto não-vazio
        for idx in range(len(self.mapping)):
            t = (self.mapping.iloc[idx].get("text","") or "").strip()
            if t:
                print("[WARN] Usando primeiro texto não-vazio do mapping como último fallback.")
                return self._select_docs_by_ids([idx], limit_tokens)

        return "", []

    # --------------------- Prompt & Formatação ---------------------
    def build_prompt(self, question: str, context: str) -> str:
        system = (
            "Você é um assistente especializado em transformação digital. "
            "Responda em português, cite fontes como [Fonte N], "
            "e não invente informações que não estejam no contexto."
        )
        user = (
            f"PERGUNTA:\n{question}\n\n"
            f"CONTEXTO:\n{context}\n\n"
            "INSTRUÇÕES:\n"
            "- Responda em 1-3 parágrafos curtos.\n"
            "- Liste bullets quando fizer sentido.\n"
            "- Inclua seção 'Fontes' no fim."
        )
        return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"

    def format_with_sources(self, answer_text: str, used_docs: List[Dict]) -> str:
        if not used_docs:
            return answer_text + "\n\n_Fontes: (nenhuma fonte)_"
        lines = ["\n\nFontes:"]
        for i, d in enumerate(used_docs, 1):
            title = d.get("title") or "(sem título)"
            url = d.get("url") or "sem URL"
            lines.append(f"- [Fonte {i}] {title} — {url}")
        return answer_text.rstrip() + "\n" + "\n".join(lines)

    # --------------------- Pipeline principal ---------------------
    def answer(self, question: str) -> Dict[str, Any]:
        ids = self.retrieve(question)
        context, used = self.build_context(ids)

        # recall extra se contexto vazio
        if not used:
            print("[INFO] Tentando recall maior (top_k=20) por contexto vazio.")
            q = encode_query(question)
            D, I = self.index.search(q, min(20, len(self.mapping)))
            ids2 = I[0].tolist()
            if self.reranker:
                ids2 = self.reranker.rerank(question, ids2)
            context, used = self.build_context(ids2)

        print("==== CONTEXTO PASSADO ====")
        print(context[:800])
        print("==========================")
        prompt = self.build_prompt(question, context)
        cfg = GenerationConfig(temperature=0.15, top_p=0.9, max_tokens=512)
        text = self.backend.generate(prompt, cfg)
        final = self.format_with_sources(text, used)
        return {"answer": final, "used_docs": used}

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Quais são pilares comuns de transformação digital no setor público?"
    rag = RAGChatbot()
    out = rag.answer(q)
    print(out["answer"])
    print("\n[DEBUG] Fontes usadas:")
    for i, d in enumerate(out["used_docs"], 1):
        print(f"- [Fonte {i}] id={d.get('id')} | {d.get('title')} | {d.get('url')}")
