# chatbot/ml/tfidf_build.py
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import scipy.sparse as sp

from .settings import MAPPING_PARQUET, TFIDF_VECT, TFIDF_MAT

def main():
    assert Path(MAPPING_PARQUET).exists(), f"Mapping não encontrado: {MAPPING_PARQUET}"
    df = pd.read_parquet(MAPPING_PARQUET)
    texts = df["text"].fillna("").astype(str).tolist()

    # TF-IDF simples e robusto para PT/EN
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=200_000,
        ngram_range=(1,2),
        token_pattern=r"(?u)\b\w\w+\b"
    )
    X = vectorizer.fit_transform(texts)

    TFIDF_VECT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, TFIDF_VECT)
    sp.save_npz(TFIDF_MAT, X)

    print(f"[OK] TF-IDF vectorizer: {TFIDF_VECT}")
    print(f"[OK] TF-IDF matrix:     {TFIDF_MAT}  shape={X.shape}")

if __name__ == "__main__":
    main()
