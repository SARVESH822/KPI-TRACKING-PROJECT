"""
Local, dependency-light retrieval over the unstructured document set.

Uses TF-IDF + cosine similarity (scikit-learn) rather than an embedding model
or vector database. This is a deliberate choice for a prototype: it is fully
offline, has zero external dependencies or API calls, is fast to build and
explain, and is reliable for a fixed demo corpus of ~70-100 documents. Swapping
in a proper embedding model + vector index (see docs/architecture.md) is a
documented future improvement once corpus size and query variety justify it.
"""
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedDoc:
    document_id: str
    date: str
    region: str
    category: Optional[str]
    source: str
    title: str
    text: str
    score: float


class DocumentRetriever:
    def __init__(self, docs_df: pd.DataFrame):
        self.docs = docs_df.reset_index(drop=True)
        corpus = (self.docs["title"].fillna("") + ". " + self.docs["text"].fillna(""))
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=4000)
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, region: Optional[str] = None, category: Optional[str] = None,
               start=None, end=None, top_k: int = 10, min_score: float = 0.05) -> List[RetrievedDoc]:
        candidates = self.docs.copy()
        candidates["_row"] = candidates.index

        if region and region != "All Regions":
            candidates = candidates[candidates["region"] == region]
        if category and category != "All Categories":
            candidates = candidates[(candidates["category"] == category) | (candidates["category"].isna())]
        if start is not None and end is not None:
            d = pd.to_datetime(candidates["date"])
            candidates = candidates[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]

        if candidates.empty:
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix[candidates["_row"].values]).flatten()
        candidates = candidates.assign(score=sims)
        candidates = candidates[candidates["score"] >= min_score]
        candidates = candidates.sort_values("score", ascending=False).head(top_k)

        return [
            RetrievedDoc(
                document_id=r.document_id, date=str(r.date), region=r.region,
                category=(r.category if pd.notna(r.category) else None),
                source=r.source, title=r.title, text=r.text, score=float(r.score),
            )
            for r in candidates.itertuples()
        ]

    def count_by_topic(self, topic_query: str, region: Optional[str], category: Optional[str],
                        start, end, min_score: float = 0.08) -> int:
        return len(self.search(topic_query, region, category, start, end, top_k=1000, min_score=min_score))


def load_documents(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
