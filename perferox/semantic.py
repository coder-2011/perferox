"""Cached semantic index for the immutable packaged SGLang documentation."""

import json
from functools import cache
from pathlib import Path

import numpy as np

from perferox import db

DOCS_DB = Path(__file__).with_name("sglang_docs") / "perferox-docs.sqlite"
Document = tuple[str, str, str, str]
DocumentIndex = tuple[tuple[Document, ...], np.ndarray]


@cache
def document_index(path: str | Path = DOCS_DB) -> DocumentIndex:
  """Load and cache documents beside one immutable float32 matrix."""
  with db.open_db(path, readonly=True) as conn:
    rows = conn.execute("SELECT source, title, url, text, embedding FROM doc_chunks").fetchall()
  documents = tuple((row["source"], row["title"], row["url"], row["text"]) for row in rows)
  vectors = np.asarray([json.loads(row["embedding"]) for row in rows], dtype=np.float32)
  vectors.setflags(write=False)
  return documents, vectors


def search_documents(query: list[float], limit: int, path: str | Path = DOCS_DB) -> list[tuple[float, Document]]:
  """Return the highest-scoring cached documents with stable tie ordering."""
  documents, vectors = document_index(path)
  if not documents:
    return []
  scores = vectors @ np.asarray(query, dtype=np.float32)
  indices = np.lexsort((np.arange(scores.size), -scores))[:limit]
  return [(float(scores[index]), documents[index]) for index in indices]
