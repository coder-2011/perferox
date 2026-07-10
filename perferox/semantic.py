"""Cached semantic index for the immutable packaged SGLang documentation."""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from perferox import db

DOCS_DB = Path(__file__).with_name("sglang_docs") / "perferox-docs.sqlite"
Document = tuple[str, str, str, str]


@dataclass(frozen=True, slots=True)
class DocumentIndex:
  """Keep documents aligned with one immutable float32 matrix."""

  documents: tuple[Document, ...]
  vectors: np.ndarray

  @classmethod
  def load(cls, path: str | Path) -> "DocumentIndex":
    """Load and freeze one SQLite document index."""
    with db.open_db(path, readonly=True) as conn:
      rows = conn.execute("SELECT source, title, url, text, embedding FROM doc_chunks").fetchall()
    documents = tuple((row["source"], row["title"], row["url"], row["text"]) for row in rows)
    vectors = np.asarray([json.loads(row["embedding"]) for row in rows], dtype=np.float32)
    vectors.setflags(write=False)
    return cls(documents, vectors)

  def search(self, query: list[float], limit: int) -> list[tuple[float, Document]]:
    """Return the highest-scoring rows with stable tie ordering."""
    if not self.documents:
      return []
    scores = self.vectors @ np.asarray(query, dtype=np.float32)
    indices = np.lexsort((np.arange(scores.size), -scores))[:limit]
    return [(float(scores[index]), self.documents[index]) for index in indices]


@cache
def document_index() -> DocumentIndex:
  """Load the immutable packaged documentation database once per process."""
  return DocumentIndex.load(DOCS_DB)
