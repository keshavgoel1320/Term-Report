"""Dense embedding backend using sentence-transformers."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Prefixes for models that require them
_QUERY_PREFIXES: dict[str, str] = {
    "bge": "Represent this sentence for searching relevant passages: ",
    "e5": "query: ",
}
_DOCUMENT_PREFIXES: dict[str, str] = {
    "e5": "passage: ",
}


class EmbeddingBackend:
    """Wraps a sentence-transformer model for document and query embedding."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded (dim=%d)", self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def _get_prefix(self, prefix_map: dict[str, str]) -> str:
        model_lower = self.model_name.lower()
        for key, prefix in prefix_map.items():
            if key in model_lower:
                return prefix
        return ""

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of document texts, returns L2-normalised vectors."""
        prefix = self._get_prefix(_DOCUMENT_PREFIXES)
        prefixed = [prefix + t for t in texts] if prefix else texts
        vectors = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_queries(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of query texts, returns L2-normalised vectors."""
        prefix = self._get_prefix(_QUERY_PREFIXES)
        prefixed = [prefix + t for t in texts] if prefix else texts
        vectors = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_single_query(self, text: str) -> np.ndarray:
        """Embed a single query, returns a 1D L2-normalised vector."""
        return self.embed_queries([text])[0]
