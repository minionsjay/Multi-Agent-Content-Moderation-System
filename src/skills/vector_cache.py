import logging
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.config import CHROMA_PERSIST_DIR, CACHE_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

COLLECTION_NAME = "moderation_cache"


class VectorCache:
    """Semantic cache for content moderation results using ChromaDB."""

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self._client = None
        self._collection = None

    def _ensure_collection(self):
        if self._collection is not None:
            return
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection ready: %s (%d docs)",
                     COLLECTION_NAME, self._collection.count())

    def lookup(self, embedding: list[float]) -> dict | None:
        """Search for a semantically similar cached result.

        Returns cached decision dict if similarity > threshold, else None.
        """
        if not embedding or all(v == 0.0 for v in embedding):
            return None

        self._ensure_collection()
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["metadatas", "distances"],
        )

        if not results["ids"][0]:
            return None

        distance = results["distances"][0][0]
        # ChromaDB with cosine returns distance in [0, 2]; convert to similarity
        similarity = 1.0 - (distance / 2.0)

        if similarity >= CACHE_SIMILARITY_THRESHOLD:
            metadata = results["metadatas"][0][0]
            logger.info("Cache HIT (similarity=%.4f)", similarity)
            return {
                "decision": metadata["decision"],
                "confidence": metadata["confidence"],
                "reason": metadata.get("reason", ""),
                "similarity": similarity,
            }

        logger.debug("Cache MISS (similarity=%.4f < %.4f)", similarity, CACHE_SIMILARITY_THRESHOLD)
        return None

    def store(
        self,
        embedding: list[float],
        text: str,
        decision: str,
        confidence: float,
        reason: str,
    ):
        """Store a moderation result in the cache."""
        if not embedding or all(v == 0.0 for v in embedding):
            return

        self._ensure_collection()
        doc_id = f"mod_{hash(text) & 0x7FFFFFFF:08x}"
        self._collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text[:2000]],  # Truncate for storage
            metadatas=[{
                "decision": decision,
                "confidence": confidence,
                "reason": reason,
            }],
        )

    def invalidate(self, text: str):
        """Remove a cached entry (called when human review overrides)."""
        self._ensure_collection()
        doc_id = f"mod_{hash(text) & 0x7FFFFFFF:08x}"
        self._collection.delete(ids=[doc_id])

    def count(self) -> int:
        self._ensure_collection()
        return self._collection.count()


# Singleton
vector_cache = VectorCache()
