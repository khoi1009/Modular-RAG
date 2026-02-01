"""Semantic similarity metrics using embeddings."""
from typing import List, Optional

import numpy as np

from backend.modules.model_gateway.model_gateway import model_gateway
from backend.modules.model_gateway.types import EmbedderConfig


class SemanticSimilarity:
    """Compute semantic similarity between texts using embeddings."""

    def __init__(self, embedder_config: Optional[EmbedderConfig] = None):
        """Initialize semantic similarity calculator.

        Args:
            embedder_config: Configuration for embedding model
        """
        self.embedder_config = embedder_config
        self.embedder = None

    async def _get_embedder(self):
        """Lazy load embedder instance."""
        if self.embedder is None:
            if self.embedder_config:
                self.embedder = model_gateway.get_embedder_from_config(
                    self.embedder_config
                )
            else:
                # Use default embedder from config
                self.embedder = model_gateway.get_default_embedder()
        return self.embedder

    async def cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score between -1 and 1
        """
        embedder = await self._get_embedder()

        # Get embeddings
        embedding1 = await embedder.aembed_query(text1)
        embedding2 = await embedder.aembed_query(text2)

        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        return float(similarity)

    async def bert_score(self, predicted: str, ground_truth: str) -> float:
        """Calculate BERTScore using embedding similarity.

        Simplified version that computes sentence-level similarity.

        Args:
            predicted: Generated answer
            ground_truth: Reference answer

        Returns:
            BERTScore (F1) between 0 and 1
        """
        similarity = await self.cosine_similarity(predicted, ground_truth)

        # Normalize from [-1, 1] to [0, 1]
        normalized = (similarity + 1) / 2

        return normalized

    async def semantic_f1(
        self,
        predicted: str,
        ground_truth: str,
        threshold: float = 0.7
    ) -> float:
        """Calculate semantic F1 using token-level embedding similarity.

        Args:
            predicted: Generated answer
            ground_truth: Reference answer
            threshold: Similarity threshold for considering tokens matched

        Returns:
            Semantic F1 score between 0 and 1
        """
        pred_tokens = predicted.split()
        gt_tokens = ground_truth.split()

        if not pred_tokens or not gt_tokens:
            return 0.0

        embedder = await self._get_embedder()

        # Get embeddings for all tokens
        pred_embeddings = []
        for token in pred_tokens:
            emb = await embedder.aembed_query(token)
            pred_embeddings.append(np.array(emb))

        gt_embeddings = []
        for token in gt_tokens:
            emb = await embedder.aembed_query(token)
            gt_embeddings.append(np.array(emb))

        # Find matches using similarity threshold
        pred_matches = 0
        for pred_emb in pred_embeddings:
            max_sim = 0.0
            for gt_emb in gt_embeddings:
                sim = self._cosine_sim_vectors(pred_emb, gt_emb)
                max_sim = max(max_sim, sim)

            if max_sim >= threshold:
                pred_matches += 1

        gt_matches = 0
        for gt_emb in gt_embeddings:
            max_sim = 0.0
            for pred_emb in pred_embeddings:
                sim = self._cosine_sim_vectors(pred_emb, gt_emb)
                max_sim = max(max_sim, sim)

            if max_sim >= threshold:
                gt_matches += 1

        precision = pred_matches / len(pred_tokens) if pred_tokens else 0
        recall = gt_matches / len(gt_tokens) if gt_tokens else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)

        return f1

    @staticmethod
    def _cosine_sim_vectors(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    async def batch_similarity(
        self,
        texts1: List[str],
        texts2: List[str]
    ) -> List[float]:
        """Calculate pairwise similarities for batches of texts.

        Args:
            texts1: First list of texts
            texts2: Second list of texts (same length as texts1)

        Returns:
            List of similarity scores
        """
        if len(texts1) != len(texts2):
            raise ValueError("texts1 and texts2 must have same length")

        similarities = []

        for text1, text2 in zip(texts1, texts2):
            sim = await self.cosine_similarity(text1, text2)
            similarities.append(sim)

        return similarities
