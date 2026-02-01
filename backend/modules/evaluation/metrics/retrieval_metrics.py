"""Retrieval quality metrics for RAG evaluation."""
import math
from typing import List


class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate Precision@K.

        Precision@K measures the proportion of retrieved documents in top-K
        that are relevant.

        Args:
            retrieved: List of retrieved document IDs in ranked order
            relevant: List of relevant document IDs (ground truth)
            k: Number of top results to consider

        Returns:
            Precision@K score between 0 and 1
        """
        if k <= 0 or not retrieved:
            return 0.0

        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)

        relevant_retrieved = len([doc_id for doc_id in retrieved_k if doc_id in relevant_set])

        return relevant_retrieved / k

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate Recall@K.

        Recall@K measures the proportion of all relevant documents
        that appear in top-K retrieved results.

        Args:
            retrieved: List of retrieved document IDs in ranked order
            relevant: List of relevant document IDs (ground truth)
            k: Number of top results to consider

        Returns:
            Recall@K score between 0 and 1
        """
        if not relevant or k <= 0:
            return 0.0

        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)

        relevant_retrieved = len([doc_id for doc_id in retrieved_k if doc_id in relevant_set])

        return relevant_retrieved / len(relevant_set)

    @staticmethod
    def mrr(retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Mean Reciprocal Rank.

        MRR measures where the first relevant document appears in the ranking.
        Score is 1/rank of first relevant document.

        Args:
            retrieved: List of retrieved document IDs in ranked order
            relevant: List of relevant document IDs (ground truth)

        Returns:
            MRR score between 0 and 1
        """
        relevant_set = set(relevant)

        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def ndcg(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain.

        NDCG measures ranking quality with position-based discounting.
        Higher ranked relevant documents contribute more to the score.

        Args:
            retrieved: List of retrieved document IDs in ranked order
            relevant: List of relevant document IDs (ground truth)
            k: Number of top results to consider

        Returns:
            NDCG@K score between 0 and 1
        """
        if not relevant or k <= 0:
            return 0.0

        relevant_set = set(relevant)
        retrieved_k = retrieved[:k]

        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_k, start=1):
            if doc_id in relevant_set:
                # rel_i = 1 for relevant, 0 for not relevant
                dcg += 1.0 / math.log2(rank + 1)

        # Calculate IDCG (Ideal DCG)
        # Ideal ranking has all relevant docs at top positions
        ideal_ranks = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_ranks + 1))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def average_precision(retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Average Precision.

        AP measures precision at each relevant document position,
        averaged over all relevant documents.

        Args:
            retrieved: List of retrieved document IDs in ranked order
            relevant: List of relevant document IDs (ground truth)

        Returns:
            Average Precision score between 0 and 1
        """
        if not relevant:
            return 0.0

        relevant_set = set(relevant)
        num_relevant_seen = 0
        sum_precisions = 0.0

        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                num_relevant_seen += 1
                precision_at_rank = num_relevant_seen / rank
                sum_precisions += precision_at_rank

        if num_relevant_seen == 0:
            return 0.0

        return sum_precisions / len(relevant_set)

    @staticmethod
    def f1_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate F1 score at K.

        F1@K is the harmonic mean of Precision@K and Recall@K.

        Args:
            retrieved: List of retrieved document IDs in ranked order
            relevant: List of relevant document IDs (ground truth)
            k: Number of top results to consider

        Returns:
            F1@K score between 0 and 1
        """
        precision = RetrievalMetrics.precision_at_k(retrieved, relevant, k)
        recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)
