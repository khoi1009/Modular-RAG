"""Generation quality metrics for RAG answer evaluation."""
import re
from collections import Counter
from typing import Dict, List


class GenerationMetrics:
    """Metrics for evaluating generated answer quality."""

    @staticmethod
    def exact_match(predicted: str, ground_truth: str, ignore_case: bool = True) -> float:
        """Calculate exact match score.

        Args:
            predicted: Generated answer
            ground_truth: Expected answer
            ignore_case: Whether to ignore case differences

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        pred = predicted.strip()
        gt = ground_truth.strip()

        if ignore_case:
            pred = pred.lower()
            gt = gt.lower()

        return 1.0 if pred == gt else 0.0

    @staticmethod
    def f1_score(predicted: str, ground_truth: str, ignore_case: bool = True) -> float:
        """Calculate token-level F1 score.

        Measures overlap between predicted and ground truth tokens.

        Args:
            predicted: Generated answer
            ground_truth: Expected answer
            ignore_case: Whether to ignore case differences

        Returns:
            F1 score between 0 and 1
        """
        pred = predicted.strip()
        gt = ground_truth.strip()

        if ignore_case:
            pred = pred.lower()
            gt = gt.lower()

        pred_tokens = pred.split()
        gt_tokens = gt.split()

        if not pred_tokens or not gt_tokens:
            return 0.0

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)

        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def bleu_score(predicted: str, ground_truth: str, n: int = 4) -> float:
        """Calculate BLEU score.

        BLEU measures n-gram overlap with brevity penalty.

        Args:
            predicted: Generated answer
            ground_truth: Expected answer
            n: Maximum n-gram size to consider

        Returns:
            BLEU score between 0 and 1
        """
        from math import exp, log

        pred_tokens = predicted.lower().split()
        ref_tokens = ground_truth.lower().split()

        if not pred_tokens:
            return 0.0

        # Brevity penalty
        bp = 1.0
        if len(pred_tokens) < len(ref_tokens):
            bp = exp(1 - len(ref_tokens) / len(pred_tokens))

        # Calculate n-gram precisions
        precisions = []
        for i in range(1, n + 1):
            pred_ngrams = GenerationMetrics._get_ngrams(pred_tokens, i)
            ref_ngrams = GenerationMetrics._get_ngrams(ref_tokens, i)

            if not pred_ngrams:
                continue

            matches = sum((Counter(pred_ngrams) & Counter(ref_ngrams)).values())
            precision = matches / len(pred_ngrams) if pred_ngrams else 0

            if precision > 0:
                precisions.append(log(precision))

        if not precisions:
            return 0.0

        # Geometric mean of precisions
        avg_precision = exp(sum(precisions) / len(precisions))

        return bp * avg_precision

    @staticmethod
    def rouge_score(predicted: str, ground_truth: str) -> Dict[str, float]:
        """Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).

        ROUGE measures recall-oriented n-gram overlap.

        Args:
            predicted: Generated answer
            ground_truth: Expected answer

        Returns:
            Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
        """
        pred_tokens = predicted.lower().split()
        ref_tokens = ground_truth.lower().split()

        if not pred_tokens or not ref_tokens:
            return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}

        # ROUGE-1 (unigram overlap)
        rouge_1 = GenerationMetrics._rouge_n(pred_tokens, ref_tokens, 1)

        # ROUGE-2 (bigram overlap)
        rouge_2 = GenerationMetrics._rouge_n(pred_tokens, ref_tokens, 2)

        # ROUGE-L (Longest Common Subsequence)
        rouge_l = GenerationMetrics._rouge_l(pred_tokens, ref_tokens)

        return {
            "rouge-1": rouge_1,
            "rouge-2": rouge_2,
            "rouge-l": rouge_l,
        }

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> List[tuple]:
        """Extract n-grams from token list."""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    @staticmethod
    def _rouge_n(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        """Calculate ROUGE-N score."""
        pred_ngrams = GenerationMetrics._get_ngrams(pred_tokens, n)
        ref_ngrams = GenerationMetrics._get_ngrams(ref_tokens, n)

        if not ref_ngrams:
            return 0.0

        matches = sum((Counter(pred_ngrams) & Counter(ref_ngrams)).values())
        recall = matches / len(ref_ngrams)

        return recall

    @staticmethod
    def _rouge_l(pred_tokens: List[str], ref_tokens: List[str]) -> float:
        """Calculate ROUGE-L score using LCS."""
        lcs_length = GenerationMetrics._lcs_length(pred_tokens, ref_tokens)

        if not ref_tokens:
            return 0.0

        recall = lcs_length / len(ref_tokens)
        precision = lcs_length / len(pred_tokens) if pred_tokens else 0

        if recall + precision == 0:
            return 0.0

        f1 = 2 * recall * precision / (recall + precision)
        return f1

    @staticmethod
    def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    @staticmethod
    def contains_answer(predicted: str, ground_truth: str, ignore_case: bool = True) -> float:
        """Check if predicted answer contains the ground truth.

        Args:
            predicted: Generated answer
            ground_truth: Expected answer
            ignore_case: Whether to ignore case differences

        Returns:
            1.0 if ground truth is substring of predicted, 0.0 otherwise
        """
        pred = predicted.strip()
        gt = ground_truth.strip()

        if ignore_case:
            pred = pred.lower()
            gt = gt.lower()

        return 1.0 if gt in pred else 0.0
