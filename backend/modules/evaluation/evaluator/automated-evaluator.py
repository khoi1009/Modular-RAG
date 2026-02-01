"""Automated evaluator for running datasets through RAG pipelines."""
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional

from backend.modules.evaluation.datasets.schemas import (
    EvaluationDataset,
    EvaluationReport,
    EvaluationResult,
)
from backend.modules.evaluation.metrics.generation_metrics import GenerationMetrics
from backend.modules.evaluation.metrics.llm_judge import LLMJudge
from backend.modules.evaluation.metrics.retrieval_metrics import RetrievalMetrics
from backend.modules.evaluation.metrics.semantic_similarity import SemanticSimilarity
from backend.modules.model_gateway.types import ModelConfig


class AutomatedEvaluator:
    """Runs evaluation datasets through RAG pipeline and computes metrics."""

    def __init__(
        self,
        pipeline_executor=None,
        llm_judge_config: Optional[ModelConfig] = None,
        enable_semantic_similarity: bool = True,
        max_concurrent: int = 10
    ):
        """Initialize automated evaluator.

        Args:
            pipeline_executor: Pipeline executor for running queries
            llm_judge_config: Optional config for LLM-as-judge evaluation
            enable_semantic_similarity: Whether to compute semantic similarity
            max_concurrent: Maximum concurrent evaluations
        """
        self.pipeline_executor = pipeline_executor
        self.llm_judge = LLMJudge(llm_judge_config) if llm_judge_config else None
        self.semantic_sim = SemanticSimilarity() if enable_semantic_similarity else None
        self.max_concurrent = max_concurrent

    async def evaluate_dataset(
        self,
        dataset: EvaluationDataset,
        pipeline_config_path: str,
        collection_name: str,
        output_dir: Optional[str] = None
    ) -> EvaluationReport:
        """Evaluate entire dataset through pipeline.

        Args:
            dataset: Evaluation dataset to run
            pipeline_config_path: Path to pipeline configuration YAML
            collection_name: Collection to query against
            output_dir: Optional directory to save detailed results

        Returns:
            Evaluation report with aggregated metrics
        """
        results = []

        # Load pipeline config
        pipeline_def = self._load_pipeline_config(pipeline_config_path)

        # Process samples with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def evaluate_sample(sample):
            async with semaphore:
                return await self._evaluate_single_sample(
                    sample,
                    pipeline_def,
                    collection_name
                )

        # Run evaluations concurrently
        tasks = [evaluate_sample(sample) for sample in dataset.samples]
        results = await asyncio.gather(*tasks)

        # Compute aggregate metrics
        aggregate_metrics = self._aggregate_metrics(results)

        report = EvaluationReport(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            pipeline_config=pipeline_config_path,
            collection_name=collection_name,
            results=results,
            aggregate_metrics=aggregate_metrics,
            total_samples=len(results),
        )

        # Save detailed results if output dir specified
        if output_dir:
            self._save_report(report, output_dir)

        return report

    async def _evaluate_single_sample(
        self,
        sample,
        pipeline_def,
        collection_name: str
    ) -> EvaluationResult:
        """Evaluate a single sample.

        Args:
            sample: EvaluationSample to evaluate
            pipeline_def: Pipeline definition
            collection_name: Collection name

        Returns:
            Evaluation result with metrics
        """
        start_time = time.time()

        try:
            # Execute pipeline
            pipeline_result = await self._execute_pipeline(
                pipeline_def,
                sample.query,
                collection_name
            )

            predicted_answer = pipeline_result.get("answer", "")
            retrieved_sources = self._extract_source_ids(
                pipeline_result.get("sources", [])
            )

        except Exception as e:
            # Handle pipeline failures gracefully
            predicted_answer = f"[ERROR: {str(e)}]"
            retrieved_sources = []

        latency_ms = int((time.time() - start_time) * 1000)

        # Compute metrics
        metrics = await self._compute_metrics(
            sample,
            predicted_answer,
            retrieved_sources
        )

        return EvaluationResult(
            sample_id=sample.id,
            query=sample.query,
            predicted_answer=predicted_answer,
            ground_truth_answer=sample.ground_truth_answer,
            retrieved_sources=retrieved_sources,
            metrics=metrics,
            latency_ms=latency_ms,
        )

    async def _compute_metrics(
        self,
        sample,
        predicted_answer: str,
        retrieved_sources: List[str]
    ) -> Dict[str, float]:
        """Compute all metrics for a sample.

        Args:
            sample: Original evaluation sample
            predicted_answer: Generated answer
            retrieved_sources: Retrieved document IDs

        Returns:
            Dictionary of metric scores
        """
        metrics = {}

        # Generation metrics
        metrics["exact_match"] = GenerationMetrics.exact_match(
            predicted_answer,
            sample.ground_truth_answer
        )

        metrics["f1"] = GenerationMetrics.f1_score(
            predicted_answer,
            sample.ground_truth_answer
        )

        metrics["bleu"] = GenerationMetrics.bleu_score(
            predicted_answer,
            sample.ground_truth_answer
        )

        rouge_scores = GenerationMetrics.rouge_score(
            predicted_answer,
            sample.ground_truth_answer
        )
        metrics.update({f"rouge_{k.split('-')[1]}": v for k, v in rouge_scores.items()})

        # Retrieval metrics (if ground truth sources available)
        if sample.ground_truth_sources and sample.requires_retrieval:
            metrics["precision@5"] = RetrievalMetrics.precision_at_k(
                retrieved_sources,
                sample.ground_truth_sources,
                5
            )

            metrics["recall@5"] = RetrievalMetrics.recall_at_k(
                retrieved_sources,
                sample.ground_truth_sources,
                5
            )

            metrics["mrr"] = RetrievalMetrics.mrr(
                retrieved_sources,
                sample.ground_truth_sources
            )

            metrics["ndcg@10"] = RetrievalMetrics.ndcg(
                retrieved_sources,
                sample.ground_truth_sources,
                10
            )

        # Semantic similarity
        if self.semantic_sim:
            metrics["semantic_similarity"] = await self.semantic_sim.cosine_similarity(
                predicted_answer,
                sample.ground_truth_answer
            )

        # LLM-as-judge scores
        if self.llm_judge:
            judge_scores = await self.llm_judge.evaluate(
                query=sample.query,
                predicted=predicted_answer,
                ground_truth=sample.ground_truth_answer,
                sources=retrieved_sources[:3]
            )
            metrics.update({f"judge_{k}": v for k, v in judge_scores.items()})

        return metrics

    def _aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Aggregate metrics across all results.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary of aggregated metric values
        """
        if not results:
            return {}

        # Collect all metric keys
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())

        # Compute averages
        aggregated = {}
        for metric_name in all_metrics:
            values = [
                result.metrics.get(metric_name, 0.0)
                for result in results
                if metric_name in result.metrics
            ]

            if values:
                aggregated[f"{metric_name}_mean"] = sum(values) / len(values)
                aggregated[f"{metric_name}_min"] = min(values)
                aggregated[f"{metric_name}_max"] = max(values)

        # Add latency statistics
        latencies = [result.latency_ms for result in results]
        aggregated["latency_mean_ms"] = sum(latencies) / len(latencies)
        aggregated["latency_p50_ms"] = self._percentile(latencies, 50)
        aggregated["latency_p95_ms"] = self._percentile(latencies, 95)
        aggregated["latency_p99_ms"] = self._percentile(latencies, 99)

        return aggregated

    @staticmethod
    def _percentile(values: List[float], percentile: int) -> float:
        """Calculate percentile of values.

        Args:
            values: List of numeric values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    async def _execute_pipeline(
        self,
        pipeline_def,
        query: str,
        collection_name: str
    ) -> Dict:
        """Execute pipeline for a query.

        Args:
            pipeline_def: Pipeline definition
            query: User query
            collection_name: Collection to query

        Returns:
            Pipeline execution result
        """
        if self.pipeline_executor:
            result = await self.pipeline_executor.execute(
                pipeline_def,
                {"query": query, "collection_name": collection_name}
            )
            return {
                "answer": result.answer,
                "sources": result.sources,
            }
        else:
            # Mock execution for testing
            return {
                "answer": "Mock answer",
                "sources": []
            }

    @staticmethod
    def _extract_source_ids(sources: List) -> List[str]:
        """Extract document IDs from source objects.

        Args:
            sources: List of source documents

        Returns:
            List of document ID strings
        """
        source_ids = []

        for source in sources:
            if isinstance(source, dict):
                doc_id = source.get("metadata", {}).get("_id") or source.get("id")
            else:
                doc_id = getattr(source, "id", None) or getattr(
                    getattr(source, "metadata", {}), "_id", None
                )

            if doc_id:
                source_ids.append(str(doc_id))

        return source_ids

    def _load_pipeline_config(self, config_path: str):
        """Load pipeline configuration from YAML file.

        Args:
            config_path: Path to pipeline config

        Returns:
            Pipeline definition object
        """
        import yaml

        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Pipeline config not found: {config_path}")

        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)

        return config_data

    def _save_report(self, report: EvaluationReport, output_dir: str):
        """Save evaluation report to file.

        Args:
            report: Evaluation report to save
            output_dir: Directory to save report
        """
        import json

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_file = output_path / f"{report.dataset_name}_v{report.dataset_version}_report.json"

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2, default=str)
