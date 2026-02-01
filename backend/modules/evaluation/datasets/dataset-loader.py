"""Loader utilities for evaluation datasets from various formats."""
import json
from pathlib import Path
from typing import List, Optional

from backend.modules.evaluation.datasets.schemas import (
    EvaluationDataset,
    EvaluationSample,
)


class DatasetLoader:
    """Utilities for loading evaluation datasets from various formats."""

    @staticmethod
    async def from_json_file(file_path: str) -> EvaluationDataset:
        """Load evaluation dataset from JSON file.

        Args:
            file_path: Path to JSON dataset file

        Returns:
            Loaded evaluation dataset
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return EvaluationDataset(**data)

    @staticmethod
    async def from_json_lines(file_path: str, name: str, version: str = "1") -> EvaluationDataset:
        """Load evaluation dataset from JSONL file format.

        Each line should be a JSON object with EvaluationSample fields.

        Args:
            file_path: Path to JSONL file
            name: Dataset name
            version: Dataset version

        Returns:
            Created evaluation dataset
        """
        samples = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample_data = json.loads(line)
                    samples.append(EvaluationSample(**sample_data))

        return EvaluationDataset(
            name=name,
            version=version,
            samples=samples,
        )

    @staticmethod
    async def from_qa_pairs(
        qa_pairs: List[dict],
        name: str,
        version: str = "1",
        domain: Optional[str] = None
    ) -> EvaluationDataset:
        """Create evaluation dataset from list of QA pair dictionaries.

        Args:
            qa_pairs: List of dicts with 'query' and 'answer' keys
            name: Dataset name
            version: Dataset version
            domain: Optional domain classification

        Returns:
            Created evaluation dataset
        """
        samples = []

        for idx, pair in enumerate(qa_pairs):
            sample = EvaluationSample(
                id=f"sample_{idx:04d}",
                query=pair["query"],
                ground_truth_answer=pair["answer"],
                ground_truth_sources=pair.get("sources", []),
                difficulty=pair.get("difficulty", "medium"),
                domain=domain,
            )
            samples.append(sample)

        return EvaluationDataset(
            name=name,
            version=version,
            domain=domain,
            samples=samples,
        )

    @staticmethod
    async def merge_datasets(
        datasets: List[EvaluationDataset],
        name: str,
        version: str = "1"
    ) -> EvaluationDataset:
        """Merge multiple evaluation datasets into one.

        Args:
            datasets: List of datasets to merge
            name: Name for merged dataset
            version: Version for merged dataset

        Returns:
            Merged evaluation dataset
        """
        all_samples = []

        for dataset in datasets:
            all_samples.extend(dataset.samples)

        return EvaluationDataset(
            name=name,
            version=version,
            samples=all_samples,
            description=f"Merged from {len(datasets)} datasets",
        )
