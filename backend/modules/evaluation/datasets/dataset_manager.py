"""Manager for storing and loading evaluation datasets."""
import json
from pathlib import Path
from typing import Dict, List, Optional

from backend.modules.evaluation.datasets.schemas import (
    EvaluationDataset,
    EvaluationSample,
)


class DatasetManager:
    """Manages evaluation dataset storage and retrieval."""

    def __init__(self, storage_path: str = "data/evaluation"):
        """Initialize dataset manager.

        Args:
            storage_path: Base directory for storing evaluation datasets
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def create_dataset(self, dataset: EvaluationDataset) -> str:
        """Create and save a new evaluation dataset.

        Args:
            dataset: Evaluation dataset to save

        Returns:
            Path to saved dataset file
        """
        domain_dir = self.storage_path / (dataset.domain or "general")
        domain_dir.mkdir(parents=True, exist_ok=True)

        file_path = domain_dir / f"{dataset.name}_v{dataset.version}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dataset.model_dump(mode="json"), f, indent=2, default=str)

        return str(file_path)

    async def load_dataset(
        self,
        name: str,
        version: str,
        domain: Optional[str] = None
    ) -> EvaluationDataset:
        """Load an existing evaluation dataset.

        Args:
            name: Dataset name
            version: Dataset version
            domain: Optional domain subdirectory

        Returns:
            Loaded evaluation dataset

        Raises:
            FileNotFoundError: If dataset file doesn't exist
        """
        domain_dir = self.storage_path / (domain or "general")
        file_path = domain_dir / f"{name}_v{version}.json"

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return EvaluationDataset(**data)

    async def list_datasets(self) -> List[Dict[str, str]]:
        """List all available evaluation datasets.

        Returns:
            List of dataset metadata dictionaries
        """
        datasets = []

        for domain_dir in self.storage_path.iterdir():
            if not domain_dir.is_dir():
                continue

            for file_path in domain_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    datasets.append({
                        "name": data.get("name"),
                        "version": data.get("version"),
                        "domain": data.get("domain"),
                        "description": data.get("description"),
                        "sample_count": len(data.get("samples", [])),
                        "path": str(file_path),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue

        return datasets

    async def import_from_csv(
        self,
        csv_path: str,
        dataset_name: str,
        version: str = "1",
        domain: Optional[str] = None
    ) -> EvaluationDataset:
        """Import evaluation dataset from CSV file.

        CSV format expected:
        id,query,ground_truth_answer,ground_truth_sources,difficulty

        Args:
            csv_path: Path to CSV file
            dataset_name: Name for the created dataset
            version: Dataset version
            domain: Optional domain classification

        Returns:
            Created evaluation dataset
        """
        import csv

        samples = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                sources = []
                if row.get("ground_truth_sources"):
                    sources = [s.strip() for s in row["ground_truth_sources"].split("|")]

                sample = EvaluationSample(
                    id=row["id"],
                    query=row["query"],
                    ground_truth_answer=row["ground_truth_answer"],
                    ground_truth_sources=sources,
                    difficulty=row.get("difficulty", "medium"),
                    domain=domain,
                )
                samples.append(sample)

        dataset = EvaluationDataset(
            name=dataset_name,
            version=version,
            domain=domain,
            samples=samples,
        )

        await self.create_dataset(dataset)
        return dataset

    async def delete_dataset(
        self,
        name: str,
        version: str,
        domain: Optional[str] = None
    ) -> bool:
        """Delete an evaluation dataset.

        Args:
            name: Dataset name
            version: Dataset version
            domain: Optional domain subdirectory

        Returns:
            True if deleted, False if not found
        """
        domain_dir = self.storage_path / (domain or "general")
        file_path = domain_dir / f"{name}_v{version}.json"

        if file_path.exists():
            file_path.unlink()
            return True

        return False
