"""A/B experiment manager for testing RAG pipeline variants."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field

from backend.types import ConfiguredBaseModel


class Variant(ConfiguredBaseModel):
    """A/B test variant configuration."""
    name: str
    pipeline_config: str
    traffic_percentage: float  # 0-100
    description: Optional[str] = None


class Experiment(ConfiguredBaseModel):
    """A/B experiment definition."""
    name: str
    variants: List[Variant]
    metrics: List[str]  # Metrics to track for this experiment
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, paused, completed
    metadata: Dict = Field(default_factory=dict)


class ExperimentAssignment(ConfiguredBaseModel):
    """Record of user assignment to experiment variant."""
    experiment_name: str
    user_id: str
    variant_name: str
    assigned_at: datetime = Field(default_factory=datetime.utcnow)


class ExperimentStorage:
    """Storage for experiment definitions and assignments."""

    def __init__(self, storage_dir: str = "data/evaluation/experiments"):
        """Initialize experiment storage.

        Args:
            storage_dir: Directory for storing experiment data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def save_experiment(self, experiment: Experiment):
        """Save experiment definition.

        Args:
            experiment: Experiment to save
        """
        exp_file = self.storage_dir / f"{experiment.name}.json"

        with open(exp_file, "w", encoding="utf-8") as f:
            json.dump(experiment.model_dump(mode="json"), f, indent=2, default=str)

    async def load_experiment(self, name: str) -> Optional[Experiment]:
        """Load experiment by name.

        Args:
            name: Experiment name

        Returns:
            Experiment or None if not found
        """
        exp_file = self.storage_dir / f"{name}.json"

        if not exp_file.exists():
            return None

        with open(exp_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return Experiment(**data)

    async def list_experiments(self) -> List[Experiment]:
        """List all experiments.

        Returns:
            List of all experiments
        """
        experiments = []

        for exp_file in self.storage_dir.glob("*.json"):
            try:
                with open(exp_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                experiments.append(Experiment(**data))
            except (json.JSONDecodeError, ValueError):
                continue

        return experiments


class ExperimentManager:
    """Manages A/B experiments for RAG pipeline testing."""

    def __init__(self, storage: Optional[ExperimentStorage] = None):
        """Initialize experiment manager.

        Args:
            storage: Experiment storage instance
        """
        self.storage = storage or ExperimentStorage()
        self.active_experiments: Dict[str, Experiment] = {}

    async def create_experiment(
        self,
        name: str,
        variants: List[Variant],
        metrics: List[str],
        start_immediately: bool = True
    ) -> Experiment:
        """Create a new A/B experiment.

        Args:
            name: Experiment name
            variants: List of variants to test
            metrics: Metrics to track
            start_immediately: Whether to start experiment immediately

        Returns:
            Created experiment

        Raises:
            ValueError: If variant traffic doesn't sum to 100%
        """
        # Validate traffic allocation
        total_traffic = sum(v.traffic_percentage for v in variants)
        if abs(total_traffic - 100.0) > 0.01:
            raise ValueError(
                f"Variant traffic must sum to 100%, got {total_traffic}%"
            )

        experiment = Experiment(
            name=name,
            variants=variants,
            metrics=metrics,
            start_time=datetime.utcnow(),
            status="running" if start_immediately else "paused"
        )

        await self.storage.save_experiment(experiment)

        if start_immediately:
            self.active_experiments[name] = experiment

        return experiment

    async def get_variant_for_user(
        self,
        experiment_name: str,
        user_id: str
    ) -> Variant:
        """Get assigned variant for a user.

        Uses deterministic hashing for consistent assignment.

        Args:
            experiment_name: Experiment name
            user_id: User identifier

        Returns:
            Assigned variant

        Raises:
            ValueError: If experiment not found or not running
        """
        # Load experiment if not in active cache
        if experiment_name not in self.active_experiments:
            experiment = await self.storage.load_experiment(experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment '{experiment_name}' not found")

            if experiment.status != "running":
                raise ValueError(
                    f"Experiment '{experiment_name}' is not running (status: {experiment.status})"
                )

            self.active_experiments[experiment_name] = experiment
        else:
            experiment = self.active_experiments[experiment_name]

        # Deterministic assignment based on hash
        hash_input = f"{experiment_name}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 100.0  # 0-100 range

        # Find variant based on traffic allocation
        cumulative = 0.0
        for variant in experiment.variants:
            cumulative += variant.traffic_percentage
            if bucket < cumulative:
                return variant

        # Fallback to last variant
        return experiment.variants[-1]

    async def get_pipeline_config_for_user(
        self,
        experiment_name: str,
        user_id: str
    ) -> str:
        """Get pipeline config path for user in experiment.

        Args:
            experiment_name: Experiment name
            user_id: User identifier

        Returns:
            Pipeline configuration path
        """
        variant = await self.get_variant_for_user(experiment_name, user_id)
        return variant.pipeline_config

    async def pause_experiment(self, experiment_name: str):
        """Pause a running experiment.

        Args:
            experiment_name: Experiment name
        """
        experiment = await self.storage.load_experiment(experiment_name)
        if experiment:
            experiment.status = "paused"
            await self.storage.save_experiment(experiment)

            if experiment_name in self.active_experiments:
                del self.active_experiments[experiment_name]

    async def resume_experiment(self, experiment_name: str):
        """Resume a paused experiment.

        Args:
            experiment_name: Experiment name
        """
        experiment = await self.storage.load_experiment(experiment_name)
        if experiment:
            experiment.status = "running"
            await self.storage.save_experiment(experiment)
            self.active_experiments[experiment_name] = experiment

    async def complete_experiment(
        self,
        experiment_name: str,
        winning_variant: Optional[str] = None
    ):
        """Mark experiment as completed.

        Args:
            experiment_name: Experiment name
            winning_variant: Optional name of winning variant
        """
        experiment = await self.storage.load_experiment(experiment_name)
        if experiment:
            experiment.status = "completed"
            experiment.end_time = datetime.utcnow()

            if winning_variant:
                experiment.metadata["winning_variant"] = winning_variant

            await self.storage.save_experiment(experiment)

            if experiment_name in self.active_experiments:
                del self.active_experiments[experiment_name]

    async def list_active_experiments(self) -> List[Experiment]:
        """List all active experiments.

        Returns:
            List of active experiments
        """
        all_experiments = await self.storage.list_experiments()
        return [exp for exp in all_experiments if exp.status == "running"]

    async def get_experiment_status(self, experiment_name: str) -> Dict:
        """Get current status of an experiment.

        Args:
            experiment_name: Experiment name

        Returns:
            Status dictionary with experiment details
        """
        experiment = await self.storage.load_experiment(experiment_name)

        if not experiment:
            return {"error": "Experiment not found"}

        return {
            "name": experiment.name,
            "status": experiment.status,
            "variants": [
                {
                    "name": v.name,
                    "traffic": v.traffic_percentage,
                    "pipeline": v.pipeline_config
                }
                for v in experiment.variants
            ],
            "start_time": experiment.start_time.isoformat(),
            "end_time": experiment.end_time.isoformat() if experiment.end_time else None,
            "metrics": experiment.metrics,
        }
