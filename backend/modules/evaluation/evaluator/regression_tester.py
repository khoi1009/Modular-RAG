"""Regression tester for catching quality degradations in RAG pipeline."""
import json
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field

from backend.modules.evaluation.datasets.schemas import (
    EvaluationDataset,
    EvaluationReport,
)
from backend.modules.evaluation.evaluator.automated_evaluator import AutomatedEvaluator
from backend.types import ConfiguredBaseModel


class RegressionResult(ConfiguredBaseModel):
    """Result of regression testing."""
    passed: bool
    regressions: List[Dict] = Field(default_factory=list)
    improvements: List[Dict] = Field(default_factory=list)
    current_report: EvaluationReport
    baseline_report: Optional[EvaluationReport] = None
    summary: str = ""


class RegressionTester:
    """Tests for quality regressions by comparing against baseline."""

    def __init__(
        self,
        evaluator: AutomatedEvaluator,
        baseline_dir: str = "data/evaluation/baselines",
        thresholds: Optional[Dict[str, float]] = None
    ):
        """Initialize regression tester.

        Args:
            evaluator: Automated evaluator instance
            baseline_dir: Directory storing baseline evaluation reports
            thresholds: Metric thresholds for regression detection
        """
        self.evaluator = evaluator
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

        # Default thresholds for regression detection
        self.thresholds = thresholds or {
            "f1_mean": 0.05,  # 5% drop is regression
            "exact_match_mean": 0.05,
            "precision@5_mean": 0.05,
            "recall@5_mean": 0.05,
            "mrr_mean": 0.05,
            "semantic_similarity_mean": 0.05,
            "judge_overall_mean": 0.05,
            "latency_p95_ms": 500,  # 500ms increase is regression
        }

    async def run_regression_test(
        self,
        dataset: EvaluationDataset,
        pipeline_config_path: str,
        collection_name: str,
        save_as_baseline: bool = False
    ) -> RegressionResult:
        """Run regression test against baseline.

        Args:
            dataset: Evaluation dataset
            pipeline_config_path: Pipeline configuration path
            collection_name: Collection to test against
            save_as_baseline: Whether to save this run as new baseline

        Returns:
            Regression test result
        """
        # Run current evaluation
        current_report = await self.evaluator.evaluate_dataset(
            dataset,
            pipeline_config_path,
            collection_name
        )

        # Try to load baseline
        baseline_report = self._load_baseline(
            dataset.name,
            dataset.version,
            pipeline_config_path
        )

        if baseline_report is None:
            # No baseline exists - save current as baseline if requested
            if save_as_baseline:
                self._save_baseline(current_report, pipeline_config_path)

            return RegressionResult(
                passed=True,
                current_report=current_report,
                baseline_report=None,
                summary="No baseline found. Run saved as new baseline."
            )

        # Compare against baseline
        regressions = []
        improvements = []

        for metric_name, threshold in self.thresholds.items():
            current_val = current_report.aggregate_metrics.get(metric_name, 0.0)
            baseline_val = baseline_report.aggregate_metrics.get(metric_name, 0.0)

            delta = current_val - baseline_val

            # For latency metrics, increase is bad
            if "latency" in metric_name:
                if delta > threshold:
                    regressions.append({
                        "metric": metric_name,
                        "baseline": baseline_val,
                        "current": current_val,
                        "delta": delta,
                        "threshold": threshold,
                        "severity": self._calculate_severity(delta, threshold)
                    })
                elif delta < -threshold:
                    improvements.append({
                        "metric": metric_name,
                        "baseline": baseline_val,
                        "current": current_val,
                        "delta": delta
                    })
            else:
                # For quality metrics, decrease is bad
                if delta < -threshold:
                    regressions.append({
                        "metric": metric_name,
                        "baseline": baseline_val,
                        "current": current_val,
                        "delta": delta,
                        "threshold": threshold,
                        "severity": self._calculate_severity(abs(delta), threshold)
                    })
                elif delta > threshold:
                    improvements.append({
                        "metric": metric_name,
                        "baseline": baseline_val,
                        "current": current_val,
                        "delta": delta
                    })

        # Generate summary
        summary = self._generate_summary(regressions, improvements)

        # Save as new baseline if requested and passed
        if save_as_baseline and len(regressions) == 0:
            self._save_baseline(current_report, pipeline_config_path)

        return RegressionResult(
            passed=len(regressions) == 0,
            regressions=regressions,
            improvements=improvements,
            current_report=current_report,
            baseline_report=baseline_report,
            summary=summary
        )

    def _load_baseline(
        self,
        dataset_name: str,
        dataset_version: str,
        pipeline_config: str
    ) -> Optional[EvaluationReport]:
        """Load baseline report from storage.

        Args:
            dataset_name: Dataset name
            dataset_version: Dataset version
            pipeline_config: Pipeline configuration path

        Returns:
            Baseline report or None if not found
        """
        baseline_file = self._get_baseline_path(
            dataset_name,
            dataset_version,
            pipeline_config
        )

        if not baseline_file.exists():
            return None

        try:
            with open(baseline_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return EvaluationReport(**data)
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    def _save_baseline(
        self,
        report: EvaluationReport,
        pipeline_config: str
    ):
        """Save evaluation report as baseline.

        Args:
            report: Evaluation report to save
            pipeline_config: Pipeline configuration path
        """
        baseline_file = self._get_baseline_path(
            report.dataset_name,
            report.dataset_version,
            pipeline_config
        )

        baseline_file.parent.mkdir(parents=True, exist_ok=True)

        with open(baseline_file, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2, default=str)

    def _get_baseline_path(
        self,
        dataset_name: str,
        dataset_version: str,
        pipeline_config: str
    ) -> Path:
        """Get baseline file path.

        Args:
            dataset_name: Dataset name
            dataset_version: Dataset version
            pipeline_config: Pipeline configuration path

        Returns:
            Path to baseline file
        """
        # Create safe filename from pipeline config
        config_name = Path(pipeline_config).stem
        baseline_filename = f"{dataset_name}_v{dataset_version}_{config_name}_baseline.json"

        return self.baseline_dir / baseline_filename

    @staticmethod
    def _calculate_severity(delta: float, threshold: float) -> str:
        """Calculate regression severity.

        Args:
            delta: Absolute delta from baseline
            threshold: Regression threshold

        Returns:
            Severity level: low, medium, high, critical
        """
        ratio = delta / threshold

        if ratio >= 3.0:
            return "critical"
        elif ratio >= 2.0:
            return "high"
        elif ratio >= 1.5:
            return "medium"
        else:
            return "low"

    def _generate_summary(
        self,
        regressions: List[Dict],
        improvements: List[Dict]
    ) -> str:
        """Generate human-readable summary.

        Args:
            regressions: List of detected regressions
            improvements: List of detected improvements

        Returns:
            Summary string
        """
        if not regressions and not improvements:
            return "No significant changes detected."

        summary_parts = []

        if regressions:
            critical = [r for r in regressions if r["severity"] == "critical"]
            high = [r for r in regressions if r["severity"] == "high"]

            summary_parts.append(
                f"REGRESSIONS DETECTED: {len(regressions)} metrics degraded."
            )

            if critical:
                summary_parts.append(
                    f"  - {len(critical)} critical regressions"
                )

            if high:
                summary_parts.append(
                    f"  - {len(high)} high severity regressions"
                )

        if improvements:
            summary_parts.append(
                f"IMPROVEMENTS: {len(improvements)} metrics improved."
            )

        return "\n".join(summary_parts)

    async def run_ci_test(
        self,
        dataset: EvaluationDataset,
        pipeline_config_path: str,
        collection_name: str,
        fail_on_regression: bool = True
    ) -> bool:
        """Run regression test for CI/CD pipeline.

        Args:
            dataset: Evaluation dataset
            pipeline_config_path: Pipeline configuration
            collection_name: Collection name
            fail_on_regression: Whether to fail on detected regressions

        Returns:
            True if passed, False if failed
        """
        result = await self.run_regression_test(
            dataset,
            pipeline_config_path,
            collection_name,
            save_as_baseline=False
        )

        print(f"\n{'='*60}")
        print(f"REGRESSION TEST RESULTS")
        print(f"{'='*60}")
        print(f"Dataset: {dataset.name} v{dataset.version}")
        print(f"Pipeline: {pipeline_config_path}")
        print(f"\n{result.summary}")

        if result.regressions:
            print(f"\nRegressions:")
            for reg in result.regressions:
                print(
                    f"  - {reg['metric']}: {reg['baseline']:.4f} → {reg['current']:.4f} "
                    f"(Δ {reg['delta']:.4f}) [{reg['severity']}]"
                )

        if result.improvements:
            print(f"\nImprovements:")
            for imp in result.improvements:
                print(
                    f"  + {imp['metric']}: {imp['baseline']:.4f} → {imp['current']:.4f} "
                    f"(Δ +{imp['delta']:.4f})"
                )

        print(f"{'='*60}\n")

        if fail_on_regression:
            return result.passed
        else:
            return True
