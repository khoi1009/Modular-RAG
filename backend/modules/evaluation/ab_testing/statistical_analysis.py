"""Statistical analysis for A/B experiment evaluation."""
from typing import Any, Dict, List

import numpy as np
from pydantic import Field

from backend.modules.evaluation.ab_testing.experiment_manager import Experiment
from backend.types import ConfiguredBaseModel


class ExperimentAnalysis(ConfiguredBaseModel):
    """Analysis results for an A/B experiment."""
    experiment: Experiment
    variant_stats: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    comparisons: Dict[str, Dict] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)


class StatisticalAnalyzer:
    """Statistical analysis for A/B experiments."""

    def __init__(self, confidence_level: float = 0.95):
        """Initialize statistical analyzer.

        Args:
            confidence_level: Confidence level for significance testing
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def compute_significance(
        self,
        control_values: List[float],
        treatment_values: List[float]
    ) -> Dict[str, Any]:
        """Compute statistical significance between two variants.

        Uses independent t-test to determine if difference is significant.

        Args:
            control_values: Metric values for control variant
            treatment_values: Metric values for treatment variant

        Returns:
            Dictionary with statistical analysis results
        """
        if not control_values or not treatment_values:
            return {
                "error": "Insufficient data",
                "significant": False
            }

        control_array = np.array(control_values)
        treatment_array = np.array(treatment_values)

        control_mean = np.mean(control_array)
        treatment_mean = np.mean(treatment_array)
        control_std = np.std(control_array, ddof=1)
        treatment_std = np.std(treatment_array, ddof=1)

        # Compute t-statistic and p-value
        t_stat, p_value = self._ttest_ind(control_array, treatment_array)

        # Calculate lift
        if control_mean != 0:
            lift = (treatment_mean - control_mean) / control_mean
        else:
            lift = 0.0

        # Determine significance
        significant = p_value < self.alpha

        return {
            "control_mean": float(control_mean),
            "treatment_mean": float(treatment_mean),
            "control_std": float(control_std),
            "treatment_std": float(treatment_std),
            "control_n": len(control_values),
            "treatment_n": len(treatment_values),
            "lift": float(lift),
            "lift_pct": float(lift * 100),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": significant,
            "confidence_level": self.confidence_level,
        }

    def _ttest_ind(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> tuple[float, float]:
        """Independent samples t-test.

        Args:
            a: First sample
            b: Second sample

        Returns:
            Tuple of (t-statistic, p-value)
        """
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        var_a = np.var(a, ddof=1)
        var_b = np.var(b, ddof=1)
        n_a = len(a)
        n_b = len(b)

        # Pooled standard error
        se = np.sqrt(var_a / n_a + var_b / n_b)

        if se == 0:
            return 0.0, 1.0

        # T-statistic
        t_stat = (mean_a - mean_b) / se

        # Degrees of freedom (Welch-Satterthwaite)
        df = (var_a / n_a + var_b / n_b) ** 2 / (
            (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        )

        # P-value (two-tailed)
        p_value = self._t_distribution_pvalue(abs(t_stat), df)

        return t_stat, p_value

    @staticmethod
    def _t_distribution_pvalue(t: float, df: float) -> float:
        """Approximate p-value for t-distribution.

        Uses approximation for two-tailed test.

        Args:
            t: T-statistic
            df: Degrees of freedom

        Returns:
            Approximate p-value
        """
        # Simple approximation - in production, use scipy.stats
        try:
            from scipy import stats
            return 2 * (1 - stats.t.cdf(abs(t), df))
        except ImportError:
            # Fallback approximation using normal distribution
            # Only valid for large df (>30)
            if df > 30:
                from math import erf, sqrt
                z = t / sqrt(1 + t**2 / df)
                p = 2 * (1 - (1 + erf(z / sqrt(2))) / 2)
                return p
            else:
                # Conservative estimate
                return 1.0 if t < 2 else 0.05

    async def analyze_experiment(
        self,
        experiment: Experiment,
        variant_data: Dict[str, Dict[str, List[float]]]
    ) -> ExperimentAnalysis:
        """Analyze experiment results.

        Args:
            experiment: Experiment definition
            variant_data: Dictionary mapping variant names to metric data
                Format: {variant_name: {metric_name: [values]}}

        Returns:
            Experiment analysis with statistical results
        """
        analysis = ExperimentAnalysis(experiment=experiment)

        # Compute statistics for each variant
        for variant_name, metrics in variant_data.items():
            analysis.variant_stats[variant_name] = {}

            for metric_name, values in metrics.items():
                if values:
                    analysis.variant_stats[variant_name][metric_name] = {
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "std": float(np.std(values, ddof=1)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "n": len(values),
                    }

        # Compare all treatments against control (first variant)
        control_name = experiment.variants[0].name
        control_data = variant_data.get(control_name, {})

        for variant in experiment.variants[1:]:
            treatment_name = variant.name
            treatment_data = variant_data.get(treatment_name, {})

            comparison_key = f"{control_name}_vs_{treatment_name}"
            analysis.comparisons[comparison_key] = {}

            for metric_name in experiment.metrics:
                control_values = control_data.get(metric_name, [])
                treatment_values = treatment_data.get(metric_name, [])

                if control_values and treatment_values:
                    sig_result = self.compute_significance(
                        control_values,
                        treatment_values
                    )
                    analysis.comparisons[comparison_key][metric_name] = sig_result

        # Generate recommendations
        analysis.recommendations = self._generate_recommendations(
            experiment,
            analysis.comparisons
        )

        return analysis

    def _generate_recommendations(
        self,
        experiment: Experiment,
        comparisons: Dict[str, Dict]
    ) -> List[str]:
        """Generate recommendations based on analysis.

        Args:
            experiment: Experiment definition
            comparisons: Comparison results

        Returns:
            List of recommendation strings
        """
        recommendations = []

        for comparison_key, metrics in comparisons.items():
            control_name, treatment_name = comparison_key.split("_vs_")

            significant_improvements = []
            significant_regressions = []

            for metric_name, result in metrics.items():
                if result.get("significant"):
                    lift_pct = result.get("lift_pct", 0)

                    if lift_pct > 0:
                        significant_improvements.append(
                            f"{metric_name} (+{lift_pct:.1f}%)"
                        )
                    else:
                        significant_regressions.append(
                            f"{metric_name} ({lift_pct:.1f}%)"
                        )

            if significant_improvements:
                recommendations.append(
                    f"✓ {treatment_name} shows significant improvements in: "
                    f"{', '.join(significant_improvements)}"
                )

            if significant_regressions:
                recommendations.append(
                    f"✗ {treatment_name} shows significant regressions in: "
                    f"{', '.join(significant_regressions)}"
                )

            if not significant_improvements and not significant_regressions:
                recommendations.append(
                    f"○ {treatment_name} shows no significant differences from {control_name}"
                )

        if not recommendations:
            recommendations.append("Insufficient data for recommendations")

        return recommendations

    def calculate_required_sample_size(
        self,
        baseline_mean: float,
        minimum_detectable_effect: float,
        baseline_std: float,
        power: float = 0.8
    ) -> int:
        """Calculate required sample size for experiment.

        Args:
            baseline_mean: Current baseline metric value
            minimum_detectable_effect: Minimum effect size to detect (as fraction)
            baseline_std: Standard deviation of baseline metric
            power: Statistical power (1 - Type II error rate)

        Returns:
            Required sample size per variant
        """
        # Effect size (Cohen's d)
        delta = baseline_mean * minimum_detectable_effect
        effect_size = delta / baseline_std if baseline_std > 0 else 0

        if effect_size == 0:
            return 10000  # Large number if effect size is 0

        # Z-scores for alpha and beta
        z_alpha = 1.96  # For 95% confidence (two-tailed)
        z_beta = 0.84   # For 80% power

        # Sample size formula
        n = 2 * ((z_alpha + z_beta) ** 2) * (baseline_std ** 2) / (delta ** 2)

        return int(np.ceil(n))

    def compute_confidence_interval(
        self,
        values: List[float],
        confidence_level: Optional[float] = None
    ) -> tuple[float, float]:
        """Compute confidence interval for mean.

        Args:
            values: Sample values
            confidence_level: Optional override for confidence level

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not values:
            return (0.0, 0.0)

        conf_level = confidence_level or self.confidence_level
        alpha = 1 - conf_level

        data = np.array(values)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)

        # Standard error
        se = std / np.sqrt(n)

        # Critical value (approximated with z-score for large n)
        z_critical = 1.96  # For 95% confidence

        margin = z_critical * se

        return (float(mean - margin), float(mean + margin))
