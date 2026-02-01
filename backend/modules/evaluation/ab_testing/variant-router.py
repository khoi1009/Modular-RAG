"""Router for directing traffic to A/B test variants."""
from typing import Dict, Optional

from backend.modules.evaluation.ab_testing.experiment_manager import (
    ExperimentManager,
    Variant,
)


class VariantRouter:
    """Routes user requests to appropriate experiment variants."""

    def __init__(self, experiment_manager: ExperimentManager):
        """Initialize variant router.

        Args:
            experiment_manager: Experiment manager instance
        """
        self.experiment_manager = experiment_manager

    async def route_query(
        self,
        query: str,
        user_id: str,
        collection_name: str,
        experiment_name: Optional[str] = None
    ) -> Dict:
        """Route query to appropriate variant.

        Args:
            query: User query
            user_id: User identifier
            collection_name: Collection to query
            experiment_name: Optional experiment to use

        Returns:
            Dictionary with variant info and routing decision
        """
        if experiment_name:
            # Route through experiment
            variant = await self.experiment_manager.get_variant_for_user(
                experiment_name,
                user_id
            )

            return {
                "pipeline_config": variant.pipeline_config,
                "variant_name": variant.name,
                "experiment_name": experiment_name,
                "is_experiment": True,
                "metadata": {
                    "query": query,
                    "user_id": user_id,
                    "collection_name": collection_name,
                }
            }
        else:
            # No experiment - use default pipeline
            return {
                "pipeline_config": "default",
                "variant_name": "control",
                "experiment_name": None,
                "is_experiment": False,
                "metadata": {
                    "query": query,
                    "user_id": user_id,
                    "collection_name": collection_name,
                }
            }

    async def get_pipeline_for_user(
        self,
        user_id: str,
        experiment_name: str
    ) -> str:
        """Get pipeline config for user in experiment.

        Args:
            user_id: User identifier
            experiment_name: Experiment name

        Returns:
            Pipeline configuration path
        """
        return await self.experiment_manager.get_pipeline_config_for_user(
            experiment_name,
            user_id
        )

    async def record_routing_decision(
        self,
        routing_info: Dict
    ) -> Dict:
        """Record routing decision for analytics.

        Args:
            routing_info: Routing information from route_query

        Returns:
            Recording confirmation
        """
        # This would typically log to a metrics system
        # For now, just return the info
        return {
            "recorded": True,
            "routing_info": routing_info
        }

    def get_variant_from_routing(self, routing_info: Dict) -> Optional[Variant]:
        """Extract variant from routing info.

        Args:
            routing_info: Routing information

        Returns:
            Variant or None
        """
        if not routing_info.get("is_experiment"):
            return None

        # Would need to reconstruct variant from stored info
        # In practice, this might query the experiment manager
        return None
