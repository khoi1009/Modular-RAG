"""Rule-based query router using YAML configuration."""
import os
from typing import Any, Dict, List, Optional

import yaml

from backend.logger import logger
from backend.modules.orchestration.routing.base_query_router import BaseQueryRouter
from backend.modules.orchestration.routing.schemas import RoutingDecision, RoutingRule
from backend.modules.query_analysis.schemas import QueryMetadata


class RuleBasedRouter(BaseQueryRouter):
    """Router that matches queries against predefined rules"""

    def __init__(self, rules_path: str):
        """
        Initialize router with rules from YAML file.

        Args:
            rules_path: Path to routing rules YAML file
        """
        self.rules_path = rules_path
        self.rules: List[RoutingRule] = []
        if os.path.exists(rules_path):
            self._load_rules()
        else:
            logger.warning(f"Rules file not found: {rules_path}, using defaults")

    def _load_rules(self):
        """Load routing rules from YAML file"""
        try:
            with open(self.rules_path, "r") as f:
                rules_data = yaml.safe_load(f)

            if not rules_data or "rules" not in rules_data:
                logger.warning("No rules found in YAML file")
                return

            for rule_dict in rules_data["rules"]:
                # Parse action into RoutingDecision
                action_dict = rule_dict["action"]
                action = RoutingDecision(**action_dict)

                # Create RoutingRule
                rule = RoutingRule(
                    name=rule_dict["name"],
                    conditions=rule_dict["conditions"],
                    action=action,
                    priority=rule_dict.get("priority", 0),
                )
                self.rules.append(rule)

            logger.info(f"Loaded {len(self.rules)} routing rules from {self.rules_path}")
        except Exception as e:
            logger.error(f"Error loading routing rules: {e}")
            self.rules = []

    def _evaluate_condition(
        self, condition: Dict[str, Any], query_metadata: QueryMetadata
    ) -> bool:
        """
        Evaluate a single condition against query metadata.

        Args:
            condition: Dict with 'field', 'operator', 'value'
            query_metadata: Query metadata to evaluate against

        Returns:
            True if condition matches
        """
        field = condition.get("field")
        operator = condition.get("operator")
        value = condition.get("value")

        if not field or not operator:
            return False

        # Get field value from query_metadata
        field_value = getattr(query_metadata, field, None)
        if field_value is None:
            return False

        # Evaluate based on operator
        if operator == "eq":
            return field_value == value
        elif operator == "ne":
            return field_value != value
        elif operator == "gt":
            return field_value > value
        elif operator == "gte":
            return field_value >= value
        elif operator == "lt":
            return field_value < value
        elif operator == "lte":
            return field_value <= value
        elif operator == "in":
            return field_value in value
        elif operator == "contains":
            return value in field_value
        else:
            logger.warning(f"Unknown operator: {operator}")
            return False

    def _matches_conditions(
        self, query_metadata: QueryMetadata, conditions: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if all conditions match (AND logic).

        Args:
            query_metadata: Query metadata
            conditions: List of condition dicts

        Returns:
            True if all conditions match
        """
        return all(
            self._evaluate_condition(cond, query_metadata) for cond in conditions
        )

    def _default_routing(self) -> RoutingDecision:
        """Return default routing decision when no rules match"""
        return RoutingDecision(
            controller_name="simple-retrieval",
            retrieval_strategy="vectorstore",
            preprocessing_steps=[],
            use_reranking=False,
            max_iterations=1,
            confidence=0.5,
            reasoning="No matching rules, using default simple retrieval",
        )

    async def route(
        self,
        query: str,
        query_metadata: QueryMetadata,
        context: Optional[Dict] = None,
    ) -> RoutingDecision:
        """
        Route query by matching against rules in priority order.

        Args:
            query: Raw query string
            query_metadata: Analyzed query metadata
            context: Optional additional context

        Returns:
            RoutingDecision from first matching rule or default
        """
        # Sort rules by priority (descending)
        sorted_rules = sorted(self.rules, key=lambda r: -r.priority)

        # Find first matching rule
        for rule in sorted_rules:
            if self._matches_conditions(query_metadata, rule.conditions):
                logger.info(
                    f"Query matched rule '{rule.name}' with confidence {rule.action.confidence}"
                )
                return rule.action

        # No match, return default
        logger.info("No rule matched, using default routing")
        return self._default_routing()
