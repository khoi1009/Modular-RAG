"""Safe condition evaluator for pipeline step conditions."""
import operator
import re
from typing import Any, Dict

from backend.logger import logger


class ConditionEvaluator:
    """
    Safe evaluator for pipeline step conditions.
    Supports simple expressions without eval().
    """

    # Allowed operators
    OPERATORS = {
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    def evaluate(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate a condition expression against context.

        Supported formats:
        - "field > 0.7"
        - "nested.field == 'value'"
        - "field in ['a', 'b']"
        - "field is not None"

        Args:
            condition: Condition expression string
            context: Context dictionary with variable values

        Returns:
            Boolean result of evaluation
        """
        if not condition or not condition.strip():
            return True

        condition = condition.strip()

        try:
            # Handle "is None" / "is not None"
            if " is not None" in condition:
                field = condition.replace(" is not None", "").strip()
                value = self._get_field_value(field, context)
                return value is not None

            if " is None" in condition:
                field = condition.replace(" is None", "").strip()
                value = self._get_field_value(field, context)
                return value is None

            # Handle "in" operator
            if " in " in condition:
                return self._evaluate_in_condition(condition, context)

            # Handle comparison operators
            for op_str, op_func in self.OPERATORS.items():
                if op_str in condition:
                    return self._evaluate_comparison(condition, op_str, op_func, context)

            # If no operator found, treat as boolean field
            value = self._get_field_value(condition, context)
            return bool(value)

        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False

    def _get_field_value(self, field: str, context: Dict[str, Any]) -> Any:
        """
        Get value from context using dot notation.

        Args:
            field: Field path (e.g., "query_metadata.complexity_score")
            context: Context dictionary

        Returns:
            Field value or None if not found
        """
        parts = field.split(".")
        value = context

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None

            if value is None:
                return None

        return value

    def _evaluate_comparison(
        self, condition: str, op_str: str, op_func, context: Dict[str, Any]
    ) -> bool:
        """
        Evaluate comparison condition.

        Args:
            condition: Full condition string
            op_str: Operator string
            op_func: Operator function
            context: Context dictionary

        Returns:
            Comparison result
        """
        parts = condition.split(op_str, 1)
        if len(parts) != 2:
            return False

        field = parts[0].strip()
        value_str = parts[1].strip()

        # Get field value
        field_value = self._get_field_value(field, context)
        if field_value is None:
            return False

        # Parse comparison value
        compare_value = self._parse_value(value_str)

        # Perform comparison
        return op_func(field_value, compare_value)

    def _evaluate_in_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate 'in' condition.

        Args:
            condition: Condition with 'in' operator
            context: Context dictionary

        Returns:
            Result of 'in' check
        """
        parts = condition.split(" in ", 1)
        if len(parts) != 2:
            return False

        field = parts[0].strip()
        list_str = parts[1].strip()

        # Get field value
        field_value = self._get_field_value(field, context)
        if field_value is None:
            return False

        # Parse list
        list_value = self._parse_value(list_str)
        if not isinstance(list_value, (list, tuple)):
            return False

        return field_value in list_value

    def _parse_value(self, value_str: str) -> Any:
        """
        Parse a value string to appropriate type.

        Args:
            value_str: String representation of value

        Returns:
            Parsed value
        """
        value_str = value_str.strip()

        # String (quoted)
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]

        # List
        if value_str.startswith("[") and value_str.endswith("]"):
            # Simple list parsing
            content = value_str[1:-1]
            if not content.strip():
                return []
            items = [self._parse_value(item.strip()) for item in content.split(",")]
            return items

        # Boolean
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False

        # None
        if value_str.lower() == "none":
            return None

        # Number
        try:
            if "." in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass

        # Return as string if nothing else matches
        return value_str
