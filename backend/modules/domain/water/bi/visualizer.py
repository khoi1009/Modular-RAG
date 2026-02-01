"""Data visualization formatter for water infrastructure analytics."""
from typing import Any, Dict, List, Optional


class DataVisualizer:
    """Format SQL query results for chart visualization."""

    @staticmethod
    def format_for_chart(
        data: List[Dict[str, Any]],
        chart_type: str = "auto",
    ) -> Dict:
        """
        Format query results for chart rendering.

        Args:
            data: SQL query results as list of dicts
            chart_type: Type of chart (auto, line_chart, bar_chart, pie_chart)

        Returns:
            Formatted data structure for visualization
        """
        if not data:
            return {
                "chart_type": "none",
                "data": [],
                "message": "No data available",
            }

        # Auto-detect chart type if not specified
        if chart_type == "auto":
            chart_type = DataVisualizer._detect_chart_type(data)

        if chart_type == "line_chart":
            return DataVisualizer._format_line_chart(data)
        elif chart_type == "bar_chart":
            return DataVisualizer._format_bar_chart(data)
        elif chart_type == "pie_chart":
            return DataVisualizer._format_pie_chart(data)
        else:
            return DataVisualizer._format_table(data)

    @staticmethod
    def _detect_chart_type(data: List[Dict]) -> str:
        """Auto-detect appropriate chart type based on data structure."""
        if not data:
            return "table"

        first_row = data[0]
        columns = list(first_row.keys())

        # Time series detection (date/timestamp + numeric)
        has_date = any(
            col.lower() in ["date", "timestamp", "time", "month", "day", "year"]
            for col in columns
        )
        has_numeric = any(
            isinstance(first_row[col], (int, float))
            for col in columns
        )

        if has_date and has_numeric:
            return "line_chart"

        # Categorical + numeric = bar chart
        if len(columns) == 2 and has_numeric:
            return "bar_chart"

        # Default to table
        return "table"

    @staticmethod
    def _format_line_chart(data: List[Dict]) -> Dict:
        """Format data for line chart."""
        if not data:
            return {"chart_type": "line_chart", "data": []}

        # Identify x-axis (date/time) and y-axis (numeric) columns
        columns = list(data[0].keys())
        x_col = None
        y_cols = []

        for col in columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["date", "time", "month", "day", "year"]):
                x_col = col
            elif isinstance(data[0][col], (int, float)):
                y_cols.append(col)

        if not x_col or not y_cols:
            return DataVisualizer._format_table(data)

        # Format for line chart
        series = []
        for y_col in y_cols:
            series.append({
                "name": y_col,
                "data": [
                    {"x": str(row[x_col]), "y": row[y_col]}
                    for row in data
                ],
            })

        return {
            "chart_type": "line_chart",
            "x_axis": x_col,
            "series": series,
            "row_count": len(data),
        }

    @staticmethod
    def _format_bar_chart(data: List[Dict]) -> Dict:
        """Format data for bar chart."""
        if not data:
            return {"chart_type": "bar_chart", "data": []}

        columns = list(data[0].keys())

        # First column is category, rest are values
        category_col = columns[0]
        value_cols = [
            col for col in columns[1:]
            if isinstance(data[0].get(col), (int, float))
        ]

        if not value_cols:
            return DataVisualizer._format_table(data)

        # Format for bar chart
        categories = [str(row[category_col]) for row in data]
        series = []

        for value_col in value_cols:
            series.append({
                "name": value_col,
                "data": [row[value_col] for row in data],
            })

        return {
            "chart_type": "bar_chart",
            "categories": categories,
            "series": series,
            "row_count": len(data),
        }

    @staticmethod
    def _format_pie_chart(data: List[Dict]) -> Dict:
        """Format data for pie chart."""
        if not data or len(data[0]) != 2:
            return DataVisualizer._format_table(data)

        columns = list(data[0].keys())
        label_col = columns[0]
        value_col = columns[1]

        return {
            "chart_type": "pie_chart",
            "data": [
                {"label": str(row[label_col]), "value": row[value_col]}
                for row in data
            ],
            "row_count": len(data),
        }

    @staticmethod
    def _format_table(data: List[Dict]) -> Dict:
        """Format data as table."""
        if not data:
            return {"chart_type": "table", "data": []}

        return {
            "chart_type": "table",
            "columns": list(data[0].keys()),
            "rows": data,
            "row_count": len(data),
        }
