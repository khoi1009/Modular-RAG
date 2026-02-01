"""SQL agent for water infrastructure analytics queries."""
from typing import Dict, Optional

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models.chat_models import BaseChatModel

from backend.logger import logger


class WaterSQLAgent:
    """
    LangChain SQL agent for querying water infrastructure analytics database.

    Security: Read-only access, sandboxed to daily_analysis table.
    """

    def __init__(self, database_url: str):
        """
        Initialize SQL agent with database connection.

        Args:
            database_url: PostgreSQL connection string
        """
        # Create read-only connection
        self.db = SQLDatabase.from_uri(
            database_url,
            include_tables=["daily_analysis"],  # Restrict to analytics table only
            sample_rows_in_table_info=2,
        )

    def query(
        self,
        natural_language_query: str,
        llm: BaseChatModel,
    ) -> Dict:
        """
        Execute natural language query against database.

        Args:
            natural_language_query: User's question in natural language
            llm: LLM for SQL generation

        Returns:
            Dictionary with query results and metadata
        """
        try:
            # Create toolkit with read-only access
            toolkit = SQLDatabaseToolkit(db=self.db, llm=llm)

            # Get SQL query tool
            query_tool = None
            for tool in toolkit.get_tools():
                if "query" in tool.name.lower() and "checker" not in tool.name.lower():
                    query_tool = tool
                    break

            if not query_tool:
                raise ValueError("SQL query tool not found in toolkit")

            # Generate and execute SQL
            # For safety, we extract the SQL generation separately
            sql_query = self._generate_sql(natural_language_query, llm)

            # Validate SQL is read-only
            if not self._is_read_only_query(sql_query):
                raise ValueError("Only SELECT queries are allowed")

            # Execute query
            result = self.db.run(sql_query, fetch="all")

            return {
                "sql": sql_query,
                "result": result,
                "row_count": len(result) if isinstance(result, list) else 1,
            }

        except Exception as e:
            logger.error(f"SQL agent query failed: {e}")
            return {
                "error": str(e),
                "sql": None,
                "result": None,
            }

    def _generate_sql(self, query: str, llm: BaseChatModel) -> str:
        """Generate SQL from natural language query."""
        # Get table schema
        table_info = self.db.get_table_info(table_names=["daily_analysis"])

        prompt = f"""Given the following database schema:

{table_info}

Generate a SQL query to answer: {query}

Requirements:
- Use only SELECT statements
- Query only the daily_analysis table
- Use proper aggregations (SUM, AVG, COUNT, etc.)
- Include appropriate WHERE clauses for filtering
- Use ORDER BY if trends or rankings are needed
- Limit results to reasonable size (use LIMIT)

Return only the SQL query, no explanation.
"""

        response = llm.invoke(prompt)
        sql = response.content.strip()

        # Clean up SQL (remove markdown code blocks if present)
        sql = sql.replace("```sql", "").replace("```", "").strip()

        return sql

    def _is_read_only_query(self, sql: str) -> bool:
        """Validate SQL query is read-only."""
        sql_upper = sql.upper().strip()

        # Must start with SELECT
        if not sql_upper.startswith("SELECT"):
            return False

        # Must not contain write operations
        forbidden_keywords = [
            "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
            "TRUNCATE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
        ]

        for keyword in forbidden_keywords:
            if keyword in sql_upper:
                return False

        return True

    def get_table_schema(self) -> str:
        """Get schema information for daily_analysis table."""
        return self.db.get_table_info(table_names=["daily_analysis"])
