from typing import Dict, List, Optional

from backend.logger import logger
from backend.types import ConfiguredBaseModel


class EnhancedContext(ConfiguredBaseModel):
    """Context-enriched query information"""
    query: str
    user_context: Optional[Dict] = None
    session_history: List[str] = []
    domain_context: Optional[Dict] = None
    injected_context: str = ""


class SessionDomainContextInjector:
    """
    Injects additional context into queries from:
    - User profile
    - Session history
    - Domain knowledge
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_history_items = config.get("max_history_items", 3)
        self.domain_knowledge = config.get("domain_knowledge", {})

    async def inject(
        self,
        query: str,
        user_profile: Optional[Dict] = None,
        session_history: Optional[List[str]] = None,
        domain: Optional[str] = None,
    ) -> EnhancedContext:
        """Inject context from various sources"""
        # Build context components
        user_context = self._get_user_context(user_profile)
        history_context = self._get_history_context(session_history)
        domain_context = self._get_domain_context(domain)

        # Build injected context string
        injected = self._build_injected_context(
            user_context, history_context, domain_context
        )

        return EnhancedContext(
            query=query,
            user_context=user_context,
            session_history=session_history[:self.max_history_items] if session_history else [],
            domain_context=domain_context,
            injected_context=injected,
        )

    def _get_user_context(self, user_profile: Optional[Dict]) -> Optional[Dict]:
        """Extract relevant user context"""
        if not user_profile:
            return None

        context = {}

        # Extract user preferences
        if "preferences" in user_profile:
            context["preferences"] = user_profile["preferences"]

        # Extract user role/department
        if "role" in user_profile:
            context["role"] = user_profile["role"]

        # Extract user expertise level
        if "expertise_level" in user_profile:
            context["expertise"] = user_profile["expertise_level"]

        return context if context else None

    def _get_history_context(self, session_history: Optional[List[str]]) -> List[str]:
        """Get recent session history"""
        if not session_history:
            return []

        # Return most recent queries
        return session_history[-self.max_history_items:]

    def _get_domain_context(self, domain: Optional[str]) -> Optional[Dict]:
        """Get domain-specific context"""
        if not domain or domain not in self.domain_knowledge:
            return None

        domain_info = self.domain_knowledge[domain]

        context = {
            "domain": domain,
            "key_concepts": domain_info.get("key_concepts", []),
            "common_queries": domain_info.get("common_queries", []),
        }

        return context

    def _build_injected_context(
        self,
        user_context: Optional[Dict],
        history_context: List[str],
        domain_context: Optional[Dict],
    ) -> str:
        """Build context string to inject into query"""
        context_parts = []

        # Add user role context
        if user_context and "role" in user_context:
            context_parts.append(f"User role: {user_context['role']}")

        # Add session history
        if history_context:
            recent_queries = "; ".join(history_context)
            context_parts.append(f"Recent queries: {recent_queries}")

        # Add domain context
        if domain_context and "key_concepts" in domain_context:
            concepts = ", ".join(domain_context["key_concepts"][:3])
            context_parts.append(f"Domain concepts: {concepts}")

        return " | ".join(context_parts)

    def format_query_with_context(
        self, query: str, enhanced_context: EnhancedContext
    ) -> str:
        """Format query with injected context for retrieval"""
        if not enhanced_context.injected_context:
            return query

        # Return query with context prefix
        return f"{enhanced_context.injected_context}\nQuery: {query}"
