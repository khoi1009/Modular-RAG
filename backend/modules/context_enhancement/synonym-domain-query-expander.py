import asyncio
from typing import Dict, List, Optional, Set

import async_timeout
from langchain.prompts import PromptTemplate

from backend.logger import logger
from backend.modules.model_gateway.model_gateway import model_gateway
from backend.types import ConfiguredBaseModel, ModelConfig


class ExpandedQuery(ConfiguredBaseModel):
    """Result of query expansion"""
    original_query: str
    expanded_terms: List[str]
    synonyms: Dict[str, List[str]]
    domain_terms: List[str]


class SynonymDomainQueryExpander:
    """
    Query expander that adds synonyms and domain-specific terms
    to improve retrieval coverage
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.use_llm = config.get("use_llm", True)
        self.domain_dict = config.get("domain_dictionary", {})
        self.timeout = config.get("timeout", 10)

        if self.use_llm:
            self.model_config = ModelConfig(
                name=config.get("model_name", "ollama/llama3"),
                parameters=config.get("model_parameters", {"temperature": 0.3}),
            )

    async def expand(
        self, query: str, context: Optional[Dict] = None
    ) -> ExpandedQuery:
        """Expand query with synonyms and domain terms"""
        expanded_terms = []
        synonyms = {}
        domain_terms = []

        # Extract domain from context if available
        domain = context.get("domain") if context else None

        if self.use_llm:
            # LLM-based expansion
            llm_result = await self._llm_expand(query, domain)
            expanded_terms = llm_result.get("expanded_terms", [])
            synonyms = llm_result.get("synonyms", {})
        else:
            # Simple expansion using word list
            expanded_terms = await self._simple_expand(query)

        # Add domain-specific terms
        if domain and domain in self.domain_dict:
            domain_terms = self._get_domain_terms(query, domain)

        return ExpandedQuery(
            original_query=query,
            expanded_terms=expanded_terms,
            synonyms=synonyms,
            domain_terms=domain_terms,
        )

    async def _llm_expand(self, query: str, domain: Optional[str]) -> Dict:
        """Use LLM to generate synonyms and related terms"""
        try:
            async with async_timeout.timeout(self.timeout):
                llm = model_gateway.get_llm_from_model_config(self.model_config)

                domain_context = f" in the {domain} domain" if domain else ""

                prompt = PromptTemplate(
                    input_variables=["query", "domain_context"],
                    template="""For this query{domain_context}, list:
1. Key terms and their synonyms
2. Related terminology that would help find relevant information

Query: {query}

Provide a concise list of terms separated by commas:""",
                )

                chain = prompt | llm
                response = await chain.ainvoke(
                    {"query": query, "domain_context": domain_context}
                )

                # Parse response
                terms = self._parse_terms(response.content)

                return {
                    "expanded_terms": terms,
                    "synonyms": {},  # Could parse more structured output
                }

        except asyncio.TimeoutError:
            logger.warning(f"LLM expansion timeout for query: {query}")
            return {"expanded_terms": [], "synonyms": {}}
        except Exception as e:
            logger.error(f"LLM expansion error: {e}")
            return {"expanded_terms": [], "synonyms": {}}

    async def _simple_expand(self, query: str) -> List[str]:
        """Simple expansion using basic heuristics"""
        # Basic acronym expansion
        expanded = []

        # Common acronyms
        acronyms = {
            "AI": ["artificial intelligence"],
            "ML": ["machine learning"],
            "NLP": ["natural language processing"],
            "API": ["application programming interface"],
            "DB": ["database"],
            "RAG": ["retrieval augmented generation"],
        }

        words = query.upper().split()
        for word in words:
            if word in acronyms:
                expanded.extend(acronyms[word])

        return expanded

    def _get_domain_terms(self, query: str, domain: str) -> List[str]:
        """Get domain-specific terms from dictionary"""
        domain_terms = self.domain_dict.get(domain, {})
        relevant_terms = []

        # Find domain terms related to query words
        query_lower = query.lower()
        for key, terms in domain_terms.items():
            if key.lower() in query_lower:
                relevant_terms.extend(terms)

        return relevant_terms[:5]  # Limit to 5 terms

    def _parse_terms(self, response: str) -> List[str]:
        """Parse terms from LLM response"""
        # Split by comma and newline
        terms = []
        for line in response.split("\n"):
            for term in line.split(","):
                term = term.strip().lstrip("0123456789.-) ")
                if term and len(term) > 2:
                    terms.append(term)

        return list(set(terms))[:10]  # Deduplicate and limit
