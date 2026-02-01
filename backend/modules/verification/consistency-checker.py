"""Check internal consistency and knowledge base alignment."""
import asyncio
import re
from typing import Dict, List, Optional

import async_timeout
from langchain.prompts import PromptTemplate

from backend.logger import logger
from backend.modules.model_gateway.model_gateway import model_gateway
from backend.types import ModelConfig


class ConsistencyChecker:
    """Verify internal consistency of answers and alignment with knowledge base"""

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.model_config = ModelConfig(
            name=config.get("model_name", "ollama/llama3"),
            parameters=config.get("model_parameters", {"temperature": 0.0}),
        )
        self.timeout = config.get("timeout", 15)

    async def check_internal_consistency(self, answer: str) -> float:
        """
        Check if answer contains contradictions or inconsistencies.

        Args:
            answer: Generated answer text

        Returns:
            Consistency score (0=inconsistent, 1=fully consistent)
        """
        try:
            # Quick checks for obvious issues
            if not answer or len(answer.strip()) < 10:
                return 0.0

            # Check for explicit contradictions
            has_contradiction = self._detect_explicit_contradictions(answer)
            if has_contradiction:
                logger.warning("Explicit contradiction detected in answer")
                return 0.3

            # Use LLM to check for logical consistency
            async with async_timeout.timeout(self.timeout):
                consistency_score = await self._llm_consistency_check(answer)
                return consistency_score

        except asyncio.TimeoutError:
            logger.warning("Consistency check timeout")
            return 0.5
        except Exception as e:
            logger.error(f"Internal consistency check error: {e}")
            return 0.5

    async def check_knowledge_consistency(
        self, answer: str, knowledge_base: List[str]
    ) -> float:
        """
        Check if answer aligns with knowledge base content.

        Args:
            answer: Generated answer
            knowledge_base: List of knowledge base document contents

        Returns:
            Consistency score (0=inconsistent, 1=fully consistent)
        """
        try:
            if not answer or not knowledge_base:
                return 0.5

            # Build knowledge context
            context = self._build_knowledge_context(knowledge_base)

            async with async_timeout.timeout(self.timeout):
                llm = model_gateway.get_llm_from_model_config(self.model_config)

                prompt = PromptTemplate(
                    input_variables=["answer", "context"],
                    template="""Given the knowledge base below, determine if the answer is consistent with the provided information.

Knowledge Base:
{context}

Answer to verify:
{answer}

Is the answer consistent with the knowledge base? Consider:
- Does it contradict any facts in the knowledge base?
- Does it make claims not supported by the knowledge base?
- Is it logically aligned with the knowledge base information?

Rate consistency from 0.0 (completely inconsistent) to 1.0 (fully consistent).
Respond with only the numeric score.

Consistency Score:""",
                )

                chain = prompt | llm
                response = await chain.ainvoke({"answer": answer, "context": context})

                # Parse score from response
                score = self._parse_score(response.content)
                return score

        except asyncio.TimeoutError:
            logger.warning("Knowledge consistency check timeout")
            return 0.5
        except Exception as e:
            logger.error(f"Knowledge consistency check error: {e}")
            return 0.5

    def _detect_explicit_contradictions(self, text: str) -> bool:
        """Detect explicit contradiction patterns in text"""
        # Patterns that often indicate contradictions
        contradiction_patterns = [
            r"(however|but|although).+(not|never|no|cannot)",
            r"(yes|true|correct).+(no|false|incorrect)",
            r"(is|are).+(is not|are not|isn't|aren't)",
            r"(can|could|will).+(cannot|couldn't|won't)",
        ]

        text_lower = text.lower()
        for pattern in contradiction_patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    async def _llm_consistency_check(self, answer: str) -> float:
        """Use LLM to check internal consistency"""
        try:
            llm = model_gateway.get_llm_from_model_config(self.model_config)

            prompt = PromptTemplate(
                input_variables=["answer"],
                template="""Analyze the following answer for internal consistency.

Answer:
{answer}

Check for:
1. Logical contradictions within the text
2. Inconsistent statements or claims
3. Self-contradicting information

Rate the internal consistency from 0.0 (highly inconsistent) to 1.0 (fully consistent).
Respond with only the numeric score.

Consistency Score:""",
            )

            chain = prompt | llm
            response = await chain.ainvoke({"answer": answer})

            score = self._parse_score(response.content)
            return score

        except Exception as e:
            logger.error(f"LLM consistency check error: {e}")
            return 0.5

    def _build_knowledge_context(self, knowledge_base: List[str]) -> str:
        """Build context string from knowledge base"""
        # Limit to first 5 documents to stay within token limits
        kb_parts = []
        for i, content in enumerate(knowledge_base[:5], 1):
            # Truncate long documents
            truncated = content[:400] if len(content) > 400 else content
            kb_parts.append(f"[KB Doc {i}]: {truncated}")

        return "\n\n".join(kb_parts)

    def _parse_score(self, response: str) -> float:
        """Parse numeric score from LLM response"""
        try:
            # Try to find a number between 0 and 1
            response = response.strip()

            # Look for decimal number
            match = re.search(r"(\d*\.?\d+)", response)
            if match:
                score = float(match.group(1))
                # Ensure score is in valid range
                if score > 1.0:
                    score = score / 10.0 if score <= 10.0 else 1.0
                score = max(0.0, min(1.0, score))
                return score

            # Fallback to neutral score
            return 0.5

        except Exception as e:
            logger.error(f"Score parsing error: {e}")
            return 0.5
