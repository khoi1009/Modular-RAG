"""Extract verifiable claims from generated text."""
import asyncio
import re
from typing import Dict, List, Optional

import async_timeout
from langchain.prompts import PromptTemplate

from backend.logger import logger
from backend.modules.model_gateway.model_gateway import model_gateway
from backend.modules.verification.schemas import Claim
from backend.types import ModelConfig


class ClaimExtractor:
    """Extract verifiable claims from text using LLM"""

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.model_config = ModelConfig(
            name=config.get("model_name", "ollama/llama3"),
            parameters=config.get("model_parameters", {"temperature": 0.0}),
        )
        self.timeout = config.get("timeout", 15)

    async def extract_claims(self, text: str) -> List[Claim]:
        """Extract verifiable claims from text"""
        try:
            async with async_timeout.timeout(self.timeout):
                llm = model_gateway.get_llm_from_model_config(self.model_config)

                prompt = PromptTemplate(
                    input_variables=["text"],
                    template="""Extract verifiable factual claims from the following text.
For each claim, classify it as: factual, opinion, or conditional.

Format your response as a numbered list:
1. [TYPE] Claim text
2. [TYPE] Claim text

Text: {text}

Claims:""",
                )

                chain = prompt | llm
                response = await chain.ainvoke({"text": text})

                claims = self._parse_claims(response.content, text)
                return claims

        except asyncio.TimeoutError:
            logger.warning("Claim extraction timeout")
            return self._fallback_sentence_split(text)
        except Exception as e:
            logger.error(f"Claim extraction error: {e}")
            return self._fallback_sentence_split(text)

    def _parse_claims(self, llm_response: str, original_text: str) -> List[Claim]:
        """Parse LLM response into Claim objects"""
        claims = []
        lines = llm_response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse format: "1. [factual] Some claim"
            match = re.match(r"^\d+\.\s*\[(\w+)\]\s*(.+)$", line)
            if match:
                claim_type = match.group(1).lower()
                claim_text = match.group(2).strip()

                # Find position in original text
                start_idx = original_text.lower().find(claim_text.lower())
                if start_idx == -1:
                    start_idx = 0
                end_idx = start_idx + len(claim_text)

                claims.append(
                    Claim(
                        text=claim_text,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        claim_type=claim_type,
                    )
                )

        return claims if claims else self._fallback_sentence_split(original_text)

    def _fallback_sentence_split(self, text: str) -> List[Claim]:
        """Fallback: split text into sentences as claims"""
        sentences = re.split(r"[.!?]+", text)
        claims = []

        current_pos = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            start_idx = text.find(sentence, current_pos)
            if start_idx == -1:
                start_idx = current_pos
            end_idx = start_idx + len(sentence)
            current_pos = end_idx

            claims.append(
                Claim(
                    text=sentence,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    claim_type="factual",
                )
            )

        return claims
