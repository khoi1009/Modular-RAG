"""
Vector Store Settings

Enterprise-scale configuration for vector databases with quantization support.
Supports 10M+ document datasets with sub-30ms latency via Binary Quantization.

Usage:
    from backend.config.vector_store_settings import get_vector_store_config

    # Get config with binary quantization for enterprise scale
    config = get_vector_store_config(quantization_mode="binary")
"""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

# Quantization mode options
QuantizationMode = Literal["none", "scalar", "binary"]


class VectorStoreSettings(BaseModel):
    """
    Vector store settings with quantization support for enterprise-scale datasets.

    Binary Quantization Benefits:
    - 32x memory reduction (1-bit vs 32-bit float per dimension)
    - Sub-30ms latency for 10M+ document retrieval
    - Accuracy preserved via rescoring mechanism

    Recommended Settings by Scale:
    - Small (<100K docs): quantization_mode="none"
    - Medium (100K-1M docs): quantization_mode="scalar"
    - Large (1M-10M+ docs): quantization_mode="binary"
    """

    quantization_mode: QuantizationMode = Field(
        default="none",
        description="Quantization mode: 'none', 'scalar' (4x compression), 'binary' (32x compression)",
    )

    rescore_multiplier: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Oversampling multiplier for rescoring (2-3x recommended for binary)",
    )

    always_ram: bool = Field(
        default=True,
        description="Keep quantized vectors in RAM for faster search",
    )

    hnsw_ef: int = Field(
        default=128,
        ge=16,
        le=512,
        description="HNSW search ef parameter (higher = more accurate, slower)",
    )

    on_disk: bool = Field(
        default=True,
        description="Store original vectors on disk to save RAM",
    )

    def to_config_dict(self) -> Dict[str, Any]:
        """Convert to VectorDBConfig.config dictionary format."""
        return {
            "quantization_mode": self.quantization_mode,
            "rescore_multiplier": self.rescore_multiplier,
            "always_ram": self.always_ram,
            "hnsw_ef": self.hnsw_ef,
            "on_disk": self.on_disk,
        }


# Pre-configured profiles for common use cases
VECTOR_STORE_PROFILES: Dict[str, VectorStoreSettings] = {
    # Default: No quantization, best accuracy
    "default": VectorStoreSettings(
        quantization_mode="none",
    ),
    # Development: Fast, low memory
    "development": VectorStoreSettings(
        quantization_mode="scalar",
        rescore_multiplier=2.0,
    ),
    # Production: Balanced accuracy and performance
    "production": VectorStoreSettings(
        quantization_mode="scalar",
        rescore_multiplier=2.5,
        hnsw_ef=256,
    ),
    # Enterprise: Maximum scale (10M+ docs)
    "enterprise": VectorStoreSettings(
        quantization_mode="binary",
        rescore_multiplier=3.0,
        hnsw_ef=128,
        always_ram=True,
    ),
    # Legal/Medical: High-accuracy for domain-specific
    "high_accuracy": VectorStoreSettings(
        quantization_mode="binary",
        rescore_multiplier=4.0,
        hnsw_ef=256,
    ),
}


def get_vector_store_config(
    profile: Optional[str] = None,
    quantization_mode: Optional[QuantizationMode] = None,
    rescore_multiplier: Optional[float] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """
    Get vector store configuration for VectorDBConfig.config.

    Args:
        profile: Pre-configured profile name (default, development, production, enterprise, high_accuracy)
        quantization_mode: Override quantization mode
        rescore_multiplier: Override rescore multiplier
        **overrides: Additional config overrides

    Returns:
        Configuration dictionary for VectorDBConfig.config

    Examples:
        # Use enterprise profile for 10M+ docs
        config = get_vector_store_config(profile="enterprise")

        # Custom binary quantization
        config = get_vector_store_config(
            quantization_mode="binary",
            rescore_multiplier=3.5,
        )

        # Use in VectorDBConfig
        vector_db_config = VectorDBConfig(
            provider="qdrant",
            url="http://localhost:6333",
            config=get_vector_store_config(profile="enterprise"),
        )
    """
    # Start with profile or default
    if profile and profile in VECTOR_STORE_PROFILES:
        settings = VECTOR_STORE_PROFILES[profile].model_copy()
    else:
        settings = VECTOR_STORE_PROFILES["default"].model_copy()

    # Apply overrides
    if quantization_mode is not None:
        settings.quantization_mode = quantization_mode
    if rescore_multiplier is not None:
        settings.rescore_multiplier = rescore_multiplier

    config = settings.to_config_dict()
    config.update(overrides)

    return config


def get_quantization_recommendations(doc_count: int) -> Dict[str, Any]:
    """
    Get recommended settings based on expected document count.

    Args:
        doc_count: Expected number of documents

    Returns:
        Recommended configuration and explanation
    """
    if doc_count < 100_000:
        return {
            "profile": "default",
            "quantization_mode": "none",
            "explanation": "Small dataset - full precision recommended for best accuracy",
            "expected_memory_gb": doc_count * 0.001 * 1.5,  # ~1.5KB per doc
        }
    elif doc_count < 1_000_000:
        return {
            "profile": "production",
            "quantization_mode": "scalar",
            "explanation": "Medium dataset - scalar quantization provides 4x compression with minimal accuracy loss",
            "expected_memory_gb": doc_count * 0.001 * 0.4,  # ~0.4KB per doc
        }
    else:
        return {
            "profile": "enterprise",
            "quantization_mode": "binary",
            "explanation": "Large dataset - binary quantization provides 32x compression, use rescoring for accuracy",
            "expected_memory_gb": doc_count * 0.001 * 0.05,  # ~0.05KB per doc
        }
