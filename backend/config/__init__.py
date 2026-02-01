"""
Backend Configuration Module

Provides enterprise-scale settings for vector stores, quantization, and retrieval.
"""

# Import with underscore module name for Python compatibility
import importlib.util
import os

# Load the kebab-case module
_module_path = os.path.join(os.path.dirname(__file__), "vector-store-quantization-settings.py")
_spec = importlib.util.spec_from_file_location("vector_store_quantization_settings", _module_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

# Re-export main items
VectorStoreSettings = _module.VectorStoreSettings
VECTOR_STORE_PROFILES = _module.VECTOR_STORE_PROFILES
get_vector_store_config = _module.get_vector_store_config
get_quantization_recommendations = _module.get_quantization_recommendations
QuantizationMode = _module.QuantizationMode

__all__ = [
    "VectorStoreSettings",
    "VECTOR_STORE_PROFILES",
    "get_vector_store_config",
    "get_quantization_recommendations",
    "QuantizationMode",
]
