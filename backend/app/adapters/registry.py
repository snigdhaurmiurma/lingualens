"""Model Registry for LinguaLens.

Manages catalog discovery, dynamic instantiation, and memory lifecycle
for multilingual embedding model adapters.
"""

from __future__ import annotations

import logging
from typing import Mapping, Type

from app.adapters.base import BaseEmbeddingAdapter, ModelMetadata
from app.adapters.e5 import MultilingualE5Adapter
from app.adapters.minilm import MiniLMEmbeddingAdapter

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry managing embedding model adapters with lifecycle control.

    Enables:
    1. Looking up models by identifier or alias.
    2. Lazy instantiation and loading.
    3. Unloading unused models to conserve RAM on resource-constrained hardware.
    """

    def __init__(self, default_device: str = "cpu") -> None:
        self.default_device = default_device
        self._adapter_classes: dict[str, Type[BaseEmbeddingAdapter]] = {
            MiniLMEmbeddingAdapter.MODEL_IDENTIFIER: MiniLMEmbeddingAdapter,
            MultilingualE5Adapter.MODEL_IDENTIFIER: MultilingualE5Adapter,
            # Aliases for convenience
            "minilm": MiniLMEmbeddingAdapter,
            "e5-small": MultilingualE5Adapter,
        }
        self._active_instances: dict[str, BaseEmbeddingAdapter] = {}

    def register(
        self,
        model_identifier: str,
        adapter_class: Type[BaseEmbeddingAdapter],
    ) -> None:
        """Register a new model adapter class."""
        self._adapter_classes[model_identifier] = adapter_class
        logger.info("Registered adapter for model '%s'", model_identifier)

    def is_registered(self, model_identifier: str) -> bool:
        """Check if a model identifier or alias is registered."""
        return model_identifier in self._adapter_classes

    def get_adapter(
        self,
        model_identifier: str,
        auto_load: bool = False,
    ) -> BaseEmbeddingAdapter:
        """Retrieve or instantiate an adapter instance.

        Args:
            model_identifier: Canonical model ID or alias.
            auto_load: If True, calls adapter.load() immediately.

        Returns:
            Instantiated BaseEmbeddingAdapter.

        Raises:
            KeyError: If model_identifier is not registered.
        """
        if model_identifier not in self._adapter_classes:
            raise KeyError(
                f"Model '{model_identifier}' is not registered in ModelRegistry. "
                f"Available models: {list(self._adapter_classes.keys())}"
            )

        # Retrieve or instantiate
        if model_identifier not in self._active_instances:
            adapter_cls = self._adapter_classes[model_identifier]
            instance = adapter_cls(device=self.default_device)
            self._active_instances[model_identifier] = instance

        adapter = self._active_instances[model_identifier]
        if auto_load and not adapter.is_loaded:
            adapter.load()

        return adapter

    def list_available_models(self) -> list[ModelMetadata]:
        """Return metadata for all registered models without loading their weights."""
        metadata_list: list[ModelMetadata] = []
        seen_ids: set[str] = set()

        for key, adapter_cls in self._adapter_classes.items():
            # Instantiate dummy un-loaded instance to read static metadata
            dummy = adapter_cls(device="cpu")
            meta = dummy.get_metadata()
            if meta.model_id not in seen_ids:
                seen_ids.add(meta.model_id)
                metadata_list.append(meta)

        return metadata_list

    def unload_all(self) -> None:
        """Unload all active model instances to free host memory."""
        for model_id, instance in self._active_instances.items():
            if instance.is_loaded:
                logger.info("Unloading model '%s' from memory", model_id)
                instance.unload()
        self._active_instances.clear()


# Global default registry instance
default_registry = ModelRegistry()
