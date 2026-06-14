import threading
from typing import Dict, Any, Optional
import torch

class ModelCacheManager:
    """
    Singleton manager for caching loaded deep learning models across different
    pipeline classes (Classification, Segmentation, etc.).
    Handles concurrency, lookup, and resource cleanup.
    """
    # Class attribute to hold the cached models: {job_id: {model, device, transform, ...}}
    _loaded_models: Dict[str, Dict[str, Any]] = {}
    
    # Lock for thread-safe access to the cache
    _cache_lock = threading.Lock()
    
    def get_model_bundle(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Safely retrieve a loaded model bundle."""
        with self._cache_lock:
            return self._loaded_models.get(job_id)

    def set_model_bundle(self, job_id: str, bundle: Dict[str, Any]):
        """Safely store a loaded model bundle."""
        with self._cache_lock:
            self._loaded_models[job_id] = bundle

    def unload_model(self, job_id: str) -> Dict[str, Any]:
        """Unload a specific model and free up resources."""
        with self._cache_lock:
            if job_id not in self._loaded_models:
                return {"status": "not_found", "job_id": job_id}
            
            del self._loaded_models[job_id]
            
            # Global cleanup after release
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {"status": "unloaded", "job_id": job_id}

    def unload_all_models(self) -> Dict[str, Any]:
        """Unload all models from the cache."""
        with self._cache_lock:
            count = len(self._loaded_models)
            self._loaded_models.clear()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return {"status": "unloaded", "num_models_cleared": count}

# Create a single global instance of the manager
MODEL_CACHE_MANAGER = ModelCacheManager()