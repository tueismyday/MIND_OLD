"""
Embedding models and operations for the Agentic RAG Medical Documentation System.

UPDATED: Now supports both TEI servers (GPU-accelerated) and local model loading.
Automatically selects based on INFERENCE_MODE configuration.

When using TEI mode:
- Embeddings are computed via HTTP API to TEI server
- No model loading in this process
- GPU memory stays available for vLLM

When using local mode:
- Models are loaded into this process
- Uses configured device (CPU/GPU)
"""

import os
from typing import List, Union

# Import configuration
from config.settings import (
    EMBEDDING_MODEL_NAME, 
    EMBEDDING_DEVICE,
    ACTUAL_INFERENCE_MODE,
    TEI_EMBEDDING_URL
)

# Global cache for embedding model (singleton pattern)
_embedding_model_cache = None
_embedding_device_cache = None


class TEIEmbeddingWrapper:
    """
    Wrapper that provides HuggingFaceEmbeddings-compatible interface
    using TEI server backend.
    """
    
    def __init__(self, base_url: str = None):
        """
        Initialize TEI embedding wrapper.
        
        Args:
            base_url: TEI server URL (default from config)
        """
        import requests
        
        self.base_url = (base_url or TEI_EMBEDDING_URL).rstrip('/')
        self._dimension = None
        self._requests = requests
        
        # Verify connection
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify TEI server is accessible."""
        try:
            response = self._requests.get(f"{self.base_url}/info", timeout=5)
            if response.status_code == 200:
                info = response.json()
                print(f"[TEI Embeddings] Connected to: {self.base_url}")
                print(f"[TEI Embeddings] Model: {info.get('model_id', 'unknown')}")
            else:
                print(f"[TEI WARNING] Server returned status {response.status_code}")
        except Exception as e:
            print(f"[TEI WARNING] Connection check failed: {e}")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            List of floats representing the embedding
        """
        response = self._requests.post(
            f"{self.base_url}/embed",
            json={"inputs": [text]},
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        
        # Cache dimension
        if self._dimension is None:
            self._dimension = len(result[0])
            print(f"[TEI Embeddings] Dimension: {self._dimension}")
        
        return result[0]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.
        
        Args:
            texts: List of documents to embed
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        response = self._requests.post(
            f"{self.base_url}/embed",
            json={"inputs": texts},
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version (falls back to sync)."""
        return self.embed_query(text)
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version (falls back to sync)."""
        return self.embed_documents(texts)


def get_embeddings():
    """
    Get the configured embedding model with robust error handling.
    
    Automatically selects between TEI server and local model based on
    INFERENCE_MODE configuration.
    
    Uses singleton pattern - only initializes once.
    
    Returns:
        Embedding model instance (TEI wrapper or HuggingFaceEmbeddings)
    """
    global _embedding_model_cache, _embedding_device_cache

    # Return cached model if already loaded
    if _embedding_model_cache is not None:
        return _embedding_model_cache

    # Use TEI mode
    if ACTUAL_INFERENCE_MODE == "tei":
        print(f"[INFO] Using TEI embedding server at {TEI_EMBEDDING_URL}")
        
        embeddings = TEIEmbeddingWrapper(TEI_EMBEDDING_URL)
        
        # Test the connection
        try:
            test_embedding = embeddings.embed_query("test")
            print(f"[SUCCESS] TEI embeddings ready (dimension: {len(test_embedding)})")
        except Exception as e:
            print(f"[ERROR] TEI embedding test failed: {e}")
            print(f"[INFO] Make sure TEI server is running on {TEI_EMBEDDING_URL}")
            raise
        
        _embedding_model_cache = embeddings
        _embedding_device_cache = "tei"
        return embeddings
    
    # Use local mode (original implementation)
    return _get_local_embeddings()


def _get_local_embeddings():
    """
    Load embeddings locally (original implementation).
    
    Used when INFERENCE_MODE="local".
    """
    global _embedding_model_cache, _embedding_device_cache
    
    from langchain_huggingface import HuggingFaceEmbeddings
    import torch
    
    print(f"[INFO] Loading local embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"[INFO] Target device: {EMBEDDING_DEVICE}")

    device_attempts = []
    if EMBEDDING_DEVICE == "cuda" or EMBEDDING_DEVICE.startswith("cuda:"):
        if torch.cuda.is_available():
            device_attempts = [EMBEDDING_DEVICE, "cpu"]
        else:
            print(f"[WARNING] CUDA requested but not available, using CPU")
            device_attempts = ["cpu"]
    else:
        device_attempts = [EMBEDDING_DEVICE]

    for device in device_attempts:
        # Check GPU memory before loading
        if device.startswith("cuda"):
            gpu_idx = 0 if device == "cuda" else int(device.split(":")[1])
            free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(gpu_idx)
            free_mem = free_mem_bytes / (1024**3)
            
            print(f"[EMBEDDING GPU] Free: {free_mem:.2f}GB")
            
            MIN_FREE_FOR_EMBEDDING = 2.23
            if free_mem < MIN_FREE_FOR_EMBEDDING:
                print(f"[EMBEDDING GPU] Insufficient memory, falling back to CPU")
                continue

        try:
            print(f"[INFO] Attempting to load on {device}...")
            
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': device}
            )
            
            # Verify with test embedding
            test_vector = embeddings.embed_query("Test embedding")
            print(f"[SUCCESS] Embeddings loaded on {device} (dim: {len(test_vector)})")
            
            _embedding_model_cache = embeddings
            _embedding_device_cache = device
            return embeddings

        except torch.cuda.OutOfMemoryError:
            print(f"[WARNING] GPU OOM on {device}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if device != "cpu":
                continue
            raise

        except Exception as e:
            print(f"[WARNING] Failed on {device}: {e}")
            if device != device_attempts[-1]:
                continue
            raise

    raise Exception("Failed to load embeddings on any device")

def reset_embedding_cache():
    """Force reload of embedding model (call after changing models)."""
    global _embedding_model_cache, _embedding_device_cache
    _embedding_model_cache = None
    _embedding_device_cache = None
    print("[INFO] Embedding cache cleared")

# Sentence-transformers compatible interface for hybrid_search
class SentenceTransformerWrapper:
    """
    Wrapper that provides sentence-transformers compatible encode() method
    for both TEI and local embeddings.
    """
    
    def __init__(self, embeddings=None):
        """
        Initialize wrapper.
        
        Args:
            embeddings: Embedding model (auto-detected if None)
        """
        self._embeddings = embeddings or get_embeddings()
    
    def encode(self, texts: Union[str, List[str]], **kwargs):
        """
        Encode texts into embeddings.
        
        Compatible with sentence-transformers encode() interface.
        
        Args:
            texts: Single text or list of texts
            **kwargs: Ignored (for compatibility)
            
        Returns:
            numpy array of embeddings
        """
        import numpy as np
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self._embeddings.embed_documents(texts)
        return np.array(embeddings)


def get_embedding_encoder():
    """
    Get a sentence-transformers compatible encoder.
    
    Returns:
        SentenceTransformerWrapper instance
    """
    return SentenceTransformerWrapper()
