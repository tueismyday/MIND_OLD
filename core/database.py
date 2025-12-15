"""
Vector database management for the Agentic RAG Medical Documentation System.
Handles ChromaDB initialization and operations.
"""

from langchain_chroma import Chroma
from config.settings import (
    GUIDELINE_DB_DIR,
    PATIENT_DB_DIR,
    GENERATED_DOCS_DB_DIR,
    ensure_directories,
    ACTUAL_INFERENCE_MODE
)


class DatabaseManager:
    """
    Manages all vector database instances.
    
    Uses lazy initialization for both embedding models and database connections
    to avoid issues with module import order and GPU memory management.
    """

    def __init__(self):
        """Initialize the database manager."""
        ensure_directories()
        
        # Lazy-loaded instances
        self._embeddings = None
        self._reranker = None
        self._patient_db = None
        self._guideline_db = None
        self._generated_docs_db = None
        
        # Track initialization status
        self._initialized = False
    
    def _ensure_models_loaded(self):
        """
        Ensure embedding and reranker models are loaded.
        Called lazily on first database access.
        """
        if self._initialized:
            return
        
        print(f"[DatabaseManager] Initializing models (mode: {ACTUAL_INFERENCE_MODE})...")
        
        # Import here to avoid circular imports and allow settings to be configured first
        from core.embeddings import get_embeddings
        from core.reranker import get_reranker
        
        try:
            # Load embedding model
            print("[DatabaseManager] Loading embedding model...")
            self._embeddings = get_embeddings()
            
            # Load reranker model
            print("[DatabaseManager] Loading reranker model...")
            self._reranker = get_reranker()
            
            self._initialized = True
            print("[DatabaseManager] All models loaded successfully")
            
        except Exception as e:
            print(f"[DatabaseManager] ERROR loading models: {e}")
            print("[DatabaseManager] Check that:")
            if ACTUAL_INFERENCE_MODE == "tei":
                print("  - TEI servers are running (embedding on port 8080, reranker on port 8081)")
                print("  - You can test with: curl http://localhost:8080/health")
            else:
                print("  - Sufficient GPU/CPU memory is available")
                print("  - Model names are correct in settings.py")
            raise
    
    @property
    def embeddings(self):
        """Get the embedding model (lazy-loaded)."""
        self._ensure_models_loaded()
        return self._embeddings
    
    @property
    def reranker(self):
        """Get the reranker model (lazy-loaded)."""
        self._ensure_models_loaded()
        return self._reranker
    
    @property
    def patient_db(self) -> Chroma:
        """Get the patient records vector database."""
        self._ensure_models_loaded()
        
        if self._patient_db is None:
            print(f"[DatabaseManager] Connecting to patient database at {PATIENT_DB_DIR}")
            self._patient_db = Chroma(
                persist_directory=str(PATIENT_DB_DIR), 
                embedding_function=self._embeddings
            )
        return self._patient_db
    
    @property
    def guideline_db(self) -> Chroma:
        """Get the guidelines vector database."""
        self._ensure_models_loaded()
        
        if self._guideline_db is None:
            print(f"[DatabaseManager] Connecting to guideline database at {GUIDELINE_DB_DIR}")
            self._guideline_db = Chroma(
                persist_directory=str(GUIDELINE_DB_DIR), 
                embedding_function=self._embeddings
            )
        return self._guideline_db
    
    @property
    def generated_docs_db(self) -> Chroma:
        """Get the generated documents vector database."""
        self._ensure_models_loaded()
        
        if self._generated_docs_db is None:
            print(f"[DatabaseManager] Connecting to generated docs database at {GENERATED_DOCS_DB_DIR}")
            self._generated_docs_db = Chroma(
                persist_directory=str(GENERATED_DOCS_DB_DIR), 
                embedding_function=self._embeddings
            )
        return self._generated_docs_db
    
    def get_database_info(self) -> dict:
        """Get information about all databases."""
        info = {
            "patient_record_chunks": 0,
            "guideline_chunks": 0,
            "generated_document_chunks": 0,
            "inference_mode": ACTUAL_INFERENCE_MODE,
            "models_loaded": self._initialized
        }
        
        try:
            info["patient_record_chunks"] = self.patient_db._collection.count()
        except Exception as e:
            print(f"[DatabaseManager] Warning: Could not get patient DB count: {e}")
            
        try:
            info["guideline_chunks"] = self.guideline_db._collection.count()
        except Exception as e:
            print(f"[DatabaseManager] Warning: Could not get guideline DB count: {e}")
            
        try:
            info["generated_document_chunks"] = self.generated_docs_db._collection.count()
        except Exception as e:
            pass  # Generated docs DB might not exist yet
            
        return info
    
    def print_database_info(self):
        """Print database information to console."""
        info = self.get_database_info()
        
        print(f"\n{'='*60}")
        print(f"Database Status (Inference Mode: {info['inference_mode'].upper()})")
        print(f"{'='*60}")
        print(f"  Patient records:     {info['patient_record_chunks']} chunks")
        print(f"  Guidelines:          {info['guideline_chunks']} chunks")
        print(f"  Generated documents: {info['generated_document_chunks']} chunks")
        print(f"  Models loaded:       {info['models_loaded']}")
        print(f"{'='*60}\n")
    
    def reset_connections(self):
        """
        Reset all database connections.
        Useful if underlying data has changed.
        """
        self._patient_db = None
        self._guideline_db = None
        self._generated_docs_db = None
        print("[DatabaseManager] Database connections reset")
    
    def health_check(self) -> dict:
        """
        Perform a health check on all components.
        
        Returns:
            dict with status of each component
        """
        status = {
            "embeddings": False,
            "reranker": False,
            "patient_db": False,
            "guideline_db": False,
            "generated_docs_db": False,
            "errors": []
        }
        
        # Check embeddings
        try:
            self._ensure_models_loaded()
            # Test embedding
            test_result = self._embeddings.embed_query("test")
            status["embeddings"] = len(test_result) > 0
        except Exception as e:
            status["errors"].append(f"Embeddings: {str(e)}")
        
        # Check reranker
        try:
            if self._reranker is not None:
                status["reranker"] = True
        except Exception as e:
            status["errors"].append(f"Reranker: {str(e)}")
        
        # Check databases
        try:
            _ = self.patient_db._collection.count()
            status["patient_db"] = True
        except Exception as e:
            status["errors"].append(f"Patient DB: {str(e)}")
        
        try:
            _ = self.guideline_db._collection.count()
            status["guideline_db"] = True
        except Exception as e:
            status["errors"].append(f"Guideline DB: {str(e)}")
        
        try:
            _ = self.generated_docs_db._collection.count()
            status["generated_docs_db"] = True
        except Exception as e:
            # This one might not exist, which is OK
            pass
        
        return status


# Global database manager instance
# Note: Models are loaded lazily on first access, not at import time
db_manager = DatabaseManager()
