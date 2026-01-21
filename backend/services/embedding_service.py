from ..config import settings
from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-small-en"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Test the model
            test_embedding = self.model.encode(["test"], normalize_embeddings=True)
            logger.info(f"Embedding model loaded successfully. Embedding dimension: {len(test_embedding[0])}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def get_embeddings(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not texts:
            return []
        
        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            
            return embeddings.tolist()
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return empty embeddings to avoid crashing
            return [[] for _ in texts]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return 384  # Default for bge-small-en