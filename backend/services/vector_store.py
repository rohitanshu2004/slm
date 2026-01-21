from ..config import settings
from typing import List, Dict, Any, Optional
import uuid
import logging
import os
import pickle

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, collection_name: str, embedding_service, persist_directory: str = "./data/vector_db"):
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        self.persist_directory = persist_directory
        self.index_file = os.path.join(persist_directory, f"{collection_name}_index.pkl")
        self.documents_file = os.path.join(persist_directory, f"{collection_name}_documents.pkl")

        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize FAISS index and document store
        self.index = None
        self.documents = []
        self.metadata = []

        # Load existing data if available
        self._load_index()

        logger.info(f"Vector store initialized. Collection: {collection_name}")

    def _load_index(self):
        """Load existing FAISS index and documents"""
        try:
            import faiss
            if os.path.exists(self.index_file) and os.path.exists(self.documents_file):
                # Load FAISS index
                self.index = faiss.read_index(self.index_file)

                # Load documents and metadata
                with open(self.documents_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadata = data['metadata']

                logger.info(f"Loaded existing index with {len(self.documents)} documents")
            else:
                # Create new index
                dimension = len(self.embedding_service.get_embeddings(["test"])[0])
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                logger.info("Created new FAISS index")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add documents to vector store"""
        if not chunks:
            logger.warning("No chunks to add")
            return

        try:
            import faiss
            import numpy as np

            # Generate embeddings for all chunks
            texts = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_service.get_embeddings(texts)

            # Convert to numpy array
            embeddings_np = np.array(embeddings, dtype=np.float32)

            # Add to FAISS index
            self.index.add(embeddings_np)

            # Store documents and metadata
            for chunk in chunks:
                self.documents.append(chunk['content'])
                self.metadata.append(chunk['metadata'])

            # Save index and documents
            self._save_index()

            logger.info(f"Added {len(chunks)} chunks to vector store")

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            import faiss
            import numpy as np

            if self.index is None or self.index.ntotal == 0:
                return []

            # Generate query embedding
            query_embedding = self.embedding_service.get_embeddings([query])[0]
            query_embedding_np = np.array([query_embedding], dtype=np.float32)

            # Search FAISS index
            scores, indices = self.index.search(query_embedding_np, min(top_k, self.index.ntotal))

            # Format results
            formatted_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):  # Valid index
                    # Apply metadata filter if provided
                    if filter_metadata:
                        metadata = self.metadata[idx]
                        if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                            continue

                    formatted_results.append({
                        'content': self.documents[idx],
                        'metadata': self.metadata[idx],
                        'similarity_score': float(score)
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'status': 'active'
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                'collection_name': self.collection_name,
                'document_count': 0,
                'status': 'error'
            }
    
    def clear_collection(self):
        """Clear all documents from collection"""
        try:
            # ChromaDB doesn't have a direct clear method, so we delete and recreate
            self.client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection()
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False