from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from typing import Iterable, List
from config import settings
from exceptions import VectorStoreException
import os
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, collection_name: str, persist_directory: str = "./data/vector_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.index_path = os.path.join(persist_directory, collection_name)

        # Initialize embeddings
        try:
            self.embeddings = OllamaEmbeddings(
                model=settings.EMBEDDING_MODEL,
                base_url=settings.OLLAMA_API_URL
            )
        except Exception as e:
            logger.exception("Failed to initialize OllamaEmbeddings for VectorStore")
            raise VectorStoreException(
                message="Failed to initialize embeddings",
                operation="init_embeddings",
                details={"error": str(e)}
            )

        # Load or create FAISS index
        try:
            if os.path.exists(self.index_path):
                try:
                    self.vector_store = FAISS.load_local(self.index_path, self.embeddings)
                    logger.info(f"Loaded existing FAISS index for collection: {collection_name}")
                except Exception as e:
                    logger.exception("Failed to load existing FAISS index, attempting to recreate")
                    # Try to recreate a fresh index
                    dummy_doc = Document(page_content="init")
                    self.vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
                    self.vector_store.index.reset()
                    self.vector_store.docstore._dict.clear()
                    logger.info(f"Recreated FAISS index for collection: {collection_name}")
            else:
                # Dummy document to bootstrap FAISS
                dummy_doc = Document(page_content="init")
                self.vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
                # Remove dummy entry
                self.vector_store.index.reset()
                self.vector_store.docstore._dict.clear()
                logger.info(f"Created new FAISS index for collection: {collection_name}")
        except Exception as e:
            logger.exception("Failed to initialize FAISS vector store")
            raise VectorStoreException(
                message="Failed to initialize FAISS vector store",
                operation="init_faiss",
                details={"error": str(e)}
            )

    def as_retriever(self, search_kwargs=None):
        """
        Expose LangChain retriever interface
        """
        return self.vector_store.as_retriever(
            search_kwargs=search_kwargs or {"k": 4}
        )


    def add_documents(self, documents: Iterable[Document]):
        """Add LangChain `Document` objects to the vector store.

        This method accepts an iterable of LangChain `Document` objects, extracts
        `page_content` and `metadata` from each, adds them to the FAISS index,
        and ensures the index is persisted to disk.
        """
        docs: List[Document] = list(documents)
        if not docs:
            logger.info("No documents provided to add to the vector store.")
            return

        # Extract texts and metadatas with validation
        try:
            texts = [getattr(d, "page_content", None) for d in docs]
            metadatas = [getattr(d, "metadata", {}) or {} for d in docs]
        except Exception as e:
            logger.exception("Failed to extract content/metadata from Document objects.")
            raise ValueError("Invalid Document objects passed to add_documents") from e

        # Validate extracted content
        if any(t is None for t in texts):
            logger.error("One or more Document objects are missing `page_content`.")
            raise ValueError("All Document objects must have `page_content` set.")

        # Add documents to FAISS with explicit error handling
        try:
            if hasattr(self.vector_store, "add_documents"):
                self.vector_store.add_documents(docs)
            else:
                self.vector_store.add_texts(texts, metadatas)
        except Exception as e:
            logger.exception("Failed while adding documents to FAISS index")
            raise VectorStoreException(
                message="Failed to add documents to FAISS index",
                operation="add_documents",
                details={"error": str(e), "document_count": len(docs)}
            )

        # Persist the FAISS index to disk
        try:
            self.vector_store.save_local(self.index_path)
        except Exception as e:
            logger.exception("Failed to save FAISS index after adding documents")
            # Attempt to roll back by clearing collection
            try:
                self.clear_collection()
            except Exception:
                logger.warning("Rollback: clearing FAISS index after failed save also failed")
            raise VectorStoreException(
                message="Failed to save FAISS index after adding documents",
                operation="save_local",
                details={"error": str(e)}
            )

        logger.info(f"Added {len(docs)} documents to the vector store at {self.index_path}.")

    def search(self, query: str, top_k: int = 5, relevance_threshold: float = 0.7):
        """Search for similar documents with relevance scores."""
        try:
            results = self.vector_store.similarity_search_with_relevance_scores(query, k=top_k)
            return [
                {
                    "content": result[0].page_content,
                    "metadata": result[0].metadata,
                    "relevance_score": result[1]
                }
                for result in results if result[1] >= relevance_threshold
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            dummy_doc = Document(page_content="init")
            self.vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
            # Remove dummy entry
            self.vector_store.index.reset()
            self.vector_store.docstore._dict.clear()
            # Persist empty index
            try:
                self.vector_store.save_local(self.index_path)
            except Exception as e:
                logger.exception("Failed to save FAISS index during clear operation")
                raise VectorStoreException(
                    message="Failed to save FAISS index during clear",
                    operation="clear_collection",
                    details={"error": str(e)}
                )

            logger.info("Cleared the vector store collection.")
            return True
        except VectorStoreException:
            raise
        except Exception as e:
            logger.exception("Unexpected error while clearing FAISS collection")
            return False

    def get_collection_info(self):
        """Get information about the collection."""
        try:
            # Use index_to_docstore_id mapping to get document count
            document_count = len(self.vector_store.index_to_docstore_id)
            return {
                "collection_name": self.collection_name,
                "document_count": document_count,
                "status": "active"
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "status": "error"
            }
