from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from config import settings
import os
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, collection_name: str, persist_directory: str = "./data/vector_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.index_path = os.path.join(persist_directory, collection_name)

        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_API_URL
        )

        # Load or create FAISS index
        if os.path.exists(self.index_path):
            self.vector_store = FAISS.load_local(self.index_path, self.embeddings)
            logger.info(f"Loaded existing FAISS index for collection: {collection_name}")
        else:
            # Dummy document to bootstrap FAISS
            dummy_doc = Document(page_content="init")
            self.vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
            # Remove dummy entry
            self.vector_store.index.reset()
            self.vector_store.docstore._dict.clear()
            logger.info(f"Created new FAISS index for collection: {collection_name}")

    def as_retriever(self, search_kwargs=None):
        """
        Expose LangChain retriever interface
        """
        return self.vector_store.as_retriever(
            search_kwargs=search_kwargs or {"k": 4}
        )


    def add_documents(self, texts: list, metadatas: list):
        """Add documents to the vector store."""
        try:
            self.vector_store.add_texts(texts, metadatas)
            self.vector_store.save_local(self.index_path)
            logger.info(f"Added {len(texts)} documents to the vector store.")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

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
            # Dummy document to bootstrap FAISS
            dummy_doc = Document(page_content="init")
            self.vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
            # Remove dummy entry
            self.vector_store.index.reset()
            self.vector_store.docstore._dict.clear()
            self.vector_store.save_local(self.index_path)
            logger.info("Cleared the vector store collection.")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise

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
