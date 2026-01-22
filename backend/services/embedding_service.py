from langchain_ollama import OllamaEmbeddings
from config import settings

# Replace custom EmbeddingService with OllamaEmbeddings
embedding_service = OllamaEmbeddings(
    model=settings.EMBEDDING_MODEL,
    base_url=settings.OLLAMA_API_URL
)