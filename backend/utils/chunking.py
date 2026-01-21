from typing import List, Dict, Any
import re
from ..config import settings

class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def create_semantic_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create semantic chunks from processed chunks"""
        final_chunks = []
        
        for chunk in chunks:
            content = chunk['content']
            metadata = chunk['metadata']
            
            # Skip if content is too short
            if len(content) < 50:
                continue
            
            # Create chunks based on content type
            if metadata.get('content_type') == 'table':
                # Keep tables as single chunks
                final_chunks.append(chunk)
            else:
                # Text content - apply semantic chunking
                text_chunks = self._chunk_text_semantically(content)
                
                for i, text_chunk in enumerate(text_chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata['chunk_index'] = i + 1
                    
                    final_chunks.append({
                        'content': text_chunk,
                        'metadata': chunk_metadata
                    })
        
        return final_chunks
    
    def _chunk_text_semantically(self, text: str) -> List[str]:
        """Split text into semantic chunks"""
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence exceeds chunk size and we already have content
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap_sentences = current_chunk.split('. ')
                if len(overlap_sentences) > 1:
                    # Keep last few sentences for overlap
                    overlap = '. '.join(overlap_sentences[-2:]) + '. '
                    current_chunk = overlap + sentence + ' '
                else:
                    current_chunk = sentence + ' '
            else:
                current_chunk += sentence + ' '
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Merge very small chunks
        merged_chunks = []
        temp_chunk = ""
        
        for chunk in chunks:
            if len(temp_chunk) + len(chunk) < self.chunk_size:
                temp_chunk += chunk + " "
            else:
                if temp_chunk:
                    merged_chunks.append(temp_chunk.strip())
                temp_chunk = chunk + " "
        
        if temp_chunk:
            merged_chunks.append(temp_chunk.strip())
        
        return merged_chunks
    
    def _split_by_headers(self, text: str) -> List[str]:
        """Split text by headers if present"""
        # Look for patterns like "Section 1:", "1.1 Introduction", etc.
        header_pattern = r'\n(#+|\d+\.\s+|\d+\.\d+\s+|\*\*[^*]+\*\*)\s+'
        
        parts = re.split(header_pattern, text)
        
        if len(parts) > 1:
            # Reconstruct with headers
            chunks = []
            for i in range(0, len(parts) - 1, 2):
                if i + 1 < len(parts):
                    chunk = parts[i] + parts[i + 1]
                    if chunk.strip():
                        chunks.append(chunk.strip())
            
            # Add last part if any
            if len(parts) % 2 == 1 and parts[-1].strip():
                chunks.append(parts[-1].strip())
            
            return chunks
        
        return [text]