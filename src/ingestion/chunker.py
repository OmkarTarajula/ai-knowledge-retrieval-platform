import uuid
from typing import List, Dict, Any

class TextChunker:
    """Splits document text into manageable chunks with configurable size and overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self, content_blocks: List[Dict[str, Any]], doc_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Takes raw content blocks and returns structured chunks with preserved metadata.
        Each chunk contains:
            - chunk_id: Unique UUID
            - text: The chunk content
            - metadata: {file_name, file_type, domain, page_number, section, chunk_index}
        """
        chunks: List[Dict[str, Any]] = []
        chunk_counter = 0

        for block in content_blocks:
            text = block.get("content", "").strip()
            if not text:
                continue

            words = text.split()
            # If text is smaller than chunk_size, create a single chunk
            if len(words) <= self.chunk_size:
                chunk_counter += 1
                chunks.append({
                    "chunk_id": f"{doc_metadata['file_name']}_chunk_{chunk_counter}_{uuid.uuid4().hex[:6]}",
                    "text": text,
                    "metadata": {
                        "file_name": doc_metadata.get("file_name", "unknown"),
                        "file_type": doc_metadata.get("file_type", ""),
                        "domain": doc_metadata.get("domain", "General"),
                        "page_number": block.get("page_number", 1),
                        "section": block.get("section", "Body"),
                        "chunk_index": chunk_counter
                    }
                })
            else:
                # Sliding window chunking with word overlap
                step = self.chunk_size - self.chunk_overlap
                for i in range(0, len(words), step):
                    chunk_words = words[i:i + self.chunk_size]
                    chunk_text = " ".join(chunk_words)
                    
                    if chunk_text.strip():
                        chunk_counter += 1
                        chunks.append({
                            "chunk_id": f"{doc_metadata['file_name']}_chunk_{chunk_counter}_{uuid.uuid4().hex[:6]}",
                            "text": chunk_text,
                            "metadata": {
                                "file_name": doc_metadata.get("file_name", "unknown"),
                                "file_type": doc_metadata.get("file_type", ""),
                                "domain": doc_metadata.get("domain", "General"),
                                "page_number": block.get("page_number", 1),
                                "section": block.get("section", "Body"),
                                "chunk_index": chunk_counter
                            }
                        })

        return chunks