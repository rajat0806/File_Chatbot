from typing import List, Dict, Any
import numpy as np
from django.conf import settings
from openai import OpenAI
from ..models import Document, DocumentChunk

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using OpenAI API"""
    if not texts:
        return []
    
    model = settings.OPENAI_EMBEDDING_MODEL
    
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    
    # Extract embeddings from response
    embeddings = [item.embedding for item in response.data]
    
    return embeddings

def store_document_chunks(document: Document, chunks: List[str]) -> None:
    """Store document chunks with embeddings in the database"""
    # Generate embeddings for all chunks
    embeddings = create_embeddings(chunks)
    
    # Store chunks and embeddings
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        DocumentChunk.objects.create(
            document=document,
            content=chunk,
            chunk_index=idx,
            embedding=embedding,
            metadata={"position": idx}
        )
    
    # Mark document as processed
    document.processed = True
    document.save()

def find_similar_chunks(query: str, document_id: int, top_k: int = 5) -> List[DocumentChunk]:
    """Find chunks similar to query using vector similarity search"""
    # Get query embedding
    query_embedding = create_embeddings([query])[0]
    
    # Convert to numpy array for calculations
    query_embedding_array = np.array(query_embedding)
    
    # Get all chunks for the document
    chunks = DocumentChunk.objects.filter(document_id=document_id)
    
    # Calculate similarity scores
    similarities = []
    for chunk in chunks:
        chunk_embedding = np.array(chunk.embedding)
        # Calculate cosine similarity
        similarity = np.dot(query_embedding_array, chunk_embedding) / (
            np.linalg.norm(query_embedding_array) * np.linalg.norm(chunk_embedding))
        similarities.append((chunk, similarity))
    
    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_k chunks
    return [item[0] for item in similarities[:top_k]]