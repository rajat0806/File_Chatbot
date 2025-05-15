from typing import List, Dict, Any
import numpy as np
from django.conf import settings
from openai import OpenAI
from ..models import Document, DocumentChunk
import httpx
import os
import openai
import logging
import time
from django.db.utils import DatabaseError
from django.db import transaction
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("document_processing.log"),
        logging.StreamHandler()
    ]
)   

logger = logging.getLogger("document_storage")


if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
    openai.api_key = settings.OPENAI_API_KEY
else:
    # Fallback to environment variable if settings not available
    openai.api_key = os.getenv('OPENAI_API_KEY', '')

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((openai.APIError, openai.APITimeoutError, openai.RateLimitError)),
    after=lambda retry_state: logger.warning(
        f"Retrying embedding generation after error. Attempt {retry_state.attempt_number}"
    )
)

def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using OpenAI API"""
    start_time = time.time()
    
    # Input validation
    if not texts:
        logger.warning("Empty text list provided to create_embeddings, returning empty list")
        return []
    
    batch_size = len(texts)
    total_tokens = sum(len(text.split()) for text in texts)  # Rough token estimation
    
    logger.info(f"Generating embeddings for {batch_size} texts (~{total_tokens} tokens)")
    
    # Log first few characters of each text for debugging (limit to avoid huge logs)
    if logger.isEnabledFor(logging.DEBUG):
        for i, text in enumerate(texts[:5]):  # Log only first 5 texts
            preview = text[:50] + "..." if len(text) > 50 else text
            logger.debug(f"Text {i+1}: {preview}")
        if len(texts) > 5:
            logger.debug(f"... and {len(texts) - 5} more texts")
    
    model = settings.OPENAI_EMBEDDING_MODEL
    logger.info(f"Using embedding model: {model}")
    
    try:
        # Log API request
        logger.debug(f"Sending request to OpenAI embeddings API")
        
        # Track API usage
        api_start_time = time.time()
        
        response = openai.embeddings.create(
            input=texts,
            model=model
        )
        
        api_duration = time.time() - api_start_time
        logger.info(f"OpenAI API request completed in {api_duration:.2f} seconds")
        
        # Extract embeddings from response
        embeddings = [item.embedding for item in response.data]
        
        # Validate response
        if len(embeddings) != len(texts):
            logger.error(f"API response mismatch: got {len(embeddings)} embeddings for {len(texts)} texts")
            raise ValueError(f"API response count mismatch: expected {len(texts)}, got {len(embeddings)}")
        
        # Log embedding dimensions
        if embeddings:
            embedding_dim = len(embeddings[0])
            logger.info(f"Generated {len(embeddings)} embeddings with dimension {embedding_dim}")
        
        # Log token usage if available in response
        if hasattr(response, 'usage') and response.usage:
            prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
            total_tokens = getattr(response.usage, 'total_tokens', 0)
            logger.info(f"Token usage: {prompt_tokens} prompt tokens, {total_tokens} total tokens")
        
        total_duration = time.time() - start_time
        texts_per_second = batch_size / total_duration if total_duration > 0 else 0
        logger.info(f"Embedding generation completed in {total_duration:.2f} seconds ({texts_per_second:.2f} texts/second)")
        
        return embeddings
        
    except openai.RateLimitError as e:
        logger.error(f"OpenAI API rate limit exceeded: {str(e)}")
        # This will be retried by the decorator
        raise
        
    except openai.APITimeoutError as e:
        logger.error(f"OpenAI API timeout: {str(e)}")
        # This will be retried by the decorator
        raise
        
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        # This will be retried by the decorator
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error during embedding generation: {str(e)}", exc_info=True)
        raise

def store_document_chunks(document: Document, chunks: List[str]) -> None:
    """Store document chunks with embeddings in the database"""
    start_time = time.time()
    
    logger.info(f"Starting to store chunks for document ID: {document.id}, Title: {document.title}")
    logger.info(f"Processing {len(chunks)} chunks to store in database")
    
    if not chunks:
        logger.warning(f"No chunks provided for document ID: {document.id}. Marking as processed without storing chunks.")
        document.processed = True
        document.save()
        logger.info(f"Document ID: {document.id} marked as processed")
        return
    
    try:
        # Generate embeddings for all chunks
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embedding_start_time = time.time()
        
        try:
            embeddings = create_embeddings(chunks)
            embedding_duration = time.time() - embedding_start_time
            logger.info(f"Successfully generated {len(embeddings)} embeddings in {embedding_duration:.2f} seconds")
            
            # Check if embeddings count matches chunks count
            if len(embeddings) != len(chunks):
                logger.error(f"Embedding count mismatch: {len(embeddings)} embeddings for {len(chunks)} chunks")
                raise ValueError(f"Embedding count mismatch: {len(embeddings)} embeddings for {len(chunks)} chunks")
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            raise
        
        # Store chunks and embeddings
        logger.info(f"Storing chunks and embeddings in database")
        storage_start_time = time.time()
        
        # Use transaction to ensure atomicity
        with transaction.atomic():
            successful_chunks = 0
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                try:
                    # Log the beginning of each chunk storage operation for large documents
                    if len(chunks) > 100 and idx % 50 == 0:
                        logger.debug(f"Storing chunk {idx+1}/{len(chunks)}")
                    
                    # Check if chunk is too small
                    if len(chunk) < 10:
                        logger.warning(f"Chunk {idx} is very small ({len(chunk)} chars), might be low quality")
                    
                    DocumentChunk.objects.create(
                        document=document,
                        content=chunk,
                        chunk_index=idx,
                        embedding=embedding,
                        metadata={"position": idx}
                    )
                    successful_chunks += 1
                    
                except DatabaseError as db_err:
                    logger.error(f"Database error storing chunk {idx}: {str(db_err)}")
                    raise
                except Exception as e:
                    logger.error(f"Error storing chunk {idx}: {str(e)}")
                    raise
            
            storage_duration = time.time() - storage_start_time
            logger.info(f"Successfully stored {successful_chunks} chunks in {storage_duration:.2f} seconds")
            
            # Mark document as processed
            logger.info(f"Marking document ID: {document.id} as processed")
            document.processed = True
            document.save()
            logger.info(f"Document ID: {document.id} successfully marked as processed")
            
    except Exception as e:
        logger.error(f"Failed to store chunks for document ID: {document.id}: {str(e)}", exc_info=True)
        # Optional: Update document status to indicate failure
        try:
            document.processed = False
            document.save()
            logger.info(f"Document ID: {document.id} marked as failed to process")
        except Exception as save_err:
            logger.error(f"Error updating document status: {str(save_err)}")
        raise
    
    finally:
        total_duration = time.time() - start_time
        logger.info(f"Document chunk storage process completed in {total_duration:.2f} seconds for document ID: {document.id}")
        # Log memory usage if psutil is available
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            logger.debug(f"Memory usage after processing: {memory_info.rss / 1024 / 1024:.2f} MB")
        except ImportError:
            pass

def find_similar_chunks(query: str, document_id: int, top_k: int = 5) -> List[DocumentChunk]:
    """Find chunks similar to query using vector similarity search"""
    logging.info(f"Running similarity search for document_id={document_id}, top_k={top_k}")
    logging.debug(f"Query: {query}")
    
    # Get query embedding
    query_embedding = create_embeddings([query])[0]
    logging.debug(f"Query embedding: {query_embedding[:5]}... (truncated)")

    # Convert to numpy array for calculations
    query_embedding_array = np.array(query_embedding)

    # Get all chunks for the document
    chunks = DocumentChunk.objects.filter(document_id=document_id)
    logging.info(f"Retrieved {chunks.count()} chunks for document_id={document_id}")

    # Calculate similarity scores
    similarities = []
    for chunk in chunks:
        chunk_embedding = np.array(chunk.embedding)
        similarity = np.dot(query_embedding_array, chunk_embedding) / (
            np.linalg.norm(query_embedding_array) * np.linalg.norm(chunk_embedding))
        similarities.append((chunk, similarity))
        logging.debug(f"Chunk ID {chunk.id} similarity: {similarity}")

    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    logging.info(f"Top similarity score: {similarities[0][1] if similarities else 'N/A'}")

    # Return top_k chunks
    top_chunks = [item[0] for item in similarities[:top_k]]
    logging.info(f"Returning top {len(top_chunks)} chunks")
    return top_chunks