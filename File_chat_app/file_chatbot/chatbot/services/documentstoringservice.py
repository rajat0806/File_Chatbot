from typing import List, Dict, Any
import numpy as np
from django.conf import settings
from ..models import Document
import httpx
import os
import logging
import time
from django.db.utils import DatabaseError
from django.db import transaction

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("document_processing.log"),
        logging.StreamHandler()
    ]
)   

logger = logging.getLogger("document_storage")


def store_document_text(document: Document, text: str) -> None:
    """Store the full extracted text for a document in the database."""
    start_time = time.time()
    
    logger.info(f"Starting to store full text for document ID: {document.id}, Title: {document.title}")
    
    if not text:
        logger.warning(f"No text provided for document ID: {document.id}. Marking as processed without storing text.")
        document.processed = True
        document.save()
        logger.info(f"Document ID: {document.id} marked as processed")
        return
    
    try:
        # Store the full text
        logger.info(f"Storing full text in database for document ID: {document.id}")
        storage_start_time = time.time()
        
        with transaction.atomic():
            document.extracted_text = text
            document.processed = True
            document.save()
            
            storage_duration = time.time() - storage_start_time
            logger.info(f"Successfully stored full text in {storage_duration:.2f} seconds for document ID: {document.id}")
            
        total_duration = time.time() - start_time
        logger.info(f"Full text storage completed in {total_duration:.2f} seconds for document ID: {document.id}")
        
    except DatabaseError as db_err:
        logger.error(f"Database error storing full text for document ID {document.id}: {str(db_err)}", exc_info=True)
        # Optionally, reset processed flag or handle error more gracefully
        document.processed = False # Example: mark as not processed if error occurs
        document.save() # Save the reset status
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error during full text storage for document ID {document.id}: {str(e)}", exc_info=True)
        # Optionally, reset processed flag
        document.processed = False
        document.save()
        raise