import os
from django.conf import settings
from django.shortcuts import get_object_or_404
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
import logging
from .models import Document, DocumentChunk, ChatSession, ChatMessage
from .serializers import (
    DocumentSerializer, 
    DocumentChunkSerializer, 
    ChatSessionSerializer, 
    ChatMessageSerializer,
    QuestionSerializer
)
from .services.file_processor import extract_text_from_file, chunk_text
from .services.embedding_service import store_document_chunks, find_similar_chunks
from .services.openai_service import generate_response
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import AnonymousUser





logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("document_processing.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("document_storage")

def chatbot_ui(request):
    """Renders the main chatbot UI."""
    if not isinstance(request.user, AnonymousUser) and request.user.is_authenticated:
        documents = Document.objects.filter(uploaded_by=request.user)
    else:
        documents = Document.objects.all() # Or an empty queryset: Document.objects.none()
    return render(request, 'chatbot.html', {'documents': documents})


class DocumentViewSet(viewsets.ModelViewSet):
    """API endpoint for managing documents"""
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    parser_classes = (MultiPartParser, FormParser)
    # permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        if self.request.user.is_authenticated:
            user_id = self.request.user.id
            logger.debug(f"Getting documents for user ID: {user_id}")
            return Document.objects.filter(uploaded_by=self.request.user)
        else:
            logger.debug("Getting all documents for anonymous user")
            return Document.objects.none()  # Or return all documents, depending on your requirement

    def perform_create(self, serializer):
        """Save the uploaded file and process it"""
        logger.info(f"Creating new document for user ID: {self.request.user.id}")
        
        # Save document with current user
        document = serializer.save(uploaded_by=self.request.user  if self.request.user.is_authenticated else AnonymousUser())
        logger.info(f"Document saved with ID: {document.id}")
        
        # Get file path
        file_path = document.file.path
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / 1024  # KB
        file_extension = os.path.splitext(file_path)[1].lower()[1:]  # Remove the dot
        
        logger.info(f"Processing file: {file_name}, size: {file_size:.2f} KB, type: {file_extension}")
        
        # Check if file extension is allowed
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            logger.warning(f"Unsupported file type: {file_extension}. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}")
            document.delete()
            logger.info(f"Document ID: {document.id} deleted due to unsupported file type")
            return Response(
                {"error": f"File type not supported. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Extract text from the file
            logger.info(f"Extracting text from file: {file_path}")
            text = extract_text_from_file(file_path)
            logger.info(f"Text extraction complete. Extracted {len(text)} characters")
            
            # Split text into chunks
            logger.info(f"Chunking text for document ID: {document.id}")
            chunks = chunk_text(text)
            logger.info(f"Text chunking complete. Created {len(chunks)} chunks")
            
            # Create embeddings and store chunks
            logger.info(f"Generating embeddings and storing chunks for document ID: {document.id}")
            store_document_chunks(document, chunks)
            logger.info(f"Document processing complete for ID: {document.id}")
            
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"Error processing document ID: {document.id}: {str(e)}", exc_info=True)
            # Delete document if processing fails
            document.delete()
            logger.info(f"Document ID: {document.id} deleted due to processing failure")
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class ChatSessionViewSet(viewsets.ModelViewSet):
    """API endpoint for managing chat sessions"""
    queryset = ChatSession.objects.all()
    serializer_class = ChatSessionSerializer
    # permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Filter chat sessions by the current user"""
        logging.info(f"Fetching chat sessions for user {self.request.user}")
        return ChatSession.objects.filter(user=self.request.user  if self.request.user.is_authenticated else AnonymousUser())
    
    def perform_create(self, serializer):
        """Create chat session with current user"""
        logging.info(f"Creating a new chat session for user {self.request.user}")
        serializer.save(user=self.request.user if self.request.user.is_authenticated else AnonymousUser())
    
    @action(detail=True, methods=['post'])
    def ask(self, request, pk=None):
        """Ask a question about the document in the chat session"""
        chat_session = self.get_object()
        logging.info(f"Received a question for ChatSession ID {chat_session.id}")

        # Deserialize and validate the question
        question_serializer = QuestionSerializer(data=request.data)
        if not question_serializer.is_valid():
            logging.warning(f"Invalid question input: {question_serializer.errors}")
            return Response(question_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        question = question_serializer.validated_data['question']
        logging.info(f"Question: {question}")
        
        # Save user message
        user_message = ChatMessage.objects.create(
            session=chat_session,
            role='user',
            content=question
        )
        logging.debug(f"Saved user message ID {user_message.id}")
        
        # Find relevant document chunks
        relevant_chunks = find_similar_chunks(question, chat_session.document.id, top_k=5)
        logging.info(f"Found {len(relevant_chunks)} relevant chunks")
        
        if not relevant_chunks:
            response_text = "I don't have enough information to answer this question based on the document."
            logging.info("No relevant chunks found. Returning fallback message.")
        else:
            # Generate response using OpenAI
            response_text = generate_response(question, relevant_chunks)
            logging.info("Generated response using OpenAI")
        
        # Save assistant message
        assistant_message = ChatMessage.objects.create(
            session=chat_session,
            role='assistant',
            content=response_text
        )
        logging.debug(f"Saved assistant message ID {assistant_message.id}")
        
        # Return both messages
        return Response({
            'user_message': ChatMessageSerializer(user_message).data,
            'assistant_message': ChatMessageSerializer(assistant_message).data
        })
    
