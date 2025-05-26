import os
from django.conf import settings
from django.shortcuts import render, get_object_or_404
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.contrib.auth.models import AnonymousUser # Import AnonymousUser
import logging

from .models import Document, ChatSession, ChatMessage
from .serializers import DocumentSerializer, ChatSessionSerializer, ChatMessageSerializer
from .services.file_processor import extract_text_from_file 
from .services.documentstoringservice import store_document_text 
from .services.openai_service import generate_chat_completion

logger = logging.getLogger(__name__)


def chatbot_ui(request):
    """Render the chatbot UI"""
    logger.info("Rendering chatbot UI")
    return render(request, 'chatbot.html')

class DocumentViewSet(viewsets.ModelViewSet):
    """API endpoint for managing documents"""
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    parser_classes = (MultiPartParser, FormParser)
    # permission_classes = [permissions.IsAuthenticated] # Consider re-enabling for production

    def get_queryset(self):
        """Filter documents by the current user"""
        user = self.request.user
        if user.is_authenticated:
            logger.info(f"Fetching documents for authenticated user: {user.username}")
            return Document.objects.filter(uploaded_by=user)
        logger.info("Fetching documents for anonymous user (or no user context)")
        # Handle anonymous or no user case - returning all or none based on policy
        return Document.objects.none() # Or Document.objects.all() if public access is intended

    def perform_create(self, serializer):
        """Save the uploaded file and process it"""
        user = self.request.user if self.request.user.is_authenticated else None
        logger.info(f"Attempting to create document for user: {user.username if user else 'Anonymous'}")
        
        if not user:
            logger.warning("Document upload attempt by unauthenticated user.")
            
            pass 

        document = serializer.save(uploaded_by=user if user else None) # Pass user or None
        file_path = document.file.path
        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_name)[1].lower().lstrip('.')
        file_size = document.file.size / 1024  # Size in KB
        
        logger.info(f"Processing file: {file_name}, size: {file_size:.2f} KB, type: {file_extension}")
        
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            logger.warning(f"Unsupported file type: {file_extension}. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}")
            document.delete()
            logger.info(f"Document ID: {document.id} deleted due to unsupported file type")
            return Response(
                {"error": f"File type not supported. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            logger.info(f"Extracting text from file: {file_path}")
            text = extract_text_from_file(file_path)
            logger.info(f"Text extraction complete. Extracted {len(text)} characters for document ID: {document.id}")
            
    
            
            logger.info(f"Storing full text for document ID: {document.id}")
            store_document_text(document, text) 
            logger.info(f"Document processing complete for ID: {document.id}")
            
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"Error processing document ID: {document.id}: {str(e)}", exc_info=True)
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
        user = self.request.user
        if user.is_authenticated:
            logger.info(f"Fetching chat sessions for authenticated user: {user.username}")
            return ChatSession.objects.filter(user=user)
        
        
        logger.info("No authenticated user found, returning no chat sessions.")
        return ChatSession.objects.none()
    
    def perform_create(self, serializer):
        """Create chat session with current user"""
        user = self.request.user
        if not user.is_authenticated:
            logger.error("Attempt to create chat session by unauthenticated user.")
            # Return a 400 Bad Request response instead of trying to save with an anonymous user
            raise serializers.ValidationError("Authentication required to create a chat session.")
        else:
            logger.info(f"Creating new chat session for user: {user.username}")
            serializer.save(user=user)

    @action(detail=True, methods=['post'])
    def ask(self, request, pk=None):
        """Handle sending a message in a chat session"""
        session = self.get_object()
        # Check for both 'message' and 'question' parameters for backward compatibility
        user_message_content = request.data.get('message') or request.data.get('question')
        user = request.user

        logger.info(f"Received message from user {user.username if user.is_authenticated else 'Anonymous'} for session {session.id}: '{user_message_content}'")

        if not user_message_content:
            logger.warning("Empty message received.")
            return Response({"error": "Message content cannot be empty"}, status=status.HTTP_400_BAD_REQUEST)

        # Save user message
        ChatMessage.objects.create(session=session, role='user', content=user_message_content)
        logger.debug(f"User message saved for session {session.id}")

        try:
            # Retrieve the full document text to use as context
            document = session.document
            if not document.processed or not document.extracted_text:
                logger.error(f"Document {document.id} not processed or no extracted text found for session {session.id}.")
                return Response({"error": "Document not ready or text not available."}, status=status.HTTP_400_BAD_REQUEST)
            
            context = document.extracted_text
            logger.debug(f"Using full text of document {document.id} (length: {len(context)}) as context for session {session.id}")

            # Generate assistant response
            logger.info(f"Generating assistant response for session {session.id}")
            assistant_response_content = generate_chat_completion(user_message_content, context)
            logger.info(f"Assistant response generated for session {session.id}: '{assistant_response_content[:100]}...'" ) # Log snippet

            # Save assistant response
            assistant_message = ChatMessage.objects.create(session=session, role='assistant', content=assistant_response_content)
            logger.debug(f"Assistant message saved for session {session.id}")
            
            # Get the user message that was just created
            user_message = ChatMessage.objects.filter(session=session, role='user').order_by('-timestamp').first()
            
            # Return both messages in serialized format for backward compatibility
            return Response({
                'user_message': ChatMessageSerializer(user_message).data,
                'assistant_message': ChatMessageSerializer(assistant_message).data
            }, status=status.HTTP_200_OK)
        
        except Exception as e:
            logger.error(f"Error during message processing for session {session.id}: {str(e)}", exc_info=True)
            return Response({"error": "Failed to generate response"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
