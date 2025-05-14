import os
from django.conf import settings
from django.shortcuts import get_object_or_404
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser

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

class DocumentViewSet(viewsets.ModelViewSet):
    """API endpoint for managing documents"""
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Filter documents by the current user"""
        return Document.objects.filter(uploaded_by=self.request.user)
    
    def perform_create(self, serializer):
        """Save the uploaded file and process it"""
        # Save document with current user
        document = serializer.save(uploaded_by=self.request.user)
        
        # Get file path
        file_path = document.file.path
        file_extension = os.path.splitext(file_path)[1].lower()[1:]  # Remove the dot
        
        # Check if file extension is allowed
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            document.delete()
            return Response(
                {"error": f"File type not supported. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Extract text from the file
            text = extract_text_from_file(file_path)
            
            # Split text into chunks
            chunks = chunk_text(text)
            
            # Create embeddings and store chunks
            store_document_chunks(document, chunks)
            
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        except Exception as e:
            # Delete document if processing fails
            document.delete()
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class ChatSessionViewSet(viewsets.ModelViewSet):
    """API endpoint for managing chat sessions"""
    queryset = ChatSession.objects.all()
    serializer_class = ChatSessionSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Filter chat sessions by the current user"""
        return ChatSession.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        """Create chat session with current user"""
        serializer.save(user=self.request.user)
    
    @action(detail=True, methods=['post'])
    def ask(self, request, pk=None):
        """Ask a question about the document in the chat session"""
        chat_session = self.get_object()
        
        # Deserialize and validate the question
        question_serializer = QuestionSerializer(data=request.data)
        if not question_serializer.is_valid():
            return Response(question_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        question = question_serializer.validated_data['question']
        
        # Save user message
        user_message = ChatMessage.objects.create(
            session=chat_session,
            role='user',
            content=question
        )
        
        # Find relevant document chunks
        relevant_chunks = find_similar_chunks(question, chat_session.document.id, top_k=5)
        
        if not relevant_chunks:
            response_text = "I don't have enough information to answer this question based on the document."
        else:
            # Generate response using OpenAI
            response_text = generate_response(question, relevant_chunks)
        
        # Save assistant message
        assistant_message = ChatMessage.objects.create(
            session=chat_session,
            role='assistant',
            content=response_text
        )
        
        # Return both messages
        return Response({
            'user_message': ChatMessageSerializer(user_message).data,
            'assistant_message': ChatMessageSerializer(assistant_message).data
        })