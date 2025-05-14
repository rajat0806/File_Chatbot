from rest_framework import serializers
from .models import Document, DocumentChunk, ChatSession, ChatMessage

class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['id', 'title', 'file', 'content_type', 'uploaded_at', 'processed']
        read_only_fields = ['uploaded_by', 'processed']

class DocumentChunkSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentChunk
        fields = ['id', 'document', 'content', 'chunk_index']
        read_only_fields = ['embedding']

class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ['id', 'role', 'content', 'timestamp']
        read_only_fields = ['timestamp']

class ChatSessionSerializer(serializers.ModelSerializer):
    messages = ChatMessageSerializer(many=True, read_only=True)
    
    class Meta:
        model = ChatSession
        fields = ['id', 'document', 'created_at', 'messages']
        read_only_fields = ['created_at']

class QuestionSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=1000)