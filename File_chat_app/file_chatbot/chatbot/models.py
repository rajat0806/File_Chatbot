from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField

class Document(models.Model):
    """Model for storing uploaded documents"""
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/')
    content_type = models.CharField(max_length=100)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    
    def __str__(self):
        return self.title

class DocumentChunk(models.Model):
    """Model for storing document chunks with embeddings"""
    document = models.ForeignKey(Document, related_name='chunks', on_delete=models.CASCADE)
    content = models.TextField()
    chunk_index = models.IntegerField()
    # Store embedding as a PostgreSQL vector field
    embedding = ArrayField(models.FloatField(), size=1536, null=True)
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        unique_together = ('document', 'chunk_index')
    
    def __str__(self):
        return f"{self.document.title} - Chunk {self.chunk_index}"

class ChatSession(models.Model):
    """Model for storing chat sessions"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Chat session for {self.document.title} by {self.user.username}"

class ChatMessage(models.Model):
    """Model for storing chat messages"""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]
    
    session = models.ForeignKey(ChatSession, related_name='messages', on_delete=models.CASCADE)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['timestamp']
    
    def __str__(self):
        return f"{self.role} message in {self.session}"