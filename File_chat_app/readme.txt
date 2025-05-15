# Django File Chatbot Project

This guide will help you build a Django backend for a file chatbot application using OpenAI and PostgreSQL vector database for document embeddings.

## Project Structure

```
file_chatbot/
├── file_chatbot/          # Project settings
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py 
│   └── wsgi.py
├── chatbot/               # Main application
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── migrations/
│   ├── models.py
│   ├── serializers.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding_service.py
│   │   ├── file_processor.py
│   │   └── openai_service.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
├── manage.py
└── requirements.txt
```

## Setup Instructions

### 1. Create a new Django project

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install django djangorestframework python-dotenv psycopg2-binary openai tiktoken pypdf python-docx pandas openpyxl

# Create Django project
django-admin startproject file_chatbot
cd file_chatbot
django-admin startapp chatbot
```

### 2. Install PostgreSQL with pgvector extension

For PostgreSQL with pgvector:
```bash
# Install PostgreSQL and pgvector extension
# For Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib
# For macOS with Homebrew:
brew install postgresql

# Connect to PostgreSQL
sudo -u postgres psql

# Create database and enable pgvector extension
CREATE DATABASE file_chatbot_db;
\c file_chatbot_db
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3. Configure Django settings

Update `file_chatbot/settings.py`:

```python
# file_chatbot/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Build paths inside the project
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'django-insecure-default-key-change-me')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv('DEBUG', 'True') == 'True'

ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', '127.0.0.1,localhost').split(',')

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'chatbot',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'file_chatbot.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'file_chatbot.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME', 'file_chatbot_db'),
        'USER': os.getenv('DB_USER', 'postgres'),
        'PASSWORD': os.getenv('DB_PASSWORD', ''),
        'HOST': os.getenv('DB_HOST', 'localhost'),
        'PORT': os.getenv('DB_PORT', '5432'),
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Media files (uploads)
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Rest Framework settings
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.BasicAuthentication',
    ],
}

# OpenAI configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
OPENAI_CHAT_MODEL = os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o')

# File upload settings
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = ['pdf', 'txt', 'docx', 'csv', 'xlsx']
```

### 4. Create a .env file

Create a `.env` file in the project root:

```
DJANGO_SECRET_KEY=your-secret-key
DEBUG=True
DB_NAME=file_chatbot_db
DB_USER=postgres
DB_PASSWORD=your-db-password
DB_HOST=localhost
DB_PORT=5432
OPENAI_API_KEY=your-openai-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o
```

### 5. Define Models

Create the database models in `chatbot/models.py`:

```python
# chatbot/models.py
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
```

### 6. Create Serializers

Create `chatbot/serializers.py`:

```python
# chatbot/serializers.py
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
```

### 7. Create Services

Next, let's create the services to handle document processing, embeddings, and OpenAI interaction.

First, create the file processor service:

```python
# chatbot/services/file_processor.py
import os
from typing import List, Dict, Any, Tuple
import re
import pandas as pd
from PyPDF2 import PdfReader
import docx

def extract_text_from_file(file_path: str) -> str:
    """Extract text from various file types"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.txt':
        return extract_text_from_txt(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension in ['.csv', '.xlsx', '.xls']:
        return extract_text_from_tabular(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    doc = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_tabular(file_path: str) -> str:
    """Extract text from CSV or Excel file"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    else:  # Excel files
        df = pd.read_excel(file_path)
    
    # Convert DataFrame to a text representation
    text = df.to_string(index=False)
    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into chunks with overlap"""
    if not text:
        return []
    
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text).strip()
    
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
    
    return chunks
```

Next, create the embedding service:

```python
# chatbot/services/embedding_service.py
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
```

Now, create the OpenAI service:

```python
# chatbot/services/openai_service.py
from typing import List, Dict, Any
from django.conf import settings
from openai import OpenAI
from ..models import DocumentChunk

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def generate_response(query: str, context_chunks: List[DocumentChunk]) -> str:
    """Generate a response to the query based on context chunks"""
    # Prepare context from chunks
    context = "\n\n".join([chunk.content for chunk in context_chunks])
    
    # Prepare the system message with instructions
    system_message = """
    You are a helpful assistant that answers questions based on the provided document context.
    Use only the information from the context to answer the question.
    If the answer cannot be derived from the context, say "I don't have enough information to answer this question based on the document."
    Provide detailed and accurate answers.
    """
    
    # Create messages array with system, context, and user query
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Context from document:\n\n{context}\n\nQuestion: {query}"}
    ]
    
    # Generate response using OpenAI API
    response = client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1000
    )
    
    return response.choices[0].message.content
```

### 8. Create Views

Let's create the views in `chatbot/views.py`:

```python
# chatbot/views.py
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
```

### 9. Create URLs

Update the project URLs in `file_chatbot/urls.py`:

```python
# file_chatbot/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('chatbot.urls')),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

Create app URLs in `chatbot/urls.py`:

```python
# chatbot/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DocumentViewSet, ChatSessionViewSet

router = DefaultRouter()
router.register(r'documents', DocumentViewSet)
router.register(r'chat-sessions', ChatSessionViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
```

### 10. Setup Admin Interface

Update `chatbot/admin.py`:

```python
# chatbot/admin.py
from django.contrib import admin
from .models import Document, DocumentChunk, ChatSession, ChatMessage

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'uploaded_by', 'uploaded_at', 'processed')
    list_filter = ('processed', 'uploaded_at')
    search_fields = ('title', 'uploaded_by__username')

@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ('document', 'chunk_index')
    list_filter = ('document',)
    search_fields = ('content',)

@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ('user', 'document', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('user__username', 'document__title')

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('session', 'role', 'timestamp')
    list_filter = ('role', 'timestamp')
    search_fields = ('content',)
```

### 11. Create Database Migration

Run the following commands to create and apply migrations:

```bash
python manage.py makemigrations
python manage.py migrate
```

### 12. Create SQL for Adding pgvector Extension

Create a migration to add the vector extension to PostgreSQL:

```bash
python manage.py shell
```

```python
from django.db import migrations

class Migration(migrations.Migration):
    dependencies = []
    operations = [
        migrations.RunSQL('CREATE EXTENSION IF NOT EXISTS vector;')
    ]

# Save this content to a file in chatbot/migrations/
# e.g., 0001_add_vector_extension.py
```

### 13. Create a Requirements File

Create a `requirements.txt` file in the project root:

```
django==5.0.0
djangorestframework==3.14.0
python-dotenv==1.0.0
psycopg2-binary==2.9.9
openai==1.17.0
tiktoken==0.6.0
pypdf==4.0.1
python-docx==1.0.1
pandas==2.2.0
openpyxl==3.1.2
numpy==1.26.3
```

## Usage Guide

1. Set up your environment:
   - Create a `.env` file with OpenAI API key and database settings
   - Install required packages: `pip install -r requirements.txt`
   - Run migrations: `python manage.py migrate`
   - Create a superuser: `python manage.py createsuperuser`

2. Start the server:
   ```bash
   python manage.py runserver
   ```

3. API Endpoints:
   - `POST /api/documents/`: Upload a document
   - `GET /api/documents/`: List uploaded documents
   - `POST /api/chat-sessions/`: Create a chat session for a document
   - `POST /api/chat-sessions/{session_id}/ask/`: Ask a question about the document

4. Upload a document:
   ```
   POST /api/documents/
   Content-Type: multipart/form-data
   
   {
     "title": "Sample Document",
     "file": [binary file data]
   }
   ```

5. Create a chat session:
   ```
   POST /api/chat-sessions/
   Content-Type: application/json
   
   {
     "document": 1
   }
   ```

6. Ask a question:
   ```
   POST /api/chat-sessions/1/ask/
   Content-Type: application/json
   
   {
     "question": "What is the main topic of this document?"
   }
   ```

## SQL for Vector Similarity Optimization

If you want to optimize vector similarity searches in PostgreSQL, you can add indexes:

```sql
-- Create index on document_chunk table for the embedding column
CREATE INDEX document_chunk_embedding_idx ON chatbot_documentchunk USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
```

This can be added to a migration file or executed directly in PostgreSQL.