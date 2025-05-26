from django.contrib import admin
from .models import Document, ChatSession, ChatMessage

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'uploaded_by', 'uploaded_at', 'processed')
    list_filter = ('processed', 'uploaded_at')
    search_fields = ('title', 'uploaded_by__username')

# @admin.register(DocumentChunk)
# class DocumentChunkAdmin(admin.ModelAdmin):
#     list_display = ('document', 'chunk_index')
#     list_filter = ('document',)
#     search_fields = ('content',)

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