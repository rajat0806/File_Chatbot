from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DocumentViewSet, ChatSessionViewSet
from . import views
router = DefaultRouter()
router.register(r'documents', DocumentViewSet)
router.register(r'chat-sessions', ChatSessionViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('ui/', views.chatbot_ui, name='chatbot_ui'),
    
]