from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DocumentViewSet, ChatSessionViewSet
from . import views
router = DefaultRouter()
router.register(r'documents', DocumentViewSet)
router.register(r'chat-sessions', ChatSessionViewSet)

urlpatterns = [
    path('', include(router.urls)),
    # path('', views.index, name='chatbot_index'),
    path('ui/', views.chatbot_ui, name='chatbot_ui'),

    # path('login/', views.login_view, name='chatbot_login'),
    # path('register/', views.register_view, name='chatbot_register'),
    # path('documents/', views.document_upload_view, name='chatbot_document_upload'),
    # path('chat-sessions/', views.chat_session_view, name='chatbot_chat_session'),
    # path('chat-sessions/<int:session_id>/ask/', views.ask_question_view, name='chatbot_ask_question'),
    
]