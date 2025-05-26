from typing import List, Dict, Any
from django.conf import settings
from openai import OpenAI

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def generate_chat_completion(query: str, document_text: str) -> str:
    """Generate a response to the query based on the full document text"""
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
        {"role": "user", "content": f"Document Text:\n\n{document_text}\n\nUser Query: {query}"}
    ]
    
    # Generate response using OpenAI API
    response = client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1000
    )
    
    return response.choices[0].message.content