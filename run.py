from app.core.config import get_settings

settings = get_settings()

print(f"App Name: {settings.app_name}")
print(f"App Environment: {settings.app_env}")
print(f"GEMINI API Key: {settings.GEMINI_api_key}")
print(f"Tavily API Key: {settings.tavily_api_key}")
print(f"Pinecone API Key: {settings.pinecone_api_key}")