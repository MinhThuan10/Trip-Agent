from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from app.src.config.settings import settings
import psycopg
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_postgres import PGVector
from sentence_transformers import CrossEncoder
from langfuse import get_client
from langfuse.langchain import CallbackHandler
import os
from langfuse import Langfuse

# Initialize Langfuse client
os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_BASE_URL"] = settings.LANGFUSE_BASE_URL


class BaseService:
    def __init__(self):
        # self.llm = ChatOpenAI(
        #     model="gemini-3.5-flash-lite",
        #     api_key="",
        #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        #     temperature=0,
        # )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=settings.LLM_API_KEY,
            temperature=0,
        )
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={"normalize_embeddings": False},
        )

        self.ranking_model = CrossEncoder(settings.RANKING_MODEL, trust_remote_code=True)

        self.database = psycopg.connect(
            settings.DATABASE_URL
        )

        self.vector_store = PGVector(
                embeddings=self.embedding_model,
                connection=settings.DATABASE_URL,
                collection_name=settings.COLLECTION_NAME,
                use_jsonb=True,
            )

        self.flight_search_api_url = settings.FLIGHT_SEARCH_API_URL
        self.api_v1_str = settings.API_V1_STR


        # Initialize Langfuse client
        self.langfuse = get_client()
        
        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        self.langfuse_handler = CallbackHandler()




base_service = BaseService()
        