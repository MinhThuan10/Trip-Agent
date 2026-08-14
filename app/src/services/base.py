from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from app.src.config.settings import settings
import psycopg
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_postgres import PGVector
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




base_service = BaseService()
        