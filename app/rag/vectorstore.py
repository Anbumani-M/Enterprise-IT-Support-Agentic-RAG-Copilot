import time
from importlib import import_module

from pinecone import Pinecone, ServerlessSpec
from pydantic import SecretStr
from langchain_google_genai import GoogleGenerativeAIEmbeddings

try:
    PineconeVectorStore = import_module("langchain_pinecone").PineconeVectorStore
except ModuleNotFoundError as exc:
    if exc.name != "langchain_pinecone":
        raise
    try:
        PineconeVectorStore = import_module(
            "langchain_community.vectorstores"
        ).PineconeVectorStore
    except ModuleNotFoundError as fallback_exc:
        raise ImportError(
            "Install langchain-pinecone to use the Pinecone vector store."
        ) from fallback_exc

from app.core.config import get_settings

settings = get_settings()

_embeddings = None
_vectorstore = None

# Gemini embedding-001 default output dimension
EMBEDDING_DIMENSIONS = {
    "models/gemini-embedding-001": 3072,
    "gemini-embedding-001": 3072,
}

INDEX_READY_TIMEOUT = 120  # seconds
BATCH_SIZE = 50


def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip().lower()

    if name not in EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Unsupported embedding model: {name}. "
            "Configure a supported Gemini embedding model."
        )

    return EMBEDDING_DIMENSIONS[name]

def get_embeddings():
    global _embeddings

    if _embeddings is None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is missing")

        _embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            api_key=SecretStr(settings.gemini_api_key),
        )

    return _embeddings


def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    desired_dimension = get_embedding_dimension()
    pc = Pinecone(api_key=settings.pinecone_api_key)

    if settings.pinecone_index_name in pc.list_indexes().names():
        current_dimension = pc.describe_index(
            settings.pinecone_index_name
        ).dimension

        if current_dimension != desired_dimension:
            raise RuntimeError(
                f"Pinecone dimension mismatch! "
                f"Existing: {current_dimension}, "
                f"Required: {desired_dimension}. "
                "Create a new index with the correct dimension "
                "or migrate your existing vectors."
            )
    else:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        start = time.time()
        while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
            if time.time() - start > INDEX_READY_TIMEOUT:
                raise TimeoutError(
                    f"Index not ready after {INDEX_READY_TIMEOUT}s"
                )
            time.sleep(2)

    return pc.Index(settings.pinecone_index_name)


def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        _vectorstore = PineconeVectorStore(
            index=ensure_index(),
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

    return _vectorstore


def get_retriever():
    return get_vectorstore().as_retriever(
        search_kwargs={"k": settings.top_k}
    )


def add_documents(chunks):
    store = get_vectorstore()
    ids = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        ids.extend(store.add_documents(batch))
        print(f"Indexed {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")


    return ids
