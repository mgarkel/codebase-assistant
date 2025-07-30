"""Module to hold string constants for project."""

from langgraph_flow.agents.enums import Intent

LOG_FORMAT_STYLE = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Keys
KEY_BASE_DIRECTORY = "base_directory"
KEY_REPO = "repo"
KEY_URL = "url"
KEY_LOCAL_PATH = "local_path"
KEY_EXIT = "exit"
KEY_QUIT = "quit"
KEY_QUESTION = "question"
KEY_CODE = "code"
KEY_COLLECTION = "collection"
KEY_CONFIG = "cfg"
KEY_INFO = "info"
KEY_INGEST = "ingest"
KEY_CHAT = "chat"
KEY_RESPONSE = "response"
KEY_CONFIG_TOP_K = "top_k"
KEY_SOURCE = "source"
KEY_UNKNOWN = "unknown"
KEY_CHUNK = "chunk"
KEY_OPENAI = "openai"
KEY_INFERENCE_MODEL = "inference_model"
KEY_EMBEDDING_MODEL = "embedding_model"
KEY_INTENT = "intent"
KEY_CONTENT = "content"
KEY_META = "meta"
KEY_VECTORSTORE = "vectorstore"
KEY_TYPE = "type"
KEY_SUBPATH = "subpath"
KEY_PERSIST_DIRECTORY = "persist_directory"
KEY_PROJECT_NAME = "project_name"
KEY_CHROMA = "chroma"
KEY_FAISS = "faiss"

# Values
DEFAULT_TOP_K_EXPLAINER = 3
DEFAULT_TOP_K_NAVIGATOR = 5
DEFAULT_TOK_K_RETRIEVER = 5
MODEL_INFERENCE_OPEN_AI = "gpt-4"
MODEL_EMBEDDING_OPEN_AI = "text-embedding-ada-002"
VALUES_UTF_8 = "utf-8"
ALLOWED_INTENTS = {
    Intent.RETRIEVE.value,
    Intent.NAVIGATE.value,
    Intent.EXPLAIN.value,
}
COLLECTION_NAME = "code_chunks"

# Env variables
ENV_OPENAIAPI_KEY = "OPENAPI_KEY"
