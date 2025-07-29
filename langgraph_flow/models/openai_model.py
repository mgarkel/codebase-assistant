import os
from threading import Lock

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils.constants import (
    ENV_OPENAIAPI_KEY,
    KEY_EMBEDDING_MODEL,
    KEY_INFERENCE_MODEL,
    KEY_OPENAI,
    MODEL_EMBEDDING_OPEN_AI,
    MODEL_INFERENCE_OPEN_AI,
)


class SingletonMeta(type):
    """
    Thread-safe implementation of Singleton.
    """

    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        # Double-checked locking
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class OpenAIModel(metaclass=SingletonMeta):
    def __init__(self, cfg):
        self.__openai_api_key = os.getenv(ENV_OPENAIAPI_KEY)
        self.__cfg = cfg
        self._inference_model = None
        self._embedding_model = None

    @property
    def inference_model(self):
        if self._inference_model is None:
            model_name = self.__cfg.get(KEY_OPENAI, {}).get(
                KEY_INFERENCE_MODEL, MODEL_INFERENCE_OPEN_AI
            )
            self._inference_model = ChatOpenAI(
                model=model_name,
                temperature=0,
                openai_api_key=self.__openai_api_key,
            )
        return self._inference_model

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            model_name = self.__cfg.get(KEY_OPENAI, {}).get(
                KEY_EMBEDDING_MODEL, MODEL_EMBEDDING_OPEN_AI
            )
            self._embedding_model = OpenAIEmbeddings(
                model=model_name, openai_api_key=self.__openai_api_key
            )
        return self._embedding_model
