import asyncio
from openai import OpenAI
from typing import Optional, Literal
from sentence_transformers import SentenceTransformer
import numpy as np
from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig


class AsyncTextEmbedderHuggingface:
    def __init__(self, config: Optional[BaseTextEmbedderConfig] = None):
        self.config = config
        if config.huggingface_base_url:
            self.client = OpenAI(base_url=config.huggingface_base_url)
        else:
            self.config.model = config.model or "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(config.model, **config.model_kwargs)
            self.config.embedding_dims = config.embedding_dims or self.model.get_sentence_embedding_dimension()

    @classmethod
    def from_config(cls, config):
        cls.validate_config(config)
        return cls(config)

    @staticmethod
    def validate_config(config):
        required_keys = ['model_name']
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ValueError(f"Missing required config keys for HuggingFace embedder: {missing}")

    async def embed_async(self, text):
        """
        Get the embedding for the given text using Hugging Face asynchronously.

        Args:
            text (str): The text to embed.
        Returns:
            list: The embedding vector.
        """
        if self.config.huggingface_base_url:
            return await asyncio.to_thread(
                lambda t: self.client.embeddings.create(input=t, model="tei").data[0].embedding,
                text
            )
        else:
            result = await asyncio.to_thread(self.model.encode, text, convert_to_numpy=True)
            if isinstance(result, np.ndarray):
                return result.tolist()
            else:
                return result

    async def embed_batch_async(self, texts):
        """
        Embed multiple texts in parallel.

        Args:
            texts (list): List of texts to embed.
        Returns:
            list: List of embedding vectors.
        """
        if self.config.huggingface_base_url:
            embeddings = await asyncio.gather(*[
                asyncio.to_thread(
                    lambda t: self.client.embeddings.create(input=t, model="tei").data[0].embedding,
                    text
                )
                for text in texts
            ])
            return embeddings
        else:
            results = await asyncio.to_thread(
                self.model.encode,
                texts,
                convert_to_numpy=True,
                batch_size=len(texts)
            )
            if isinstance(results, np.ndarray):
                return results.tolist()
            else:
                return results

