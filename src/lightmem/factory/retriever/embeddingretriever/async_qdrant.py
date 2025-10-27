import asyncio
import logging
import os
import shutil

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    Range,
    VectorParams,
)
from lightmem.configs.retriever.embeddingretriever.base import EmbeddingRetrieverConfig
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AsyncQdrant:
    def __init__(
        self, config: Optional[EmbeddingRetrieverConfig] = None
    ):
        if config.client:
            self.client = config.client
        else:
            params = {}
            if config.api_key:
                params["api_key"] = config.api_key
            if config.url:
                params["url"] = config.url
            if config.host and config.port:
                params["host"] = config.host
                params["port"] = config.port
            if not params:
                params["path"] = config.path
                if not config.on_disk:
                    if os.path.exists(config.path) and os.path.isdir(config.path):
                        shutil.rmtree(config.path)

            self.client = QdrantClient(**params)

        self.collection_name = config.collection_name
        self.embedding_model_dims = config.embedding_model_dims
        self.on_disk = config.on_disk
        self.create_col(config.embedding_model_dims, config.on_disk)

    def create_col(self, vector_size: int, on_disk: bool, distance: Distance = Distance.COSINE):
        response = self.list_cols()
        for collection in response.collections:
            if collection.name == self.collection_name:
                logging.debug(f"Collection {self.collection_name} already exists. Skipping creation.")
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance, on_disk=on_disk),
        )

    async def insert_async(self, vectors: list, payloads: list = None, ids: list = None):
        logger.info(f"Inserting {len(vectors)} vectors into collection {self.collection_name}")
        points = [
            PointStruct(
                id=idx if ids is None else ids[idx],
                vector=vector,
                payload=payloads[idx] if payloads else {},
            )
            for idx, vector in enumerate(vectors)
        ]
        await asyncio.to_thread(
            self.client.upsert,
            collection_name=self.collection_name,
            points=points
        )

    def _create_filter(self, filters: dict) -> Filter:
        conditions = []
        for key, value in filters.items():
            if isinstance(value, dict):
                gte = value.get("gte", None)
                lte = value.get("lte", None)
                conditions.append(FieldCondition(key=key, range=Range(gte=gte, lte=lte)))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=conditions) if conditions else None

    async def search_async(
        self,
        query_vector: list,
        limit: int = 5,
        filters: dict = None,
        return_full: bool = False,
    ) -> list:
        query_filter = self._create_filter(filters) if filters else None
        
        hits = await asyncio.to_thread(
            self.client.query_points,
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )

        results = []
        for h in hits.points:
            if return_full:
                results.append({
                    "id": h.id,
                    "score": h.score,
                    "payload": h.payload,
                })
            else:
                results.append({
                    "id": h.id,
                    "score": h.score,
                })
        return results

    async def update_async(self, vector_id: int, vector: list = None, payload: dict = None):
        update_data = {}
        if vector is not None:
            update_data["vector"] = vector
        if payload is not None:
            update_data["payload"] = payload

        if not update_data:
            return

        point = PointStruct(id=vector_id, **update_data)
        await asyncio.to_thread(
            self.client.upsert,
            collection_name=self.collection_name,
            points=[point]
        )

    async def delete_async(self, vector_id):
        await asyncio.to_thread(
            self.client.delete,
            collection_name=self.collection_name,
            points_selector=PointIdsList(
                points=[vector_id],
            ),
        )

    def list_cols(self):
        return self.client.get_collections()

    def exists(self, vector_id):
        try:
            self.client.retrieve(collection_name=self.collection_name, ids=[vector_id], with_payload=True)
            return True
        except Exception:
            return False

    async def exists_async(self, vector_id):
        return await asyncio.to_thread(self.exists, vector_id)

    async def get_all_async(self):
        all_points = await asyncio.to_thread(
            self.client.scroll,
            collection_name=self.collection_name,
            limit=None,
            with_payload=True,
            with_vectors=True
        )
        
        results = []
        for point in all_points[0]:
            results.append({
                "id": point.id,
                "vector": point.vector,
                "payload": point.payload
            })
        
        return results

    def delete_col(self):
        self.client.delete_collection(collection_name=self.collection_name)

    def col_info(self) -> dict:
        info = self.client.get_collection(self.collection_name)
        return {
            "vectors_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "segments_count": info.segments_count,
            "config": {
                "params": {
                    "vectors": {
                        "size": info.config.params.vectors.size,
                        "distance": info.config.params.vectors.distance.name,
                    }
                }
            },
        }

