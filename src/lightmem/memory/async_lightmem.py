import asyncio
import uuid
import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional, List

from lightmem.memory.lightmem import LightMemory, MessageNormalizer
from lightmem.memory.utils import MemoryEntry, assign_sequence_numbers_with_timestamps, save_memory_entries
from lightmem.memory.prompts import METADATA_GENERATE_PROMPT, UPDATE_PROMPT
from lightmem.factory.pre_compressor.async_llmlingua_2 import AsyncLlmLingua2Compressor
from lightmem.factory.text_embedder.async_huggingface import AsyncTextEmbedderHuggingface
from lightmem.factory.memory_manager.async_openai import AsyncOpenaiManager
from lightmem.factory.retriever.embeddingretriever.async_qdrant import AsyncQdrant
from lightmem.factory.memory_buffer.async_sensory_memory import AsyncSenMemBufferManager
from lightmem.factory.memory_buffer.short_term_memory import ShortMemBufferManager
from lightmem.factory.topic_segmenter.factory import TopicSegmenterFactory
from lightmem.configs.logging.utils import get_logger


class AsyncLightMemory(LightMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger("AsyncLightMemory")
        self.logger.info("Initializing AsyncLightMemory components")
        
        if self.config.pre_compress:
            self.async_compressor = AsyncLlmLingua2Compressor(self.config.pre_compressor.configs)
        if self.config.topic_segment:
            self.async_senmem_buffer_manager = AsyncSenMemBufferManager(
                max_tokens=self.segmenter.buffer_len, 
                tokenizer=self.segmenter.tokenizer
            )
        self.async_manager = AsyncOpenaiManager(self.config.memory_manager.config)
        self.async_text_embedder = AsyncTextEmbedderHuggingface(self.config.text_embedder.configs)
        if self.config.retrieve_strategy in ["embedding", "hybrid"]:
            self.async_embedding_retriever = AsyncQdrant(self.config.embedding_retriever.configs)

    async def add_memory_async(
        self,
        messages,
        *,
        force_segment: bool = False,
        force_extract: bool = False
    ):
        call_id = f"add_memory_async_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.logger.info(f"========== START {call_id} ==========")
        self.logger.info(f"force_segment={force_segment}, force_extract={force_extract}")
        
        result = {
            "add_input_prompt": [],
            "add_output_prompt": [],
            "api_call_nums": 0
        }
        
        self.logger.debug(f"[{call_id}] Raw input type: {type(messages)}")
        if isinstance(messages, list):
            self.logger.debug(f"[{call_id}] Raw input sample: {json.dumps(messages)}")
        
        normalizer = MessageNormalizer(offset_ms=500)
        msgs = normalizer.normalize_messages(messages)
        self.logger.debug(f"[{call_id}] Normalized messages sample: {json.dumps(msgs)}")
        
        if self.config.pre_compress:
            self.logger.info(f"[{call_id}] Starting async compression")
            compressed_messages = await self.async_compressor.compress_async(msgs)
            self.logger.info(f"[{call_id}] Compression completed")
            self.logger.debug(f"[{call_id}] Compressed messages sample: {json.dumps(compressed_messages)}")
        else:
            compressed_messages = msgs
            self.logger.info(f"[{call_id}] Pre-compression disabled, using normalized messages")
        
        if not self.config.topic_segment:
            self.logger.info(f"[{call_id}] Topic segmentation disabled, returning emitted messages")
            return {
                "triggered": True,
                "cut_index": len(msgs),
                "boundaries": [0, len(msgs)],
                "emitted_messages": msgs,
                "carryover_size": 0,
            }

        all_segments = await self.async_senmem_buffer_manager.add_messages_async(
            compressed_messages, self.segmenter, self.async_text_embedder
        )

        if force_segment:
            all_segments = await self.async_senmem_buffer_manager.cut_with_segmenter_async(
                self.segmenter, self.async_text_embedder, force_segment
            )
        
        if not all_segments:
            self.logger.debug(f"[{call_id}] No segments generated, returning empty result")
            return result

        self.logger.info(f"[{call_id}] Generated {len(all_segments)} segments")
        self.logger.debug(f"[{call_id}] Segments sample: {json.dumps(all_segments)}")

        extract_trigger_num, extract_list = self.shortmem_buffer_manager.add_segments(
            all_segments, self.config.messages_use, force_extract
        )

        if extract_trigger_num == 0:
            self.logger.debug(f"[{call_id}] Extraction not triggered, returning result")
            return result
        
        self.logger.info(f"[{call_id}] Extraction triggered {extract_trigger_num} times, extract_list length: {len(extract_list)}")
        extract_list, timestamps_list, weekday_list = assign_sequence_numbers_with_timestamps(extract_list)
        self.logger.info(f"[{call_id}] Assigned timestamps to {len(extract_list)} items")

        if self.config.metadata_generate and self.config.text_summary:
            self.logger.info(f"[{call_id}] Starting async metadata generation")
            extracted_results = await self.async_manager.meta_text_extract_async(
                METADATA_GENERATE_PROMPT, extract_list, self.config.messages_use
            )
            for item in extracted_results:
                if item is not None:
                    result["add_input_prompt"].append(item["input_prompt"])
                    result["add_output_prompt"].append(item["output_prompt"])
                    result["api_call_nums"] += 1
            self.logger.info(f"[{call_id}] Metadata generation completed with {result['api_call_nums']} API calls")
            extracted_memory_entry = [item["cleaned_result"] for item in extracted_results if item]
            self.logger.info(f"[{call_id}] Extracted {len(extracted_memory_entry)} memory entries")
        
        memory_entries = []
        for topic_memory in extracted_memory_entry:
            if not topic_memory:
                continue
            for entry in topic_memory:
                sequence_n = entry.get("source_id")
                try:
                    time_stamp = timestamps_list[sequence_n]
                    if not isinstance(time_stamp, float):
                        float_time_stamp = datetime.fromisoformat(time_stamp).timestamp()
                    else:
                        float_time_stamp = time_stamp
                    weekday = weekday_list[sequence_n]
                except (IndexError, TypeError) as e:
                    self.logger.warning(f"[{call_id}] Error getting timestamp for sequence {sequence_n}: {e}")
                    time_stamp = None
                    float_time_stamp = None
                    weekday = None
                mem_obj = MemoryEntry(
                    time_stamp=time_stamp,
                    float_time_stamp=float_time_stamp,
                    weekday=weekday,
                    memory=entry.get("fact", ""),
                )
                memory_entries.append(mem_obj)

        self.logger.info(f"[{call_id}] Created {len(memory_entries)} MemoryEntry objects")

        if self.config.update == "offline":
            await self.offline_update_async(memory_entries)
        
        self.logger.info(f"========== END {call_id} ==========")
        return result

    async def offline_update_async(
        self, 
        memory_list: List, 
        construct_update_queue_trigger: bool = False, 
        offline_update_trigger: bool = False
    ):
        call_id = f"offline_update_async_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.logger.info(f"========== START {call_id} ==========")
        self.logger.info(f"[{call_id}] Received {len(memory_list)} memory entries")
        
        if self.config.index_strategy in ["context", "hybrid"]:
            self.logger.info(f"[{call_id}] Saving memory entries to file")
            save_memory_entries(memory_list, "memory_entries.json")

        if self.config.index_strategy in ["embedding", "hybrid"]:
            self.logger.info(f"[{call_id}] Starting async embedding and insertion to vector database")
            
            async def process_entry(mem_obj):
                embedding_vector = await self.async_text_embedder.embed_async(mem_obj.memory)
                ids = mem_obj.id
                while await self.async_embedding_retriever.exists_async(ids):
                    ids = str(uuid.uuid4())
                    mem_obj.id = ids
                payload = {
                    "time_stamp": mem_obj.time_stamp,
                    "float_time_stamp": mem_obj.float_time_stamp,
                    "weekday": mem_obj.weekday,
                    "category": mem_obj.category,
                    "subcategory": mem_obj.subcategory,
                    "memory_class": mem_obj.memory_class,
                    "memory": mem_obj.memory,
                    "original_memory": mem_obj.original_memory,
                    "compressed_memory": mem_obj.compressed_memory,
                }
                await self.async_embedding_retriever.insert_async(
                    vectors=[embedding_vector],
                    payloads=[payload],
                    ids=[ids],
                )
                return ids
            
            inserted_ids = await asyncio.gather(*[process_entry(mem) for mem in memory_list])
            self.logger.info(f"[{call_id}] Successfully inserted {len(inserted_ids)} entries")
            
            if construct_update_queue_trigger:
                self.logger.info(f"[{call_id}] Triggering async update queue construction")
                await self.construct_update_queue_all_entries_async(top_k=20, keep_top_n=10)
            
            if offline_update_trigger:
                self.logger.info(f"[{call_id}] Triggering async offline update")
                await self.offline_update_all_entries_async(score_threshold=0.8)
        
        self.logger.info(f"========== END {call_id} ==========")

    async def construct_update_queue_all_entries_async(
        self, 
        top_k: int = 20, 
        keep_top_n: int = 10, 
        max_concurrent: int = 50
    ):
        call_id = f"construct_queue_async_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.logger.info(f"========== START {call_id} ==========")
        
        all_entries = await self.async_embedding_retriever.get_all_async()
        self.logger.info(f"[{call_id}] Retrieved {len(all_entries)} entries from vector database")
        
        if not all_entries:
            self.logger.warning(f"[{call_id}] No entries found in database")
            self.logger.info(f"========== END {call_id} ==========")
            return
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def update_entry_async(entry):
            async with semaphore:
                eid = entry["id"]
                payload = entry["payload"]
                vec = entry.get("vector")
                ts = payload.get("float_time_stamp", None)
                
                if vec is None or ts is None:
                    return
                
                hits = await self.async_embedding_retriever.search_async(
                    query_vector=vec,
                    limit=top_k,
                    filters={"float_time_stamp": {"lte": ts}}
                )
                
                candidates = []
                for h in hits:
                    hid = h["id"]
                    if hid == eid:
                        continue
                    candidates.append({"id": hid, "score": h.get("score")})
                
                candidates.sort(key=lambda x: x["score"], reverse=True)
                update_queue = candidates[:keep_top_n]
                
                new_payload = dict(payload)
                new_payload["update_queue"] = update_queue
                
                await self.async_embedding_retriever.update_async(vector_id=eid, vector=vec, payload=new_payload)
        
        await asyncio.gather(*[update_entry_async(entry) for entry in all_entries])
        self.logger.info(f"[{call_id}] Queue construction completed")
        self.logger.info(f"========== END {call_id} ==========")

    async def offline_update_all_entries_async(
        self, 
        score_threshold: float = 0.5, 
        max_concurrent: int = 20
    ):
        call_id = f"offline_update_all_async_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.logger.info(f"========== START {call_id} ==========")
        
        all_entries = await self.async_embedding_retriever.get_all_async()
        self.logger.info(f"[{call_id}] Retrieved {len(all_entries)} entries")
        
        if not all_entries:
            self.logger.warning(f"[{call_id}] No entries found")
            self.logger.info(f"========== END {call_id} ==========")
            return
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def update_entry_async(entry):
            async with semaphore:
                eid = entry["id"]
                payload = entry["payload"]
                
                candidate_sources = []
                for other in all_entries:
                    update_queue = other["payload"].get("update_queue", [])
                    for candidate in update_queue:
                        if candidate["id"] == eid and candidate["score"] >= score_threshold:
                            candidate_sources.append(other)
                            break
                
                if not candidate_sources:
                    return
                
                updated_entry = await self.async_manager._call_update_llm_async(
                    UPDATE_PROMPT, entry, candidate_sources
                )
                
                if updated_entry is None:
                    return
                
                action = updated_entry.get("action")
                if action == "delete":
                    await self.async_embedding_retriever.delete_async(eid)
                elif action == "update":
                    new_payload = dict(payload)
                    new_payload["memory"] = updated_entry.get("new_memory")
                    vector = entry.get("vector")
                    await self.async_embedding_retriever.update_async(
                        vector_id=eid, vector=vector, payload=new_payload
                    )
        
        await asyncio.gather(*[update_entry_async(entry) for entry in all_entries])
        self.logger.info(f"[{call_id}] Offline update completed")
        self.logger.info(f"========== END {call_id} ==========")
    
    async def retrieve_async(self, query: str, limit: int = 10, filters: dict = None) -> str:
        call_id = f"retrieve_async_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self.logger.info(f"========== START {call_id} ==========")
        self.logger.info(f"[{call_id}] Query: {query}")
        
        self.logger.debug(f"[{call_id}] Generating embedding for query")
        query_vector = await self.async_text_embedder.embed_async(query)
        self.logger.debug(f"[{call_id}] Query embedding dimension: {len(query_vector)}")
        
        self.logger.info(f"[{call_id}] Searching vector database")
        results = await self.async_embedding_retriever.search_async(
            query_vector=query_vector,
            limit=limit,
            filters=filters,
            return_full=True,
        )
        
        self.logger.info(f"[{call_id}] Found {len(results)} results")
        formatted_results = []
        for r in results:
            payload = r.get("payload", {})
            time_stamp = payload.get("time_stamp", "")
            weekday = payload.get("weekday", "")
            memory = payload.get("memory", "")
            formatted_results.append(f"{time_stamp} {weekday} {memory}")
        
        result_string = "\n".join(formatted_results)
        self.logger.info(f"[{call_id}] Formatted {len(formatted_results)} results")
        self.logger.info(f"========== END {call_id} ==========")
        return result_string

