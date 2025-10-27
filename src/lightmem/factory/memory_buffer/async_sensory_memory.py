import asyncio
import numpy as np
from typing import List, Dict, Optional, Any


class AsyncSenMemBufferManager:
    def __init__(self, max_tokens: int = 512, tokenizer=None):
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.buffer: List[Dict] = []
        self.big_buffer: List[Dict] = []
        self.token_count: int = 0

    def _recount_tokens(self) -> None:
        self.token_count = sum(len(self.tokenizer.encode(m["content"])) for m in self.buffer if m["role"] == "user")

    async def add_messages_async(self, messages: List[Dict], segmenter, text_embedder):
        all_segments = []
        self.big_buffer.extend(messages)

        while self.big_buffer:
            for msg in self.big_buffer:
                if msg["role"] == "user":
                    cur_token_count = len(self.tokenizer.encode(msg["content"]))
                    if self.token_count + cur_token_count <= self.max_tokens:
                        self.buffer.append(msg)
                        self.token_count += cur_token_count
                        self.big_buffer.remove(msg)
                    else:
                        segments = await self.cut_with_segmenter_async(segmenter, text_embedder)
                        all_segments.extend(segments)
                        break
                else:
                    self.buffer.append(msg)
                    self.big_buffer.remove(msg)

        return all_segments

    def should_trigger(self) -> bool:
        return self.token_count >= self.max_tokens

    async def cut_with_segmenter_async(self, segmenter, text_embedder, force_segment: bool = False):
        segments = []
        buffer_texts = [m["content"] for m in self.buffer if m["role"] == "user"]
        boundaries = segmenter.propose_cut(buffer_texts)

        if not boundaries:
            segments.append(self.buffer.copy())
            self.buffer.clear()
            self.token_count = 0
            return segments

        turns = []
        for i in range(0, len(self.buffer), 2):
            user_msg = self.buffer[i]["content"]
            assistant_msg = self.buffer[i + 1]["content"]
            turns.append(user_msg + " " + assistant_msg)

        embeddings_list = await text_embedder.embed_batch_async(turns)
        embeddings = np.vstack([np.array(emb, dtype=np.float32) for emb in embeddings_list])

        fine_boundaries = []
        threshold = 0.2
        while threshold <= 0.5 and not fine_boundaries:
            for i in range(len(turns) - 1):
                sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
                if sim < threshold:
                    fine_boundaries.append(i + 1)
            if not fine_boundaries:
                threshold += 0.05

        if not fine_boundaries:
            segments.append(self.buffer.copy())
            self.buffer.clear()
            self.token_count = 0
            return segments

        adjusted_boundaries = []
        for fb in fine_boundaries:
            for cb in boundaries:
                if abs(fb - cb) <= 3:
                    adjusted_boundaries.append(fb)
                    break
        if not adjusted_boundaries:
            adjusted_boundaries = fine_boundaries

        boundaries = sorted(set(adjusted_boundaries))

        start_idx = 0
        for i, boundary in enumerate(boundaries):
            end_idx = 2 * boundary
            seg = self.buffer[start_idx:end_idx]
            segments.append(seg)
            start_idx = 2 * boundary

        if force_segment:
            segments.append(self.buffer[start_idx:])
            start_idx = len(boundaries)

        if start_idx > 0:
            del self.buffer[:start_idx]
            self._recount_tokens()

        return segments

    def _cosine_similarity(self, vec1, vec2):
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

