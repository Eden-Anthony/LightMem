import asyncio
from openai import AsyncOpenAI
from typing import List, Dict, Optional, Literal
import json
import os
import httpx
from lightmem.configs.memory_manager.base_config import BaseMemoryManagerConfig
from lightmem.memory.utils import clean_response

model_name_context_windows = {
    "gpt-4o-mini": 128000,
    "qwen3-30b-a3b-instruct-2507": 128000
}


class AsyncOpenaiManager:
    def __init__(self, config: BaseMemoryManagerConfig):
        self.config = config

        if not self.config.model:
            self.config.model = "gpt-4o-mini"

        self.context_windows = model_name_context_windows[self.config.model]

        async_http_client = httpx.AsyncClient(verify=False, timeout=300.0)

        if os.environ.get("OPENROUTER_API_KEY"):
            self.async_client = AsyncOpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url=self.config.openrouter_base_url
                or os.getenv("OPENROUTER_API_BASE")
                or "https://openrouter.ai/api/v1",
                http_client=async_http_client
            )
        else:
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            base_url = (
                self.config.openai_base_url
                or os.getenv("OPENAI_API_BASE")
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.openai.com/v1"
            )

            self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=async_http_client)

        self.config = config

    def _parse_response(self, response, tools):
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content

    async def generate_response_async(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        if os.getenv("OPENROUTER_API_KEY"):
            openrouter_params = {}
            if self.config.models:
                openrouter_params["models"] = self.config.models
                openrouter_params["route"] = self.config.route
                params.pop("model")

            if self.config.site_url and self.config.app_name:
                extra_headers = {
                    "HTTP-Referer": self.config.site_url,
                    "X-Title": self.config.app_name,
                }
                openrouter_params["extra_headers"] = extra_headers

            params.update(**openrouter_params)

        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = await self.async_client.chat.completions.create(**params)
        return self._parse_response(response, tools)

    async def meta_text_extract_async(
        self,
        system_prompt: str,
        extract_list: List[List[List[Dict]]],
        messages_use: Literal["user_only", "assistant_only", "hybrid"] = "user_only"
    ) -> List[Optional[Dict]]:
        if not extract_list:
            return []

        def concatenate_messages(segment: List[Dict], messages_use: str) -> str:
            role_filter = {
                "user_only": {"user"},
                "assistant_only": {"assistant"},
                "hybrid": {"user", "assistant"}
            }

            if messages_use not in role_filter:
                raise ValueError(f"Invalid messages_use value: {messages_use}")

            allowed_roles = role_filter[messages_use]
            message_lines = []

            for mes in segment:
                if mes.get("role") in allowed_roles:
                    sequence_id = mes["sequence_number"]
                    role = mes["role"]
                    content = mes.get("content", "")
                    message_lines.append(f"{sequence_id}.{role}: {content}")

            return "\n".join(message_lines)

        semaphore = asyncio.Semaphore(20)

        async def process_segment_async(api_call_segments: List[List[Dict]]):
            async with semaphore:
                try:
                    user_prompt_parts = []
                    for idx, topic_segment in enumerate(api_call_segments, start=1):
                        topic_text = concatenate_messages(topic_segment, messages_use)
                        user_prompt_parts.append(f"--- Topic {idx} ---\n{topic_text}")

                    user_prompt = "\n".join(user_prompt_parts)

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                    raw_response = await self.generate_response_async(
                        messages=messages,
                        response_format={"type": "json_object"}
                    )
                    cleaned_result = clean_response(raw_response)
                    return {
                        "input_prompt": messages,
                        "output_prompt": raw_response,
                        "cleaned_result": cleaned_result
                    }
                except Exception as e:
                    print(f"Error processing API call: {e}")
                    return None

        results = await asyncio.gather(*[process_segment_async(segments) for segments in extract_list])
        return results

    async def _call_update_llm_async(self, system_prompt, target_entry, candidate_sources):
        target_memory = target_entry["payload"]["memory"]
        candidate_memories = [c["payload"]["memory"] for c in candidate_sources]

        user_prompt = (
            f"Target memory:{target_memory}\n"
            f"Candidate memories:\n" + "\n".join([f"- {m}" for m in candidate_memories])
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response_text = await self.generate_response_async(
            messages=messages,
            response_format={"type": "json_object"}
        )

        try:
            result = json.loads(response_text)
            if "action" not in result:
                return {"action": "ignore"}
            return result
        except json.JSONDecodeError:
            return {"action": "ignore"}

