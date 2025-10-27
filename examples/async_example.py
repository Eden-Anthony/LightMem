import asyncio
import json
from lightmem.memory.async_lightmem import AsyncLightMemory


async def main():
    config = {
        "pre_compress": True,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": "/path/to/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                    "device_map": "cuda",
                    "use_llmlingua2": True,
                },
            }
        },
        "topic_segment": True,
        "precomp_topic_shared": True,
        "topic_segmenter": {
            "model_name": "llmlingua-2",
        },
        "messages_use": "user_only",
        "metadata_generate": True,
        "text_summary": True,
        "memory_manager": {
            "model_name": "openai",
            "configs": {
                "model": "gpt-4o-mini",
                "api_key": "sk-...",
                "max_tokens": 16000,
                "openai_base_url": "https://api.openai.com/v1"
            }
        },
        "extract_threshold": 0.5,
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": "/path/to/all-MiniLM-L6-v2",
                "embedding_dims": 384,
                "model_kwargs": {"device": "cuda"},
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": "user_memory",
                "embedding_model_dims": 384,
                "path": "./data/user_memory",
            }
        },
        "update": "offline",
    }

    lightmem = AsyncLightMemory.from_config(config)

    session_1_messages = [
        {"role": "user", "content": "My name is Alex, I'm a software engineer working at Google.", "time_stamp": "2024-01-15 (Mon) 10:30"},
        {"role": "assistant", "content": "Nice to meet you, Alex! That sounds like an interesting career.", "time_stamp": "2024-01-15 (Mon) 10:30"},
        {"role": "user", "content": "I've been there for 3 years now. Before that, I worked at Microsoft.", "time_stamp": "2024-01-15 (Mon) 10:30"},
        {"role": "assistant", "content": "That's impressive experience.", "time_stamp": "2024-01-15 (Mon) 10:30"},
        {"role": "user", "content": "My favorite programming language is Python, and I also enjoy cooking in my free time.", "time_stamp": "2024-01-15 (Mon) 10:30"},
        {"role": "assistant", "content": "Python is great! What do you like to cook?", "time_stamp": "2024-01-15 (Mon) 10:30"},
        {"role": "user", "content": "I love making Italian food, especially pasta carbonara. My girlfriend Sarah introduced me to it.", "time_stamp": "2024-01-15 (Mon) 10:30"},
        {"role": "assistant", "content": "Pasta carbonara is delicious! It's nice that Sarah got you into cooking.", "time_stamp": "2024-01-15 (Mon) 10:30"}
    ]

    print("Adding memories asynchronously...")
    result = await lightmem.add_memory_async(session_1_messages, force_segment=True, force_extract=True)
    print(f"API calls made: {result['api_call_nums']}")

    session_2_messages = [
        {"role": "user", "content": "Good news! I got promoted to Senior Software Engineer last week.", "time_stamp": "2024-01-17 (Wed) 14:00"},
        {"role": "assistant", "content": "Congratulations! That's fantastic news.", "time_stamp": "2024-01-17 (Wed) 14:00"},
        {"role": "user", "content": "I'm also moving to a new apartment next month in downtown Seattle.", "time_stamp": "2024-01-17 (Wed) 14:00"},
        {"role": "assistant", "content": "That's exciting! Good luck with the move.", "time_stamp": "2024-01-17 (Wed) 14:00"},
        {"role": "user", "content": "I'm allergic to peanuts, by the way, so never recommend places that might use them.", "time_stamp": "2024-01-17 (Wed) 14:00"},
        {"role": "assistant", "content": "Noted. I'll keep that in mind.", "time_stamp": "2024-01-17 (Wed) 14:00"}
    ]

    result2 = await lightmem.add_memory_async(session_2_messages, force_segment=True, force_extract=True)
    print(f"API calls made: {result2['api_call_nums']}")

    session_3_messages = [
        {"role": "user", "content": "Sarah and I broke up last week. It's been tough.", "time_stamp": "2024-01-19 (Fri) 16:30"},
        {"role": "assistant", "content": "I'm sorry to hear that. Take your time.", "time_stamp": "2024-01-19 (Fri) 16:30"},
        {"role": "user", "content": "I've been trying to keep busy. Signed up for a marathon training program. My goal is to run the Boston Marathon in April.", "time_stamp": "2024-01-19 (Fri) 16:30"},
        {"role": "assistant", "content": "That's a great goal. Training will keep you focused.", "time_stamp": "2024-01-19 (Fri) 16:30"},
        {"role": "user", "content": "Also, I'm now vegetarian. Changed my diet completely last month.", "time_stamp": "2024-01-19 (Fri) 16:30"},
        {"role": "assistant", "content": "That's a big change. How are you finding it?", "time_stamp": "2024-01-19 (Fri) 16:30"}
    ]

    result3 = await lightmem.add_memory_async(session_3_messages, force_segment=True, force_extract=True)
    print(f"API calls made: {result3['api_call_nums']}")

    await lightmem.construct_update_queue_all_entries_async(top_k=20, keep_top_n=10)
    await lightmem.offline_update_all_entries_async(score_threshold=0.8)

    query_1 = "What is Alex's job and where do they work?"
    memories_1 = await lightmem.retrieve_async(query_1, limit=5)
    print(f"\nQuery 1: {query_1}")
    print(f"Results:\n{memories_1}")

    query_2 = "What are Alex's dietary restrictions?"
    memories_2 = await lightmem.retrieve_async(query_2, limit=5)
    print(f"\nQuery 2: {query_2}")
    print(f"Results:\n{memories_2}")

    query_3 = "What does Alex enjoy doing?"
    memories_3 = await lightmem.retrieve_async(query_3, limit=10)
    print(f"\nQuery 3: {query_3}")
    print(f"Results:\n{memories_3}")


if __name__ == "__main__":
    asyncio.run(main())

