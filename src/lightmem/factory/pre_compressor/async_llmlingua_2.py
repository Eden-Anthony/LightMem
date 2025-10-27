import asyncio
from typing import Dict, Optional, List
from llmlingua import PromptCompressor
from lightmem.configs.pre_compressor.llmlingua_2 import LlmLingua2Config


class AsyncLlmLingua2Compressor:
    def __init__(self, config: Optional[LlmLingua2Config] = None):
        self.config = config

        if config.llmlingua_config['use_llmlingua2'] is True:
            self._compressor = PromptCompressor(
                model_name=config.llmlingua_config['model_name'],
                device_map=config.llmlingua_config['device_map'],
                use_llmlingua2=config.llmlingua_config['use_llmlingua2'],
                llmlingua2_config=config.llmlingua2_config
            )
        else:
            self._compressor = PromptCompressor(
                model_name=config.llmlingua_config['model_name'],
                device_map=config.llmlingua_config['device_map']
            )

    async def compress_async(
        self,
        messages: List[Dict[str, str]],
        tokenizer: None = None
    ):
        async def compress_one_message(msg):
            compress_config = {
                'context': [msg['content']],
                **self.config.compress_config
            }
            comp_content = await asyncio.to_thread(
                self._compressor.compress_prompt, 
                **compress_config
            )['compressed_prompt']
            
            while tokenizer is not None and len(tokenizer.encode(comp_content)) >= 512:
                new_compress_config = {
                    'context': comp_content,
                    **self.config.compress_config
                }
                comp_content = await asyncio.to_thread(
                    self._compressor.compress_prompt,
                    **new_compress_config
                )['compressed_prompt']
            
            if comp_content:
                msg['content'] = comp_content.strip()
            else:
                msg['content'] = msg['content'].strip()
            
            return msg
        
        compressed_messages = await asyncio.gather(*[compress_one_message(mes) for mes in messages])
        return compressed_messages

    @property
    def inner_compressor(self):
        return self._compressor

