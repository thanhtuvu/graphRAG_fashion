"""Shared Claude wrapper for neo4j-graphrag (Text2Cypher etc.).

neo4j-graphrag 1.7.0 reads response.content[0].text, which crashes when a
thinking block comes first. This subclass takes the first *text* block instead.
Same class as Stage 1's unstructured_ingest.py, just reusable.
"""
import os

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import AnthropicLLM
from neo4j_graphrag.llm.types import LLMResponse


class TextOnlyAnthropicLLM(AnthropicLLM):
    @staticmethod
    def _first_text(response) -> LLMResponse:
        for block in response.content:
            if block.type == "text":
                return LLMResponse(content=block.text)
        raise LLMGenerationError("LLM returned no text block.")

    def invoke(self, input, message_history=None, system_instruction=None):
        try:
            response = self.client.messages.create(
                model=self.model_name
                , system=system_instruction or self.anthropic.NOT_GIVEN
                , messages=self.get_messages(input, message_history)
                , **self.model_params
            )
            return self._first_text(response)
        except self.anthropic.APIError as e:
            raise LLMGenerationError(e)

    async def ainvoke(self, input, message_history=None, system_instruction=None):
        try:
            response = await self.async_client.messages.create(
                model=self.model_name
                , system=system_instruction or self.anthropic.NOT_GIVEN
                , messages=self.get_messages(input, message_history)
                , **self.model_params
            )
            return self._first_text(response)
        except self.anthropic.APIError as e:
            raise LLMGenerationError(e)


def build_claude_llm() -> TextOnlyAnthropicLLM:
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
    params = {"max_tokens": 8192}  # required by the Anthropic API
    if model.startswith("claude-sonnet-5-5"):
        params["thinking"] = {"type": "between_tools"}  # same setting as Stage 1
    return TextOnlyAnthropicLLM(model_name=model, model_params=params)