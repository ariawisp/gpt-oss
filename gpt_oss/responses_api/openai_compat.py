"""Utilities for adapting Responses API payloads to OpenAI-compatible schemas."""

from __future__ import annotations

import datetime
import json
import uuid
from typing import Any, AsyncGenerator, Optional, Tuple

from fastapi import HTTPException

from .events import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseOutputItemDone,
    ResponseOutputTextDelta,
    ResponseReasoningTextDelta,
)
from .types import (
    BrowserToolConfig,
    CodeInterpreterToolConfig,
    FunctionCallItem,
    FunctionCallOutputItem,
    FunctionToolDefinition,
    Item,
    ReasoningConfig,
    ReasoningItem,
    ResponseObject,
    ResponsesRequest,
    Usage,
)


def _normalize_message_content(content: Any) -> str:
    """Flatten OpenAI-style message content into a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type in {"text", "input_text", "output_text"}:
                    text_parts.append(str(part.get("text", "")))
                elif part_type == "tool_result":
                    if isinstance(part.get("content"), str):
                        text_parts.append(part["content"])
                    elif "text" in part:
                        text_parts.append(str(part["text"]))
                    elif "result" in part:
                        text_parts.append(json.dumps(part["result"]))
                elif part_type in {"image_url", "image"}:
                    raise HTTPException(
                        status_code=400,
                        detail="Image content is not supported by this server.",
                    )
                elif "text" in part:
                    text_parts.append(str(part.get("text", "")))
            else:
                text_parts.append(str(part))
        return "".join(text_parts)
    return str(content)


def _ensure_function_arguments(arguments: Any) -> str:
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments)


def _convert_tool_definitions(
    tools_payload: Any,
) -> list[FunctionToolDefinition | BrowserToolConfig | CodeInterpreterToolConfig]:
    converted: list[
        FunctionToolDefinition | BrowserToolConfig | CodeInterpreterToolConfig
    ] = []
    if not isinstance(tools_payload, list):
        return converted
    for tool in tools_payload:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            name = fn.get("name")
            if not name:
                continue
            description = fn.get("description", "")
            parameters = fn.get("parameters") or {}
            strict = bool(tool.get("strict", fn.get("strict", False)))
            converted.append(
                FunctionToolDefinition(
                    type="function",
                    name=name,
                    description=description,
                    parameters=parameters,
                    strict=strict,
                )
            )
        elif tool_type == "browser_search":
            converted.append(BrowserToolConfig(type="browser_search"))
        elif tool_type == "code_interpreter":
            converted.append(CodeInterpreterToolConfig(type="code_interpreter"))
    return converted


def convert_chat_request(
    payload: dict[str, Any]
) -> Tuple[ResponsesRequest, bool, bool]:
    """Translate an OpenAI chat completions payload into a Responses request."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

    n_value = payload.get("n", 1)
    try:
        if int(n_value) != 1:
            raise HTTPException(
                status_code=400,
                detail="Only n=1 is supported for chat completions.",
            )
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid value for 'n' in chat completions request.",
        )

    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        raise HTTPException(
            status_code=400,
            detail="The 'messages' field must be a non-empty list.",
        )

    stream = bool(payload.get("stream", False))
    stream_options = payload.get("stream_options") or {}
    include_usage = bool(stream_options.get("include_usage"))

    instructions_parts: list[str] = []
    input_items: list[Item | FunctionCallItem | FunctionCallOutputItem | ReasoningItem] = []

    for message in messages:
        if not isinstance(message, dict):
            raise HTTPException(
                status_code=400,
                detail="Each message must be an object with role and content.",
            )
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            instructions_parts.append(_normalize_message_content(content))
            continue

        if role == "user":
            text_content = _normalize_message_content(content)
            input_items.append(Item(role="user", content=text_content))
            continue

        if role == "assistant":
            text_content = _normalize_message_content(content)
            if text_content:
                input_items.append(Item(role="assistant", content=text_content))

            tool_calls_data: list[dict[str, Any]] = []
            message_tool_calls = message.get("tool_calls")
            if isinstance(message_tool_calls, list):
                tool_calls_data.extend(tc for tc in message_tool_calls if isinstance(tc, dict))

            function_call = message.get("function_call")
            if isinstance(function_call, dict) and function_call.get("name"):
                tool_calls_data.append(
                    {
                        "id": message.get("id"),
                        "type": "function",
                        "function": function_call,
                    }
                )

            for tool_call in tool_calls_data:
                fn = tool_call.get("function") or {}
                name = fn.get("name")
                if not name:
                    continue
                call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex}"
                arguments = _ensure_function_arguments(fn.get("arguments"))
                input_items.append(
                    FunctionCallItem(
                        name=name,
                        arguments=arguments,
                        call_id=call_id,
                        id=call_id,
                    )
                )
            continue

        if role in {"tool", "function"}:
            tool_call_id = (
                message.get("tool_call_id")
                or message.get("id")
                or message.get("name")
            )
            if not tool_call_id:
                raise HTTPException(
                    status_code=400,
                    detail="Tool messages must specify 'tool_call_id'.",
                )
            output_text = _normalize_message_content(content)
            input_items.append(
                FunctionCallOutputItem(call_id=tool_call_id, output=output_text)
            )
            continue

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported message role '{role}'.",
        )

    if not input_items:
        raise HTTPException(
            status_code=400,
            detail="At least one non-system message must be provided.",
        )

    instructions = "\n".join(part for part in instructions_parts if part and part.strip())
    instructions = instructions or None

    tool_definitions = _convert_tool_definitions(payload.get("tools"))

    tool_choice_param = payload.get("tool_choice")
    mapped_tool_choice: Optional[str] = None
    if isinstance(tool_choice_param, str):
        if tool_choice_param in {"auto", "none"}:
            mapped_tool_choice = tool_choice_param
    elif isinstance(tool_choice_param, dict):
        choice_type = tool_choice_param.get("type")
        if choice_type in {"auto", "none"}:
            mapped_tool_choice = choice_type
        elif choice_type == "function":
            mapped_tool_choice = "auto"

    reasoning_param = payload.get("reasoning")
    reasoning_config: Optional[ReasoningConfig] = None
    if isinstance(reasoning_param, dict):
        effort = reasoning_param.get("effort")
        if effort in {"low", "medium", "high"}:
            reasoning_config = ReasoningConfig(effort=effort)

    metadata_param = payload.get("metadata")
    metadata_dict = metadata_param if isinstance(metadata_param, dict) else {}

    request_kwargs: dict[str, Any] = {
        "instructions": instructions,
        "input": input_items,
        "tools": tool_definitions,
        "model": payload.get("model"),
        "stream": stream,
        "temperature": payload.get("temperature"),
        "tool_choice": mapped_tool_choice,
        "parallel_tool_calls": payload.get("parallel_tool_calls"),
        "metadata": metadata_dict,
    }
    max_tokens_param = payload.get("max_tokens")
    if max_tokens_param is not None:
        request_kwargs["max_output_tokens"] = max_tokens_param
    if reasoning_config is not None:
        request_kwargs["reasoning"] = reasoning_config

    responses_request = ResponsesRequest(**request_kwargs)
    return responses_request, stream, include_usage


def convert_completion_request(
    payload: dict[str, Any]
) -> Tuple[ResponsesRequest, bool, bool]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

    n_value = payload.get("n", 1)
    try:
        if int(n_value) != 1:
            raise HTTPException(
                status_code=400,
                detail="Only n=1 is supported for completions.",
            )
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid value for 'n' in completions request.",
        )

    prompt = payload.get("prompt")
    if isinstance(prompt, list):
        prompt_text = "".join(str(item) for item in prompt)
    elif isinstance(prompt, str):
        prompt_text = prompt
    else:
        raise HTTPException(
            status_code=400, detail="The 'prompt' field must be provided."
        )

    stream = bool(payload.get("stream", False))
    stream_options = payload.get("stream_options") or {}
    include_usage = bool(stream_options.get("include_usage"))

    request_kwargs: dict[str, Any] = {
        "instructions": None,
        "input": [Item(role="user", content=prompt_text)],
        "model": payload.get("model"),
        "stream": stream,
        "temperature": payload.get("temperature"),
        "metadata": payload.get("metadata")
        if isinstance(payload.get("metadata"), dict)
        else {},
    }

    max_tokens_param = payload.get("max_tokens")
    if max_tokens_param is not None:
        request_kwargs["max_output_tokens"] = max_tokens_param

    responses_request = ResponsesRequest(**request_kwargs)
    return responses_request, stream, include_usage


def _collect_response_parts(
    response: ResponseObject,
) -> tuple[Optional[str], list[dict[str, Any]], Optional[str]]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    reasoning_parts: list[str] = []

    for item in response.output:
        if isinstance(item, Item) and item.role == "assistant":
            if isinstance(item.content, list):
                for content_item in item.content:
                    if hasattr(content_item, "text") and content_item.text:
                        text_parts.append(content_item.text)
            elif isinstance(item.content, str):
                text_parts.append(item.content)
        elif isinstance(item, FunctionCallItem):
            tool_calls.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": item.arguments,
                    },
                }
            )
        elif isinstance(item, ReasoningItem) and item.content:
            reasoning_parts.extend(
                part.text for part in item.content if hasattr(part, "text")
            )

    text = "".join(text_parts) if text_parts else None
    reasoning_text = "\n".join(reasoning_parts).strip() if reasoning_parts else None
    return text, tool_calls, reasoning_text


def _build_usage_dict(usage: Optional[Usage]) -> Optional[dict[str, int]]:
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _resolve_finish_reason(text: Optional[str], tool_calls: list[dict[str, Any]]) -> str:
    if tool_calls and not (text and text.strip()):
        return "tool_calls"
    return "stop"


def _format_chat_completion_id(response_id: str) -> str:
    if response_id.startswith("resp_"):
        return f"chatcmpl-{response_id[len('resp_') :]}"
    return response_id


def _format_completion_id(response_id: str) -> str:
    if response_id.startswith("resp_"):
        return f"cmpl-{response_id[len('resp_') :]}"
    return response_id


def build_chat_completion_response(
    response: ResponseObject, model_name: str
) -> dict[str, Any]:
    if response.error is not None:
        raise HTTPException(status_code=500, detail=response.error.message)

    text, tool_calls, reasoning_text = _collect_response_parts(response)
    finish_reason = _resolve_finish_reason(text, tool_calls)

    message: dict[str, Any] = {
        "role": "assistant",
        "content": text,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_text:
        message["reasoning"] = reasoning_text
        message["reasoning_content"] = reasoning_text

    response_id = response.id or f"resp_{uuid.uuid4().hex}"

    payload = {
        "id": _format_chat_completion_id(response_id),
        "object": "chat.completion",
        "created": response.created_at,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }

    usage_dict = _build_usage_dict(response.usage)
    if usage_dict is not None:
        payload["usage"] = usage_dict

    return payload


def build_completion_response(
    response: ResponseObject, model_name: str
) -> dict[str, Any]:
    if response.error is not None:
        raise HTTPException(status_code=500, detail=response.error.message)

    text, tool_calls, _ = _collect_response_parts(response)
    completion_text = text or ""
    finish_reason = _resolve_finish_reason(text, tool_calls)
    response_id = response.id or f"resp_{uuid.uuid4().hex}"

    payload = {
        "id": _format_completion_id(response_id),
        "object": "text_completion",
        "created": response.created_at,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "text": completion_text,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }

    usage_dict = _build_usage_dict(response.usage)
    if usage_dict is not None:
        payload["usage"] = usage_dict

    return payload


def _format_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _build_chat_chunk(
    response_id: str,
    model_name: str,
    created: int,
    delta: dict[str, Any],
    *,
    finish_reason: Optional[str] = None,
    usage: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": _format_chat_completion_id(response_id),
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _build_completion_chunk(
    response_id: str,
    model_name: str,
    created: int,
    text_delta: str,
    *,
    finish_reason: Optional[str] = None,
    usage: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": _format_completion_id(response_id),
        "object": "text_completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "text": text_delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


async def stream_chat_events(
    event_stream: "StreamResponsesEvents",
    response_id: str,
    model_name: str,
    *,
    include_usage: bool,
) -> AsyncGenerator[str, None]:
    created = int(datetime.datetime.now().timestamp())
    final_response: Optional[ResponseObject] = None
    first_chunk_sent = False
    tool_call_index = 0

    async for event in event_stream.run():
        if isinstance(event, ResponseCreatedEvent):
            created = event.response.created_at
            if event.response.id:
                response_id = event.response.id
        elif isinstance(event, ResponseOutputTextDelta):
            delta_payload = {"content": event.delta}
            if not first_chunk_sent:
                delta_payload["role"] = "assistant"
                first_chunk_sent = True
            chunk = _build_chat_chunk(response_id, model_name, created, delta_payload)
            yield _format_sse(chunk)
        elif isinstance(event, ResponseOutputItemDone) and isinstance(
            event.item, FunctionCallItem
        ):
            tool_call = {
                "index": tool_call_index,
                "id": event.item.call_id,
                "type": "function",
                "function": {
                    "name": event.item.name,
                    "arguments": event.item.arguments,
                },
            }
            delta_payload = {"tool_calls": [tool_call]}
            if not first_chunk_sent:
                delta_payload["role"] = "assistant"
                first_chunk_sent = True
            chunk = _build_chat_chunk(response_id, model_name, created, delta_payload)
            yield _format_sse(chunk)
            tool_call_index += 1
        elif isinstance(event, ResponseReasoningTextDelta):
            reasoning_delta = event.delta
            if reasoning_delta:
                delta_payload = {
                    "reasoning": reasoning_delta,
                    "reasoning_content": reasoning_delta,
                }
                if not first_chunk_sent:
                    delta_payload["role"] = "assistant"
                    first_chunk_sent = True
                chunk = _build_chat_chunk(response_id, model_name, created, delta_payload)
                yield _format_sse(chunk)
        elif isinstance(event, ResponseCompletedEvent):
            final_response = event.response
            if event.response.id:
                response_id = event.response.id

    if final_response is None:
        return

    text, tool_calls, _ = _collect_response_parts(final_response)
    finish_reason = _resolve_finish_reason(text, tool_calls)
    usage = _build_usage_dict(final_response.usage) if include_usage else None
    final_chunk = _build_chat_chunk(
        response_id,
        model_name,
        final_response.created_at,
        {},
        finish_reason=finish_reason,
        usage=usage,
    )
    yield _format_sse(final_chunk)
    yield "data: [DONE]\n\n"


async def stream_completion_events(
    event_stream: "StreamResponsesEvents",
    response_id: str,
    model_name: str,
    *,
    include_usage: bool,
) -> AsyncGenerator[str, None]:
    created = int(datetime.datetime.now().timestamp())
    final_response: Optional[ResponseObject] = None

    async for event in event_stream.run():
        if isinstance(event, ResponseCreatedEvent):
            created = event.response.created_at
            if event.response.id:
                response_id = event.response.id
        elif isinstance(event, ResponseOutputTextDelta):
            chunk = _build_completion_chunk(
                response_id, model_name, created, event.delta
            )
            yield _format_sse(chunk)
        elif isinstance(event, ResponseCompletedEvent):
            final_response = event.response
            if event.response.id:
                response_id = event.response.id

    if final_response is None:
        return

    text, tool_calls, _ = _collect_response_parts(final_response)
    finish_reason = _resolve_finish_reason(text, tool_calls)
    usage = _build_usage_dict(final_response.usage) if include_usage else None
    final_chunk = _build_completion_chunk(
        response_id,
        model_name,
        final_response.created_at,
        "",
        finish_reason=finish_reason,
        usage=usage,
    )
    yield _format_sse(final_chunk)
    yield "data: [DONE]\n\n"


__all__ = [
    "build_chat_completion_response",
    "build_completion_response",
    "convert_chat_request",
    "convert_completion_request",
    "stream_chat_events",
    "stream_completion_events",
]
