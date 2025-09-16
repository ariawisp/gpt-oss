import os
import json
import datetime
import uuid
from typing import Any, AsyncGenerator, Callable, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncoding,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    StreamState,
    SystemContent,
    ToolDescription,
)

from gpt_oss.tools.python_docker.docker_tool import PythonTool
from gpt_oss.tools.simple_browser import SimpleBrowserTool
from gpt_oss.tools.simple_browser.backend import YouComBackend, ExaBackend

from .events import (
    ResponseCodeInterpreterCallCompleted,
    ResponseCodeInterpreterCallInProgress,
    ResponseCompletedEvent,
    ResponseContentPartAdded,
    ResponseContentPartDone,
    ResponseCreatedEvent,
    ResponseEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAdded,
    ResponseOutputItemDone,
    ResponseOutputTextAnnotationAdded,
    ResponseOutputTextDelta,
    ResponseOutputTextDone,
    ResponseReasoningTextDelta,
    ResponseReasoningTextDone,
    ResponseWebSearchCallCompleted,
    ResponseWebSearchCallInProgress,
    ResponseWebSearchCallSearching,
)
from .types import (
    BrowserToolConfig,
    CodeInterpreterCallItem,
    CodeInterpreterToolConfig,
    Error,
    FunctionCallItem,
    FunctionCallOutputItem,
    FunctionToolDefinition,
    Item,
    MODEL_IDENTIFIER,
    ReasoningConfig,
    ReasoningItem,
    ReasoningTextContentItem,
    ResponseObject,
    ResponsesRequest,
    TextContentItem,
    UrlCitation,
    Usage,
    WebSearchActionFind,
    WebSearchActionOpenPage,
    WebSearchActionSearch,
    WebSearchCallItem,
)

DEFAULT_TEMPERATURE = 0.0


def get_reasoning_effort(effort: Literal["low", "medium", "high"]) -> ReasoningEffort:
    if effort == "low":
        return ReasoningEffort.LOW
    if effort == "medium":
        return ReasoningEffort.MEDIUM
    if effort == "high":
        return ReasoningEffort.HIGH
    raise ValueError(f"Invalid reasoning effort: {effort}")


def is_not_builtin_tool(recipient: str) -> bool:
    return (
        not recipient.startswith("browser.")
        and not recipient == "python"
        and not recipient == "assistant"
    )


def create_api_server(
    infer_next_token: Callable[[list[int], float], int],
    encoding: HarmonyEncoding,
    model_id: Optional[str] = None,
) -> FastAPI:
    app = FastAPI()
    responses_store: dict[str, tuple[ResponsesRequest, ResponseObject]] = {}
    reported_model_id = model_id or MODEL_IDENTIFIER

    def _maybe_override_model_id(body: ResponsesRequest) -> None:
        if body.model in (None, "", MODEL_IDENTIFIER):
            body.model = reported_model_id

    def _normalize_message_content(content: Any) -> str:
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

    def _convert_tool_definitions(tools_payload: Any) -> list[
        FunctionToolDefinition | BrowserToolConfig | CodeInterpreterToolConfig
    ]:
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

    def _convert_chat_request(
        payload: dict[str, Any]
    ) -> tuple[ResponsesRequest, bool, bool]:
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400, detail="Request body must be a JSON object."
            )

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
        input_items: list[
            Item | FunctionCallItem | FunctionCallOutputItem | ReasoningItem
        ] = []

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
                    tool_calls_data.extend(
                        tc for tc in message_tool_calls if isinstance(tc, dict)
                    )

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

        instructions = "\n".join(
            part for part in instructions_parts if part and part.strip()
        )
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
            "max_output_tokens": payload.get("max_tokens"),
            "temperature": payload.get("temperature"),
            "tool_choice": mapped_tool_choice,
            "parallel_tool_calls": payload.get("parallel_tool_calls"),
            "metadata": metadata_dict,
        }
        if reasoning_config is not None:
            request_kwargs["reasoning"] = reasoning_config

        responses_request = ResponsesRequest(**request_kwargs)
        return responses_request, stream, include_usage

    def _convert_completion_request(
        payload: dict[str, Any]
    ) -> tuple[ResponsesRequest, bool, bool]:
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400, detail="Request body must be a JSON object."
            )

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
            "max_output_tokens": payload.get("max_tokens"),
            "temperature": payload.get("temperature"),
            "metadata": payload.get("metadata")
            if isinstance(payload.get("metadata"), dict)
            else {},
        }

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

    def _resolve_finish_reason(
        text: Optional[str], tool_calls: list[dict[str, Any]]
    ) -> str:
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

    def _build_chat_completion_response(
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

    def _build_completion_response(
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

    def _prepare_event_stream(
        body: ResponsesRequest,
        request: Optional[Request],
        *,
        stream_override: Optional[bool] = None,
        as_sse_override: Optional[bool] = None,
        store_response: bool = True,
    ) -> tuple["StreamResponsesEvents", bool, str]:
        print("request received")

        _maybe_override_model_id(body)

        use_browser_tool = any(
            getattr(tool, "type", None) == "browser_search"
            for tool in (body.tools or [])
        )
        use_code_interpreter = any(
            getattr(tool, "type", None) == "code_interpreter"
            for tool in (body.tools or [])
        )

        if use_browser_tool:
            tool_backend = os.getenv("BROWSER_BACKEND", "exa")
            if tool_backend == "youcom":
                backend = YouComBackend(source="web")
            elif tool_backend == "exa":
                backend = ExaBackend(source="web")
            else:
                raise ValueError(f"Invalid tool backend: {tool_backend}")
            browser_tool: Optional[SimpleBrowserTool] = SimpleBrowserTool(backend=backend)
        else:
            browser_tool = None

        if use_code_interpreter:
            python_tool: Optional[PythonTool] = PythonTool()
        else:
            python_tool = None

        if body.previous_response_id:
            prev = responses_store.get(body.previous_response_id)
            if prev:
                prev_req, prev_resp = prev

                def _ensure_list(inp):
                    if isinstance(inp, str):
                        return [
                            Item(
                                type="message",
                                role="user",
                                content=[TextContentItem(type="input_text", text=inp)],
                            )
                        ]
                    return list(inp)

                merged_input = _ensure_list(prev_req.input) + list(prev_resp.output)
                merged_input.extend(_ensure_list(body.input))

                if body.instructions is None:
                    body.instructions = prev_req.instructions
                body.input = merged_input

        system_message_content = SystemContent.new().with_conversation_start_date(
            datetime.datetime.now().strftime("%Y-%m-%d")
        )

        if body.reasoning is not None:
            try:
                reasoning_effort = get_reasoning_effort(body.reasoning.effort)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))
            system_message_content = system_message_content.with_reasoning_effort(
                reasoning_effort
            )

        if use_browser_tool:
            system_message_content = system_message_content.with_tools(
                browser_tool.tool_config
            )
        if use_code_interpreter:
            system_message_content = system_message_content.with_tools(
                python_tool.tool_config
            )

        system_message = Message.from_role_and_content(
            Role.SYSTEM, system_message_content
        )
        messages = [system_message]

        if body.instructions or body.tools:
            developer_message_content = DeveloperContent.new().with_instructions(
                body.instructions
            )

            tools = []
            for tool in body.tools or []:
                if getattr(tool, "type", None) == "function":
                    tools.append(
                        ToolDescription.new(
                            tool.name,
                            tool.description,
                            tool.parameters,
                        )
                    )

            if tools:
                developer_message_content = developer_message_content.with_function_tools(
                    tools
                )

            developer_message = Message.from_role_and_content(
                Role.DEVELOPER, developer_message_content
            )

            messages.append(developer_message)

        if isinstance(body.input, str):
            user_message = Message.from_role_and_content(Role.USER, body.input)
            messages.append(user_message)
        else:
            is_last_message_function_call_output = (
                len(body.input) > 0 and body.input[-1].type == "function_call_output"
            )
            function_call_map: dict[str, FunctionCallItem] = {}
            last_assistant_idx = -1
            for idx, item in enumerate(body.input):
                if item.type == "message" and item.role == Role.ASSISTANT:
                    last_assistant_idx = idx

            for idx, item in enumerate(body.input):
                if item.type == "message":
                    if isinstance(item.content, str):
                        messages.append(
                            Message.from_role_and_content(item.role, item.content)
                        )
                    else:
                        for content_item in item.content:
                            messages.append(
                                Message.from_role_and_content(
                                    item.role, content_item.text
                                )
                            )
                    if item.role == Role.ASSISTANT:
                        messages[-1] = messages[-1].with_channel("final")
                elif item.type == "reasoning":
                    if idx > last_assistant_idx and is_last_message_function_call_output:
                        for content_item in item.content:
                            messages.append(
                                Message.from_role_and_content(
                                    Role.ASSISTANT, content_item.text
                                ).with_channel("analysis")
                            )
                elif item.type == "function_call":
                    function_call_map[item.call_id] = item
                    messages.append(
                        Message.from_role_and_content(Role.ASSISTANT, item.arguments)
                        .with_recipient(f"functions.{item.name}")
                        .with_channel("commentary")
                    )
                elif item.type == "function_call_output":
                    function_call = function_call_map.get(item.call_id, None)
                    if not function_call:
                        raise ValueError(f"Function call {item.call_id} not found")

                    messages.append(
                        Message.from_author_and_content(
                            Author.new(Role.TOOL, f"functions.{function_call.name}"),
                            item.output,
                        )
                        .with_recipient("assistant")
                        .with_channel("commentary")
                    )

        conversation = Conversation.from_messages(messages)

        initial_tokens = encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )
        print(encoding.decode_utf8(initial_tokens))
        response_id = f"resp_{uuid.uuid4().hex}"

        stream = stream_override if stream_override is not None else bool(body.stream)
        body.stream = stream
        as_sse = as_sse_override if as_sse_override is not None else stream

        def store_callback(rid: str, req: ResponsesRequest, resp: ResponseObject):
            responses_store[rid] = (req, resp)

        event_stream = StreamResponsesEvents(
            initial_tokens,
            body,
            as_sse=as_sse,
            request=request,
            response_id=response_id,
            store_callback=store_callback if store_response and body.store else None,
            browser_tool=browser_tool,
            python_tool=python_tool,
        )

        return event_stream, stream, response_id

    async def _stream_chat_events(
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
                    chunk = _build_chat_chunk(
                        response_id, model_name, created, delta_payload
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
        usage = (
            _build_usage_dict(final_response.usage) if include_usage else None
        )
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

    async def _stream_completion_events(
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
        usage = (
            _build_usage_dict(final_response.usage) if include_usage else None
        )
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

    def generate_response(
        input_tokens: list[int],
        output_tokens: list[int],
        request_body: ResponsesRequest,
        debug_mode: bool = False,
        function_call_ids: Optional[list[tuple[str, str]]] = None,
        response_id: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        browser_tool: Optional[SimpleBrowserTool] = None,
        browser_call_ids: Optional[list[str]] = None,
        python_tool: Optional[PythonTool] = None,
        python_call_ids: Optional[list[str]] = None,
    ) -> ResponseObject:
        output = []
        error = None
        if len(output_tokens) > 0:
            if debug_mode:
                try:
                    entries = encoding.parse_messages_from_completion_tokens(
                        output_tokens, Role.ASSISTANT
                    )
                except Exception as e:
                    print(f"Error parsing tokens: {e}")
                    error = Error(
                        code="invalid_function_call",
                        message=f"{e}",
                    )
                    entries = []
            else:
                entries = encoding.parse_messages_from_completion_tokens(
                    output_tokens, Role.ASSISTANT
                )

            fc_index = 0
            browser_tool_index = 0
            python_tool_index = 0
            for entry in entries:
                entry_dict = entry.to_dict()
                if len(entry_dict.get("recipient", "")) > 0 and is_not_builtin_tool(
                    entry_dict["recipient"]
                ):
                    call = entry_dict["content"][0]
                    arguments = call["text"]
                    name = entry_dict["recipient"]

                    if name.startswith("functions."):
                        name = name[len("functions.") :]
                    if function_call_ids and fc_index < len(function_call_ids):
                        fc_id, call_id = function_call_ids[fc_index]
                    else:
                        fc_id, call_id = (
                            f"fc_{uuid.uuid4().hex}",
                            f"call_{uuid.uuid4().hex}",
                        )
                    fc_index += 1
                    output.append(
                        FunctionCallItem(
                            type="function_call",
                            name=name,
                            arguments=arguments,
                            id=fc_id,
                            call_id=call_id,
                        )
                    )
                elif (
                    len(entry_dict.get("recipient", "")) > 0
                    and entry_dict["recipient"].startswith("browser.")
                    and browser_tool is not None
                ):
                    # Mirror event-based creation of WebSearchCallItems when the browser tool is invoked
                    name = entry_dict["recipient"]
                    call = entry_dict["content"][0]
                    arguments = call["text"]
                    function_name = name[len("browser.") :]

                    # Reconstruct a Message for argument parsing
                    tool_msg = (
                        Message.from_role_and_content(Role.ASSISTANT, arguments)
                        .with_recipient(name)
                        .with_channel("analysis")
                    )

                    action = None
                    try:
                        parsed_args = browser_tool.process_arguments(tool_msg)
                        if function_name == "search":
                            action = WebSearchActionSearch(
                                type="search",
                                query=parsed_args["query"],
                            )
                        elif function_name == "open":
                            action = WebSearchActionOpenPage(
                                type="open_page",
                                url=parsed_args["url"],
                            )
                        elif function_name == "find":
                            action = WebSearchActionFind(
                                type="find",
                                pattern=parsed_args["pattern"],
                                url=parsed_args["url"],
                            )
                    except Exception as e:
                        print(f"Error processing browser tool arguments: {e}")
                        action = None

                    if action is not None:
                        if browser_call_ids and browser_tool_index < len(
                            browser_call_ids
                        ):
                            web_search_call_id = browser_call_ids[browser_tool_index]
                        else:
                            web_search_call_id = f"ws_{uuid.uuid4().hex}"
                        browser_tool_index += 1
                        output.append(
                            WebSearchCallItem(
                                type="web_search_call",
                                id=web_search_call_id,
                                action=action,
                            )
                        )
                elif (
                    len(entry_dict.get("recipient", "")) > 0
                    and entry_dict["recipient"].startswith("python")
                    and python_tool is not None
                ):
                    if python_call_ids and python_tool_index < len(python_call_ids):
                        code_call_id = python_call_ids[python_tool_index]
                    else:
                        code_call_id = f"ci_{uuid.uuid4().hex}"
                    python_tool_index += 1
                    output.append(
                        CodeInterpreterCallItem(
                            type="code_interpreter_call",
                            id=code_call_id,
                        )
                    )
                elif entry_dict["channel"] == "final":
                    content = []
                    for content_entry in entry_dict["content"]:
                        if browser_tool:
                            text_content, annotation_entries, _has_partial_citations = (
                                browser_tool.normalize_citations(content_entry["text"])
                            )
                            annotations = [UrlCitation(**a) for a in annotation_entries]
                        else:
                            text_content = content_entry["text"]
                            annotations = []

                        content.append(
                            TextContentItem(
                                type="output_text",
                                text=text_content,
                                annotations=annotations,
                            )
                        )

                    output.append(
                        Item(
                            type="message",
                            role="assistant",
                            content=content,
                            status="completed",
                        )
                    )
                elif entry_dict["channel"] == "analysis":
                    summary = []
                    content = [
                        ReasoningTextContentItem(
                            type="reasoning_text",
                            text=entry["text"],
                        )
                        for entry in entry_dict["content"]
                    ]
                    output.append(
                        ReasoningItem(
                            type="reasoning",
                            summary=summary,
                            content=content,
                        )
                    )
        else:
            output = []

        usage = (
            Usage(
                input_tokens=len(input_tokens),
                output_tokens=len(output_tokens),
                total_tokens=len(input_tokens) + len(output_tokens),
            )
            if len(output_tokens) > 0
            else None
        )

        try:
            debug_str = encoding.decode_utf8(input_tokens + output_tokens)
        except Exception:
            debug_str = input_tokens + output_tokens
        try:
            debug_input_str = encoding.decode_utf8(input_tokens)
        except Exception:
            debug_input_str = input_tokens
        try:
            debug_output_str = encoding.decode_utf8(output_tokens)
        except Exception:
            debug_output_str = output_tokens

        metadata = (
            {
                "__debug": debug_str,
                "__debug_input": debug_input_str,
                "__debug_output": debug_output_str,
            }
            if debug_mode
            else {}
        )

        return ResponseObject(
            created_at=int(datetime.datetime.now().timestamp()),
            status="completed",
            output=output,
            text={"format": {"type": "text"}},
            usage=usage,
            max_output_tokens=request_body.max_output_tokens,
            error=error,
            metadata=metadata,
            id=response_id,
            previous_response_id=previous_response_id,
        )

    class StreamResponsesEvents:
        initial_tokens: list[int]
        tokens: list[int]
        output_tokens: list[int]
        output_text: str
        request_body: ResponsesRequest
        request: Request
        sequence_number: int

        def __init__(
            self,
            initial_tokens,
            request_body: ResponsesRequest,
            as_sse: bool = False,
            request: Optional[Request] = None,
            response_id: Optional[str] = None,
            store_callback: Optional[
                Callable[[str, ResponsesRequest, ResponseObject], None]
            ] = None,
            browser_tool: Optional[SimpleBrowserTool] = None,
            python_tool: Optional[PythonTool] = None,
        ):
            self.initial_tokens = initial_tokens
            self.tokens = initial_tokens.copy()
            self.output_tokens = []
            self.output_text = ""
            self.request_body = request_body
            self.parser = StreamableParser(encoding, role=Role.ASSISTANT)
            self.as_sse = as_sse
            self.debug_mode = request_body.metadata.get(
                "__debug", False
            )  # we use this for demo purposes
            # Set temperature for this stream, fallback to DEFAULT_TEMPERATURE if not set
            self.temperature = (
                request_body.temperature
                if request_body.temperature is not None
                else DEFAULT_TEMPERATURE
            )
            self.request = request
            self.sequence_number = 0
            self.function_call_ids: list[tuple[str, str]] = []
            self.response_id = response_id
            self.store_callback = store_callback
            self.new_request = True
            self.browser_tool = browser_tool
            self.use_browser_tool = browser_tool is not None
            self.browser_call_ids: list[str] = []
            self.python_tool = python_tool
            self.use_code_interpreter = python_tool is not None
            self.python_call_ids: list[str] = []

        def _send_event(self, event: ResponseEvent):
            event.sequence_number = self.sequence_number
            self.sequence_number += 1
            if self.as_sse:
                return f"event: {event.type}\ndata: {event.model_dump_json(indent=None)}\n\n"
            else:
                return event

        async def run(self):
            browser_tool = self.browser_tool
            self.new_request = True
            initial_response = generate_response(
                self.initial_tokens,
                self.output_tokens,
                self.request_body,
                function_call_ids=self.function_call_ids,
                response_id=self.response_id,
                previous_response_id=self.request_body.previous_response_id,
                browser_tool=self.browser_tool,
                browser_call_ids=self.browser_call_ids,
                python_tool=self.python_tool,
                python_call_ids=self.python_call_ids,
            )
            initial_response.status = "in_progress"
            yield self._send_event(
                ResponseCreatedEvent(
                    type="response.created",
                    response=initial_response,
                )
            )
            yield self._send_event(
                ResponseInProgressEvent(
                    type="response.in_progress",
                    response=initial_response,
                )
            )

            current_content_index = (
                0  # for this implementation we will always have one content item only
            )
            current_output_index = -1
            sent_output_item_added = False

            # we use this if the model outputs a citation to buffer until completed
            output_delta_buffer = ""
            # we use this to track the current output text content for things like providing the right indices in citations
            current_output_text_content = ""
            current_annotations = []

            while True:
                # Check for client disconnect
                if self.request is not None and await self.request.is_disconnected():
                    print("Client disconnected, stopping token generation.")
                    break
                next_tok = infer_next_token(
                    self.tokens,
                    temperature=self.temperature,
                    new_request=self.new_request,
                )
                self.new_request = False
                self.tokens.append(next_tok)
                try:
                    self.parser.process(next_tok)
                except Exception:
                    pass

                if self.parser.state == StreamState.EXPECT_START:
                    current_output_index += 1
                    sent_output_item_added = False

                    if len(self.parser.messages) > 0:
                        previous_item = self.parser.messages[-1]
                        if previous_item.recipient is not None:
                            recipient = previous_item.recipient
                            if (
                                not recipient.startswith("browser.")
                                and not recipient == "python"
                            ):
                                fc_id = f"fc_{uuid.uuid4().hex}"
                                call_id = f"call_{uuid.uuid4().hex}"
                                self.function_call_ids.append((fc_id, call_id))
                                yield self._send_event(
                                    ResponseOutputItemDone(
                                        type="response.output_item.done",
                                        output_index=current_output_index,
                                        item=FunctionCallItem(
                                            type="function_call",
                                            name=(
                                                previous_item.recipient[
                                                    len("functions.") :
                                                ]
                                                if previous_item.recipient.startswith(
                                                    "functions."
                                                )
                                                else previous_item.recipient
                                            ),
                                            arguments=previous_item.content[0].text,
                                            id=fc_id,
                                            call_id=call_id,
                                        ),
                                    )
                                )
                        if previous_item.channel == "analysis":
                            yield self._send_event(
                                ResponseReasoningTextDone(
                                    type="response.reasoning_text.done",
                                    output_index=current_output_index,
                                    content_index=current_content_index,
                                    text=previous_item.content[0].text,
                                )
                            )
                            yield self._send_event(
                                ResponseContentPartDone(
                                    type="response.content_part.done",
                                    output_index=current_output_index,
                                    content_index=current_content_index,
                                    part=ReasoningTextContentItem(
                                        type="reasoning_text",
                                        text=previous_item.content[0].text,
                                    ),
                                )
                            )
                            yield self._send_event(
                                ResponseOutputItemDone(
                                    type="response.output_item.done",
                                    output_index=current_output_index,
                                    item=ReasoningItem(
                                        type="reasoning",
                                        summary=[],
                                        content=[
                                            ReasoningTextContentItem(
                                                type="reasoning_text",
                                                text=previous_item.content[0].text,
                                            )
                                        ],
                                    ),
                                )
                            )
                        if previous_item.channel == "final":
                            annotations = [
                                UrlCitation(**a) for a in current_annotations
                            ]
                            if browser_tool:
                                (
                                    normalized_text,
                                    _annotations,
                                    _has_partial_citations,
                                ) = browser_tool.normalize_citations(
                                    previous_item.content[0].text
                                )
                            else:
                                normalized_text = previous_item.content[0].text
                                annotations = []
                            text_content = TextContentItem(
                                type="output_text",
                                text=normalized_text,
                                annotations=annotations,
                            )
                            yield self._send_event(
                                ResponseOutputTextDone(
                                    type="response.output_text.done",
                                    output_index=current_output_index,
                                    content_index=current_content_index,
                                    text=normalized_text,
                                )
                            )
                            yield self._send_event(
                                ResponseContentPartDone(
                                    type="response.content_part.done",
                                    output_index=current_output_index,
                                    content_index=current_content_index,
                                    part=text_content,
                                )
                            )
                            yield self._send_event(
                                ResponseOutputItemDone(
                                    type="response.output_item.done",
                                    output_index=current_output_index,
                                    item=Item(
                                        type="message",
                                        role="assistant",
                                        content=[text_content],
                                    ),
                                )
                            )
                            current_annotations = []
                            current_output_text_content = ""

                if (
                    self.parser.last_content_delta
                    and self.parser.current_channel == "final"
                    and self.parser.current_recipient is None
                ):
                    if not sent_output_item_added:
                        sent_output_item_added = True
                        yield self._send_event(
                            ResponseOutputItemAdded(
                                type="response.output_item.added",
                                output_index=current_output_index,
                                item=Item(type="message", role="assistant", content=[]),
                            )
                        )
                        yield self._send_event(
                            ResponseContentPartAdded(
                                type="response.content_part.added",
                                output_index=current_output_index,
                                content_index=current_content_index,
                                part=TextContentItem(type="output_text", text=""),
                            )
                        )

                    output_delta_buffer += self.parser.last_content_delta
                    should_send_output_text_delta = True
                    if browser_tool:
                        # we normalize on the full current text to get the right indices in citations
                        updated_output_text, annotations, has_partial_citations = (
                            browser_tool.normalize_citations(
                                current_output_text_content + output_delta_buffer
                            )
                        )
                        # remove the current text to get back the delta but now normalized
                        output_delta_buffer = updated_output_text[
                            len(current_output_text_content) :
                        ]

                        # Filter annotations to only include those whose start_index is not already present in current_annotations
                        # this is to avoid sending duplicate annotations as multiple annotations can't be in the same place
                        existing_start_indices = {
                            a["start_index"] for a in current_annotations
                        }
                        new_annotations = [
                            a
                            for a in annotations
                            if a["start_index"] not in existing_start_indices
                        ]
                        for a in new_annotations:
                            current_annotations.append(a)
                            citation = UrlCitation(**a)
                            yield self._send_event(
                                ResponseOutputTextAnnotationAdded(
                                    type="response.output_text.annotation.added",
                                    output_index=current_output_index,
                                    content_index=current_content_index,
                                    annotation_index=len(current_annotations),
                                    annotation=citation,
                                )
                            )

                        if has_partial_citations:
                            should_send_output_text_delta = False

                    if should_send_output_text_delta:
                        yield self._send_event(
                            ResponseOutputTextDelta(
                                type="response.output_text.delta",
                                output_index=current_output_index,
                                content_index=current_content_index,
                                delta=output_delta_buffer,
                            )
                        )
                        current_output_text_content += output_delta_buffer
                        output_delta_buffer = ""

                if (
                    self.parser.last_content_delta
                    and self.parser.current_channel == "analysis"
                    and self.parser.current_recipient is None
                ):
                    if not sent_output_item_added:
                        sent_output_item_added = True
                        yield self._send_event(
                            ResponseOutputItemAdded(
                                type="response.output_item.added",
                                output_index=current_output_index,
                                item=ReasoningItem(
                                    type="reasoning", summary=[], content=[]
                                ),
                            )
                        )
                        yield self._send_event(
                            ResponseContentPartAdded(
                                type="response.content_part.added",
                                output_index=current_output_index,
                                content_index=current_content_index,
                                part=ReasoningTextContentItem(
                                    type="reasoning_text", text=""
                                ),
                            )
                        )
                    yield self._send_event(
                        ResponseReasoningTextDelta(
                            type="response.reasoning_text.delta",
                            output_index=current_output_index,
                            content_index=current_content_index,
                            delta=self.parser.last_content_delta,
                        )
                    )

                try:
                    # purely for debugging purposes
                    output_token_text = encoding.decode_utf8([next_tok])
                    self.output_text += output_token_text
                    print(output_token_text, end="", flush=True)

                except RuntimeError:
                    pass

                if next_tok in encoding.stop_tokens_for_assistant_actions():
                    if len(self.parser.messages) > 0:
                        last_message = self.parser.messages[-1]
                        if (
                            self.use_browser_tool
                            and last_message.recipient is not None
                            and last_message.recipient.startswith("browser.")
                        ):
                            function_name = last_message.recipient[len("browser.") :]
                            action = None
                            parsed_args = browser_tool.process_arguments(last_message)
                            if function_name == "search":
                                action = WebSearchActionSearch(
                                    type="search",
                                    query=parsed_args["query"],
                                )
                            elif function_name == "open":
                                action = WebSearchActionOpenPage(
                                    type="open_page",
                                    url=(
                                        parsed_args["url"]
                                        if "url" in parsed_args
                                        else None
                                    ),
                                )
                            elif function_name == "find":
                                action = WebSearchActionFind(
                                    type="find",
                                    pattern=parsed_args["pattern"],
                                    url=(
                                        parsed_args["url"]
                                        if "url" in parsed_args
                                        else None
                                    ),
                                )

                            if action is not None:
                                web_search_call_id = f"ws_{uuid.uuid4().hex}"
                                self.browser_call_ids.append(web_search_call_id)
                                yield self._send_event(
                                    ResponseOutputItemAdded(
                                        type="response.output_item.added",
                                        output_index=current_output_index,
                                        item=WebSearchCallItem(
                                            type="web_search_call",
                                            id=web_search_call_id,
                                            action=action,
                                        ),
                                    )
                                )
                                yield self._send_event(
                                    ResponseWebSearchCallInProgress(
                                        type="response.web_search_call.in_progress",
                                        output_index=current_output_index,
                                        id=web_search_call_id,
                                    )
                                )

                            async def run_tool():
                                results = []
                                async for msg in browser_tool.process(last_message):
                                    results.append(msg)
                                return results

                            yield self._send_event(
                                ResponseWebSearchCallSearching(
                                    type="response.web_search_call.searching",
                                    output_index=current_output_index,
                                    id=web_search_call_id,
                                )
                            )
                            result = await run_tool()

                            new_tokens = encoding.render_conversation_for_completion(
                                Conversation.from_messages(result), Role.ASSISTANT
                            )

                            print(encoding.decode_utf8(new_tokens))
                            self.output_tokens.append(next_tok)
                            self.tokens.append(
                                encoding.encode("<|end|>", allowed_special="all")[0]
                            )

                            for token in new_tokens:
                                self.parser.process(token)
                                self.output_tokens.append(token)
                                self.tokens.append(token)

                            yield self._send_event(
                                ResponseWebSearchCallCompleted(
                                    type="response.web_search_call.completed",
                                    output_index=current_output_index,
                                    id=web_search_call_id,
                                )
                            )
                            yield self._send_event(
                                ResponseOutputItemDone(
                                    type="response.output_item.done",
                                    output_index=current_output_index,
                                    item=WebSearchCallItem(
                                        type="web_search_call",
                                        id=web_search_call_id,
                                        action=action,
                                    ),
                                )
                            )

                            current_output_index += 1
                            self.new_request = True

                            continue

                        elif (
                            self.use_code_interpreter
                            and last_message.recipient is not None
                            and last_message.recipient.startswith("python")
                        ):
                            code_call_id = f"ci_{uuid.uuid4().hex}"
                            self.python_call_ids.append(code_call_id)
                            yield self._send_event(
                                ResponseOutputItemAdded(
                                    type="response.output_item.added",
                                    output_index=current_output_index,
                                    item=CodeInterpreterCallItem(
                                        type="code_interpreter_call",
                                        id=code_call_id,
                                    ),
                                )
                            )
                            yield self._send_event(
                                ResponseCodeInterpreterCallInProgress(
                                    type="response.code_interpreter_call.in_progress",
                                    output_index=current_output_index,
                                    id=code_call_id,
                                )
                            )

                            async def run_python_tool():
                                results = []
                                async for msg in self.python_tool.process(last_message):
                                    results.append(msg)
                                return results

                            result = await run_python_tool()

                            print(result)

                            new_tokens = encoding.render_conversation_for_completion(
                                Conversation.from_messages(result), Role.ASSISTANT
                            )

                            print(encoding.decode_utf8(new_tokens))
                            self.output_tokens.append(next_tok)
                            self.tokens.append(
                                encoding.encode("<|end|>", allowed_special="all")[0]
                            )

                            for token in new_tokens:
                                self.parser.process(token)
                                self.output_tokens.append(token)
                                self.tokens.append(token)

                            yield self._send_event(
                                ResponseCodeInterpreterCallCompleted(
                                    type="response.code_interpreter_call.completed",
                                    output_index=current_output_index,
                                    id=code_call_id,
                                )
                            )
                            yield self._send_event(
                                ResponseOutputItemDone(
                                    type="response.output_item.done",
                                    output_index=current_output_index,
                                    item=CodeInterpreterCallItem(
                                        type="code_interpreter_call",
                                        id=code_call_id,
                                    ),
                                )
                            )

                            current_output_index += 1
                            self.new_request = True

                            continue

                        else:
                            break
                    else:
                        raise ValueError("No messages to process")
                if len(self.output_tokens) >= self.request_body.max_output_tokens:
                    break

                # Adding in the end if we know we are not done
                self.output_tokens.append(next_tok)

            if self.request is None or not await self.request.is_disconnected():
                response = generate_response(
                    self.initial_tokens,
                    self.output_tokens,
                    self.request_body,
                    debug_mode=self.debug_mode,
                    function_call_ids=self.function_call_ids,
                    response_id=self.response_id,
                    previous_response_id=self.request_body.previous_response_id,
                    browser_tool=self.browser_tool,
                    browser_call_ids=self.browser_call_ids,
                )
                if self.store_callback and self.request_body.store:
                    self.store_callback(self.response_id, self.request_body, response)
                yield self._send_event(
                    ResponseCompletedEvent(
                        type="response.completed",
                        response=response,
                    )
                )

    @app.post("/v1/responses", response_model=ResponseObject)
    async def generate(body: ResponsesRequest, request: Request):
        event_stream, stream, _ = _prepare_event_stream(body, request)
        if stream:
            return StreamingResponse(event_stream.run(), media_type="text/event-stream")

        final_event: Optional[ResponseCompletedEvent] = None
        async for event in event_stream.run():
            if isinstance(event, ResponseCompletedEvent):
                final_event = event

        if final_event is None:
            raise HTTPException(status_code=500, detail="No response generated")

        return final_event.response

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": reported_model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "gpt-oss",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: Request) -> Any:
        payload = await request.json()
        responses_request, stream, include_usage = _convert_chat_request(payload)
        event_stream, _, response_id = _prepare_event_stream(
            responses_request,
            request,
            stream_override=stream,
            as_sse_override=False,
            store_response=False,
        )

        model_name = responses_request.model or reported_model_id

        if stream:
            async def chat_event_generator() -> AsyncGenerator[str, None]:
                async for chunk in _stream_chat_events(
                    event_stream,
                    response_id,
                    model_name,
                    include_usage=include_usage,
                ):
                    yield chunk

            return StreamingResponse(chat_event_generator(), media_type="text/event-stream")

        final_response: Optional[ResponseObject] = None
        async for event in event_stream.run():
            if isinstance(event, ResponseCompletedEvent):
                final_response = event.response

        if final_response is None:
            raise HTTPException(status_code=500, detail="No response generated")

        return _build_chat_completion_response(final_response, model_name)

    @app.post("/v1/completions")
    async def create_completion(request: Request) -> Any:
        payload = await request.json()
        responses_request, stream, include_usage = _convert_completion_request(payload)
        event_stream, _, response_id = _prepare_event_stream(
            responses_request,
            request,
            stream_override=stream,
            as_sse_override=False,
            store_response=False,
        )

        model_name = responses_request.model or reported_model_id

        if stream:
            async def completion_event_generator() -> AsyncGenerator[str, None]:
                async for chunk in _stream_completion_events(
                    event_stream,
                    response_id,
                    model_name,
                    include_usage=include_usage,
                ):
                    yield chunk

            return StreamingResponse(
                completion_event_generator(), media_type="text/event-stream"
            )

        final_response: Optional[ResponseObject] = None
        async for event in event_stream.run():
            if isinstance(event, ResponseCompletedEvent):
                final_response = event.response

        if final_response is None:
            raise HTTPException(status_code=500, detail="No response generated")

        return _build_completion_response(final_response, model_name)

    return app
