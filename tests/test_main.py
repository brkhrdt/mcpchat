import json
from unittest.mock import MagicMock

import pytest

from mcp_simple_chatbot.clients import LLMClient
from mcp_simple_chatbot.core import ChatSession, LLMResponse, Server, ToolCall


@pytest.mark.asyncio
async def test_valid_json_command_execution():
    # Create mock server and tool
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.input_schema = {"type": "object", "properties": {"arg1": {"type": "string"}}}
    mock_tool.format_for_llm.return_value = {
        "name": "test_tool",
        "description": "A test tool",
        "parameters": mock_tool.input_schema,
    }

    mock_server = MagicMock(spec=Server)
    mock_server.name = "test_server"
    mock_server.list_tools.return_value = [mock_tool]
    mock_server.execute_tool.return_value = MagicMock(
        model_dump=lambda: {"status": "success", "output": "tool executed"}
    )

    # Create mock LLM client
    mock_llm_client = MagicMock(spec=LLMClient)
    # The LLMClient's get_response method should return a string that
    # ChatSession._parse_llm_response can handle
    # For this test, we are directly testing process_llm_response, so the LLMClient mock's
    # return value isn't directly used here.
    # However, if we were testing the full chat session flow, get_response would return a string.
    # For the purpose of this test, we'll assume the parsing has already happened and we have an
    # LLMResponse object.

    # Create ChatSession instance
    chat_session = ChatSession(servers=[mock_server], llm_client=mock_llm_client)
    await chat_session.start()  # Initialize servers and tools

    # Simulate LLM response that requests a tool call
    tool_call_args = {"arg1": "value1"}
    llm_response_with_tool_call = LLMResponse(
        thinking="I need to use the test_tool.",
        tool_call=ToolCall(tool="test_tool", args=tool_call_args),
    )

    # Mock the follow-up LLM call after tool execution
    mock_llm_client.get_response.return_value = (
        "<|channel|>final<|message|>Tool executed successfully."
    )

    # Process the LLM response
    final_message = await chat_session.process_llm_response(llm_response_with_tool_call)

    # Assertions
    mock_server.execute_tool.assert_called_once_with("test_tool", tool_call_args)
    mock_llm_client.get_response.assert_called_once()
    assert final_message == "Tool executed successfully."


@pytest.mark.asyncio
async def test_invalid_json_response_printed():
    # Create mock server (not directly used in this test, but needed for ChatSession init)
    mock_server = MagicMock(spec=Server)
    mock_server.name = "test_server"
    mock_server.list_tools.return_value = []  # No tools for this test

    # Mock LLM response with invalid JSON
    # Create an LLMResponse object with a message, as this is what
    # _parse_llm_response would return for non-tool calls
    mock_llm_response_obj = LLMResponse(
        message="This is not valid JSON - just a regular text response"
    )

    # Create mock LLM client
    mock_llm_client = MagicMock(spec=LLMClient)
    # Again, LLMClient's get_response would return a string, which would then be parsed into an
    # LLMResponse object.
    # For this test, we directly provide the parsed LLMResponse object.

    # Create ChatSession instance
    chat_session = ChatSession(servers=[mock_server], llm_client=mock_llm_client)
    await chat_session.start()  # Initialize servers

    # Process the LLM response
    final_message = await chat_session.process_llm_response(mock_llm_response_obj)

    # Assertions
    assert final_message == "This is not valid JSON - just a regular text response"
    mock_llm_client.get_response.assert_not_called()  # No follow-up LLM call expected
