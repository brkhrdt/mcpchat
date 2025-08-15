from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import CallToolResult, TextContent

from mcp_simple_chatbot import ChatSession, LLMClient, Server, Tool
from mcp_simple_chatbot.core.chat_session import LLMResponse, ToolCall


@pytest.mark.asyncio
async def test_valid_json_command_execution():
    """Test that a valid JSON command is correctly parsed and executed on the MCP
    server."""

    # Mock LLM response with valid JSON tool call
    # Create an LLMResponse object instead of a raw JSON string
    mock_llm_response_obj = LLMResponse(
        tool_call=ToolCall(
            tool="test_tool", args={"param1": "value1", "param2": "value2"}
        )
    )

    # Mock tool execution result
    mock_tool_result = CallToolResult(
        content=[TextContent(type="text", text="Tool executed successfully")],
        isError=False,
    )

    # Create mock LLM client
    mock_llm_client = MagicMock(spec=LLMClient)
    # The LLMClient's get_response method should return a string that ChatSession._parse_llm_response can handle
    # For this test, we are directly testing process_llm_response, so the LLMClient mock's return value isn't directly used here.
    # However, if we were testing the full chat session flow, get_response would return a string.
    # For the purpose of this test, we'll assume the parsing has already happened and we have an LLMResponse object.

    # Create mock server with tool
    mock_server = MagicMock(spec=Server)
    mock_server.name = "test_server"

    # Mock tool
    mock_tool = Tool(
        name="test_tool",
        description="Test tool description",
        input_schema={
            "properties": {
                "param1": {"description": "Parameter 1"},
                "param2": {"description": "Parameter 2"},
            },
            "required": ["param1"],
        },
    )

    # Configure server mocks
    mock_server.initialize = AsyncMock()
    mock_server.list_tools = AsyncMock(return_value=[mock_tool])
    mock_server.execute_tool = AsyncMock(return_value=mock_tool_result)
    mock_server.cleanup = AsyncMock()

    # Create chat session
    chat_session = ChatSession([mock_server], mock_llm_client)

    # Test the process_llm_response method directly
    # Pass the LLMResponse object
    result = await chat_session.process_llm_response(mock_llm_response_obj)

    # Verify tool was executed with correct arguments
    mock_server.execute_tool.assert_called_once_with(
        "test_tool", {"param1": "value1", "param2": "value2"}
    )

    # Verify result contains tool execution output
    assert "Tool execution result:" in result
    assert "Tool executed successfully" in result

    # Clean up
    await chat_session.cleanup_servers()


@pytest.mark.asyncio
async def test_invalid_json_response_printed():
    """Test that invalid JSON responses don't trigger tool execution."""

    # Mock LLM response with invalid JSON
    # Create an LLMResponse object with a message, as this is what _parse_llm_response would return for non-tool calls
    mock_llm_response_obj = LLMResponse(
        message="This is not valid JSON - just a regular text response"
    )

    # Create mock LLM client
    mock_llm_client = MagicMock(spec=LLMClient)
    # Again, LLMClient's get_response would return a string, which would then be parsed into an LLMResponse object.
    # For this test, we directly provide the parsed LLMResponse object.

    # Create mock server
    mock_server = MagicMock(spec=Server)
    mock_server.name = "test_server"
    mock_server.initialize = AsyncMock()
    mock_server.list_tools = AsyncMock(return_value=[])
    mock_server.execute_tool = AsyncMock()
    mock_server.cleanup = AsyncMock()

    # Create chat session
    chat_session = ChatSession([mock_server], mock_llm_client)

    # Process the invalid JSON response
    # Pass the LLMResponse object
    result = await chat_session.process_llm_response(mock_llm_response_obj)

    # Verify that execute_tool was never called (no valid JSON to execute)
    mock_server.execute_tool.assert_not_called()

    # Verify the result is the original message from the LLMResponse object
    assert result == "This is not valid JSON - just a regular text response"

    # Clean up
    await chat_session.cleanup_servers()
