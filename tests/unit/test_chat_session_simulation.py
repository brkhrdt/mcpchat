from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_simple_chatbot.core.chat_session import (
    ChatSession,
    ToolResult,
    UserInput,
)


@pytest.fixture
async def mock_chat_session():
    # Mock servers
    mock_server = AsyncMock()
    mock_server.name = "test_server"
    mock_server.initialize = AsyncMock()

    # Mock tool
    mock_tool = MagicMock()
    mock_tool.name = "get_weather"
    mock_tool.format_for_llm.return_value = {
        "name": "get_weather",
        "description": "Gets the current weather",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
        },
    }
    mock_server.list_tools = AsyncMock(return_value=[mock_tool])
    mock_server.execute_tool = AsyncMock(
        return_value={"temperature": "15°C", "condition": "partly cloudy"}
    )
    mock_server.cleanup = AsyncMock()

    # Mock LLM client
    mock_llm_client = MagicMock()

    # Create chat session
    chat_session = ChatSession([mock_server], mock_llm_client)
    chat_session._monitor_user_input_enabled = False

    return chat_session, mock_llm_client, mock_server


async def test_simulated_conversation(mock_chat_session):
    chat_session, mock_llm_client, mock_server = mock_chat_session

    # Map of LLM raw responses to expected parsed message content
    llm_response_map = {
        # First response: tool call
        '<|channel|>analysis<|message|>User is asking for weather. I need to call the get_weather tool.<|channel|>commentary to=functions.get_weather json<|message|>{"location": "London"}': None,  # Tool calls don't add message content directly
        # Second response: final answer after tool execution
        "<|channel|>final<|message|>The weather in London is currently 15°C and partly cloudy.": "The weather in London is currently 15°C and partly cloudy.",
        # Third response: joke
        "<|channel|>final<|message|>Why don't scientists trust atoms? Because they make up everything!": "Why don't scientists trust atoms? Because they make up everything!",
    }

    # Configure LLM responses
    mock_llm_client.get_response.side_effect = list(llm_response_map.keys())

    # Initialize chat session
    await chat_session.start_initialization()

    # Expected final message history
    expected_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant",
        },  # Will check contains
        {"role": "user", "content": "What's the weather like in London?"},
        {
            "role": "system",
            "content": '{"temperature": "15°C", "condition": "partly cloudy"}',
        },
        {
            "role": "assistant",
            "content": "The weather in London is currently 15°C and partly cloudy.",
        },
        {"role": "user", "content": "Tell me a joke."},
        {
            "role": "assistant",
            "content": "Why don't scientists trust atoms? Because they make up everything!",
        },
    ]

    # Simulate conversation
    # Step 1: User asks about weather
    await chat_session.input_queue.put(UserInput("What's the weather like in London?"))
    await (
        chat_session._process_conversation_turn()
    )  # Processes user input, calls LLM, executes tool

    # Simulate tool result
    await chat_session.input_queue.put(
        ToolResult(
            "tool_1",
            {"temperature": "15°C", "condition": "partly cloudy"},
            "get_weather",
        )
    )
    await (
        chat_session._process_conversation_turn()
    )  # Processes tool result, calls LLM for final response

    # Step 2: User asks for joke
    await chat_session.input_queue.put(UserInput("Tell me a joke."))
    await chat_session._process_conversation_turn()  # Processes user input, calls LLM

    # Verify final message history
    assert len(chat_session.messages) == len(expected_messages)

    for i, expected_msg in enumerate(expected_messages):
        actual_msg = chat_session.messages[i]
        assert actual_msg["role"] == expected_msg["role"]

        if expected_msg["role"] == "system" and i == 0:
            # Initial system message contains tool schema
            assert "You are a helpful assistant" in actual_msg["content"]
        else:
            assert actual_msg["content"] == expected_msg["content"]

    # Verify LLM was called with correct responses and parsing worked
    assert mock_llm_client.get_response.call_count == 3

    # Verify tool was executed
    mock_server.execute_tool.assert_called_once_with(
        "get_weather", {"location": "London"}
    )
