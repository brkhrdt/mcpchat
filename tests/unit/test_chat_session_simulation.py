import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_simple_chatbot.core.chat_session import (
    ChatSession,
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
        return_value={"temperature": "15C", "condition": "partly cloudy"}
    )
    mock_server.cleanup = AsyncMock()

    # Mock LLM client
    mock_llm_client = MagicMock()

    # Create chat session
    chat_session = ChatSession([mock_server], mock_llm_client)
    chat_session._monitor_user_input_enabled = (
        False  # Disable real user input for testing
    )

    return chat_session, mock_llm_client, mock_server


async def test_simulated_conversation(mock_chat_session):
    chat_session, mock_llm_client, mock_server = mock_chat_session

    # Configure LLM responses in order
    mock_llm_client.get_response.side_effect = [
        # First response: tool call for weather (triggered by user's weather question)
        '<|channel|>analysis<|message|>User is asking for weather. I need to call the get_weather tool.<|channel|>commentary to=functions.get_weather json<|message|>{"location": "London"}',
        # Second response: final answer after tool execution (triggered by tool result)
        "<|channel|>final<|message|>The weather in London is currently 15C and partly cloudy.",
        # Third response: joke (triggered by user's joke question)
        "<|channel|>final<|message|>Why don't scientists trust atoms? Because they make up everything!",
    ]

    # Start the chat session's main loop in a background task
    # This task will handle initialization and continuously pull from input_queue
    chat_session_task = asyncio.create_task(chat_session.start())

    # --- Simulate the conversation turn by turn, waiting for conditions ---

    # Wait for initial system message to be added after initialization
    # The chat_session.start() calls start_initialization, which adds the first system message.
    # We need to wait for this to happen before we can put user input.
    # We can check the length of chat_session.messages.
    await asyncio.wait_for(
        _wait_for_condition(lambda: len(chat_session.messages) >= 1),
        timeout=1,  # Add a timeout to prevent infinite waits in case of logic errors
    )
    assert chat_session.messages[0]["role"] == "system"
    assert "You are a helpful assistant" in chat_session.messages[0]["content"]

    # 1. User asks about weather
    await chat_session.input_queue.put(UserInput("What's the weather like in London?"))

    # Wait for the first LLM call (tool call) and the tool execution to start,
    # and for the tool result to be processed, leading to the second LLM call.
    # We expect mock_llm_client.get_response to be called twice by this point.
    await asyncio.wait_for(
        _wait_for_condition(lambda: mock_llm_client.get_response.call_count >= 2),
        timeout=1,
    )

    # Verify tool was executed once for the weather query
    mock_server.execute_tool.assert_called_once_with(
        "get_weather", {"location": "London"}
    )
    mock_server.execute_tool.reset_mock()  # Reset mock for potential future tool calls if any

    # 2. User asks for joke
    await chat_session.input_queue.put(UserInput("Tell me a joke."))

    # Wait for the third LLM call (the joke)
    await asyncio.wait_for(
        _wait_for_condition(lambda: mock_llm_client.get_response.call_count >= 3),
        timeout=1,
    )

    # 3. Add exit command to end the conversation
    await chat_session.input_queue.put(UserInput("quit"))

    # Wait for the chat session task to complete (it will exit after processing "quit")
    await chat_session_task

    # Verify the conversation happened as expected
    expected_messages = [
        # System message (contains tool schema)
        {"role": "system"},  # We'll just check role since content is complex
        {"role": "user", "content": "What's the weather like in London?"},
        {
            "role": "system",
            "content": '{"temperature": "15C", "condition": "partly cloudy"}',
        },
        {
            "role": "assistant",
            "content": "The weather in London is currently 15C and partly cloudy.",
        },
        {"role": "user", "content": "Tell me a joke."},
        {
            "role": "assistant",
            "content": "Why don't scientists trust atoms? Because they make up everything!",
        },
    ]

    # Verify final message history
    # from pprint import pprint # Keep for debugging if needed
    # pprint(chat_session.messages)
    # print('\n\n')
    # pprint(expected_messages)
    assert len(chat_session.messages) == len(expected_messages)

    for i, expected_msg in enumerate(expected_messages):
        actual_msg = chat_session.messages[i]
        assert actual_msg["role"] == expected_msg["role"]

        if expected_msg["role"] == "system" and i == 0:
            # Initial system message contains tool schema
            assert "You are a helpful assistant" in actual_msg["content"]
        elif "content" in expected_msg:
            assert actual_msg["content"] == expected_msg["content"]

    # Verify LLM was called 3 times
    assert mock_llm_client.get_response.call_count == 3

    # Verify tool was executed once (already asserted above, but good to have final check)
    mock_server.execute_tool.assert_called_once_with(
        "get_weather", {"location": "London"}
    )


# Helper function to wait for a condition
async def _wait_for_condition(condition_func, interval=0.01):
    """Waits for a condition_func to return True."""
    while not condition_func():
        await asyncio.sleep(interval)
