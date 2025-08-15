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
    # Mock list_tools to return a mock tool for testing tool calls
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
    mock_server.cleanup = AsyncMock()

    # Mock LLM client
    mock_llm_client = MagicMock()

    # Create chat session
    chat_session = ChatSession([mock_server], mock_llm_client)

    # Disable the user input monitoring for tests
    chat_session._monitor_user_input_enabled = False

    return chat_session, mock_llm_client, mock_server


async def test_simulated_conversation(mock_chat_session):
    chat_session, mock_llm_client, mock_server = mock_chat_session

    # Define conversation script
    # Each step represents an input to the chat session and the expected LLM response(s)
    conversation_script = [
        {
            "input_type": "user",
            "input_content": "What's the weather like in London?",
            "llm_responses": [
                # LLM's first response: thinking and a tool call
                '<|channel|>analysis<|message|>User is asking for weather. I need to call the get_weather tool.<|channel|>commentary to=functions.get_weather json<|message|>{"location": "London"}',
                # LLM's second response: final message after tool execution
                "<|channel|>final<|message|>The weather in London is currently 15°C and partly cloudy.",
            ],
            "expected_tool_call": {
                "tool": "get_weather",
                "args": {"location": "London"},
            },
            "tool_result": {"temperature": "15°C", "condition": "partly cloudy"},
            "expected_messages_after_turn_1": [
                {"role": "system", "content": MagicMock()},  # Initial system message
                {"role": "user", "content": "What's the weather like in London?"},
                {
                    "role": "assistant",
                    "content": MagicMock(),
                },  # LLM's thinking/tool call (content will be parsed)
            ],
            "expected_messages_after_turn_2": [
                {"role": "system", "content": MagicMock()},  # Initial system message
                {"role": "user", "content": "What's the weather like in London?"},
                {
                    "role": "assistant",
                    "content": MagicMock(),
                },  # LLM's thinking/tool call
                {
                    "role": "system",
                    "content": '{"temperature": "15°C", "condition": "partly cloudy"}',
                },  # Tool result
                {
                    "role": "assistant",
                    "content": "The weather in London is currently 15°C and partly cloudy.",
                },
            ],
        },
        {
            "input_type": "user",
            "input_content": "Tell me a joke.",
            "llm_responses": [
                # LLM's response: direct message, no tool call
                "<|channel|>final<|message|>Why don't scientists trust atoms? Because they make up everything!"
            ],
            "expected_messages_after_turn_1": [
                {"role": "system", "content": MagicMock()},  # Initial system message
                {"role": "user", "content": "What's the weather like in London?"},
                {
                    "role": "assistant",
                    "content": MagicMock(),
                },  # LLM's thinking/tool call
                {
                    "role": "system",
                    "content": '{"temperature": "15°C", "condition": "partly cloudy"}',
                },  # Tool result
                {
                    "role": "assistant",
                    "content": "The weather in London is currently 15°C and partly cloudy.",
                },
                {"role": "user", "content": "Tell me a joke."},
                {
                    "role": "assistant",
                    "content": "Why don't scientists trust atoms? Because they make up everything!",
                },
            ],
        },
    ]

    # Configure LLM responses in sequence for the entire conversation
    all_llm_responses = []
    for step in conversation_script:
        all_llm_responses.extend(step["llm_responses"])
    mock_llm_client.get_response.side_effect = all_llm_responses

    # Configure tool execution
    # We'll set this up dynamically per step if needed, or use a side_effect list

    # Initialize chat session (discovers tools, sets up system message)
    await chat_session.start_initialization()

    # Verify initial system message is added
    assert len(chat_session.messages) == 1
    assert chat_session.messages[0]["role"] == "system"
    assert (
        "You are a helpful assistant with access to these tools"
        in chat_session.messages[0]["content"]
    )

    # Simulate conversation steps
    for i, step in enumerate(conversation_script):
        print(f"\n--- Simulating Step {i + 1}: {step['input_content']} ---")

        # 1. Simulate user input
        if step["input_type"] == "user":
            await chat_session.input_queue.put(UserInput(step["input_content"]))

            # Process the turn that handles user input and potentially calls LLM/tool
            should_continue = await chat_session._process_conversation_turn()
            assert should_continue is True

            # If a tool call was expected, verify it and simulate its result
            if "expected_tool_call" in step:
                mock_llm_client.get_response.assert_called_with(chat_session.messages)

                # Verify tool execution was attempted
                mock_server.execute_tool.assert_called_once_with(
                    step["expected_tool_call"]["tool"],
                    step["expected_tool_call"]["args"],
                )
                mock_server.execute_tool.reset_mock()  # Reset for next potential call

                # Simulate tool result being put back into the queue
                # This mimics the background task completing
                tool_id_pattern = (
                    r"tool_\d+"  # We don't know the exact ID, just the pattern
                )
                # Find the tool call in the messages to get its ID
                tool_call_message = next(
                    (
                        msg
                        for msg in chat_session.messages
                        if "commentary to=functions" in msg.get("content", "")
                    ),
                    None,
                )
                assert tool_call_message is not None, (
                    "Tool call message not found in history"
                )

                # Extract the tool ID from the assistant's response (e.g., "Started get_weather (ID: tool_1)")
                # This is a bit brittle, ideally the tool_id would be more directly accessible
                # For now, we'll assume the print_system_message format
                # A more robust way would be to capture the return value of _start_tool_execution

                # For this test, we'll just use a placeholder ID as we're directly putting to queue
                # In a real scenario, you'd await the task created by _start_tool_execution
                # and then put the result with the correct ID.
                # For now, we'll just use a dummy ID and ensure the tool result is processed.
                dummy_tool_id = "tool_1"  # This will be the first tool ID generated

                await chat_session.input_queue.put(
                    ToolResult(
                        dummy_tool_id,
                        step["tool_result"],
                        step["expected_tool_call"]["tool"],
                    )
                )

                # Process the turn that handles the tool result and gets final LLM response
                should_continue = await chat_session._process_conversation_turn()
                assert should_continue is True

            # Verify the final state of messages for this step
            # We need to be careful with the system message content as it contains dynamic tool schema
            # So we'll compare roles and user/assistant content, and check system content for keywords

            # For the first step, check against expected_messages_after_turn_2
            # For subsequent steps, check against expected_messages_after_turn_1 (which is cumulative)
            expected_messages = step.get(
                "expected_messages_after_turn_2"
                if "expected_tool_call" in step
                else "expected_messages_after_turn_1"
            )

            assert len(chat_session.messages) == len(expected_messages)
            for j, expected_msg in enumerate(expected_messages):
                actual_msg = chat_session.messages[j]
                assert actual_msg["role"] == expected_msg["role"]
                if expected_msg["role"] == "system":
                    # For system messages, just check if it contains the expected keywords
                    if j == 0:  # Initial system message
                        assert "You are a helpful assistant" in actual_msg["content"]
                    else:  # Tool result system message
                        assert json.loads(actual_msg["content"]) == json.loads(
                            expected_msg["content"]
                        )
                else:
                    assert actual_msg["content"] == expected_msg["content"]

        # Add more input types (e.g., tool_result) if your script needs to simulate them directly
        # For this example, tool_results are handled internally after a tool_call
