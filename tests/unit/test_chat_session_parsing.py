from unittest.mock import MagicMock

import pytest

from mcp_simple_chatbot.core.chat_session import ChatSession, ToolCall
from mcp_simple_chatbot.core.server import Server


@pytest.fixture
def chat_session_parser():
    # Mock LLMClient and Server as they are dependencies for ChatSession
    mock_llm_client = MagicMock()
    mock_server = MagicMock(spec=Server)
    mock_server.name = "mock_server"
    mock_server.list_tools.return_value = []  # No tools needed for parsing tests
    return ChatSession(servers=[mock_server], llm_client=mock_llm_client)


def test_parse_llm_response_greeting(chat_session_parser):
    """Test parsing an LLM response for a simple greeting."""
    llm_response_string = (
        '<|channel|>analysis<|message|>User says "hi". Likely just greeting. '
        "No tool needed.<|start|>assistant<|channel|>final<|message|>Hello! "
        "How can I help you today?"
    )
    parsed_response = chat_session_parser._parse_llm_response(llm_response_string)

    assert parsed_response.role == "assistant"
    assert (
        parsed_response.thinking
        == 'User says "hi". Likely just greeting. No tool needed.'
    )
    assert parsed_response.message == "Hello! How can I help you today?"
    assert parsed_response.tool_call is None
    assert parsed_response.commentary == llm_response_string


def test_parse_llm_response_tool_call(chat_session_parser):
    """Test parsing an LLM response that includes a tool call."""
    llm_response_string = (
        "<|channel|>analysis<|message|>Need to list directory /projects. "
        "Use list_directory tool."
        "<|start|>assistant<|channel|>commentary to=functions.list_directory json"
        "<|message|>"
        '{"path":"/projects"}'
    )
    parsed_response = chat_session_parser._parse_llm_response(llm_response_string)

    assert parsed_response.role == "assistant"
    assert parsed_response.thinking == (
        "Need to list directory /projects. Use list_directory tool."
    )
    assert parsed_response.message is None
    assert isinstance(parsed_response.tool_call, ToolCall)
    assert parsed_response.tool_call.tool == "list_directory"
    assert parsed_response.tool_call.args == {"path": "/projects"}
    assert parsed_response.commentary == llm_response_string


def test_parse_llm_response_only_message(chat_session_parser):
    """Test parsing an LLM response that only contains a final message."""
    llm_response_string = "<|channel|>final<|message|>Here is your answer."
    parsed_response = chat_session_parser._parse_llm_response(llm_response_string)

    assert parsed_response.role == "assistant"
    assert parsed_response.thinking is None
    assert parsed_response.message == "Here is your answer."
    assert parsed_response.tool_call is None
    assert parsed_response.commentary == llm_response_string


def test_parse_llm_response_only_thinking(chat_session_parser):
    """Test parsing an LLM response that only contains thinking/analysis."""
    llm_response_string = (
        "<|channel|>analysis<|message|>I am thinking about the next step."
    )
    parsed_response = chat_session_parser._parse_llm_response(llm_response_string)

    assert parsed_response.role == "assistant"
    assert parsed_response.thinking == "I am thinking about the next step."
    assert parsed_response.message is None
    assert parsed_response.tool_call is None
    assert parsed_response.commentary == llm_response_string


def test_parse_llm_response_full_cycle_with_tool_and_final_message(chat_session_parser):
    """
    Test a more complex scenario with thinking, tool call, and a subsequent final
    message.
    """
    llm_response_string = (
        "<|channel|>analysis<|message|>User wants to know the weather. "
        "I will use the get_weather tool."
        "<|start|>assistant<|channel|>commentary to=functions.get_weather json"
        "<|message|>"
        '{"location":"London"}'
        "<|channel|>final<|message|>The weather in London is 15 degrees Celsius and "
        "partly cloudy."
    )
    parsed_response = chat_session_parser._parse_llm_response(llm_response_string)

    assert parsed_response.role == "assistant"
    assert parsed_response.thinking == (
        "User wants to know the weather. I will use the get_weather tool."
    )
    assert isinstance(parsed_response.tool_call, ToolCall)
    assert parsed_response.tool_call.tool == "get_weather"
    assert parsed_response.tool_call.args == {"location": "London"}
    assert parsed_response.message == (
        "The weather in London is 15 degrees Celsius and partly cloudy."
    )
    assert parsed_response.commentary == llm_response_string


def test_parse_llm_response_only_commentary_fallback(chat_session_parser):
    """
    Test parsing a response that doesn't fit other patterns, falling back to
    commentary.
    """
    llm_response_string = (
        "This is just some raw commentary from the LLM without specific channels."
    )
    parsed_response = chat_session_parser._parse_llm_response(llm_response_string)

    assert parsed_response.role == "assistant"
    assert parsed_response.thinking is None
    assert parsed_response.message is None
    assert parsed_response.tool_call is None
    assert parsed_response.commentary == llm_response_string


def test_parse_llm_response_empty_string(chat_session_parser):
    """Test parsing an empty string response."""
    llm_response_string = ""
    parsed_response = chat_session_parser._parse_llm_response(llm_response_string)

    assert parsed_response.role == "assistant"
    assert parsed_response.thinking is None
    assert parsed_response.message is None
    assert parsed_response.tool_call is None
    assert parsed_response.commentary == ""


def test_parse_llm_response_tool_call_no_thinking(chat_session_parser):
    """Test parsing a tool call without preceding thinking."""
    llm_response_string = (
        "<|start|>assistant<|channel|>commentary to=functions.search json<|message|>"
        '{"query":"latest news"}'
    )
    parsed_response = chat_session_parser._parse_llm_response(llm_response_string)

    assert parsed_response.role == "assistant"
    assert parsed_response.thinking is None
    assert parsed_response.message is None
    assert isinstance(parsed_response.tool_call, ToolCall)
    assert parsed_response.tool_call.tool == "search"
    assert parsed_response.tool_call.args == {"query": "latest news"}
    assert parsed_response.commentary == llm_response_string


def test_parse_llm_response_malformed_tool_json(chat_session_parser):
    """Test parsing a tool call with malformed JSON arguments."""
    llm_response_string = (
        "<|channel|>analysis<|message|>I will try to use a tool."
        "<|start|>assistant<|channel|>commentary to=functions.bad_tool json<|message|>"
        '{"arg1":"value1", "arg2":}'  # Malformed JSON
    )
    parsed_response = chat_session_parser._parse_llm_response(llm_response_string)

    assert parsed_response.role == "assistant"
    assert parsed_response.thinking == "I will try to use a tool."
    assert parsed_response.tool_call is None  # Tool call should be invalidated
    assert "Error: LLM provided malformed tool arguments." in parsed_response.message
    assert parsed_response.commentary == llm_response_string
