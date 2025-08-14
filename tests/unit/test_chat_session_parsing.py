import json
import pytest

from mcp_simple_chatbot.core.chat_session import ChatSession, ToolCall, LLMResponse


@pytest.fixture
def chat_session_parser():
    """Fixture to provide a ChatSession instance for testing parsing methods."""
    # We only need a dummy ChatSession instance to call the _parse_llm_response method.
    # Its dependencies (servers, llm_client) are not relevant for this specific test.
    return ChatSession(servers=[], llm_client=None)


def test_parse_llm_response_greeting(chat_session_parser):
    """Test parsing an LLM response for a simple greeting."""
    llm_response_string = (
        "<|channel|>analysis<|message|>User says \"hi\". Likely just greeting. No tool needed."
        "<|start|>assistant<|channel|>final<|message|>Hello! How can I help you today?"
    )
    parsed_response = chat_session_parser._parse_llm_response(llm_response_string)

    assert parsed_response.thinking == 'User says "hi". Likely just greeting. No tool needed.'
    assert parsed_response.role == "assistant"
    assert parsed_response.message == "Hello! How can I help you today?"
    assert parsed_response.tool_call is None
    assert parsed_response.commentary is None


def test_parse_llm_response_tool_call(chat_session_parser):
    """Test parsing an LLM response that includes a tool call."""
    llm_response_string = (
        "<|channel|>analysis<|message|>Need to list directory /projects. Use list_directory tool."
        "<|start|>assistant<|channel|>commentary to=functions.list_directory json<|message|>{\"path\":\"/projects\"}"
    )
    parsed_response = chat_session_parser._parse_llm_response(llm_response_string)

    assert parsed_response.thinking == "Need to list directory /projects. Use list_directory tool."
    assert parsed_response.role == "assistant"
    assert parsed_response.message is None
    assert isinstance(parsed_response.tool_call, ToolCall)
    assert parsed_response.tool_call.tool == "list_directory"
    assert parsed_response.tool_call.args == {"path": "/projects"}
    assert parsed_response.commentary is None


def test_parse_llm_response_only_message(chat_session_parser):
    """Test parsing an LLM response with only a final message."""
    llm_response_string = "<|channel|>final<|message|>This is a direct message from the assistant."
    parsed_response = chat_session_parser._parse_llm_response(llm_response_string)

    assert parsed_response.thinking is None
    assert parsed_response.role is None # Role is only set if <|start|> is present
    assert parsed_response.message == "This is a direct message from the assistant."
    assert parsed_response.tool_call is None
    assert parsed_response.commentary is None


def test_parse_llm_response_only_thinking(chat_session_parser):
    """Test parsing an LLM response with only thinking."""
    llm_response_string = "<|channel|>analysis<|message|>Just thinking out loud."
    parsed_response = chat_session_parser._parse_llm_response(llm_response_string)

    assert parsed_response.thinking == "Just thinking out loud."
    assert parsed_response.role is None
    assert parsed_response.message is None
    assert parsed_response.tool_call is None
    assert parsed_response.commentary is None


def test_parse_llm_response_full_cycle_with_tool_and_final_message(chat_session_parser):
    """Test a more complex scenario with thinking, tool call, and a subsequent final message."""
    llm_response_string = (
        "<|channel|>analysis<|message|>User wants to know the weather. I will use the get_weather tool."
        "<|start|>assistant<|channel|>commentary to=functions.get_weather json<|message|>{\"location\":\"London\"}"
        "<|channel|>final<|message|>The weather in London is 15 degrees Celsius and partly cloudy."
    )
    parsed_response = chat_session_parser._parse_llm_response(llm_response_string)

    assert parsed_response.thinking == "User wants to know the weather. I will use the get_weather tool."
    assert parsed_response.role == "assistant"
    # Note: The current parsing logic prioritizes tool_call if present, and then message.
    # If both are in the string, and the tool call regex matches first, the message might be missed
    # unless the regexes are ordered or designed to capture all parts independently.
    # Based on the current implementation, if a tool call is found, the 'final' message might not be captured
    # as the primary 'message' field, but rather the tool call takes precedence.
    # Let's adjust the expectation based on the current code's behavior.
    assert parsed_response.message == "The weather in London is 15 degrees Celsius and partly cloudy."
    assert isinstance(parsed_response.tool_call, ToolCall)
    assert parsed_response.tool_call.tool == "get_weather"
    assert parsed_response.tool_call.args == {"location": "London"}
    assert parsed_response.commentary is None


def test_parse_llm_response_only_commentary_fallback(chat_session_parser):
    """Test parsing a response that doesn't fit other patterns, falling back to commentary."""
    llm_response_string = "This is just some raw commentary from the LLM without specific channels."
    parsed_response = chat_session_parser._parse_llm_response(llm_response_string)

    assert parsed_response.thinking is None
    assert parsed_response.role is None
    assert parsed_response.message is None
    assert parsed_response.tool_call is None
    assert parsed_response.commentary == "This is just some raw commentary from the LLM without specific channels."


def test_parse_llm_response_empty_string(chat_session_parser):
    """Test parsing an empty string."""
    llm_response_string = ""
    parsed_response = chat_session_parser._parse_llm_response(llm_response_string)

    assert parsed_response.thinking is None
    assert parsed_response.role is None
    assert parsed_response.message is None
    assert parsed_response.tool_call is None
    assert parsed_response.commentary == "" # Empty string is still commentary if nothing else matches


def test_parse_llm_response_tool_call_no_thinking(chat_session_parser):
    """Test parsing a tool call without preceding thinking."""
    llm_response_string = (
        "<|start|>assistant<|channel|>commentary to=functions.search json<|message|>{\"query\":\"latest news\"}"
    )
    parsed_response = chat_session_parser._parse_llm_response(llm_response_string)

    assert parsed_response.thinking is None
    assert parsed_response.role == "assistant"
    assert parsed_response.message is None
    assert isinstance(parsed_response.tool_call, ToolCall)
    assert parsed_response.tool_call.tool == "search"
    assert parsed_response.tool_call.args == {"query": "latest news"}
    assert parsed_response.commentary is None

