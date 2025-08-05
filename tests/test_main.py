import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from mcp_simple_chatbot.main import ChatSession, LLMClient, Server, Tool


@pytest.mark.asyncio
async def test_valid_json_command_execution():
    """Test that a valid JSON command is correctly parsed and executed on the MCP server."""
    
    # Mock LLM response with valid JSON tool call
    mock_llm_response = json.dumps({
        "tool": "test_tool",
        "arguments": {"param1": "value1", "param2": "value2"}
    })
    
    # Mock tool execution result
    mock_tool_result = {"status": "success", "data": "test result"}
    
    # Create mock LLM client
    mock_llm_client = MagicMock(spec=LLMClient)
    mock_llm_client.get_response.return_value = mock_llm_response
    
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
                "param2": {"description": "Parameter 2"}
            },
            "required": ["param1"]
        }
    )
    
    # Configure server mocks
    mock_server.initialize = AsyncMock()
    mock_server.list_tools = AsyncMock(return_value=[mock_tool])
    mock_server.execute_tool = AsyncMock(return_value=mock_tool_result)
    mock_server.cleanup = AsyncMock()
    
    # Create chat session
    chat_session = ChatSession([mock_server], mock_llm_client)
    
    # Test the process_llm_response method directly
    result = await chat_session.process_llm_response(mock_llm_response)
    
    # Verify tool was executed with correct arguments
    mock_server.execute_tool.assert_called_once_with(
        "test_tool", 
        {"param1": "value1", "param2": "value2"}
    )
    
    # Verify result contains tool execution output
    assert "Tool execution result:" in result
    assert str(mock_tool_result) in result
    
    # Clean up
    await chat_session.cleanup_servers()
