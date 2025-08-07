"""Main entry point for MCP Simple Chatbot."""

import asyncio
import logging

from mcp_simple_chatbot.clients.llm_client import LLMClient
from mcp_simple_chatbot.config.configuration import Configuration
from mcp_simple_chatbot.core.chat_session import ChatSession
from mcp_simple_chatbot.core.server import Server
from mcp_simple_chatbot.utils.console import print_system_message
from mcp_simple_chatbot.utils.logging_config import setup_logging

# Set up logging
setup_logging()

logger = logging.getLogger("mcp_simple_chatbot.main")


async def main() -> None:
    """Initialize and run the chat session."""
    print_system_message("🤖 Starting MCP Simple Chatbot...")
    
    config = Configuration()
    server_config = config.load_config("servers_config.json")
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]
    logger.info(str(config.llm_url_base))
    llm_client = LLMClient(config.llm_api_key, config.llm_url_base)
    chat_session = ChatSession(servers, llm_client)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())
