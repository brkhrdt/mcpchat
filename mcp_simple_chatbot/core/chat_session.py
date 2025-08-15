"""Chat session orchestration."""

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Optional

from mcp_simple_chatbot.utils import (
    print_assistant_response,
    print_error_message,
    print_system_message,
    print_tool_execution,
)

from .command_handler import CommandHandler
from .server import Server

if TYPE_CHECKING:
    from mcp_simple_chatbot.clients import (
        LLMClient,
    )  # Import for type hinting
    # from mcp_simple_chatbot.core.chat_session import LLMResponse # This is in the same
    # file, no need to import

logger = logging.getLogger("mcp_simple_chatbot.chat_session")


class ToolCall:
    def __init__(self, tool: str, args: dict):
        self.tool = tool
        self.args = args

    def __repr__(self):
        return f"ToolCall(tool='{self.tool}', args={self.args})"


class LLMResponse:
    def __init__(
        self,
        role: Optional[str] = None,  # Changed to Optional[str]
        thinking: Optional[str] = None,  # Changed to Optional[str]
        message: Optional[str] = None,  # Changed to Optional[str]
        tool_call: Optional[ToolCall] = None,  # Changed to Optional[ToolCall]
        commentary: Optional[str] = None,  # Changed to Optional[str]
    ):
        self.role = role
        self.thinking = thinking
        self.message = message
        self.tool_call = tool_call
        self.commentary = commentary

    def __repr__(self):
        return (
            f"LLMResponse(role='{self.role}', thinking='{self.thinking}', "
            f"message='{self.message}', tool_call={self.tool_call}, "
            f"commentary='{self.commentary}')"
        )


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(
        self, servers: list[Server], llm_client: "LLMClient", debug_mode: bool = False
    ) -> None:
        self.servers = {server.name: server for server in servers}
        self.llm_client = llm_client
        self.debug_mode = debug_mode
        self.messages: list[dict[str, str]] = []
        self.available_tools_schema: list[dict[str, Any]] = []
        self.command_handler = CommandHandler()

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        for server in self.servers.values():
            try:
                await server.cleanup()
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    def _parse_llm_response(self, llm_response: str) -> LLMResponse:
        """
        Parses the LLM's raw string response into a structured LLMResponse object.
        This method handles different message channels and extracts relevant
        information.
        """
        logger.debug(f"Parsing LLM response:\n{llm_response}")
        # Default role to assistant and commentary to the full raw response
        parsed_response = LLMResponse(role="assistant", commentary=llm_response)

        # Regex to capture thinking/analysis
        analysis_match = re.search(
            r"<\|channel\|>analysis<\|message\|>(.*?)(?=<\||$)",
            llm_response,
            re.DOTALL,
        )
        if analysis_match:
            parsed_response.thinking = analysis_match.group(1).strip()

        # Regex to capture tool call
        tool_call_match = re.search(
            r"<\|channel\|>commentary to=functions\.(.*?) ?json<\|message\|>"
            r"(.*?)(?=<\||$)",
            llm_response,
            re.DOTALL,
        )
        if tool_call_match:
            tool_name = tool_call_match.group(1).strip()
            try:
                tool_args = json.loads(tool_call_match.group(2).strip())
                parsed_response.tool_call = ToolCall(tool_name, tool_args)
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to parse tool arguments JSON: {tool_call_match.group(2)}"
                )
                # If tool args are malformed, treat it as a message error
                parsed_response.message = (
                    "Error: LLM provided malformed tool arguments. "
                    "Please try rephrasing your request."
                )
                parsed_response.tool_call = None  # Invalidate tool call if args are bad

        # Regex to capture final message
        final_message_match = re.search(
            r"<\|channel\|>final<\|message\|>(.*?)(?=<\||$)",
            llm_response,
            re.DOTALL,
        )
        if final_message_match:
            parsed_response.message = final_message_match.group(1).strip()

        logger.debug("Parsed LLM response: %s", parsed_response)
        return parsed_response

    async def _execute_tool_call(self, tool_call: ToolCall) -> Optional[Any]:
        """
        Execute a single tool call and return the result.

        Args:
            tool_call: The tool call to execute

        Returns:
            The tool execution result, or None if execution failed
        """
        tool = tool_call.tool
        arguments = tool_call.args

        logging.info(f"Executing tool: {tool}")
        logging.info(f"With arguments: {arguments}")

        # Find the server that has this tool
        found_tool_server = None
        for server in self.servers.values():
            available_tools = await server.list_tools()
            if tool in [t.name for t in available_tools]:
                found_tool_server = server
                break

        if not found_tool_server:
            error_msg = f"No server found with tool: {tool}"
            logger.warning(error_msg)
            print_error_message(error_msg)
            self.messages.append({"role": "system", "content": error_msg})
            return None

        try:
            result = await found_tool_server.execute_tool(tool, arguments)

            # Handle progress reporting if present
            if isinstance(result, dict) and "progress" in result:
                progress = result["progress"]
                total = result["total"]
                percentage = (progress / total) * 100
                logging.info(f"Progress: {progress}/{total} ({percentage:.1f}%)")

            print_tool_execution(tool, result)
            logger.info("Tool execution completed.")
            return result

        except Exception as e:
            error_msg = f"Error executing tool: {str(e)}"
            print_error_message(error_msg)
            logging.error(error_msg)
            self.messages.append({"role": "system", "content": error_msg})
            return None

    async def _process_conversation_turn(self) -> None:
        """
        Process a complete conversation turn, handling LLM responses and tool executions
        until a final message is provided or no further action is needed.
        """
        while True:
            # Get LLM response
            llm_response_raw = self.llm_client.get_response(self.messages)
            parsed = self._parse_llm_response(llm_response_raw)

            # Display the LLM's response (thinking, commentary, etc.)
            print_assistant_response(parsed)

            # If there's a tool call, execute it and continue the loop
            if parsed.tool_call:
                tool_result = await self._execute_tool_call(parsed.tool_call)
                if tool_result:
                    # Add tool result to messages and continue processing
                    self.messages.append(
                        {
                            "role": "system",
                            "content": json.dumps(tool_result.model_dump()),
                        }
                    )
                    continue
                else:
                    # Tool execution failed, break the loop
                    break

            # If there's a final message, add it to history and end the turn
            if parsed.message:
                self.messages.append({"role": "assistant", "content": parsed.message})
                break

            # If no tool call and no final message, log warning and end turn
            logger.warning("LLM provided no tool call or final message. Ending turn.")
            break

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            print_system_message("Initializing servers and discovering tools...")
            for server in self.servers.values():
                try:
                    logger.info(f"Initializing server: {server.name}")
                    await server.initialize()
                    logger.info(f"Server {server.name} initialized successfully.")
                    tools = await server.list_tools()
                    for tool in tools:
                        self.available_tools_schema.append(tool.format_for_llm())
                        logger.info(f"Discovered tool: {tool.name} from {server.name}")
                except Exception as e:
                    logging.error(f"Failed to initialize server {server.name}: {e}")
                    print_error_message(
                        f"Failed to initialize server {server.name}: {e}"
                    )
                    await self.cleanup_servers()
                    return

            if not self.available_tools_schema:
                print_system_message(
                    "No tools discovered. The chatbot will operate without tools."
                )
            else:
                print_system_message(
                    f"Discovered {len(self.available_tools_schema)} tools. "
                    "Type /help for more information."
                )

            system_message = (
                "You are a helpful assistant with access to these tools:\n\n"
                f"{json.dumps(self.available_tools_schema, indent=2)}\n\n"
                "Choose the appropriate tool based on the user's question. "
                "If no tool is needed, reply directly.\n\n"
                "After receiving a tool's response:\n"
                "1. Transform the raw data into a natural, conversational response\n"
                "2. Keep responses concise but informative\n"
                "3. Focus on the most relevant information\n"
                "4. Use appropriate context from the user's question\n"
                "5. Avoid simply repeating the raw data\n\n"
                "Please use only the tools that are explicitly defined above."
            )
            logger.debug("System message: %s", system_message)

            self.messages.append({"role": "system", "content": system_message})

            while True:
                try:
                    user_input = input("You: ")
                    logger.info("User input: %s", user_input)

                    if user_input.strip().lower() in ["quit", "exit"]:
                        print_system_message("👋 Goodbye!")
                        logger.info("Chat session ended by user.")
                        break

                    # Check if input is a command - handle immediately and continue
                    if self.command_handler.is_command(user_input):
                        logger.info(f"Executing command: {user_input}")
                        command_response = await self.command_handler.execute_command(
                            user_input
                        )
                        print_system_message(command_response)
                        logger.info(f"Command response: {command_response}")
                        continue

                    # Add user message to conversation history
                    self.messages.append({"role": "user", "content": user_input})

                    # Process the conversation until LLM provides a final response
                    await self._process_conversation_turn()

                except KeyboardInterrupt:
                    print_system_message("👋 Goodbye!")
                    logger.info("Chat session interrupted by user (KeyboardInterrupt).")
                    break
                except Exception as e:
                    logger.exception(
                        "An unexpected error occurred during chat session."
                    )
                    print_error_message(f"An unexpected error occurred: {e}")

        finally:
            logger.info("Cleaning up servers.")
            await self.cleanup_servers()
            logger.info("Server cleanup complete.")
