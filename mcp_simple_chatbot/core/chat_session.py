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

    async def process_llm_response(self, parsed_response: LLMResponse) -> str:
        """
        Process the LLM response and execute tools if needed.

        Args:
            parsed_response: The parsed response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        logger.info("Processing LLM response...")

        # Always print thinking if present
        if parsed_response.thinking:
            print_assistant_response(parsed_response, thinking_only=True)

        if parsed_response.tool_call:
            tool = parsed_response.tool_call.tool
            arguments = parsed_response.tool_call.args

            logging.info(f"Executing tool: {tool}")
            logging.info(f"With arguments: {arguments}")

            found_tool_server = None
            for server in self.servers.values():
                available_tools = await server.list_tools()
                if tool in [t.name for t in available_tools]:
                    found_tool_server = server
                    break

            if found_tool_server:
                try:
                    result = await found_tool_server.execute_tool(tool, arguments)

                    if isinstance(result, dict) and "progress" in result:
                        progress = result["progress"]
                        total = result["total"]
                        percentage = (progress / total) * 100
                        logging.info(
                            f"Progress: {progress}/{total} ({percentage:.1f}%)"
                        )

                    print_tool_execution(tool, result)
                    logger.info("Tool execution completed.")

                    # Append tool output to messages for LLM to process
                    self.messages.append(
                        {
                            "role": "tool",
                            "content": json.dumps(result.model_dump()),
                            "name": tool,
                        }
                    )

                    # Call LLM again to interpret the tool result
                    logger.info("Calling LLM to interpret tool result...")
                    llm_follow_up_response_raw = self.llm_client.get_response(
                        self.messages
                    )
                    parsed_follow_up_response = self._parse_llm_response(
                        llm_follow_up_response_raw
                    )
                    # Recursively process the follow-up response
                    return await self.process_llm_response(parsed_follow_up_response)

                except Exception as e:
                    error_msg = f"Error executing tool: {str(e)}"
                    print_error_message(error_msg)
                    logging.error(error_msg)
                    # Append error to messages for LLM to potentially handle
                    self.messages.append({"role": "system", "content": error_msg})
                    return error_msg
            else:
                logger.warning(f"No server found with tool: {tool}")
                error_msg = f"No server found with tool: {tool}"
                print_error_message(error_msg)
                self.messages.append({"role": "system", "content": error_msg})
                return error_msg
        else:
            # If there's a final message, print it
            if parsed_response.message:
                print_assistant_response(parsed_response)
                return parsed_response.message
            # If no tool call and no final message, but there was commentary
            # (e.g., malformed response)
            elif parsed_response.commentary:
                print_assistant_response(parsed_response)
                return parsed_response.commentary
            else:
                logger.info("No tool call or final message detected in LLM response.")
                return ""

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
                    user_input = input(
                        "You: "
                    )  # Using input for now, will switch to get_user_input
                    logger.info("User input: %s", user_input)
                    if user_input.strip().lower() in ["quit", "exit"]:
                        print_system_message("👋 Goodbye!")
                        logger.info("Chat session ended by user.")
                        break

                    # Check if input is a command
                    if self.command_handler.is_command(user_input):
                        logger.info(f"Executing command: {user_input}")
                        command_response = await self.command_handler.execute_command(
                            user_input
                        )
                        print_system_message(command_response)
                        logger.info(f"Command response: {command_response}")
                        continue

                    self.messages.append({"role": "user", "content": user_input})

                    llm_response_raw = self.llm_client.get_response(self.messages)

                    parsed = self._parse_llm_response(llm_response_raw)

                    final_response_content = await self.process_llm_response(parsed)

                    # Append the assistant's final message to history if it's not a
                    # tool call. Tool calls and their results are handled within
                    # process_llm_response.
                    if not parsed.tool_call and final_response_content:
                        self.messages.append(
                            {"role": "assistant", "content": final_response_content}
                        )

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
