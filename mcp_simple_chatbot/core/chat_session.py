"""Chat session orchestration."""

import json
import logging
import re
from typing import TYPE_CHECKING, Optional # Import Optional

from ..utils.console import (
    get_user_input,
    print_assistant_response, # MODIFIED: Changed import
    print_error_message,
    print_system_message,
    print_tool_execution,
)
from .command_handler import CommandHandler
from .server import Server
# from typing import TYPE_CHECKING # For type hinting LLMResponse - already imported above

if TYPE_CHECKING:
    from mcp_simple_chatbot.clients.llm_client import LLMClient # Import for type hinting
    # from mcp_simple_chatbot.core.chat_session import LLMResponse # This is in the same file, no need to import

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
        role: Optional[str] = None, # Changed to Optional[str]
        thinking: Optional[str] = None, # Changed to Optional[str]
        message: Optional[str] = None, # Changed to Optional[str]
        tool_call: Optional[ToolCall] = None, # Changed to Optional[ToolCall]
        commentary: Optional[str] = None, # Changed to Optional[str]
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

    def __init__(self, servers: list[Server], llm_client: "LLMClient") -> None: # Added type hint
        self.servers: list[Server] = servers
        self.llm_client = llm_client
        self.command_handler = CommandHandler()

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        for server in reversed(self.servers):
            try:
                await server.cleanup()
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    def _parse_llm_response(self, llm_response: str) -> LLMResponse:
        logger.debug("Raw LLM response: %s", llm_response)
        parsed_response = LLMResponse()

        # Regex to capture thinking/analysis
        thinking_match = re.search(
            r"<\|channel\|>analysis<\|message\|>(.*?)(?=<\||$)", llm_response, re.DOTALL
        )
        if thinking_match:
            parsed_response.thinking = thinking_match.group(1).strip()

        # Regex to capture thinking/analysis
        message_match = re.search(
            r"<\|channel\|>final<\|message\|>(.*?)(?=<\||$)", llm_response, re.DOTALL
        )
        if message_match:
            parsed_response.message = message_match.group(1).strip()

        # Regex to capture role
        role_match = re.search(r"<\|start\|>(.*?)(?=<\||$)", llm_response)
        if role_match:
            parsed_response.role = role_match.group(1).strip()

        # Regex to capture tool call
        tool_call_match = re.search(
            r"<\|channel\|>commentary to=functions\.(.*?) ?json<\|message\|>(.*?)(?=<\||$)",
            llm_response,
            re.DOTALL,
        )
        if tool_call_match:
            tool_name = tool_call_match.group(1).strip()
            tool_args_str = tool_call_match.group(2).strip()
            try:
                parsed_response.tool_call = ToolCall(tool_name, json.loads(tool_args_str))
            except json.JSONDecodeError:
                logger.error(f"Failed to parse tool arguments JSON: {tool_args_str}")
                # Fallback to commentary if tool args are malformed
                parsed_response.commentary = llm_response.strip()


        if (
            not parsed_response.thinking
            and not parsed_response.tool_call
            and not parsed_response.message
        ):
            # Fallback for commentary if no specific message or thinking was found
            parsed_response.commentary = llm_response.strip()

        logger.debug("Parsed LLM response: %s", parsed_response)
        return parsed_response

    async def process_llm_response(self, parsed_response: LLMResponse) -> str:
        """Process the LLM response and execute tools if needed.

        Args:
            parsed_response: The parsed response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        logger.info("Processing LLM response...")
        
        # MODIFIED: Call the new print function
        print_assistant_response(parsed_response)

        if parsed_response.tool_call:
            tool = parsed_response.tool_call.tool
            arguments = parsed_response.tool_call.args
            
            logging.info(f"Executing tool: {tool}")
            logging.info(f"With arguments: {arguments}")

            logging.debug(f"Available servers: {self.servers}")
            for server in self.servers:
                available_tools = await server.list_tools()
                logging.debug(f"Available tools: {available_tools}")
                if tool in [t.name for t in available_tools]:
                    try:
                        result = await server.execute_tool(tool, arguments)

                        if isinstance(result, dict) and "progress" in result:
                            progress = result["progress"]
                            total = result["total"]
                            percentage = (progress / total) * 100
                            logging.info(
                                f"Progress: {progress}/{total} ({percentage:.1f}%)"
                            )

                        print_tool_execution(tool, result)
                        logger.info("Tool execution completed.")
                        return f"Tool execution result: {result}"
                    except Exception as e:
                        error_msg = f"Error executing tool: {str(e)}"
                        print_error_message(error_msg)
                        logging.error(error_msg)
                        return error_msg

            logger.warning(f"No server found with tool: {tool}")
            return f"No server found with tool: {tool}"
        else:
            # No tool call, so the message/thinking/commentary was already printed by print_assistant_response
            pass 
        logger.info("No tool call detected in LLM response.")
        return parsed_response.message if parsed_response.message else ""

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    logger.info(f"Initializing server: {server.name}")
                    await server.initialize()
                    logger.info(f"Server {server.name} initialized successfully.")
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)

            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
            logger.debug("Tools description for LLM: \n%s", tools_description)

            system_message = (
                "You are a helpful assistant with access to these tools:\n\n"
                f"{tools_description}\n"
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

            messages = [{"role": "system", "content": system_message}]

            while True:
                try:
                    # TODO user input async, command output async
                    user_input = get_user_input()
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

                    messages.append({"role": "user", "content": user_input})

                    llm_response_raw = self.llm_client.get_response(messages)
                    # Debug print for raw LLM response is now in _parse_llm_response

                    parsed = self._parse_llm_response(llm_response_raw)
                    # Debug print for parsed LLM response is now in _parse_llm_response

                    result = await self.process_llm_response(parsed)

                    if parsed.tool_call:
                        logger.info(
                            "Tool call detected. Appending tool response to messages."
                        )
                        messages.append(
                            {"role": "assistant", "content": str(parsed.tool_call)}
                        )
                        messages.append({"role": "system", "content": result})
                        logger.debug(
                            "Messages after tool execution: %s",
                            json.dumps(messages, indent=2),
                        )

                        final_response_raw = self.llm_client.get_response(messages)
                        logger.info(
                            "\nFinal response from LLM after tool execution: %s",
                            final_response_raw,
                        )
                        parsed_final_response = self._parse_llm_response(final_response_raw)

                        # MODIFIED: Call the new print function
                        print_assistant_response(parsed_final_response)

                        messages.append(
                            {
                                "role": "assistant",
                                "content": parsed_final_response.message,
                            }  # do not add thinking into message history
                        )
                    else:
                        logger.info("No tool call. Appending LLM response to messages.")
                        messages.append(
                            {
                                "role": "assistant",
                                "content": parsed.message,
                            }  # do not add thinking into message history
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
