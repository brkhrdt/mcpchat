"""Chat session orchestration."""

import json
import logging

from ..utils.console import (
    get_user_input,
    print_assistant_message,
    print_error_message,
    print_system_message,
    print_tool_execution,
)
from .command_handler import CommandHandler
from .server import Server

logger = logging.getLogger("mcp_simple_chatbot.chat_session")


class ToolCall:
    # tool = str
    # args = dict
    pass


class LLMResponse:
    # role: str = None
    # thinking str = None
    # message str = None
    # tool = ToolCall = none
    pass


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client) -> None:
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

    def _parse_llm_response(self, llm_response: str) -> LlmResponse:
        # split llm_response into these possible substrings
        # first split on this string:
        # <|start|>([a-z]+)
        # the capture is the role name

        # then match on
        # <|channel|>([a-z]+)<|message|>(.*)
        # the first capture is the channel
        # if analysis then save the right capture group to thinking
        # message in message
        # commentary in LLMRespones.commentary
        # a specail case for commentary is the toolcall:
        # <|channel|>commentary to=function.([^ ]+) json<|message|>(.*)
        pass

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        # Extract JSON content by removing everything before first { and after last }
        # TODO just look for json block
        first_brace = llm_response.find("{")
        last_brace = llm_response.rfind("}")

        if first_brace != -1 and last_brace != -1 and first_brace <= last_brace:
            json_content = llm_response[first_brace : last_brace + 1]
        else:
            json_content = llm_response

        try:
            logging.debug(f"Testing for valid json:\n{json_content}")
            tool_call = json.loads(json_content)
            if "tool" in tool_call:
                tool = tool_call["tool"]
                arguments = tool_call.get("arguments", {})
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
                            return f"Tool execution result: {result}"
                        except Exception as e:
                            error_msg = f"Error executing tool: {str(e)}"
                            print_error_message(error_msg)
                            logging.error(error_msg)
                            return error_msg

                return f"No server found with tool: {tool}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)

            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])

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

            messages = [{"role": "system", "content": system_message}]

            while True:
                try:
                    # TODO user input async, command output async
                    user_input = get_user_input()
                    if user_input.strip().lower() in ["quit", "exit"]:
                        print_system_message("👋 Goodbye!")
                        break

                    # Check if input is a command
                    if self.command_handler.is_command(user_input):
                        command_response = await self.command_handler.execute_command(
                            user_input
                        )
                        print_system_message(command_response)
                        continue

                    messages.append({"role": "user", "content": user_input})
                    logging.debug(json.dumps(messages, indent=2))

                    llm_response = self.llm_client.get_response(messages)

                    parsed = _parse_llm_response(llm_response)
                    print_assistant_message(llm_response)

                    result = await self.process_llm_response(llm_response)

                    if result != llm_response:
                        messages.append({"role": "assistant", "content": llm_response})
                        messages.append({"role": "system", "content": result})

                        final_response = self.llm_client.get_response(messages)
                        logging.info("\nFinal response: %s", final_response)
                        print_assistant_message(final_response)
                        messages.append(
                            {"role": "assistant", "content": final_response}
                        )
                    else:
                        messages.append({"role": "assistant", "content": llm_response})

                except KeyboardInterrupt:
                    print_system_message("👋 Goodbye!")
                    break

        finally:
            await self.cleanup_servers()
