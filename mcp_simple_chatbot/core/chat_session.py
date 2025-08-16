"""Chat session orchestration."""

import asyncio
import json
import logging
import re
from asyncio import Queue
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from mcp.types import TextContent

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
        role: Optional[str] = None,
        thinking: Optional[str] = None,
        message: Optional[str] = None,
        tool_call: Optional[ToolCall] = None,
        commentary: Optional[str] = None,
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


class UserInput:
    def __init__(self, content: str):
        self.content = content
        self.type = "user"

    def __repr__(self):
        return f"UserInput(content='{self.content}')"


class ToolResult:
    def __init__(self, tool_id: str, result: Any, tool_name: str):
        self.tool_id = tool_id
        self.result = result
        self.tool_name = tool_name
        self.type = "tool_result"

    def __repr__(self):
        return f"ToolResult(tool_id='{self.tool_id}', tool_name='{self.tool_name}')"


InputType = Union[UserInput, ToolResult]


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
        self.input_queue: Queue = Queue()
        self.running_tools: Dict[str, asyncio.Task] = {}
        self.tool_counter = 0
        self.user_input_task: Optional[asyncio.Task] = None
        self._monitor_user_input_enabled = True

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
            r"<\|channel\|>commentary to=(?:functions\.)?(.*?) ?json<\|message\|>"
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

    async def _start_tool_execution(self, tool_call: ToolCall) -> str:
        """Start tool execution in background and return task ID."""
        self.tool_counter += 1
        task_id = f"tool_{self.tool_counter}"

        # Create background task that will put result in queue when done
        task = asyncio.create_task(
            self._execute_tool_and_queue_result(tool_call, task_id)
        )
        self.running_tools[task_id] = task

        print_system_message(f"🔄 Started {tool_call.tool} (ID: {task_id})")
        return task_id

    async def _execute_tool_and_queue_result(self, tool_call: ToolCall, task_id: str):
        """Execute tool and put result in input queue."""
        try:
            result = await self._execute_tool_call(tool_call)
            if result:
                tool_result = ToolResult(task_id, result, tool_call.tool)
                await self.input_queue.put(tool_result)
                print_system_message(f"✅ {tool_call.tool} completed (ID: {task_id})")
        except Exception as e:
            logger.error(f"Tool {task_id} failed: {e}")
            # Put error result in queue
            error_result = {"error": str(e), "tool": tool_call.tool}
            tool_result = ToolResult(task_id, error_result, tool_call.tool)
            await self.input_queue.put(tool_result)
            print_system_message(f"❌ {tool_call.tool} failed (ID: {task_id})")
        finally:
            # Clean up completed task
            if task_id in self.running_tools:
                del self.running_tools[task_id]

    async def _execute_tool_call(self, tool_call: ToolCall) -> Optional[str]:
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
            return None

        try:
            result = await found_tool_server.execute_tool(tool, arguments)

            if isinstance(result.content[0], TextContent):
                print_tool_execution(tool, result)
                logger.info("Tool execution completed.")
                return result.content[0].text
            else:
                raise ValueError(
                    f"Unsupported tool result type: {type(result.content[0])}"
                )

        except Exception as e:
            error_msg = f"Error executing tool: {str(e)}"
            print_error_message(error_msg)
            logging.error(error_msg)
            return None

    async def _monitor_user_input(self):
        """Background task to monitor user input and add to queue."""
        while True:
            try:
                user_input = await self._get_user_input_async()
                if user_input is not None:
                    await self.input_queue.put(UserInput(user_input))
                else:
                    # EOF received, exit gracefully
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring user input: {e}")

    async def _get_user_input_async(self) -> Optional[str]:
        """Get user input asynchronously."""
        try:
            return await asyncio.to_thread(input, "You: ")
        except EOFError:
            return None
        except KeyboardInterrupt:
            return None

    async def _process_conversation_turn(self) -> bool:
        """
        Process a single input from the queue.

        Returns:
            True to continue processing, False to exit
        """
        # Get next input (user or tool result)
        input_item: InputType = await self.input_queue.get()

        if isinstance(input_item, UserInput):
            logger.info("Processing user input: %s", input_item.content)

            # Check for exit commands
            if input_item.content.strip().lower() in ["quit", "exit"]:
                print_system_message("👋 Goodbye!")
                logger.info("Chat session ended by user.")
                return False

            if input_item.content.strip().lower() in ["/history"]:
                import pprint
                pprint.pprint(self.messages)
                return True

            # Handle commands
            if self.command_handler.is_command(input_item.content):
                command_response = await self.command_handler.execute_command(
                    input_item.content
                )
                print_system_message(command_response)
                return True

            # Add to conversation history
            self.messages.append({"role": "user", "content": input_item.content})

        elif isinstance(input_item, ToolResult):
            logger.info("Processing tool result: %s", input_item.tool_id)
            # Add tool result to conversation
            if hasattr(input_item.result, "model_dump"):
                content = json.dumps(input_item.result.model_dump())
            else:
                content = json.dumps(input_item.result)
            self.messages.append({"role": "system", "content": content})

        # Get LLM response
        llm_response_raw = self.llm_client.get_response(self.messages)
        parsed = self._parse_llm_response(llm_response_raw)
        print_assistant_response(parsed)

        # Handle LLM response
        if parsed.tool_call:
            # Start tool execution (non-blocking)
            self.messages.append({"role": "assistant", "content": f"Calling tool `{parsed.tool_call.tool}` with arguments: `{parsed.tool_call.args}`"})
            await self._start_tool_execution(parsed.tool_call)

        if parsed.message:
            # Add final message to history
            self.messages.append({"role": "assistant", "content": parsed.message})

        return True

    async def start_initialization(self) -> None:
        """Initializes servers and discovers tools."""
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
                print_error_message(f"Failed to initialize server {server.name}: {e}")
                await self.cleanup_servers()
                raise  # Re-raise to stop further execution if init fails

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

    async def start(self) -> None:
        """Main chat session with event-driven processing."""
        input_monitor = None
        try:
            await self.start_initialization()

            # Start user input monitoring if enabled
            if self._monitor_user_input_enabled:
                input_monitor = asyncio.create_task(self._monitor_user_input())

            try:
                # Main event loop - process inputs as they arrive
                while True:
                    should_continue = await self._process_conversation_turn()
                    if not should_continue:
                        break

            except KeyboardInterrupt:
                print_system_message("👋 Goodbye!")
                logger.info("Chat session interrupted by user (KeyboardInterrupt).")

        finally:
            # Cleanup
            if input_monitor:
                input_monitor.cancel()
                try:
                    await input_monitor
                except asyncio.CancelledError:
                    pass

            # Cancel any running tools
            for task in self.running_tools.values():
                task.cancel()

            # Wait for running tools to complete cancellation
            if self.running_tools:
                await asyncio.gather(
                    *self.running_tools.values(), return_exceptions=True
                )

            logger.info("Cleaning up servers.")
            await self.cleanup_servers()
            logger.info("Server cleanup complete.")
