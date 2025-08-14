"""Rich console utilities for beautiful chat interface."""

import logging
import json # Added for json.dumps

from mcp.types import CallToolResult, TextContent
from rich.box import Box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.theme import Theme
from rich.text import Text # Import Text for granular styling

# Custom theme for the chat interface
CHAT_THEME = Theme(
    {
        "user": "bold cyan",
        "assistant": "bold green",
        "system": "bold yellow",
        "error": "bold red",
        "tool": "bold magenta",
        "info": "dim blue",
        "thinking": "dim white", # Add a new style for thinking text
    }
)

console = Console(theme=CHAT_THEME)

# fmt: off
LEFT_BAR = Box(
    "┃   \n"
    "┃   \n"
    "┃   \n"
    "┃   \n"
    "┃   \n"
    "┃   \n"
    "┃   \n"
    "    \n"
)
# fmt: on


def print_user_message(message: str) -> None:
    """Print user message with rich formatting."""
    panel = Panel(
        Markdown(message), box=LEFT_BAR, title="[user]You[/user]", border_style="cyan"
    )
    console.print(panel)


# MODIFIED: Now takes LLMResponse object
def print_assistant_response(parsed_response) -> None: # Type hint will be added in chat_session.py
    """Print assistant message with rich formatting based on LLMResponse."""
    
    # Create a Rich Text object to build the message
    full_message = Text()

    if parsed_response.thinking:
        full_message.append("_[thinking] ", style="thinking") # Apply 'thinking' style
        full_message.append(parsed_response.thinking, style="thinking")
        full_message.append("_\n\n", style="thinking") # Add newline for separation

    if parsed_response.message:
        # Markdown can be applied to a Text object, but it's often easier
        # to render Markdown separately if it's a large block.
        # For simplicity, we'll just append the message text.
        # If you need full Markdown rendering for the message part,
        # you might need to render it as a separate segment or use a more complex Rich layout.
        full_message.append(parsed_response.message)
        full_message.append("\n\n") # Add newline for separation

    if parsed_response.tool_call:
        tool = parsed_response.tool_call.tool
        arguments = parsed_response.tool_call.args
        tool_json = f'```json\n{{"tool": "{tool}", "arguments": {json.dumps(arguments, indent=2)}}}\n```'
        # Render tool_json as Markdown (code block)
        full_message.append(Markdown(tool_json))
        full_message.append("\n\n") # Add newline for separation

    if parsed_response.commentary and not (parsed_response.thinking or parsed_response.message or parsed_response.tool_call):
        # Only show commentary if no other specific channels were found
        full_message.append(parsed_response.commentary)
        full_message.append("\n\n") # Add newline for separation

    # Remove trailing newlines if any
    if str(full_message).endswith("\n\n"):
        full_message = Text(str(full_message).rstrip("\n"))

    panel = Panel(
        full_message,
        box=LEFT_BAR,
        title="[assistant]Assistant[/assistant]",
        title_align="left",
        border_style="green",
    )
    console.print(panel)


def print_system_message(message: str) -> None:
    """Print system message with rich formatting."""
    console.print(f"[system]{message}[/system]")


def print_error_message(message: str) -> None:
    """Print error message with rich formatting."""
    console.print(f"[error]Error: {message}[/error]")


def print_tool_execution(tool_name: str, result: CallToolResult) -> None:
    """Print tool execution result with rich formatting."""
    logging.debug(result)
    content = result.content[0]
    if isinstance(content, TextContent):
        text = content.text
    else:
        text = str(content)

    text = text.strip()
    text = text.replace("\r", "")

    try:
        # Try to auto-detect and highlight
        syntax = Syntax(text, lexer="guess", theme="monokai", line_numbers=False)
        panel = Panel(
            syntax,
            title="[tool]Tool Execution[/tool]",
            title_align="left",
            box=LEFT_BAR,
            border_style="magenta",
        )
    except Exception:  # Changed from bare except
        # Fallback to plain text if syntax highlighting fails
        panel = Panel(
            text,
            title="[tool]Tool Execution[/tool]",
            title_align="left",
            box=LEFT_BAR,
            border_style="magenta",
        )
    console.print(panel)


def get_user_input(prompt: str = "You") -> str:
    """Get user input with rich prompt."""
    return Prompt.ask(f"[user]{prompt}[/user]")
