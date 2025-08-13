"""Rich console utilities for beautiful chat interface."""

import logging

from mcp.types import CallToolResult, TextContent
from rich.box import Box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.theme import Theme

# Custom theme for the chat interface
CHAT_THEME = Theme(
    {
        "user": "bold cyan",
        "assistant": "bold green",
        "system": "bold yellow",
        "error": "bold red",
        "tool": "bold magenta",
        "info": "dim blue",
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


def print_assistant_message(message: str) -> None:
    """Print assistant message with rich formatting."""
    panel = Panel(
        Markdown(message),
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
