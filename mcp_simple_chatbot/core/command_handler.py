"""Command handler for chat commands."""

import logging
from typing import Callable, Dict


class CommandHandler:
    """Handles chat commands that start with '/'."""

    def __init__(self):
        self.commands: Dict[str, Callable] = {
            "debug": self._toggle_debug,
            "help": self._show_help,
        }
        self._debug_enabled = False

    def is_command(self, message: str) -> bool:
        """Check if message is a command."""
        return message.strip().startswith("/")

    async def execute_command(self, message: str) -> str:
        """Execute a command and return response."""
        command_parts = message.strip()[1:].split()
        command_name = command_parts[0].lower()

        if command_name in self.commands:
            return await self.commands[command_name](command_parts[1:])
        else:
            return (
                f"Unknown command: /{command_name}. Type /help for available commands."
            )

    async def _toggle_debug(self, args: list) -> str:
        """Toggle debug logging."""
        self._debug_enabled = not self._debug_enabled
        level = logging.DEBUG if self._debug_enabled else logging.ERROR

        # Set logging level for all mcp_simple_chatbot loggers
        for name in logging.root.manager.loggerDict:
            if name.startswith("mcp_simple_chatbot"):
                logging.getLogger(name).setLevel(level)

        status = "enabled" if self._debug_enabled else "disabled"
        return f"Debug logging {status}."

    async def _show_help(self, args: list) -> str:
        """Show available commands."""
        return """Available commands:
/debug - Toggle debug logging on/off
/help - Show this help message"""
