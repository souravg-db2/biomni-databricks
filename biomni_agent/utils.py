"""Shared utilities for Biomni Databricks agent."""

import queue
import threading
from typing import Any, Callable


def run_with_timeout(
    func: Callable,
    args: list | None = None,
    kwargs: dict | None = None,
    timeout: int = 600,
) -> Any:
    """Run a function with a timeout. Returns result or error string."""
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    result_queue: queue.Queue = queue.Queue()

    def thread_func():
        try:
            result = func(*args, **kwargs)
            result_queue.put(("success", result))
        except Exception as e:
            result_queue.put(("error", str(e)))

    thread = threading.Thread(target=thread_func, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return (
            f"ERROR: Code execution timed out after {timeout} seconds. "
            "Please try with simpler inputs or break your task into smaller steps."
        )
    try:
        status, result = result_queue.get(block=False)
        return result if status == "success" else f"Error in execution: {result}"
    except queue.Empty:
        return "Error: Execution completed but no result was returned"


def pretty_print(message, printout: bool = True) -> str:
    """Format a message for display. Returns formatted string."""
    try:
        content = getattr(message, "content", message)
        msg_type = getattr(message, "type", "message")
        name = getattr(message, "name", None)
    except Exception:
        return str(message)
    title = f"{msg_type.title()} Message"
    if name:
        title += f"\nName: {name}"
    if isinstance(content, list):
        for i in content:
            if isinstance(i, dict):
                if i.get("type") == "text":
                    title += f"\n{i.get('text', '')}\n"
                elif i.get("type") == "tool_use":
                    title += f"\nTool: {i.get('name', '')}\nInput: {i.get('input', '')}"
            else:
                title += f"\n{str(i)}"
    else:
        title += f"\n\n{content}"
    if printout:
        print(title)
    return title
