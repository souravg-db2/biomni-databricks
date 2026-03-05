"""
Python REPL for execute node. Persistent namespace; UC function wrappers are injected.
"""

import sys
from io import StringIO

# Persistent namespace shared across all executions
_persistent_namespace: dict = {}


def run_python_repl(command: str) -> str:
    """Execute Python code in a persistent environment and return the output."""
    command = command.strip("```").strip()
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    try:
        exec(command, _persistent_namespace)
        output = mystdout.getvalue()
    except Exception as e:
        output = f"Error: {str(e)}"
    finally:
        sys.stdout = old_stdout
    return output


def get_persistent_namespace() -> dict:
    """Return the persistent namespace (for injecting UC wrappers)."""
    return _persistent_namespace
