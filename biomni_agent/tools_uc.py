"""
Discover biomni.agent.* UC Functions and build module2api-style dict + REPL wrappers.

Tools are invoked via spark.sql("SELECT biomni.agent.<name>(...)") and results
returned to the agent. No read_module2api() from Biomni-Old; tools come only from UC.
"""

from __future__ import annotations

import json
from typing import Any

# Schema for UC functions (catalog.schema)
BIOMNI_AGENT_SCHEMA = "biomni.agent"


def discover_uc_functions(spark) -> dict[str, list[dict]]:
    """
    Discover all functions in biomni.agent schema and build module2api-like structure.

    Returns:
        Dict mapping module name (e.g. "biomni.agent") to list of tool dicts with
        keys: name, description, required_parameters, optional_parameters, etc.
    """
    try:
        df = spark.sql(f"SHOW USER FUNCTIONS IN {BIOMNI_AGENT_SCHEMA}")
    except Exception:
        try:
            df = spark.sql(f"SHOW FUNCTIONS IN {BIOMNI_AGENT_SCHEMA}")
        except Exception as e:
            print(f"Warning: Could not SHOW FUNCTIONS IN {BIOMNI_AGENT_SCHEMA}: {e}")
            return {BIOMNI_AGENT_SCHEMA: []}

    tools = []
    for row in df.collect():
        rd = row.asDict()
        func_name = rd.get("function") or rd.get("functionName") or str(row[0])
        if not func_name or "." in func_name.split()[-1]:
            continue
        if not func_name.isidentifier():
            continue
        # Describe function for params (optional; some runtimes support DESCRIBE FUNCTION)
        try:
            desc_df = spark.sql(f"DESCRIBE FUNCTION EXTENDED {BIOMNI_AGENT_SCHEMA}.{func_name}")
            desc_rows = desc_df.collect()
            description = ""
            params = []
            for r in desc_rows:
                d = r.asDict()
                key = (d.get("info_name") or d.get("col_name") or str(r[0])).strip()
                val = (d.get("info_value") or d.get("data_type") or str(r[1])).strip()
                if key == "Description" or key == "desc":
                    description = val
                if key == "Input" or "Parameter" in key:
                    params.append(val)
        except Exception:
            description = f"UC function {BIOMNI_AGENT_SCHEMA}.{func_name}"
            params = []

        tool = {
            "name": func_name,
            "description": description or f"UC function: {func_name}",
            "required_parameters": [{"name": p, "type": "string", "description": p} for p in params[:3]],
            "optional_parameters": [],
            "module": BIOMNI_AGENT_SCHEMA,
        }
        tools.append(tool)

    return {BIOMNI_AGENT_SCHEMA: tools}


def textify_api_dict(api_dict: dict[str, list[dict]]) -> str:
    """Convert module2api dict to a formatted string for the system prompt."""
    lines = []
    for category, methods in api_dict.items():
        lines.append(f"Import file: {category}")
        lines.append("=" * (len("Import file: ") + len(category)))
        for method in methods:
            lines.append(f"Method: {method.get('name', 'N/A')}")
            lines.append(f"  Description: {method.get('description', 'No description provided.')}")
            req = method.get("required_parameters", [])
            if req:
                lines.append("  Required Parameters:")
                for p in req:
                    name = p.get("name", "N/A")
                    typ = p.get("type", "N/A")
                    desc = p.get("description", "No description")
                    lines.append(f"    - {name} ({typ}): {desc}")
            lines.append("")
        lines.append("")
    return "\n".join(lines)


def make_uc_function_wrappers(spark, module2api: dict[str, list[dict]]) -> dict[str, Any]:
    """
    Build a dict of callable wrappers for each UC function.
    Each wrapper runs spark.sql("SELECT biomni.agent.<name>(...)") and returns the result.

    Returns:
        Dict mapping function name -> callable(**kwargs) -> result (str or dict).
    """
    wrappers = {}
    for _module, tools in module2api.items():
        for tool in tools:
            name = tool.get("name")
            if not name:
                continue
            full_name = f"{BIOMNI_AGENT_SCHEMA}.{name}"

            def _make_wrapper(fn_name: str, full_fn: str, desc: str):
                def _wrapper(**kwargs):
                    # Build SQL: SELECT biomni.agent.fn('arg1', 'arg2') or named args
                    args_str = ", ".join(
                        repr(v) if isinstance(v, str) else str(v) for v in kwargs.values()
                    )
                    if not kwargs:
                        args_str = ""
                    sql = f"SELECT {full_fn}({args_str}) AS result"
                    try:
                        result_df = spark.sql(sql)
                        row = result_df.first()
                        if row is None:
                            return ""
                        out = row.result
                        if hasattr(out, "asDict"):
                            return json.dumps(out.asDict())
                        return str(out)
                    except Exception as e:
                        return json.dumps({"error": str(e)})

                _wrapper.__name__ = fn_name
                _wrapper.__doc__ = desc
                return _wrapper

            wrappers[name] = _make_wrapper(name, full_name, tool.get("description", ""))
    return wrappers


def inject_uc_wrappers_into_namespace(wrappers: dict[str, Any], namespace: dict) -> None:
    """Inject UC function wrappers into the given namespace (e.g. REPL persistent namespace)."""
    namespace.update(wrappers)
