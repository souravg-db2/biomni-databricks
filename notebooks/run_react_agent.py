# Databricks notebook source
# MAGIC %md
# MAGIC # Biomni ReAct Agent – Run
# MAGIC
# MAGIC Entry point for the ReAct agent on Databricks: reads widget values (prompt, payload, attachments_path, llm_model),
# MAGIC builds attachment context if provided, configures the agent, and runs it.

# COMMAND ----------

# Widgets: WebApp or job sets these when launching the notebook
dbutils.widgets.text("prompt", "You are a helpful biologist and expert geneticist", "User prompt")
dbutils.widgets.text("payload", "", "JSON payload (optional; may contain prompt, trial_id, attachments_path)")
dbutils.widgets.text("attachments_path", "", "UC Volume path to user-uploaded files (optional)")
dbutils.widgets.dropdown(
    "llm_model",
    "databricks-claude-sonnet-4-5",
    [
        "databricks-claude-sonnet-4-5",
        "databricks-claude-opus-4-5",
        "databricks-claude-haiku-4-5",
        "databricks-gemini-2-5-pro",
        "databricks-gemini-2-5-flash",
        "databricks-gpt-5-2",
        "databricks-gpt-5-1",
    ],
    "LLM model",
)
dbutils.widgets.dropdown(
    "plan",
    "false",
    ["true", "false"],
    "Enable planning",
)
dbutils.widgets.dropdown(
    "reflect",
    "false",
    ["true", "false"],
    "Enable reflection",
)
dbutils.widgets.dropdown(
    "data_lake",
    "true",
    ["true", "false"],
    "Enable data lake access",
)
dbutils.widgets.dropdown(
    "library_access",
    "true",
    ["true", "false"],
    "Enable library access",
)

# COMMAND ----------

# MAGIC %pip install -q "langchain-core>=1.0.0,<2"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import sys

try:
    import biomni_agent
except ImportError:
    repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if os.path.isdir(os.path.join(repo_root, "biomni_agent")) and repo_root not in sys.path:
        sys.path.insert(0, repo_root)

# COMMAND ----------

def get_prompt_and_context():
    """Read widgets: prompt (or from payload), optional attachments_path. Build attachments_context."""
    prompt = dbutils.widgets.get("prompt") or ""
    payload_str = dbutils.widgets.get("payload") or ""
    attachments_path = dbutils.widgets.get("attachments_path") or ""

    if payload_str:
        try:
            payload = json.loads(payload_str)
            prompt = prompt or payload.get("prompt", "")
            attachments_path = attachments_path or payload.get("attachments_path", "")
        except json.JSONDecodeError:
            pass

    attachments_context = None
    if attachments_path:
        try:
            files = dbutils.fs.ls(attachments_path)
            lines = [f"Files at {attachments_path}:"]
            for f in files:
                name = os.path.basename(f.path.rstrip("/"))
                lines.append(f"  - {name}")
            attachments_context = "\n".join(lines)
        except Exception as e:
            attachments_context = f"Could not list attachments at {attachments_path}: {e}"

    return prompt.strip(), attachments_context

# COMMAND ----------

prompt, attachments_context = get_prompt_and_context()
if not prompt:
    raise ValueError("Prompt is required. Set the 'prompt' widget or include 'prompt' in the 'payload' JSON.")

# COMMAND ----------

llm_model = dbutils.widgets.get("llm_model")
use_plan = dbutils.widgets.get("plan") == "true"
use_reflect = dbutils.widgets.get("reflect") == "true"
use_data_lake = dbutils.widgets.get("data_lake") == "true"
use_library_access = dbutils.widgets.get("library_access") == "true"

spark = spark  # noqa: F821

spark.sql("USE CATALOG biomni")

# COMMAND ----------

import importlib
import biomni_agent.llm
import biomni_agent.tools_uc
import biomni_agent.db_react
importlib.reload(biomni_agent.llm)
importlib.reload(biomni_agent.tools_uc)
importlib.reload(biomni_agent.db_react)
from biomni_agent.db_react import DBReact

agent = DBReact(
    path="/Volumes/biomni/agent/raw_files",
    llm=llm_model,
    spark=spark,
    dbutils=dbutils,  # noqa: F821
    use_tool_retriever=False,
    timeout_seconds=600,
)

agent.configure(
    plan=use_plan,
    reflect=use_reflect,
    data_lake=use_data_lake,
    library_access=use_library_access,
)

# COMMAND ----------

log, final_content = agent.go(prompt, attachments_context=attachments_context)

# COMMAND ----------

# Display final answer (log is available as agent.log)
display(final_content)
