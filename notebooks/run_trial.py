# Databricks notebook source
# MAGIC %md
# MAGIC # Biomni A1 Agent – Run Trial
# MAGIC
# MAGIC Entry point for the WebApp: reads widget values (prompt, payload, attachments_path, llm_model), builds attachment context if provided, and runs the agent.

# COMMAND ----------

# Widgets: WebApp sets these when launching the notebook
dbutils.widgets.text("prompt", "", "User prompt")
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

# COMMAND ----------

import json
import os
import sys

# Add repo root so we can import biomni_agent (adjust if your repo is elsewhere)
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
spark = spark  # noqa: F821

from biomni_agent import A1

agent = A1(
    path="/Volumes/biomni/agent/raw_files",
    llm=llm_model,
    spark=spark,
    use_tool_retriever=True,
    timeout_seconds=600,
)

# COMMAND ----------

log, final_content = agent.go(prompt, attachments_context=attachments_context)

# COMMAND ----------

# Display final answer (and log is available as agent.log)
display(final_content)
