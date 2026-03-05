"""
A1 Agent for Databricks: LangGraph ReAct + UC Functions + KnowHow.

Uses DSPy-optimized prompt (placeholder: override _get_system_prompt for compiled prompt).
"""

import glob
import os
import re
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from biomni_agent.config import BiomniConfig, default_config


class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_step: str | None
from biomni_agent.env_desc import data_lake_dict, library_content_dict
from biomni_agent.know_how.loader import KnowHowLoader
from biomni_agent.llm import get_llm
from biomni_agent.repl_support import get_persistent_namespace, run_python_repl
from biomni_agent.tools_uc import (
    discover_uc_functions,
    inject_uc_wrappers_into_namespace,
    make_uc_function_wrappers,
    textify_api_dict,
)
from biomni_agent.utils import pretty_print, run_with_timeout


class A1:
    """Biomni A1 agent: LangGraph ReAct + UC Functions + volume data path."""

    def __init__(
        self,
        path: str | None = None,
        llm: str | None = None,
        spark: Any = None,
        use_tool_retriever: bool | None = None,
        timeout_seconds: int | None = None,
        commercial_mode: bool | None = None,
        config: BiomniConfig | None = None,
    ):
        cfg = config or default_config
        self.path = path or cfg.path
        self.llm_name = llm or cfg.llm
        self.spark = spark
        self.use_tool_retriever = use_tool_retriever if use_tool_retriever is not None else cfg.use_tool_retriever
        self.timeout_seconds = timeout_seconds or cfg.timeout_seconds
        self.commercial_mode = commercial_mode if commercial_mode is not None else cfg.commercial_mode
        self.config = cfg

        self.data_lake_dict = data_lake_dict
        self.library_content_dict = library_content_dict

        if self.spark is None:
            try:
                self.spark = __import__("pyspark.sql", fromlist=["SparkSession"]).SparkSession.getActiveSession()
            except Exception:
                pass
        if self.spark is None:
            print("Warning: Spark not available; UC function tools will be empty until Spark is set.")

        self.module2api = discover_uc_functions(self.spark) if self.spark else {"biomni.agent": []}
        self._uc_wrappers = make_uc_function_wrappers(self.spark, self.module2api) if self.spark else {}

        self.llm = get_llm(model=self.llm_name, temperature=cfg.temperature, stop_sequences=["</execute>", "</solution>"], config=cfg)
        self.know_how_loader = KnowHowLoader()
        print(f"Loaded {len(self.know_how_loader.documents)} know-how documents")
        self.timeout_seconds = self.timeout_seconds
        self.system_prompt = ""
        self.app = None
        self.checkpointer = MemorySaver()
        self.configure()

    def _generate_system_prompt(
        self,
        tool_desc: dict,
        data_lake_content: list,
        library_content_list: list,
        know_how_docs: list[dict] | None = None,
    ) -> str:
        """Build system prompt for generate node (DSPy-optimized prompt can override via _get_system_prompt)."""
        data_lake_formatted = []
        for item in data_lake_content:
            name = item.get("name", item) if isinstance(item, dict) else item
            desc = self.data_lake_dict.get(name, f"Data lake item: {name}")
            data_lake_formatted.append(f"{name}: {desc}")
        library_formatted = []
        for lib in library_content_list:
            name = lib.get("name", lib) if isinstance(lib, dict) else lib
            desc = self.library_content_dict.get(name, f"Software library: {name}")
            library_formatted.append(f"{name}: {desc}")

        know_how_block = ""
        if know_how_docs:
            know_how_block = "\n\n".join(
                f"📚 {d.get('name', '')}:\n{d.get('content', '')}" for d in know_how_docs
            )
            know_how_block = (
                "\n\n📚 KNOW-HOW DOCUMENTS (BEST PRACTICES & PROTOCOLS - ALREADY LOADED):\n"
                + know_how_block
            )

        base = """
You are a helpful biomedical assistant. Use the interactive coding environment with the provided tools and data.

Given a task, make a plan (numbered steps with checkboxes). At each turn, provide thinking then either:
1) Run code in <execute>...</execute> tags (Python default, or #!R for R, #!BASH for Bash).
2) Give the final answer in <solution>...</solution>.

You must include EITHER <execute> or <solution> in every response. When calling tools, save and print the result, e.g. result = some_tool(...); print(result).

Environment Resources:

- Function Dictionary (UC Functions in biomni.agent - call by name in Python):
{tool_desc}

- Biological data lake path: {data_lake_path}
Datasets:
{data_lake_content}

- Software libraries:
{library_content}
"""
        prompt = base.format(
            tool_desc=textify_api_dict(tool_desc),
            data_lake_path=self.path,
            data_lake_content="\n".join(data_lake_formatted),
            library_content="\n".join(library_formatted),
        ) + know_how_block
        return prompt

    def _get_system_prompt(self) -> str:
        """Override for DSPy-compiled prompt; default returns configured system_prompt."""
        return self.system_prompt

    def configure(self, self_critic: bool = False):
        """Build system prompt and LangGraph workflow."""
        data_lake_path = self.path
        try:
            if data_lake_path.startswith("/Volumes/"):
                try:
                    dbutils = __import__("dbutils").dbutils
                    files = dbutils.fs.ls(data_lake_path)
                    data_lake_items = [os.path.basename(f.path.rstrip("/")) for f in files]
                except Exception:
                    data_lake_items = list(self.data_lake_dict.keys())
            else:
                data_lake_content = glob.glob(os.path.join(data_lake_path, "*"))
                data_lake_items = [os.path.basename(x) for x in data_lake_content]
        except Exception:
            data_lake_items = list(self.data_lake_dict.keys())

        data_lake_with_desc = [{"name": x, "description": self.data_lake_dict.get(x, x)} for x in data_lake_items]
        tool_desc = {k: [t for t in v if t.get("name") != "run_python_repl"] for k, v in self.module2api.items()}
        library_list = list(self.library_content_dict.keys())

        know_how_docs = []
        for _id, doc in self.know_how_loader.documents.items():
            know_how_docs.append({
                "id": doc["id"],
                "name": doc["name"],
                "description": doc["description"],
                "content": doc.get("content_without_metadata", doc["content"]),
            })

        self.system_prompt = self._generate_system_prompt(
            tool_desc=tool_desc,
            data_lake_content=data_lake_with_desc,
            library_content_list=library_list,
            know_how_docs=know_how_docs or None,
        )

        def generate(state: dict) -> dict:
            system_prompt = self._get_system_prompt()
            if hasattr(self.llm, "model_name") and "gpt" in str(getattr(self.llm, "model_name", "")).lower():
                system_prompt += "\n\nIMPORTANT FOR GPT MODELS: Use <execute> or <solution> in EVERY response."
            messages = [SystemMessage(content=system_prompt)] + state["messages"]
            response = self.llm.invoke(messages)
            content = response.content
            if isinstance(content, list):
                msg = "".join(
                    (b.get("text") or b.get("content") or "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") in ("text", "output_text", "redacted_text")
                )
            else:
                msg = str(content)
            if "<execute>" in msg and "</execute>" not in msg:
                msg += "</execute>"
            if "<solution>" in msg and "</solution>" not in msg:
                msg += "</solution>"
            state["messages"] = state["messages"] + [AIMessage(content=msg.strip())]
            execute_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL | re.IGNORECASE)
            answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL | re.IGNORECASE)
            if answer_match:
                state["next_step"] = "end"
            elif execute_match:
                state["next_step"] = "execute"
            else:
                state["next_step"] = "generate"
            return state

        def execute(state: dict) -> dict:
            last_message = state["messages"][-1].content
            if "<execute>" in last_message and "</execute>" not in last_message:
                last_message += "</execute>"
            execute_match = re.search(r"<execute>(.*?)</execute>", last_message, re.DOTALL)
            if not execute_match:
                state["messages"].append(HumanMessage(content="<observe>No execute block found.</observe>"))
                state["next_step"] = "generate"
                return state
            code = execute_match.group(1)
            inject_uc_wrappers_into_namespace(self._uc_wrappers, get_persistent_namespace())
            result = run_with_timeout(run_python_repl, [code], timeout=self.timeout_seconds)
            if len(result) > 10000:
                result = "The output is too long. First 10K chars:\n" + result[:10000]
            state["messages"].append(HumanMessage(content=f"<observe>\n{result}\n</observe>"))
            state["next_step"] = "generate"
            return state

        workflow = StateGraph(AgentState)
        workflow.add_node("generate", generate)
        workflow.add_node("execute", execute)
        workflow.add_conditional_edges("generate", lambda s: s["next_step"], {"execute": "execute", "generate": "generate", "end": END})
        workflow.add_edge("execute", "generate")
        workflow.add_edge(START, "generate")
        self.app = workflow.compile(checkpointer=self.checkpointer)

    def go(self, prompt: str, attachments_context: str | None = None):
        """Run the agent. Optional attachments_context is prepended to the first user message."""
        user_content = prompt
        if attachments_context:
            user_content = prompt + "\n\nAdditional context from uploaded files:\n" + attachments_context
        inputs = {"messages": [HumanMessage(content=user_content)], "next_step": None}
        config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
        self.log = []
        final_state = None
        for s in self.app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            self.log.append(pretty_print(message))
            final_state = s
        self._conversation_state = final_state
        last_message = final_state["messages"][-1] if final_state else None
        return self.log, (last_message.content if last_message else "")
