"""
Databricks-compatible ReAct Agent: LangGraph tool-calling + UC Functions.

Adapts the original react.py to Databricks by:
- Using Databricks Foundation Model API for LLM
- Discovering tools from Unity Catalog functions
- Using UC Volume paths for data lake
- Thread-based timeout instead of multiprocessing
"""

import glob
import json
import os
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, create_model

from biomni_agent.config import BiomniConfig, default_config
from biomni_agent.env_desc import data_lake_dict, library_content_dict
from biomni_agent.llm import get_llm
from biomni_agent.repl_support import get_persistent_namespace, run_python_repl
from biomni_agent.tools_uc import (
    discover_uc_functions,
    inject_uc_wrappers_into_namespace,
    make_uc_function_wrappers,
    textify_api_dict,
)
from biomni_agent.utils import pretty_print, run_with_timeout


def _uc_tool_to_langchain(tool_spec: dict, uc_wrappers: dict) -> Optional[StructuredTool]:
    """Convert a UC function spec + wrapper into a langchain StructuredTool for bind_tools."""
    name = tool_spec.get("name")
    if not name or name not in uc_wrappers:
        return None

    description = tool_spec.get("description", f"UC function: {name}")
    wrapper_fn = uc_wrappers[name]

    fields: dict[str, Any] = {}
    for param in tool_spec.get("required_parameters", []):
        p_name = param.get("name", "arg")
        p_desc = param.get("description", "")
        fields[p_name] = (str, Field(description=p_desc))

    for param in tool_spec.get("optional_parameters", []):
        p_name = param.get("name", "arg")
        p_desc = param.get("description", "")
        fields[p_name] = (Optional[str], Field(default=None, description=p_desc))

    if not fields:
        fields["input"] = (Optional[str], Field(default=None, description="Input argument"))

    input_model = create_model(f"{name}_input", **fields)

    return StructuredTool.from_function(
        func=wrapper_fn,
        name=name,
        description=description,
        args_schema=input_model,
    )


def _make_repl_tool(timeout_seconds: int) -> StructuredTool:
    """Create a langchain StructuredTool for the Python REPL with timeout."""

    def _repl_with_timeout(code: str) -> str:
        return run_with_timeout(run_python_repl, [code], timeout=timeout_seconds)

    class ReplInput(BaseModel):
        code: str = Field(description="Python code to execute")

    return StructuredTool.from_function(
        func=_repl_with_timeout,
        name="run_python_repl",
        description=(
            "Execute Python code in a persistent environment. "
            "Use for data analysis, computations, and accessing data lake files."
        ),
        args_schema=ReplInput,
    )


class DBReact:
    """Databricks-compatible ReAct agent using LangGraph tool-calling pattern."""

    def __init__(
        self,
        path: str | None = None,
        llm: str | None = None,
        spark: Any = None,
        dbutils: Any = None,
        use_tool_retriever: bool | None = None,
        timeout_seconds: int | None = None,
        config: BiomniConfig | None = None,
    ):
        cfg = config or default_config
        self.path = path or cfg.path
        self.llm_name = llm or cfg.llm
        self.spark = spark
        self.dbutils = dbutils
        self.use_tool_retriever = (
            use_tool_retriever if use_tool_retriever is not None else cfg.use_tool_retriever
        )
        self.timeout_seconds = timeout_seconds or cfg.timeout_seconds
        self.config = cfg

        self.data_lake_dict = data_lake_dict
        self.library_content_dict = library_content_dict

        if self.spark is None:
            try:
                self.spark = (
                    __import__("pyspark.sql", fromlist=["SparkSession"])
                    .SparkSession.getActiveSession()
                )
            except Exception:
                pass
        if self.spark is None:
            print("Warning: Spark not available; UC function tools will be empty until Spark is set.")

        self.module2api = discover_uc_functions(self.spark) if self.spark else {"biomni.agent": []}
        self._uc_wrappers = make_uc_function_wrappers(self.spark, self.module2api) if self.spark else {}

        self.tools = self._build_tools()

        self.llm = get_llm(model=self.llm_name, temperature=cfg.temperature, config=cfg)
        self.prompt = ""
        self.system_prompt = ""

    def _build_tools(self) -> list[StructuredTool]:
        """Build langchain StructuredTool list from UC functions + Python REPL."""
        tools: list[StructuredTool] = []

        for _module, api_list in self.module2api.items():
            for api_spec in api_list:
                if api_spec.get("name") == "run_python_repl":
                    continue
                tool = _uc_tool_to_langchain(api_spec, self._uc_wrappers)
                if tool:
                    tools.append(tool)

        tools.append(_make_repl_tool(self.timeout_seconds))
        return tools

    def configure(
        self,
        plan=False,
        reflect=False,
        data_lake=False,
        react_code_search=False,
        library_access=False,
    ):
        """Build system prompt and compile the LangGraph ReAct workflow."""
        data_lake_path = self.path

        try:
            if data_lake_path.startswith("/Volumes/") and self.dbutils:
                try:
                    files = self.dbutils.fs.ls(data_lake_path)
                    data_lake_items = [os.path.basename(f.path.rstrip("/")) for f in files]
                except Exception:
                    data_lake_items = list(self.data_lake_dict.keys())
            else:
                data_lake_content = glob.glob(os.path.join(data_lake_path, "*"))
                data_lake_items = [os.path.basename(x) for x in data_lake_content]
        except Exception:
            data_lake_items = list(self.data_lake_dict.keys())

        if react_code_search:
            tools = [t for t in self.tools if t.name in ["run_python_repl", "search_google"]]
            prompt_modifier = """
You are a helpful biomedical assistant assigned with the task of problem-solving.

You have access to two tools:
1) run_python_repl: to write and run python code
2) search_google: to search google for information

You can use them to solve the problem.
            """
        else:
            tools = self.tools
            if (not plan) and (not reflect):
                prompt_modifier = """You are a helpful biologist and expert geneticist.
                """
            elif plan and (not reflect):
                prompt_modifier = """You are a helpful biologist and expert geneticist.
                Given the question from the user,
                - First, come up with a high level plan based on your understanding of the problem and available tools and record it in the Research Plan and Status. You can revise the plan later.
                - Research Plan and Status should well organized and succinctly keep track of 1) high level plan (can be revised), 2) what steps have been done and what steps are in progress, 3) short results and conclusions of each step after it has been performed. Do not perform action in research plan.
                - Research Plan and Status must only include progress that has been made by previous steps. It should not include results not directly confirmed by the previous observation.
                - Follow the plan and try to achieve the goal as straightforwardly as possible. Use tools as necessary.
                """
            elif (not plan) and reflect:
                prompt_modifier = """You are a helpful biologist and expert geneticist.
                In each round after the tool is used, conduct "reflection" step: reflect on the current state of the problem and the results of the last round. What does the observation mean? If there is an error, what caused the error and how to debug?
                """
            else:
                prompt_modifier = """You are a helpful biologist and expert geneticist.
                Given the question from the user,
                - First, come up with a high level plan based on your understanding of the problem and available tools and record it in the Research Plan and Status. You can revise the plan later.
                - Research Plan and Status should well organized and succinctly keep track of 1) high level plan (can be revised), 2) what steps have been done and what steps are in progress, 3) short results and conclusions of each step after it has been performed. Do not perform action in research plan.
                - Research Plan and Status must only include progress that has been made by previous steps. It should not include results not directly confirmed by the previous observation.
                - Follow the plan and try to achieve the goal as straightforwardly as possible. Use tools as necessary.
                In each round after the tool is used, conduct "reflection" step: reflect on the current state of the problem and the results of the last round. What does the observation mean? If there is an error, what caused the error and how to debug?
                You have access to write_python_code and run_python_repl tool to write and run your own code if tools fail, or if the given tools are not enough. Please always make sure to write code when dealing with substantial data, including finding the length of long sequences or elements at different positions.
                """

        if data_lake:
            data_lake_formatted = []
            for item in data_lake_items:
                description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                data_lake_formatted.append(f"{item}: {description}")

            prompt_modifier += """
You can also access a biological data lake at the following path: {data_lake_path}. You can use the run_python_repl tool to write code to understand the data, process and utilize it for the task.
Here is the list of datasets with their descriptions:
----
{data_lake_formatted}
----
            """.format(
                data_lake_path=data_lake_path,
                data_lake_formatted="\n".join(data_lake_formatted),
            )

        if library_access:
            library_formatted = []
            for lib_name, lib_desc in self.library_content_dict.items():
                library_formatted.append(f"{lib_name}: {lib_desc}")

            prompt_modifier += """
You also have access to a list of software packages that can be used to perform various tasks.
You can use the run_python_repl tool to write code to access and utilize the library for the task.
Don't forget the import statement.
Here is the list of available libraries with their descriptions:
----
{library_formatted}
----
            """.format(library_formatted="\n".join(library_formatted))

        tool_desc = textify_api_dict(
            {k: [t for t in v if t.get("name") != "run_python_repl"] for k, v in self.module2api.items()}
        )
        if tool_desc.strip():
            prompt_modifier += f"""
Available UC Functions (callable via tools):
{tool_desc}
"""

        print("=" * 25 + "System Prompt" + "=" * 25)
        print(prompt_modifier)
        self.system_prompt = prompt_modifier
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_modifier),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        self.active_tools = tools

        inject_uc_wrappers_into_namespace(self._uc_wrappers, get_persistent_namespace())

        self.app = create_react_agent(self.llm, tools)

    def go(self, prompt: str, attachments_context: str | None = None):
        """Execute the agent with the given prompt.

        Args:
            prompt: The user's query.
            attachments_context: Optional context from uploaded files.
        """
        user_content = prompt
        if attachments_context:
            user_content = prompt + "\n\nAdditional context from uploaded files:\n" + attachments_context

        config = {"recursion_limit": 50}
        inputs = {"messages": [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content),
        ]}
        self.log = []
        for s in self.app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            out = pretty_print(message)
            self.log.append(out)
        return self.log, s["messages"][-1].content

    def result_formatting(self, output_class, task_intention):
        """Extract structured output from agent history using an LLM."""
        self.format_check_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are evaluateGPT, tasked with extract and parse the task output "
                        "based on the history of an agent. "
                        "Review the entire history of messages provided. "
                        "Here is the task output requirement: \n"
                        f"'{task_intention.replace('{', '{{').replace('}', '}}')}'.\n"
                    ),
                ),
                ("placeholder", "{messages}"),
            ]
        )
        checker_llm = self.format_check_prompt | self.llm.with_structured_output(output_class)
        raw = checker_llm.invoke({"messages": [HumanMessage(content=str(self.log))]})
        # result = raw.model_dump() if hasattr(raw, "model_dump") else raw.dict()
        return raw
