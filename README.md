# Biomni - A1 Agent

Biomni is a general-purpose biomedical AI agent designed to autonomously execute a wide range of research tasks across diverse biomedical subfields. By integrating cutting-edge large language model (LLM) reasoning with retrieval-augmented planning and code-based execution, Biomni helps scientists dramatically enhance research productivity and generate testable hypotheses.

## Scope

- **A1 Agent**: LangGraph ReAct loop with DSPy prompt optimization (placeholder), UC Function tools (`biomni.agent.*`), KnowHow, and optional attachment context.
- **Data**: Datasets land in `/Volumes/biomni/agent/raw_files` (ingestion notebook provided).
- **Tools**: Use only UC Functions in schema `biomni.agent`; discovered at runtime and invoked via Spark SQL from the REPL.
- **LLMs**: Databricks Foundation Model API; model selectable via widget (Claude, Gemini, GPT).
- **WebApp**: Notebook widgets for `prompt`, `payload`, `attachments_path`, `llm_model`; optional user uploads via volume path.

## Pre-install (cluster or job)

All dependencies must be **pre-installed** on the Databricks cluster. Do not rely on `pip install` in the agent notebook.

1. Attach the packages in `requirements.txt` to your cluster (Libraries → PyPI) or install them in your job’s environment:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure the repo (or `biomni_agent` package) is on the Python path when running `notebooks/run_trial.py` (e.g. clone repo into workspace and add project root to path, or install as a package).

## Repo layout

```
biomni-databricks/
  biomni_agent/           # Agent package
    __init__.py
    a1.py                # LangGraph A1 + UC tools + go(prompt, attachments_context)
    config.py
    llm.py               # Databricks Foundation Model API
    tools_uc.py          # Discover biomni.agent.*, REPL wrappers
    repl_support.py
    utils.py
    env_desc.py          # data_lake_dict, library_content_dict
    know_how/
      loader.py
      *.md
  notebooks/
    run_trial.py         # Widgets → agent.go(prompt, attachments_context)
    data_ingestion.py    # S3 → /Volumes/biomni/agent/raw_files
  requirements.txt
  README.md
```

## Deployment (Databricks Asset Bundle)

1. Configure `databricks.yml` targets: set `workspace.host` (and optionally `workspace.profile`) for your workspace.
2. From the bundle root: `databricks bundle deploy -t dev` (or `-t prod`).
3. Jobs **Biomni Run Trial** and **Biomni Data Ingestion** will appear in the workspace. Ensure the cluster has `requirements.txt` installed and the repo root is on `PYTHONPATH` if needed (e.g. run from Repos).

## Running a trial

1. **Volume and data**: Create catalog `biomni`, schema `agent`, volume `raw_files`. Run `notebooks/data_ingestion.py` once to download data lake files into `/Volumes/biomni/agent/raw_files`.
2. **UC Functions**: Register tools as UC Functions in `biomni.agent`; the agent discovers them via `SHOW FUNCTIONS IN biomni.agent`.
3. **Run**: Execute `notebooks/run_trial.py`. Set widgets:
   - `prompt`: user query (or pass JSON in `payload` with `prompt` and optional `attachments_path`).
   - `attachments_path`: optional volume path where the WebApp wrote user uploads.
   - `llm_model`: e.g. `databricks-claude-sonnet-4-5`, `databricks-gemini-2-5-pro`, `databricks-gpt-5-2`.

## Attachments

If the WebApp sets `attachments_path` to a UC Volume path (e.g. `/Volumes/biomni/agent/uploads/<trial_id>/`), the notebook lists files there and passes a short description to `agent.go(prompt, attachments_context=...)`. The agent includes this in the first user message so the LLM can reference the uploaded files.

## DSPy

The generate node uses `_get_system_prompt()`; override it in a subclass or replace the default system prompt with a DSPy-compiled prompt (e.g. from a separate optimization job) to enable prompt optimization.
