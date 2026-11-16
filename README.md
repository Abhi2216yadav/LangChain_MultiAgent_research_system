# Multi-agent research assistant

A research system built with **LangChain**, **LangGraph**, **Google Gemini**, and **Tavily**. Ask a question and the system plans the research, searches and scrapes the web, writes a cited Markdown report, and reviews its own work.

## How it works

```text
START
  |
  v
Planner        breaks the question into 2-5 searchable sub-questions
  |
  v
Researcher     runs Tavily searches + scrapes the best pages
  |
  v
Writer         writes a cited Markdown report
  |
  v
Reviewer       checks accuracy, structure, and citations
  |                 |
  | approved        | needs work (and writing passes remain)
  v                 v
 END              Writer (revision loop)
```

The graph lives in `src/piplines/pipline.py`; the agent prompts and nodes live in `src/agents/agent.py`.

## Setup

Requirements: Python 3.11+ and a [Gemini API key](https://aistudio.google.com/apikey) plus a [Tavily API key](https://app.tavily.com).

```powershell
conda create -n langgraph-test python=3.11 -y
conda activate langgraph-test

# Install into the active environment. Using "python -m pip" avoids
# accidentally installing packages into a different Python.
python -m pip install -r requirements.txt
```

Create your local secrets file:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` and set at least:

```dotenv
GEMINI_API_KEY=...
TAVILY_API_KEY=...
GEMINI_MODEL=gemini-3.5-flash
MAX_REVISIONS=1
```

## Run

### Command line

```powershell
python main.py "What are the most important developments in AI agents in 2026?"
```

Options:

```powershell
python main.py "How is AI regulation changing?" --max-revisions 2 --save
python main.py "Compare small and frontier language models" --model gemini-3.5-flash-lite
```

- `--max-revisions N` controls how many writing passes the writer may make. `1` means one draft and one review; `2` or more enables the revision loop.
- `--save` writes the finished report to `./reports/`.
- `--model` overrides `GEMINI_MODEL` for that run.

### Streamlit app

```powershell
streamlit run app.py
```

The app shows live progress for each agent, keeps the finished result across reruns, exposes the plan/sources/notes under **Research trail**, and lets you download the report plus a portable README.

## Project structure

```text
.
├── app.py                  # Streamlit UI
├── main.py                 # CLI entry point
├── requirements.txt
├── .env.example            # Template for local secrets
└── src
    ├── config.py           # Env loading + LLM factory
    ├── utils.py            # Report saving / filename helpers
    ├── agents
    │   └── agent.py        # Planner, researcher, writer, reviewer nodes
    ├── piplines
    │   └── pipline.py      # LangGraph state machine
    └── tools
        └── tools.py        # Tavily search + multi-strategy page scraping
```

## Configuration

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes (for Gemini) | - | Google Gemini API key |
| `TAVILY_API_KEY` | Yes | - | Tavily web search API key |
| `GEMINI_MODEL` | No | `gemini-3.5-flash` | Gemini model name |
| `MAX_REVISIONS` | No | `1` | Default number of writing passes |
| `LLM_PROVIDER` | No | `google` | `google` or `openai` |
| `OPENAI_API_KEY` | Only for OpenAI | - | OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI model name |

## Troubleshooting

**`ModuleNotFoundError: No module named 'bs4'`**

This means the `python` running the script is not the environment where the dependencies were installed. Check which interpreter is being used and install into that same interpreter:

```powershell
python -c "import sys; print(sys.executable)"
python -m pip install -r requirements.txt
```

Make sure the `(langgraph-test)` prefix is visible in your terminal before running. If `python` opens the Microsoft Store, disable its app execution alias or use the full path to `miniconda3\envs\langgraph-test\python.exe`.

**Gemini returns `404 NOT_FOUND` or says a model is no longer available**

Google retires models periodically. List the models available to your key and set `GEMINI_MODEL` to one of them. The default in this repo is `gemini-3.5-flash`.

**Gemini request hangs**

Some preview/newer models can be slow or unavailable. Prefer `gemini-3.5-flash`, `gemini-3.5-flash-lite`, or `gemini-3.1-flash-lite`.

**`TAVILY_API_KEY is missing` / search errors**

Create a key at <https://app.tavily.com>, add it to `.env`, and restart the app or CLI. Search failures are returned to the agents as text so one bad query does not crash the run.

**The report says evidence is limited**

The writer is instructed not to invent facts. If sources are blocked, paywalled, or thin, the report will say so. Increase `--max-revisions` or ask a narrower question.

## Security

- `.env` is ignored by git. If you previously committed API keys, rotate them.
- API keys are only read from the environment; they are never written to reports.
- Research output is generated by an LLM and should be checked before being used for decisions.

## License

See [LICENSE](LICENSE).
