# Drama script writer (episodes)

Multi-agent **drama episode script** flow (planning, drafting, per-episode writing with editor rounds) in a **Streamlit** UI. The default entrypoint launches that app.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or another environment manager

## Setup

```bash
uv sync
```

Create a `.env` file in the project root with your keys (see table below). It is loaded automatically by the apps.

## Environment variables

| Variable | Required | Description |
| -------- | -------- | ----------- |
| `GOOGLE_API_KEY` | Yes (for default Gemini) | API key from [Google AI Studio](https://aistudio.google.com/apikey) for the Gemini API. |
| `OPENAI_API_KEY` | If using OpenAI | Set when `NOVEL_LLM` points at an OpenAI model (e.g. `openai:gpt-4.1-mini`). |
| `NOVEL_LLM` | No | Chat model id for LangChain `init_chat_model` (e.g. `google_genai:gemini-2.5-flash`). Defaults to `google_genai:gemini-2.5-flash` when unset. |

The agents load these via `python-dotenv` from `.env` when present (see `agents/_llm.py`).

## Run the app

From the project root (starts Streamlit):

```bash
uv run python main.py
```

Equivalent:

```bash
streamlit run streamlit_app.py
```

Extra Streamlit CLI flags after `main.py` are forwarded (for example `uv run python main.py --server.port 8502`).
