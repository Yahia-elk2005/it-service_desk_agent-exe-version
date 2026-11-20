# IT Service Desk Triage & Routing Agent (Windows EXE Version)

Turns a plain-text IT problem into a structured ticket, like a small ServiceNow/ITSM triage desk.
Built with a free LLM API (Groq by default), a Gradio web UI, and full PyInstaller support for packaging into a standalone Windows `.exe` with a custom icon.

## What it does

1. **Receive request** – A user describes a tech issue in plain text, for example:
   *"My printer won't connect and I can't print payroll"*.
2. **Classify and prioritize** – The agent sets the priority (`Low`, `Medium`, `Critical`)
   and a department (`Network`, `Hardware`, `Software`, `Security`, `Access`).
3. **Route and draft** – Returns a JSON payload with priority, tags, assigned queue,
   response target (SLA), and numbered draft fix steps.

### Extra features
- **Accounts:** Create an account and sign in with a password (stored only as a salted PBKDF2 hash; 5 wrong attempts in a minute lock that username for the rest of the minute).
- **Guardrail checks:** Blocks unrelated requests, prompt-injection attempts, and requests for other people's tickets. Very short messages are rejected before any model call.
- **Per-user memory:** Ask *"What was my last ticket?"* or *"How many Critical tickets do I have?"*. The model only builds a structured plan; the code runs fixed, parameterized, read-only SQL for the signed-in user.
- **Model selector & auto-switch:** Pick the AI model per request (models appear when their provider key is in `.env`) with automatic fallback if a model is busy.
- **UI & desktop controls:** Live progress strip, priority bar, ticket card, JSON tab, history table, custom favicon, and a **Quit app** button to cleanly stop the background server when running as a windowed `.exe`.

## Structure (7 blocks in `service_desk_agent.py`)

1. Config and LLM client
2. Data models
3. Guardrail agent
4. Classifier agent
5. Router and resolution agent
6. Orchestrator, memory, and accounts
7. Gradio UI / UX

## Run from Python source

```bash
python -m venv venv
venv\Scripts\activate        # Windows (macOS/Linux: source venv/bin/activate)
pip install openai pydantic python-dotenv gradio
copy env.example .env        # macOS/Linux: cp env.example .env
# Open .env and paste your free API key
python service_desk_agent.py