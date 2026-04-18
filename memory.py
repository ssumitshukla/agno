import os
from agno.agent import Agent
from agno.models.openai import OpenAILike
from agno.db.sqlite.sqlite import SqliteDb
from agno.memory.manager import MemoryManager
from agno.session.summary import SessionSummaryManager

# ── 0. Config ──────────────────────────────────────────────────────────────────

  # or load from .env

USER_ID    = "sumit_001"        # unique per user  (drives memory scoping)
SESSION_ID = "session_abc"      # unique per conversation

# ── 1. SQLite DB ───────────────────────────────────────────────────────────────
# Single .db file holds sessions, memories, summaries — everything.


import dotenv
dotenv.load_dotenv()

llm = OpenAILike(
    id="openai/gpt-4o",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

db = SqliteDb(
    db_file="agent_data.db",
    session_table="agent_sessions",     # stores conversation turns
    memory_table="user_memories",       # stores extracted user facts
)

# ── 2. User Memory ─────────────────────────────────────────────────────────────
# After each turn, gpt-4o-mini scans the conversation and extracts
# persistent facts about the user (name, preferences, goals, etc.)
# These are stored in SQLite and injected in ALL future sessions.

memory_manager = MemoryManager(
    model=llm,   # cheap model — just extracting facts
    db=db,
    add_memories=True,
    update_memories=True,
    delete_memories=False,
)

# ── 3. Session Summary Manager ─────────────────────────────────────────────────
# Once history exceeds `last_n_runs` raw turns, older turns get compressed
# into a tight summary by gpt-4o-mini. Only summary + last N turns are
# sent to gpt-4o — token cost stays flat even at turn 100+.

session_summary_manager = SessionSummaryManager(
    model=llm,   # cheap model — just summarizing
    last_n_runs=3,                         # keep last 3 turns raw; rest → summary
    conversation_limit=20,                 # compress after 20 messages
)

# ── 4. Agent ───────────────────────────────────────────────────────────────────

agent = Agent(
    model=llm,         # main model for responses

    # Storage
    db=db,
    num_history_messages=10,          # inject past turns into context

    # Memory
    memory_manager=memory_manager,
    enable_user_memories=True,             # auto-extract facts after each turn

    # Session Summary
    session_summary_manager=session_summary_manager,

    # Prompt
    description=(
        "You are a helpful personal learning assistant. "
        "Remember what the user has shared about themselves "
        "and personalize every response accordingly."
    ),

    markdown=True,
)

# ── 5. Chat Loop ───────────────────────────────────────────────────────────────

agent.print_response(
            input="Hi, I'm Sumit. I love learning about AI and building cool projects. Can you help me stay organized and remember what I tell you?",
            user_id=USER_ID,
            session_id=SESSION_ID,
            stream=True,
        )

