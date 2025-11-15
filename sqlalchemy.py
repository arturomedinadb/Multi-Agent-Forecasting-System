import json
import sqlite3
from pathlib import Path

SESSION_ID = "20251114T172648Z"
AGENT_NAME = "evaluate_column_mapping_agent"
REPO_ROOT = Path(__file__).resolve().parent
DB_PATH = REPO_ROOT / "output" / "workflow_sessions.db"

conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()
cur.execute(
    """
    SELECT message_data
    FROM agent_messages
    WHERE session_id = ?
    ORDER BY id
    """,
    (SESSION_ID,),
)
rows = cur.fetchall()
conn.close()

for (message_data,) in rows:
    if not message_data:
        continue
    try:
        item = json.loads(message_data)
    except json.JSONDecodeError:
        continue

    if item.get("name") != AGENT_NAME:
        continue

    content_blocks = item.get("content") or []
    if not isinstance(content_blocks, list):
        content_blocks = [content_blocks]

    for block in content_blocks:
        if isinstance(block, dict):
            text = block.get("text")
            if text:
                print(text)
        else:
            print(block)