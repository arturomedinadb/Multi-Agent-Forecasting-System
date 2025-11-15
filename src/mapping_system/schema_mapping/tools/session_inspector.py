"""Tools for inspecting workflow session history stored in SQLAlchemy database."""
import asyncio
import json
from pathlib import Path
from typing import Optional

from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DB_PATH = PROJECT_ROOT / "output" / "workflow_sessions.db"
DEFAULT_MESSAGES_TABLE = "agent_messages"


async def inspect_session(
    session_id: str,
    db_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict:
    """
    Inspect the conversation history for a workflow session.
    
    Args:
        session_id: The session ID to inspect (e.g., "20251114T120000Z")
        db_path: Path to the session database (defaults to output/workflow_sessions.db)
        limit: Maximum number of messages to retrieve (None = all)
    
    Returns:
        Dictionary with session information and message history
    """
    resolved_db_path = db_path or str(DEFAULT_DB_PATH)
    
    session = SQLAlchemySession.from_url(
        session_id=session_id,
        url=f"sqlite+aiosqlite:///{resolved_db_path}",
    )
    
    items = await session.get_items(limit=limit)
    
    # Analyze message flow
    agent_sequence = []
    agent_counts = {}
    
    for msg in items:
        agent_name = msg.get("name", "unknown")
        role = msg.get("role", "unknown")
        
        agent_sequence.append(agent_name)
        agent_counts[agent_name] = agent_counts.get(agent_name, 0) + 1
    
    # Build detailed message list
    messages = []
    for i, msg in enumerate(items):
        messages.append({
            "index": i,
            "role": msg.get("role"),
            "agent": msg.get("name", "unknown"),
            "content_preview": str(msg.get("content", ""))[:200] + "..." if len(str(msg.get("content", ""))) > 200 else str(msg.get("content", "")),
            "timestamp": msg.get("timestamp", "unknown"),
        })
    
    return {
        "session_id": session_id,
        "db_path": resolved_db_path,
        "total_messages": len(items),
        "agent_counts": agent_counts,
        "agent_sequence": agent_sequence,
        "messages": messages,
    }


async def list_all_sessions(db_path: Optional[str] = None) -> list:
    """
    List all sessions in the database.
    
    Args:
        db_path: Path to the session database
    
    Returns:
        List of session IDs
    """
    resolved_db_path = db_path or str(DEFAULT_DB_PATH)
    
    import sqlite3
    conn = sqlite3.connect(resolved_db_path)
    cursor = conn.cursor()
    
    try:
        # Ensure the expected table exists. If not, inform the caller gracefully.
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (DEFAULT_MESSAGES_TABLE,),
        )
        if cursor.fetchone() is None:
            return []
        
        cursor.execute(
            f"SELECT DISTINCT session_id FROM {DEFAULT_MESSAGES_TABLE} ORDER BY session_id DESC"
        )
        sessions = [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()
    
    return sessions


async def print_session_summary(
    session_id: str,
    db_path: Optional[str] = None,
    show_messages: bool = False,
) -> None:
    """
    Print a formatted summary of a session.
    
    Args:
        session_id: The session ID to inspect
        db_path: Path to the session database
        show_messages: Whether to print individual messages
    """
    summary = await inspect_session(session_id, db_path)
    
    print("=" * 70)
    print(f"SESSION SUMMARY: {session_id}")
    print("=" * 70)
    print(f"\nDatabase: {summary['db_path']}")
    print(f"Total messages: {summary['total_messages']}")
    
    print("\nAgent Interactions:")
    for agent_name, count in sorted(summary['agent_counts'].items()):
        print(f"  {agent_name}: {count} messages")
    
    print("\nAgent Sequence:")
    sequence_preview = " -> ".join(summary['agent_sequence'][:10])
    if len(summary['agent_sequence']) > 10:
        sequence_preview += " -> ..."
    print(f"  {sequence_preview}")
    
    if show_messages:
        print("\nDetailed Messages:")
        for msg in summary['messages']:
            print(f"\n  [{msg['index']}] {msg['role']} ({msg['agent']}):")
            print(f"    {msg['content_preview']}")
    
    print("\n" + "=" * 70)


def main():
    """CLI for session inspection."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m mapping_system.schema_mapping.tools.session_inspector list")
        print("  python -m mapping_system.schema_mapping.tools.session_inspector inspect <session_id>")
        print("  python -m mapping_system.schema_mapping.tools.session_inspector inspect <session_id> --verbose")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "list":
        sessions = asyncio.run(list_all_sessions())
        if not sessions:
            resolved_db = DEFAULT_DB_PATH
            print(
                f"\nNo sessions found. Ensure at least one workflow has been run and that "
                f"\"{resolved_db}\" exists with table '{DEFAULT_MESSAGES_TABLE}'."
            )
        else:
            print(f"\nFound {len(sessions)} sessions:")
            for session_id in sessions:
                print(f"  - {session_id}")
    
    elif command == "inspect":
        if len(sys.argv) < 3:
            print("Error: Please provide a session ID")
            sys.exit(1)
        
        session_id = sys.argv[2]
        show_verbose = "--verbose" in sys.argv or "-v" in sys.argv
        
        asyncio.run(print_session_summary(session_id, show_messages=show_verbose))
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()

