"""
Example script demonstrating direct SQL queries on the session database.

This shows how to analyze workflow patterns using SQL queries.
"""
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "output" / "workflow_sessions.db"


def run_example_queries():
    """Run example SQL queries on the session database."""
    
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("Run the workflow first: poetry run schema-mapper")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("=" * 70)
    print("SESSION DATABASE QUERIES")
    print("=" * 70)
    print(f"\nDatabase: {DB_PATH}\n")
    
    # Query 1: List all sessions
    print("1. All Workflow Sessions:")
    cursor.execute("""
        SELECT session_id, COUNT(*) as message_count
        FROM messages
        GROUP BY session_id
        ORDER BY session_id DESC
    """)
    for session_id, count in cursor.fetchall():
        print(f"   {session_id}: {count} messages")
    
    # Query 2: Agent message counts
    print("\n2. Message Counts by Agent:")
    cursor.execute("""
        SELECT name, role, COUNT(*) as count
        FROM messages
        GROUP BY name, role
        ORDER BY count DESC
    """)
    for agent, role, count in cursor.fetchall():
        print(f"   {agent} ({role}): {count} messages")
    
    # Query 3: Latest session details
    print("\n3. Latest Session Agent Flow:")
    cursor.execute("""
        SELECT session_id
        FROM messages
        ORDER BY rowid DESC
        LIMIT 1
    """)
    result = cursor.fetchone()
    if result:
        latest_session = result[0]
        cursor.execute("""
            SELECT name, role, substr(content, 1, 100) as content_preview
            FROM messages
            WHERE session_id = ?
            ORDER BY rowid
        """, (latest_session,))
        
        print(f"   Session: {latest_session}")
        for i, (agent, role, preview) in enumerate(cursor.fetchall(), 1):
            print(f"   [{i}] {agent} ({role}): {preview}...")
    
    # Query 4: Average messages per session
    print("\n4. Session Statistics:")
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT session_id) as total_sessions,
            COUNT(*) as total_messages,
            AVG(msg_count) as avg_messages_per_session
        FROM (
            SELECT session_id, COUNT(*) as msg_count
            FROM messages
            GROUP BY session_id
        )
    """)
    total_sessions, total_messages, avg_messages = cursor.fetchone()
    print(f"   Total sessions: {total_sessions}")
    print(f"   Total messages: {total_messages}")
    print(f"   Avg messages per session: {avg_messages:.1f}")
    
    # Query 5: Agent handoff patterns
    print("\n5. Most Common Agent Transitions:")
    cursor.execute("""
        SELECT 
            m1.name as from_agent,
            m2.name as to_agent,
            COUNT(*) as frequency
        FROM messages m1
        JOIN messages m2 ON m1.session_id = m2.session_id 
            AND m2.rowid = m1.rowid + 1
        WHERE m1.name != m2.name
        GROUP BY m1.name, m2.name
        ORDER BY frequency DESC
        LIMIT 10
    """)
    for from_agent, to_agent, freq in cursor.fetchall():
        print(f"   {from_agent} -> {to_agent}: {freq} times")
    
    conn.close()
    
    print("\n" + "=" * 70)
    print("\nCustom Query Example:")
    print("  sqlite3 output/workflow_sessions.db")
    print('  SELECT * FROM messages WHERE name = "WorkflowOrchestrator";')
    print("=" * 70)


if __name__ == "__main__":
    run_example_queries()

