"""
Example script demonstrating how to inspect workflow sessions.

This shows how to use the session inspector to analyze workflow runs.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mapping_system.schema_mapping.tools.session_inspector import (
    list_all_sessions,
    print_session_summary,
    inspect_session,
)


async def main():
    print("=" * 70)
    print("WORKFLOW SESSION INSPECTOR EXAMPLE")
    print("=" * 70)
    
    # List all sessions
    print("\n1. Listing all sessions...")
    sessions = await list_all_sessions()
    
    if not sessions:
        print("  No sessions found yet. Run the workflow first:")
        print("  poetry run schema-mapper")
        return
    
    print(f"  Found {len(sessions)} sessions:")
    for session_id in sessions:
        print(f"    - {session_id}")
    
    # Get latest session
    latest_session = sessions[0]
    print(f"\n2. Inspecting latest session: {latest_session}")
    
    # Get session details
    summary = await inspect_session(latest_session)
    
    print(f"\n  Total messages: {summary['total_messages']}")
    print(f"  Database: {summary['db_path']}")
    
    print("\n  Agent interactions:")
    for agent_name, count in sorted(summary['agent_counts'].items()):
        print(f"    {agent_name}: {count} messages")
    
    print("\n  Agent sequence (first 10):")
    sequence = summary['agent_sequence'][:10]
    print(f"    {' -> '.join(sequence)}")
    if len(summary['agent_sequence']) > 10:
        print(f"    ... and {len(summary['agent_sequence']) - 10} more")
    
    # Show first few messages
    print("\n3. First 5 messages:")
    for msg in summary['messages'][:5]:
        print(f"\n  [{msg['index']}] {msg['role']} ({msg['agent']}):")
        preview = msg['content_preview']
        if len(preview) > 100:
            preview = preview[:100] + "..."
        print(f"    {preview}")
    
    print("\n" + "=" * 70)
    print("\nTo see full session details, run:")
    print(f"  python -m mapping_system.schema_mapping.tools.session_inspector inspect {latest_session} --verbose")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

