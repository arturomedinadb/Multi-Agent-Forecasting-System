"""
Simple script to run the Flask development server.

Usage:
    cd src/ui
    python run_server.py
    
Or from project root:
    python -m src.ui.run_server
"""
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add src to path for mapping_system imports
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ui.app import app

if __name__ == "__main__":
    print("=" * 70)
    print("Schema Mapping Workflow UI")
    print("=" * 70)
    print("\nStarting Flask development server...")
    print("Open your browser to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server\n")
    app.run(debug=True, host="0.0.0.0", port=5000)

