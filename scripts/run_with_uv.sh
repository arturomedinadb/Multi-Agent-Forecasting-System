#!/bin/bash

# Script to run the schema mapping agent with uv, working around parent pyproject.toml issues

echo "🚀 Running Schema Mapping Agent with uv"
echo "======================================"

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Temporarily move the problematic parent pyproject.toml
PARENT_TOML="/Users/kkkatheriine/pyproject.toml"
BACKUP_TOML="/Users/kkkatheriine/pyproject.toml.backup"

if [ -f "$PARENT_TOML" ]; then
    echo "🔧 Temporarily moving parent pyproject.toml..."
    mv "$PARENT_TOML" "$BACKUP_TOML"
fi

# Run the agent
echo "🤖 Starting agent..."
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✅ API key found, running real agent..."
    uv run python run_agent.py
else
    echo "⚠️  No API key found, running mock version..."
    uv run python test_without_api.py
fi

# Restore the parent pyproject.toml
if [ -f "$BACKUP_TOML" ]; then
    echo "🔄 Restoring parent pyproject.toml..."
    mv "$BACKUP_TOML" "$PARENT_TOML"
fi

echo "✅ Done!"
