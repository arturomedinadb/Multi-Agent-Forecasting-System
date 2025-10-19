# Understanding RunResult (OpenAI Agents SDK)

When you run an agent, the OpenAI Agents SDK returns a `RunResult` object that contains comprehensive information about what happened during the agent's execution.

---

## 📊 What You Saw in Your Output

```
RunResult:
- Last agent: Agent(name="DataPrepAgent", ...)
- Final output (str):
    [JSON array with metadata...]
    Data preparation complete - 6 files analyzed
- 13 new item(s)
- 2 raw response(s)
- 0 input guardrail result(s)
- 0 output guardrail result(s)
```

---

## 🔍 Breaking Down Each Component

### 1. **Last agent**
```
- Last agent: Agent(name="DataPrepAgent", ...)
```
**What it is**: The last agent that was executed  
**Your case**: `DataPrepAgent` - the only agent you ran  
**Why useful**: In multi-agent workflows with handoffs, this tells you which agent completed the work

---

### 2. **Final output**
```
- Final output (str):
    [
      {
        "file_path": "...",
        "shape": [3, 14],
        "columns": [...],
        ...
      }
    ]
    Data preparation complete - 6 files analyzed
```
**What it is**: The text content the agent produced  
**Your case**: JSON metadata array + completion message  
**Why useful**: This is the actual result you care about - the agent's work product

---

### 3. **13 new item(s)**
```
- 13 new item(s)
```
**What it is**: Number of conversation turns/messages added during execution  
**Your case**: 13 items = Initial user message + agent thinking + 6 tool calls + 6 tool results + final response  

**Breakdown**:
- 1 user message (your rendered prompt)
- 6 tool call messages (agent calling `load_and_describe_dataset` for each file)
- 6 tool result messages (responses from each tool call)
- 1+ assistant response messages (agent's reasoning and final output)

**Why useful**: Shows how much interaction happened between user, agent, and tools

---

### 4. **2 raw response(s)**
```
- 2 raw response(s)
```
**What it is**: Number of times the LLM generated a response  
**Your case**: 2 responses

**Likely breakdown**:
1. **First response**: Agent analyzes the prompt, decides to call tools, generates 6 tool calls
2. **Second response**: After receiving all 6 tool results, agent compiles them into the final JSON output

**Why useful**: Shows how many "thinking cycles" the agent went through. More responses = more complex reasoning path.

---

### 5. **0 input guardrail result(s)**
```
- 0 input guardrail result(s)
```
**What it is**: Results from input validation/safety checks  
**Your case**: 0 = No input guardrails configured  
**What they do**: Filter or block unsafe/inappropriate inputs before the agent processes them

**Example use cases**:
- Block prompts with PII (personal identifiable information)
- Prevent prompt injection attacks
- Validate input format

---

### 6. **0 output guardrail result(s)**
```
- 0 output guardrail result(s)
```
**What it is**: Results from output validation/safety checks  
**Your case**: 0 = No output guardrails configured  
**What they do**: Filter or block unsafe/inappropriate outputs before returning to user

**Example use cases**:
- Prevent leaking sensitive information
- Ensure outputs meet quality standards
- Block inappropriate content

---

## 🎯 Why These Matter

### For Debugging
- **new item(s)**: Too many items? Agent might be stuck in a loop
- **raw response(s)**: Multiple responses show the agent's thought process
- **guardrail results**: If non-zero, something was blocked/filtered

### For Optimization
- **new item(s)**: Fewer items = faster execution, lower cost
- **raw response(s)**: Each response costs API calls, optimize to reduce

### For Production
- **guardrails**: Essential for safety in production systems
- **Final output**: The validated result ready for use

---

## 📝 Accessing RunResult Properties

```python
result = await Runner.run(agent, input=messages)

# Access different parts:
print(result.output)                    # The final text output
print(result.final_response)            # Last agent response
print(result.agent)                     # Last agent that ran
print(result.to_input_list())           # Full conversation history
print(len(result.all_items))            # Count of all items

# Detailed breakdown:
for item in result.all_items:
    print(f"Type: {item.type}, Role: {item.role}")
```

---

## 🔄 Example: Multi-Agent Workflow

```python
# Agent 1 runs
r1 = await Runner.run(data_prep_agent, inputs)
print(r1)
# - 13 new item(s)  ← 1 user + 6 tools + 6 results + agent response
# - 2 raw response(s)

# Agent 2 receives handoff
r2 = await Runner.run(column_mapping_agent, r1.to_input_list())
print(r2)
# - 21 new item(s)  ← Previous 13 + new interactions
# - 3 raw response(s)  ← Previous 2 + new response
```

Each agent builds on the conversation history, adding more items!

---

## 🎨 Visualizing Your Execution

```
┌─────────────────────────────────────────────────────┐
│ 1. User Message (from Jinja2 template)             │
│    "# DATA PREPARATION TASK\nAnalyze 6 files..."   │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ 2. Agent Response 1 (First raw response)           │
│    Decides to call 6 tools                          │
│    ├─ Tool call: load_and_describe_dataset(file1)  │
│    ├─ Tool call: load_and_describe_dataset(file2)  │
│    ├─ Tool call: load_and_describe_dataset(file3)  │
│    ├─ Tool call: load_and_describe_dataset(file4)  │
│    ├─ Tool call: load_and_describe_dataset(file5)  │
│    └─ Tool call: load_and_describe_dataset(file6)  │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ 3. Tool Results (6 items)                           │
│    ├─ Result 1: {"file_path": "...", "shape": ...} │
│    ├─ Result 2: {"file_path": "...", "shape": ...} │
│    ├─ Result 3: {"file_path": "...", "shape": ...} │
│    ├─ Result 4: {"file_path": "...", "shape": ...} │
│    ├─ Result 5: {"file_path": "...", "shape": ...} │
│    └─ Result 6: {"file_path": "...", "shape": ...} │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ 4. Agent Response 2 (Second raw response)          │
│    Compiles results into final JSON array           │
│    "[{...}, {...}, {...}, {...}, {...}, {...}]"    │
│    "Data preparation complete - 6 files analyzed"  │
└─────────────────────────────────────────────────────┘

Total Items: 1 + 6 + 6 + 1 = ~13 items ✓
Total Raw Responses: 2 ✓
```

---

## 💡 Pro Tips

### 1. Debug Mode - See All Items
```python
result = await Runner.run(agent, input=messages)

print("\n=== Detailed Execution Trace ===")
for i, item in enumerate(result.all_items, 1):
    print(f"{i}. Type: {item.type}, Role: {getattr(item, 'role', 'N/A')}")
    if hasattr(item, 'content'):
        content_preview = str(item.content)[:100]
        print(f"   Content: {content_preview}...")
```

### 2. Extract Just the Text
```python
# Instead of printing the whole RunResult:
result = await Runner.run(agent, input=messages)
print(result.output)  # Just the final text output
```

### 3. Count Costs
```python
# More responses = more API calls = higher cost
print(f"Agent made {len(result.raw_responses)} API calls")
print(f"Total conversation items: {len(result.all_items)}")
```

---

## 🚀 Your Updated Script

Now when you run:
```bash
export AGENT_ROW_LIMIT=5
python scripts/examples/run_data_prep_agent.py
```

You'll see:
```
=== Running DataPrepAgent (Top 5 Rows) ===
```

Change it:
```bash
export AGENT_ROW_LIMIT=20
python scripts/examples/run_data_prep_agent.py
```

You'll see:
```
=== Running DataPrepAgent (Top 20 Rows) ===
```

---

## 📚 Further Reading

- OpenAI Agents SDK: [github.com/openai/openai-agents-sdk](https://github.com/openai/openai-agents-sdk)
- Guardrails: Add safety checks to your agents
- Streaming: Use `run_streamed()` to see responses in real-time

---

**TL;DR**:
- **RunResult** = Complete execution summary
- **new item(s)** = Conversation turns (user messages + tool calls + responses)
- **raw response(s)** = How many times the LLM generated output
- **guardrails** = Safety checks (you have none configured, which is fine for testing)
- **Final output** = The actual result you care about!

