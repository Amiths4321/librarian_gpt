import json
import re

# =========================================================================
# 🛠️ 1. DEFINING THE AVAILABLE TOOLS
# =========================================================================
def check_mongodb_status(form_id: str) -> str:
    """Checks the validation status of an Account Opening Form in the database."""
    database = {
        "AOF-9921": {"status": "FLAGGED", "reason": "Blurry signature image match"},
        "AOF-4402": {"status": "VERIFIED", "destination": "Active Production Pipeline"}
    }
    record = database.get(form_id)
    if record:
        return f"Database Record Found: {json.dumps(record)}"
    return f"Error: Form ID '{form_id}' not found in MongoDB instance."

def request_manual_review(form_id: str) -> str:
    """Escalates a flagged form to the back-office human engineering team."""
    return f"Success: Escalation ticket generated for Form '{form_id}'. Sent to manual review pool."

# Map tool names directly to their execution pointers
TOOL_REGISTRY = {
    "check_mongodb_status": check_mongodb_status,
    "request_manual_review": request_manual_review
}

SYSTEM_INSTRUCTIONS = (
    "You are an AI Agent with tool access. You solve tasks using a step-by-step loop: Thought, Action, Observation.\n\n"
    "Available Tools:\n"
    "- check_mongodb_status(form_id: str): Returns form status.\n"
    "- request_manual_review(form_id: str): Escalates form to humans.\n\n"
    "Format your responses exactly like this:\n"
    "Thought: Reason through what needs to be done next.\n"
    "Action:\n"
    "```json\n"
    '{"tool_name": "name", "parameters": {"param_name": "value"}}\n'
    "```\n\n"
    "When you get an 'Observation:' response, evaluate it. If the problem is solved, output:\n"
    "Final Answer: [Your clear human-readable summary response]"
)

# =========================================================================
# 🤖 2. MOCK LLM SIMULATOR ENGINE
# =========================================================================
class MockLLMEngine:
    """Simulates how a structured LLM responds to historical prompts over a session loop."""
    def __init__(self):
        self.step = 0

    def generate(self, complete_prompt_history: str) -> str:
        self.step += 1
        if self.step == 1:
            return """Thought: The user wants to know why form AOF-9921 is stuck. I should check its current status in MongoDB.
Action:
```json
{"tool_name": "check_mongodb_status", "parameters": {"form_id": "AOF-9921"}}
```"""
        elif self.step == 2:
            return """Thought: The database indicates the form is FLAGGED due to a blurry signature. I must escalate this to the human engineering review pool to unblock the client.
Action:
```json
{"tool_name": "request_manual_review", "parameters": {"form_id": "AOF-9921"}}
```"""
        else:
            return """Thought: The manual review ticket was successfully submitted. I can now deliver my final conclusion to the user.
Final Answer: Form AOF-9921 was stuck in the system because it was flagged for a blurry signature mismatch. I have successfully escalated this file to the back-office engineering team for human review."""

# =========================================================================
# 🚀 3. THE AGENT RUNTIME EXECUTION LOOP
# =========================================================================
class ReActAgent:
    def __init__(self):
        self.llm = MockLLMEngine()

    def run(self, user_goal: str, max_steps: int = 5):
        print(f"🚀 Starting Autonomous Agent Session...")
        print(f"🎯 Goal: {user_goal}\n" + "="*70)
        
        session_history = f"{SYSTEM_INSTRUCTIONS}\nUser Goal: {user_goal}\n"

        for step in range(1, max_steps + 1):
            print(f"\n🧠 [STEP {step}] Calling LLM Brain...")
            llm_response = self.llm.generate(session_history)
            print(llm_response)
            
            if "Final Answer:" in llm_response:
                print("\n" + "="*70 + "\n✅ AGENT EXECUTION COMPLETED SUCCESSFULLY!")
                return
            
            json_match = re.search(r"```json\s*(.*?)\s*```", llm_response, re.DOTALL)
            if json_match:
                try:
                    action_data = json.loads(json_match.group(1).strip())
                    tool_name = action_data.get("tool_name")
                    params = action_data.get("parameters", {})
                    
                    if tool_name in TOOL_REGISTRY:
                        print(f"\n⚙️ [Executing Tool: {tool_name} with params {params}]")
                        observation = TOOL_REGISTRY[tool_name](**params)
                    else:
                        observation = f"Error: Tool '{tool_name}' doesn't exist in registry."
                        
                except Exception as e:
                    observation = f"Error parsing action JSON block: {str(e)}"
            else:
                observation = "Error: Action block was missing or improperly formatted."

            print(f"👁️ [Observation: {observation}]")
            session_history += f"\n{llm_response}\nObservation: {observation}\n"

        print(f"\n❌ Execution Halting: Maximum limit of {max_steps} steps reached without terminal answer.")


# =========================================================================
# 🔥 ENTRYPOINT (Ensuring this runs explicitly)
# =========================================================================
if __name__ == "__main__":
    agent = ReActAgent()
    agent.run("Check why form AOF-9921 is stuck and fix it.")