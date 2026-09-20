"""
hastejev Autonomous Agent Control Loop
Demonstrates calibrated confidence gating (P >= 0.85) in mission-critical agent loops.
"""

from hastejev import HasteJevEngine

class AgentWorkflowRouter:
    def __init__(self, confidence_threshold: float = 0.85):
        self.engine = HasteJevEngine(d_model=256)
        self.confidence_threshold = confidence_threshold
        self.tools = [
            "Execute_Database_Migration",
            "Send_Customer_Email",
            "Trigger_Security_Lockdown",
            "Fetch_Analytics_Summary",
            "Escalate_To_Human_Supervisor"
        ]

    def handle_event(self, event_state: str):
        # 1. Evaluate Security Guardrail via Noul
        guardrail = self.engine.noul(event_state, "Proposed action is safe and complies with system security policy.")
        
        if not guardrail.is_true:
            return {
                "action": "Trigger_Security_Lockdown",
                "execution_mode": "BLOCKED_BY_GUARDRAIL",
                "confidence": guardrail.confidence,
                "reason": "Security assertion failed"
            }

        # 2. Select Tool via Choice
        choice = self.engine.choice(event_state, self.tools)
        top_tool = choice.decision
        top_prob = max(choice.probabilities.values())

        # 3. Confidence Gating
        if top_prob >= self.confidence_threshold:
            mode = "AUTONOMOUS_EXECUTION"
        else:
            mode = "ROUTED_TO_HUMAN_SUPERVISOR"
            top_tool = "Escalate_To_Human_Supervisor"

        return {
            "action": top_tool,
            "execution_mode": mode,
            "confidence": choice.confidence,
            "top_probability": top_prob
        }

if __name__ == "__main__":
    router = AgentWorkflowRouter(confidence_threshold=0.85)
    events = [
        "Unauthorized access attempt detected with multiple invalid admin tokens.",
        "Generate daily active user engagement report for marketing team.",
        "Malformed request received with conflicting parameter types."
    ]

    for i, event in enumerate(events, 1):
        result = router.handle_event(event)
        print(f"\n--- Event {i} ---")
        print(f"Input: {event}")
        print(f"Action: {result['action']} [{result['execution_mode']}]")
