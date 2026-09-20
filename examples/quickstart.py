"""
hastejev Quickstart Example
Demonstrates the 5 native System-1 decision primitives.
"""

from hastejev import HasteJevEngine

def main():
    print("🚀 Initializing hastejev Decision Engine...")
    engine = HasteJevEngine(d_model=256)

    # 1. Choice Primitive (Categorical selection with entropy confidence)
    print("\n--- 1. Choice Primitive ---")
    state = "Customer with account balance $24,500 submitted loan application on 2026-03-20."
    options = ["Auto-Approve", "Require Manual Underwriting", "Request Additional Proof", "Auto-Decline"]
    choice_res = engine.choice(state, options)
    print(f"State: {state}")
    print(f"Decision: {choice_res.decision} (Confidence: {choice_res.confidence:.3f})")
    print(f"Probabilities: {choice_res.probabilities}")

    # 2. Score Primitive (Ordinal rubric expectation)
    print("\n--- 2. Score Primitive ---")
    rubric = ["Poor", "Fair", "Good", "Excellent"]
    score_res = engine.score(state, rubric)
    print(f"Expectation Score: {score_res.expectation_score:.2f} / {score_res.max_tier}")

    # 3. Noul Primitive (Calibrated Boolean truth assertion)
    print("\n--- 3. Noul Primitive ---")
    assertion = "Account balance exceeds $20,000 threshold."
    noul_res = engine.noul(state, assertion)
    print(f"Assertion: '{assertion}'")
    print(f"Is True: {noul_res.is_true} (Probability: {noul_res.probability:.3f})")

    # 4. Range Primitive (Continuous scalar estimation with 95% CI)
    print("\n--- 4. Range Primitive ---")
    range_res = engine.range_eval(state, "Estimated Credit Risk")
    print(f"Estimated Value: {range_res.estimated_value:.2f}")
    print(f"95% Confidence Interval: [{range_res.confidence_interval_95[0]:.2f}, {range_res.confidence_interval_95[1]:.2f}]")

    # 5. SetChoice Primitive (Multi-label subset selection)
    print("\n--- 5. SetChoice Primitive ---")
    tags = ["High-Value", "VIP", "Needs-Verification", "Fraud-Suspect", "Fast-Track"]
    set_res = engine.set_choice(state, tags, threshold=0.5)
    print(f"Selected Subset: {set_res.selected_subset}")
    print(f"Marginal Probabilities: {set_res.marginal_probabilities}")

if __name__ == "__main__":
    main()
