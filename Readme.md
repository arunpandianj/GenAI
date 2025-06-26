Project Title: Personalized Treatment Plan Generation using Generative AI Guided by Reinforcement Learning

Overview:
This project demonstrates a proof-of-concept system that combines Reinforcement Learning (RL) with Generative AI (via OpenAI GPT models) to generate and optimize personalized treatment plans for patients with Type 2 Diabetes.
1. The RL agent learns which treatment strategies lead to the best patient health outcomes.
2. The GPT model acts as a medical assistant, generating treatment plans in natural language.
3. The agent uses those suggestions and improves its choices over multiple patient interactions.

Motivation:
Traditional treatment planning in chronic diseases like diabetes is often rule-based or reactive. With AI integration, we aim to:
1. Simulate personalized patient responses to treatments
2. Incorporate medically sound language from LLMs (like GPT)
3. Use reinforcement learning to adapt and choose the best long-term treatment policy

Components:
1. Synthetic Patient Dataset
    Generates patients with features:
        glucose_level (140–250 mg/dL)
        weight (60–100 kg)
        medication_status: 0 (none), 1 (low dose), 2 (high dose)
2. Treatment Plan Generator (GPT)
    Given a patient profile, GPT generates a treatment plan.
    Example GPT output:
        "Begin with 500mg Metformin once daily and encourage 30 minutes of daily walking."
3. Plan-to-Action Parser
    Converts GPT suggestions to discrete RL actions:
        0: No treatment
        1: Low dose medication
        2: High dose medication
        3: Lifestyle advice
4. Patient Simulator
    Applies the chosen treatment to a patient state and returns:
        New glucose level and weight
        A reward representing health improvement
5. Q-learning Agent
    Learns optimal actions by maximizing long-term rewards.
    Uses ε-greedy exploration, Q-table updates, and state discretization.

Workflow:
    A synthetic patient is sampled.
    GPT generates a textual treatment plan.
    The plan is parsed into an RL-compatible action.
    The treatment is applied in a simulation environment.
    The Q-learning agent observes the reward and updates its policy.
    Over many episodes, the agent learns which GPT-generated plans are most effective.

Output:
    A learning curve shows the total reward per episode.
    A sample decision is displayed:
        Input: (glucose = 180, weight = 85, no medication)
        Output: Recommended action → “Low Dose”
