import openai
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# ----------- SETUP OPENAI API KEY -----------
openai.api_key = "your-api-key-here"  # Replace with your key

# ----------- SYNTHETIC PATIENT DATA -----------
def generate_patient_data(n=100):
    return pd.DataFrame([
        [np.random.randint(140, 250), np.random.randint(60, 100), 0]
        for _ in range(n)
    ], columns=['glucose_level', 'weight', 'medication_status'])

# ----------- SIMULATED PATIENT RESPONSE -----------
def simulate_treatment(state, action):
    glucose, weight, med = state
    reward = 0

    if action == 0:
        glucose += np.random.randint(-5, 10)
        reward -= 1
    elif action == 1:
        glucose -= np.random.randint(10, 20)
        med = 1
        reward += 1
    elif action == 2:
        glucose -= np.random.randint(20, 40)
        med = 2
        reward += 2
    elif action == 3:
        glucose -= np.random.randint(5, 15)
        weight -= np.random.randint(1, 3)
        reward += 1

    if glucose < 70:
        reward -= 5

    glucose = max(50, min(300, glucose))
    weight = max(45, min(150, weight))

    return (glucose, weight, med), reward

# ----------- DISCRETIZATION FUNCTION -----------
def discretize_state(state):
    glucose_bin = state[0] // 20
    weight_bin = state[1] // 10
    return (glucose_bin, weight_bin, state[2])

# ----------- GPT-BASED PLAN GENERATOR -----------
def generate_treatment_plan(patient_info):
    prompt = f"""
A patient has the following:
- Glucose level: {patient_info[0]} mg/dL
- Weight: {patient_info[1]} kg
- Current medication: {['None', 'Low Dose', 'High Dose'][patient_info[2]]}

Suggest a medically safe 1-week treatment plan for managing Type 2 Diabetes.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or gpt-4
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print("Error with GPT:", e)
        return "No clear recommendation."

# ----------- PLAN PARSER TO ACTION -----------
def parse_plan_to_action(plan_text):
    text = plan_text.lower()
    if "metformin" in text:
        if "500" in text or "low" in text:
            return 1
        elif "1000" in text or "high" in text:
            return 2
    elif "exercise" in text or "lifestyle" in text:
        return 3
    else:
        return 0

# ----------- RL TRAINING LOOP -----------
def train_agent(patient_data, episodes=100):
    actions = [0, 1, 2, 3]
    action_names = ["No Treatment", "Low Dose", "High Dose", "Lifestyle"]
    q_table = defaultdict(lambda: [0.0] * len(actions))

    alpha, gamma, epsilon = 0.1, 0.9, 0.3
    reward_history = []

    for ep in range(episodes):
        patient = patient_data.sample(1).values[0]
        state = tuple(patient)
        total_reward = 0

        for step in range(5):
            disc_state = discretize_state(state)

            # Get GPT plan
            plan_text = generate_treatment_plan(state)
            action = parse_plan_to_action(plan_text)

            # ε-greedy override
            if random.random() < epsilon:
                action = random.choice(actions)

            next_state, reward = simulate_treatment(state, action)
            disc_next = discretize_state(next_state)
            best_next = max(q_table[disc_next])

            q_table[disc_state][action] += alpha * (
                reward + gamma * best_next - q_table[disc_state][action]
            )

            state = next_state
            total_reward += reward
            time.sleep(1.5)  # avoid OpenAI rate limits

        reward_history.append(total_reward)
        print(f"Episode {ep+1} | Reward: {total_reward}")

    return q_table, reward_history, action_names

# ----------- RUN TRAINING -----------
patient_data = generate_patient_data(50)
q_table, reward_history, action_names = train_agent(patient_data, episodes=20)

# ----------- PLOT LEARNING CURVE -----------
plt.figure(figsize=(10, 5))
plt.plot(reward_history)
plt.title("Learning Curve (RL + GPT-generated plans)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------- SAMPLE INFERENCE -----------
sample_state = (180, 85, 0)
disc_sample = discretize_state(sample_state)
best_action = np.argmax(q_table[disc_sample])

print("Sample State:", sample_state)
print("Recommended Action:", action_names[best_action])