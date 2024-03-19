import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("LunarLander-v2", render_mode="human")

model = DQN.load("models/model2.keras")

observation = env.reset()[0]

terminated = truncated = False

total_reward = 0
while not terminated and not truncated:
    action, states = model.predict(observation, deterministic=True)

    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

print(f"\nTotal reward: {total_reward}")

env.close()
