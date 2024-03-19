import gymnasium as gym

env = gym.make("LunarLander-v2", render_mode = "human")

observation = env.reset()

terminated = truncated = False

total_reward = 0
while (not terminated and not truncated):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

print(f"\nTotal reward: {total_reward}")

env.close()