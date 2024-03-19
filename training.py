import os
import torch

import matplotlib.pyplot as plt

import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.results_plotter import ts2xy, load_results
from stable_baselines3.common.callbacks import EvalCallback

import gymnasium as gym

env_name = 'LunarLander-v2'
env = gym.make(env_name)

nn_layers = [64, 64]

learning_rate = 0.001

log = "/LunarLander/"
os.makedirs(log, exist_ok=True)

env = stable_baselines3.common.monitor.Monitor(env, log)

callback = EvalCallback(env, log_path=log, deterministic=True)

policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=nn_layers)

model = DQN("MlpPolicy", env,policy_kwargs = policy_kwargs, learning_rate=learning_rate)

print('Nombre d\'états : ', env.observation_space.shape)
print('Nombre d\'actions : ', env.action_space.n)


env = gym.make(env_name, render_mode="rgb_array")
observation = env.reset()[0]

total_reward = 0
done = False
while not done:
  frame = env.render()
  action, states = model.predict(observation, deterministic=True)
  observation, reward, done, info, _ = env.step(action)
  total_reward += reward
env.close()
print(f"\nTotal reward: {total_reward}")

model.learn(total_timesteps=100000, log_interval=1, callback=callback, progress_bar=True)

model.save("model3.keras")

x, y = ts2xy(load_results(log), 'timesteps')
plt.plot(x, y)
plt.ylim([-1000, 500])
plt.xlabel('Timesteps')
plt.ylabel('Episode Rewards')
plt.show()