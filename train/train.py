import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, moving_average
from model import NatureDQN, Impala
from policy import Policy
import os

# Hyperparameters
total_steps = 8e6
num_envs = 32
num_levels = 10
num_steps = 256
num_epochs = 3
batch_size = 512
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01

# Define environment
# check the utils.py file for info on arguments
env = make_env(num_envs, num_levels=num_levels)
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)

# Define network
in_channels = env.observation_space.shape[0]
feature_dim = 512
num_actions = env.action_space.n

model = os.getenv('MODEL', False)

# if model == 'NatureDQN':
encoder = Impala(in_channels, feature_dim)

#if model == 'Impala':
 # encoder = Impala(in_channels, feature_dim)


policy = Policy(encoder, feature_dim, num_actions)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs
)

# Run training
obs = env.reset()
step = 0
rewards, losses = [], []

while step < total_steps:

  # Use policy to collect data for num_steps steps
  policy.eval()
  for _ in range(num_steps):
    # Use policy
    action, log_prob, value = policy.act(obs)
    
    # Take step in environment
    next_obs, reward, done, info = env.step(action)

    # Store data
    storage.store(obs, action, reward, done, info, log_prob, value)
    
    # Update current observation
    obs = next_obs

  # Add the last observation to collected data
  _, _, value = policy.act(obs)
  storage.store_last(obs, value)

  # Compute return and advantage
  storage.compute_return_advantage()

  # Optimize policy
  policy.train()
  for epoch in range(num_epochs):

    # Iterate over batches of transitions
    generator = storage.get_generator(batch_size)
    for batch in generator:
      b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

      # Get current policy outputs
      new_dist, new_value = policy(b_obs)
      new_log_prob = new_dist.log_prob(b_action)

      loss = policy.calculate_loss(b_log_prob, b_returns, b_advantage, new_log_prob, new_dist, new_value, eps, value_coef, entropy_coef)
      loss.mean().backward()

      # Clip gradients
      torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

      # Update policy
      losses.append(loss.mean().item())
      optimizer.step()
      optimizer.zero_grad()

  # Update stats
  step += num_envs * num_steps
  print(f'Step: {step}\tMean reward: {storage.get_reward()}')
  rewards.append(storage.get_reward())

print('Completed training!')
torch.save(policy.state_dict(), 'checkpoint.pt')

# plot results

plt.figure(figsize=(16,6))
plt.subplot(211)
plt.plot(range(1, len(rewards)+1), moving_average(rewards), label='training reward')
plt.xlabel('episode'); plt.ylabel('reward')
plt.xlim((0, len(rewards)))
plt.legend(loc=4); plt.grid()
plt.subplot(212)
plt.plot(range(1, len(losses)+1), moving_average(losses), label='loss')
plt.xlabel('episode'); plt.ylabel('loss')
plt.xlim((0, len(losses)))
plt.legend(loc=4); plt.grid()
plt.tight_layout(); plt.show()
