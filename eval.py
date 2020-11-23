import imageio
from train.utils import make_env
import torch
from train.model import Policy, Encoder

# Hyperparameters
num_envs = 32
num_levels = 10

# Make evaluation environment
eval_env = make_env(num_envs, start_level=num_levels, num_levels=num_levels)
obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
in_channels = eval_env.observation_space.shape[0]
feature_dim = 512
num_actions = eval_env.action_space.n

encoder = Encoder(in_channels, feature_dim)
policy = Policy(encoder, feature_dim, num_actions)
policy.load_state_dict(torch.load('checkpoint.pt'))
policy.eval()

for _ in range(512):

  # Use policy
  action, log_prob, value = policy.act(obs)

  # Take step in environment
  obs, reward, done, info = eval_env.step(action)
  total_reward.append(torch.Tensor(reward))

  # Render environment and store
  frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
  frames.append(frame)

# Calculate average return
total_reward = torch.stack(total_reward).sum(0).mean(0)
print('Average return:', total_reward)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave('vid.mp4', frames, fps=25)