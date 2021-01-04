import torch
import torch.nn as nn
from utils import orthogonal_init

class Policy(nn.Module):
  def __init__(self, encoder, feature_dim, num_actions):
    super().__init__()
    self.encoder = encoder
    self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

  def act(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)
    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def forward(self, x):
    x = self.encoder(x)
    logits = self.policy(x)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value

  def calculate_loss(self, b_log_prob, b_returns, b_advantage, new_log_prob, new_dist, new_value, eps, value_coef, entropy_coef):
    # Clipped policy objective
      ratio = torch.exp(new_log_prob - b_log_prob)
      p1 = ratio * b_advantage
      p2 = torch.clamp(ratio,1 - eps, 1 + eps) * b_advantage
      pi_loss = - torch.min(p1,p2)

      # Clipped value function objective
      loss_function = nn.MSELoss()
      value_loss = value_coef * loss_function(new_value, b_returns)

      # Entropy loss
      entropy_loss = - entropy_coef * new_dist.entropy()

      # Backpropagate losses
      loss = pi_loss + value_loss + entropy_loss

      return loss