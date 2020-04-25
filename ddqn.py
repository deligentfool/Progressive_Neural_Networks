import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from replay_buffer import replay_buffer


def train(buffer, target_model, eval_model, gamma, optimizer, batch_size, loss_fn, count, soft_update_freq):
    observation, action, reward, next_observation, done = buffer.sample(batch_size)

    observation = torch.FloatTensor(observation)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    next_observation = torch.FloatTensor(next_observation)
    done = torch.FloatTensor(done)

    q_values = eval_model.forward(observation)
    next_q_values = target_model.forward(next_observation)
    argmax_actions = eval_model.forward(next_observation).max(1)[1].detach()
    next_q_value = next_q_values.gather(1, argmax_actions.unsqueeze(1)).squeeze(1)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * (1 - done) * next_q_value

    #loss = loss_fn(q_value, expected_q_value.detach())
    loss = (expected_q_value.detach() - q_value).pow(2)
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if count % soft_update_freq == 0:
        target_model.load_state_dict(eval_model.state_dict())