import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import gym
from replay_buffer import replay_buffer
from ddqn import train
from model import Progressive_neural_net
from env_wrapper import reverse_action_wrapper, reverse_observation_wrapper


if __name__ == '__main__':
    gamma = 0.99
    learning_rate = 1e-3
    batch_size = 64
    soft_update_freq = 100
    capacity = 10000
    exploration = 200
    epsilon_init = 0.99
    epsilon_min = 0.2
    decay = 0.998
    episode = 1000000
    render = False
    threshold_reward = 80

    env_list = [reverse_observation_wrapper(gym.make('CartPole-v0')), gym.make('CartPole-v0'), reverse_action_wrapper(reverse_observation_wrapper(gym.make('CartPole-v0'))), reverse_action_wrapper(gym.make('CartPole-v0'))]
    observation_dim = env_list[0].observation_space.shape[0]
    action_dim = env_list[0].action_space.n
    target_net = Progressive_neural_net(3)
    eval_net = Progressive_neural_net(3)
    eval_net.load_state_dict(target_net.state_dict())
    buffer = replay_buffer(capacity)
    loss_fn = nn.MSELoss()
    sizes = [observation_dim, 64, 32, action_dim]

    for env_idx, env in enumerate(env_list):
        count = 0
        epsilon = epsilon_init
        eval_net.add_new_column(sizes)
        target_net.add_new_column(sizes)
        target_net.load_state_dict(eval_net.state_dict())
        optimizer = torch.optim.Adam(eval_net.parameters(env_idx), lr=learning_rate)
        weight_reward = None
        for i in range(episode):
            obs = env.reset()
            if epsilon > epsilon_min:
                epsilon = epsilon * decay
            reward_total = 0
            if render:
                env.render()
            while True:
                obs_tensor = torch.FloatTensor(np.expand_dims(obs, 0))
                if random.random() > epsilon:
                    q_value = eval_net.forward(obs_tensor)
                    action = q_value.max(1)[1].data[0].item()
                else:
                    action = random.choice(list(range(action_dim)))

                count += 1
                next_obs, reward, done, info = env.step(action)
                buffer.store(obs, action, reward, next_obs, done)
                reward_total += reward
                obs = next_obs
                if render:
                    env.render()
                if i > exploration:
                    train(buffer, target_net, eval_net, gamma, optimizer, batch_size, loss_fn, count, soft_update_freq)

                if done:
                    if not weight_reward:
                        weight_reward = reward_total
                    else:
                        weight_reward = 0.99 * weight_reward + 0.01 * reward_total
                    print('task: {}  episode: {}  epsilon: {:.2f}  reward: {}  weight_reward: {:.3f}'.format(env_idx, i+1, epsilon, reward_total, weight_reward))
                    break
            if weight_reward >= threshold_reward:
                buffer.clear()
                eval_net.freeze_columns()
                break
    torch.save(eval_net, './model.pkl')