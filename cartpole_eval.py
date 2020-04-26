import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import Progressive_neural_net
import gym
from env_wrapper import reverse_action_wrapper, reverse_observation_wrapper


if __name__ == '__main__':
    eval_episode = 10
    render = False

    net = torch.load('./model.pkl')
    env_list = [reverse_observation_wrapper(gym.make('CartPole-v0')), gym.make('CartPole-v0'), reverse_action_wrapper(reverse_observation_wrapper(gym.make('CartPole-v0'))), reverse_action_wrapper(gym.make('CartPole-v0'))]
    env_indice = [0, 1, 2, 3, 2, 0, 1]
    task_indice = [0, 1, 2, 3, 1, 3, 2]
    for task_id, env_id in zip(task_indice, env_indice):
        env = env_list[env_indice[env_id]]
        weight_reward = None
        for i in range(eval_episode):
            obs = env.reset()
            total_reward = 0
            if render:
                env.render()
            while True:
                obs_tensor = torch.FloatTensor(np.expand_dims(obs, 0))
                q_value = net.forward(obs_tensor, task_id)
                action = q_value.max(1)[1].data[0].item()
                next_obs, reward, done, info = env.step(action)
                total_reward += reward
                obs = next_obs
                if render:
                    env.render()
                if done:
                    if not weight_reward:
                        weight_reward = total_reward
                    else:
                        weight_reward = 0.9 * weight_reward + 0.1 * total_reward
                    break
        print('task: {}  env: {}  weight_reward: {:.3f}'.format(task_id, env_id, weight_reward))