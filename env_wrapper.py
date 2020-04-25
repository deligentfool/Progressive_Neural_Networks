import gym


class reverse_action_wrapper(gym.ActionWrapper):
    def action(self, action):
        return not action


class reverse_observation_wrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return [- obs for obs in observation]