import numpy as np
import gym

class DiscreteMountainCar():
    def __init__(self):
        self.env = gym.make("MountainCar-v0")
        """
            observation :   position :  ( -1.2,  0.6)
                            velocity :  (-0.07, 0.07)
            action      :   Discrete(3)    :  [0, 1, 2]
            reward      :
        """
        self.observation_list = [
            np.linspace(-1.2, 0.6, 17),  # 16 segments for position
            np.linspace(-0.07, 0.07, 17)  # 16 segments for velocity
        ]
        self.observation_space = 2 * 16
        self.action_space = 3
        return

    def observation_from_consecutive_to_discrete(self, observation):
        obs = np.zeros(self.observation_space, dtype="uint8")
        for i in range(2):  # len(observation)
            for j in range(16):  # 16 segments for discretization
                if self.observation_list[i][j] <= observation[i] < self.observation_list[i][j + 1]:
                    obs[i * 16 + j] = 1
        return obs

    def get_observation_space(self):
        return self.observation_space

    def action_from_discrete_to_consecutive(self, action):
        """Convert discrete action to actual environment action (in this case, they are the same)"""
        return np.array([action], dtype=float)

    def get_action_space(self):
        return self.action_space

    def reset(self):
        obs = self.env.reset()
        return self.observation_from_consecutive_to_discrete(obs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.observation_from_consecutive_to_discrete(observation)
        return observation, reward, done, info

    def render(self,mode='rgb_array'):
    #def render(self):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()
