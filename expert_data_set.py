import numpy as np

class ExpertDataset:

    def __init__(self, original_dataset, size, shuffle=True):
        self.obs = None
        self.action = None
        self.reward = None
        self.next_obs = None
        self.done = None
        self.size = size
        self.shuffle = shuffle
        self._convert(original_dataset)
        self.indices = []
        self._reset_indices()

    def _convert(self, original_dataset):
        for i in range(self.size):

            info = original_dataset.memory[i][0]

            obs = info['state'].reshape([1, -1])
            action = info['action'].reshape([1, -1])
            next_obs = info['next_state'].reshape([1, -1])
            reward = np.asarray([info['reward']])
            done = np.asarray([info['is_state_terminal']])

            self.obs = (
                obs if self.obs is None else
                np.concatenate((self.obs, obs)))
            self.action = (
                action if self.action is None else
                np.concatenate((self.action, action)))
            self.reward = (
                reward if self.reward is None else
                np.concatenate((self.reward, reward)))
            self.next_obs = (
                next_obs if self.next_obs is None else
                np.concatenate((self.next_obs, next_obs)))
            self.done = (
                done if self.done is None else
                np.concatenate((self.done, done)))

    def _reset_indices(self):
        if self.shuffle:
            self.indices = np.random.permutation(np.arange(self.size))
        else:
            self.indices = np.arange(self.size)

    def sample(self):
        index = self.indices[0]
        obs = self.obs[index]
        action = self.action[index]
        reward = self.reward[index]
        next_obs = self.next_obs[index]
        done = self.done[index]
        self.indices = np.delete(self.indices, 0)
        if len(self.indices) == 0:
            self._reset_indices()
        return obs, action, reward, next_obs, done