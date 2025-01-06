# auther: Jianzhuang Li
# date: 2025.01.02
# describe: A wrapper of d4rl  hopper dataset
import  gym
import d4rl
import numpy as np
import torch.utils.data as Data

HOPPER_DATASET_LIST =   ['hopper-random-v0', 'hopper-random-v2', 'hopper-medium-v0', 'hopper-medium-v2', 'hopper-expert-v0', 'hopper-expert-v2']
class HopperDataSet(object):
    
    def __init__(self, dataset_conf):
        self.dataset_conf = dataset_conf
        self.dataset_id = self.dataset_conf.dataset_id
        assert self.dataset_id in HOPPER_DATASET_LIST
        self._env = gym.make(self.dataset_id)

        # d4rl abides by the OpenAI gym interface
        self._env.reset()
        self._env.step(self._env.action_space.sample())
        self._len = None
        self._dataset = d4rl.qlearning_dataset(self._env)
        self._keys = [k for k in self._dataset.keys()]
        
    def __getitem__(self, index):
        item = {}
        for key in self._keys:
            item[key] = self._dataset[key][index]
        return (item["observations"], item["actions"])

    def __len__(self):
        if self._len is None:
            self._len = self._dataset['observations'].shape[0]
        return self._len

class HopperDataSetWithNext(HopperDataSet):

    def __init__(self, dataset_conf):
        super().__init__(dataset_conf)

    def __getitem__(self, index):
        item = {}
        for key in self._keys:
            item[key] = self._dataset[key][index]
        termials = np.float32(1.0) if item["terminals"] else np.float32(0.0)
        return (item["observations"], item["actions"], item["rewards"], termials, item["next_observations"])
    
if __name__ == "__main__":
    # data = HopperDataSet()
    data = HopperDataSetWithNext()
    print(data[0])