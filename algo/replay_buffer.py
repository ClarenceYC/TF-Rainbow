from typing import Any, List, Optional, Tuple, Union

import numpy as np
import random
import tensorflow as tf

from drill.keys import DECODER_MASK, ACTION, DONE
from algo.utils.segtree import SegmentTree
from algo.utils.converter import to_numpy


class ReplayBuffer(object):
    def __init__(self, size, **kwargs):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        if isinstance(size, float):
            size = int(size)
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        o1, o2, a1, a2, a3, r, no1, no2, d = [], [], [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            o1_, o2_ = obs_t
            a1_, a2_, a3_ = action
            no1_, no2_ = obs_tp1

            o1.append(np.array(o1_, copy=False))
            o2.append(np.array(o2_, copy=False))

            a1.append(np.array(a1_, copy=False))
            a2.append(np.array(a2_, copy=False))
            a3.append(np.array(a3_, copy=False))

            r.append(reward)

            no1.append(np.array(no1_, copy=False))
            no2.append(np.array(no2_, copy=False))

            d.append(done)

        state_dict = {
            'my_units': np.array(o1),
            'b_info': np.array(o2),
            'reward_dqn': np.array(r),
            'done_dqn': np.array(d),
            'nextobs_my_units': np.array(no1),
            'nextobs_b_info': np.array(no2)
        }
        if hasattr(self, 'get_weight'):
            idxes = np.array(idxes)
            weight = self.get_weight(idxes)
            weight = weight / np.max(weight) if hasattr(self, '_weight_norm') and self._weight_norm else weight
            state_dict.update({'weight': weight})
        behavior_info_dict = {
            ACTION: {
                'action_x': np.array(a1),
                'action_y': np.array(a2),
                'action_dv': np.array(a3)
            }
        }
        return state_dict, behavior_info_dict, idxes

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
            self,
            size: int,
            alpha: float = 1.0,
            beta: float = 1.0,
            weight_norm: bool = True,
            **kwargs: Any
    ):
        ReplayBuffer.__init__(self, size, **kwargs)
        assert alpha > 0.0 and beta >= 0.0
        self._alpha, self._beta = alpha, beta
        self._max_prio = self._min_prio = 1.0

        self.weight = SegmentTree(size)
        self.__eps = np.finfo(np.float32).eps.item()
        self._weight_norm = weight_norm

    def init_weight(self, index: Union[int, np.ndarray]) -> None:
        self.weight[index] = self._max_prio**self._alpha

    def add(self, obs_t, action, reward, obs_tp1, done):
        self.init_weight(self._next_idx)
        super().add(obs_t, action, reward, obs_t, done)

    def get_weight(self, index: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Get the importance sampling weight.

        The "weight" in the returned Batch is the weight on loss function to debias
        the sampling process (some transition tuples are sampled more often so their
        losses are weighted less).
        """
        # important sampling weight calculation
        # original formula: ((p_j/p_sum*N)**(-beta))/((p_min/p_sum*N)**(-beta))
        # simplified formula: (p_j/p_min)**(-beta)
        return (self.weight[index] / self._min_prio) ** (-self._beta)

    def update_weight(
        self, index: np.ndarray, new_weight: Union[np.ndarray, tf.Tensor]
    ) -> None:
        """Update priority weight by index in this buffer.

        :param np.ndarray index: index you want to update weight.
        :param np.ndarray new_weight: new priority weight you want to update.
        """
        weight = np.abs(to_numpy(new_weight)) + self.__eps
        self.weight[index] = weight**self._alpha
        self._max_prio = max(self._max_prio, weight.max())
        self._min_prio = min(self._min_prio, weight.min())

    def __getitem__(self, index: Union[slice, int, List[int], np.ndarray]):
        pass
