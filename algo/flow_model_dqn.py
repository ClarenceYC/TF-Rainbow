from __future__ import annotations

import copy
import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import tree
import tensorflow as tf

# from drill import summary
# import drill.summary as summary
from drill.flow import flow
from drill import summary
from drill.keys import DECODER_MASK, ACTION, DONE
from drill.model import Model
from drill.utils import get_hvd

if TYPE_CHECKING:
    from drill.builder import Builder

from algo.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class FlowModelDQN(flow.Model):
    """FlowModel 是 [flow.api.Model](https://docs.inspir.work/flow/flow/tutorial.html#flow.api.Model) 一个实现。

    Attributes
    ----------
    model_name : str
        model 的名字
    builder: Builder
        详见 `drill.builder.Builder`
    """

    def __init__(self, model_name: str,
                 builder: Builder,
                 dqn_type='double_DQN',
                 buffer_size=1e+6,
                 batch_size=512,
                 learning_starts=0):
        self._init(model_name, builder, buffer_size, batch_size, learning_starts, dqn_type)

    def _init(self, model_name, builder: Builder,
              buffer_size=1e+6,
              batch_size=256,
              learning_starts=0,
              dqn_type='double_DQN'):
        self._model_name = model_name
        self._model: Model = builder.build_model(model_name)
        self._builder = builder
        self._learn_step = builder.learn_step
        hvd = get_hvd(builder.backend)
        if hvd.rank() == 0 and (model_name in builder.save_params):
            self._save_params = builder.save_params[model_name]
            Path(f'{self._save_params["path"]}/{self._model_name}').mkdir(parents=True,
                                                                          exist_ok=True)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = 0
        self.dqn_type = dqn_type

        self._epsilon = {
            'value': 1.0,
            'min': 0.05,
            'decrease': 0.05}
        self.freq_renew_target_net = 32
        self.train_times = 2

        # 添加经验池
        self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size)
        self.beta_schedule = None

        # update target at init
        weights = self._model._network.get_weights()
        self._model._target_network.set_weights(weights)

    @property
    @functools.lru_cache()
    def _is_tensorflow(self):
        return self._builder.backend == "tensorflow"

    def get_weights(self) -> List[np.ndarray]:
        """ 获取模型权重

        Returns
        -------
        List[np.ndarray]
            模型权重
        """
        if self._is_tensorflow:
            return self._model.network.get_weights()
        return [p.cpu().detach().numpy() for p in self._model.network.parameters()]

    def set_weights(self, weights: List[np.ndarray]):
        """ 设置模型权重

        Parameters
        ----------
        weights : List[np.ndarray]
            模型权重
        """
        if self._is_tensorflow:
            self._model.network.set_weights(weights)
        else:
            import torch
            for target_p, p in zip(self._model.network.parameters(), weights):
                target_p.copy_(torch.from_numpy(p))

    def save_weights(self, mode='npz'):
        """ 保存模型

        Parameters
        ----------
        mode : str, optional
            模型的格式，可以是 'npz' 或者 'tf-ckpt'， by default 'npz'
        """
        from drill.utils import save_model
        model_path = f'{self._save_params["path"]}/{self._model_name}/{self._model_name}_{self._learn_step}'
        save_model(self._model.network, model_path, self._builder.backend, mode)

    def load_weights(self, model_path: str, backend: str, mode='npz'):
        """ 加载模型

        Parameters
        ----------
        model_path : str
            模型路径，注意路径应指向具体模型文件，而不是其父级目录
        mode : str, optional
            模型的格式，可以是 'npz' 或者 'tf-ckpt'， 默认为 'npz'
        """
        from drill.utils import load_model
        load_model(self._model.network, model_path, backend, mode)

    def __getstate__(self):
        if self._is_tensorflow:
            return self._model_name, self._builder, self.get_weights()
        return self._model_name, self._builder, self._model.network.state_dict()

    def setstate_learn(self, state):
        model_name, builder, weights = state
        self._init(model_name, builder)

        if self._is_tensorflow:
            self.set_weights(weights)
        else:
            import torch
            self._model.network.load_state_dict(weights)
            if torch.cuda.is_available():
                self._model.network.cuda()

    def setstate_predict(self, state):
        model_name, builder, weights = state
        self._init(model_name, builder)

        if self._is_tensorflow:
            self.set_weights(weights)
        else:
            import torch
            self._model.network.load_state_dict(weights)
            self._model.network.requires_grad_(False)
            self._model.network.eval()
            if torch.cuda.is_available():
                self._model.network.cuda()

    def learn(self, piece: List[Dict[str, Any]]) -> bool:
        """ `FlowModel` 使用批量数据 piece 进行学习，训练模型

        Parameters
        ----------
        piece : List[Dict[str, Any]]
            由 state_dict, behavior_info_dict, decoder_mask, advantage 组成。

            * state_dict 包含 state， reward， done 等信息，还可能包含 hidden_state;
            * behavior_info_dict 包含 logits, action, value;
            * decoder_mask 包含 valid action。

        Returns
        -------
        bool
            是否将数据推送给 PredictorService
        """

        if hasattr(self, "_save_params") and self._learn_step % self._save_params["interval"] == 0:
            self.save_weights(self._save_params["mode"])
        self._learn_step += 1
        flow.summary.scalar('learning-step', self._learn_step)

        value = self._epsilon['value'] - self._epsilon['decrease']
        self._epsilon['value'] = max(value, self._epsilon['min'])

        flow.summary.scalar('epsilon_predict', self._epsilon['value'])

        hvd = get_hvd(self._builder.backend)
        if hvd.rank() == 0:
            summary.sum("learn_step", 1, source="origin")
        state_dict, behavior_info_dict, mask_dict, advantage = piece

        # behavior_info_dict.update(advantage)
        behavior_info_dict[DECODER_MASK] = mask_dict[DECODER_MASK]

        state_dict_ = copy.deepcopy(
            {'my_units': state_dict['my_units'], 'b_info': state_dict['b_info']})
        action_dict = copy.deepcopy(behavior_info_dict[ACTION])
        reward = copy.deepcopy(state_dict['reward_dqn'])
        next_state_dict = copy.deepcopy({'my_units': state_dict['nextobs_my_units'],
                                         'b_info': state_dict['nextobs_b_info']})
        done = copy.deepcopy(state_dict[DONE])

        for i in range(len(state_dict_)):
            self.replay_buffer.add((state_dict_['my_units'][i], state_dict_['b_info'][i]),
                                   (action_dict['action_x'][i], action_dict['action_y'][i],
                                    action_dict['action_dv'][i]),
                                   reward[i],
                                   (next_state_dict['my_units'][i], next_state_dict['b_info'][i]),
                                   done[i])

        if len(self.replay_buffer) > self.learning_starts:
            if self._learn_step % self.freq_renew_target_net == 0:
                weights_ = self._model.network.get_weights()
                self._model.target_network.set_weights(weights_)

            for _ in range(self.train_times):
                state_dict, behavior_info_dict, indices = self.replay_buffer.sample(self.batch_size)

                summary_dict = self._model.learn(state_dict, behavior_info_dict)

                if hasattr(self.replay_buffer, 'update_weight') and 'weight' in summary_dict:
                    weights = summary_dict.pop('weight')
                    self.replay_buffer.update_weight(indices, weights)
                if 'weight' in summary_dict:
                    summary_dict.pop('weight')

                for k, v in summary_dict.items():
                    summary.scalar(k, v, step=self._learn_step)
        return True

    def predict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """ `FlowModel` 进行前向预测

        Parameters
        ----------
        state_dict : Dict[str, Any]
            模型 inference 的输入

        Returns
        -------
        Dict[str, Any]
            模型 inference 的结果

        Examples
        --------
        state_dict
        ``` python
        {
            "spatial": np.ndarray,
            "entity": np.ndarray,
            "reward": array,
            "hidden_state": Any,
            ...
        }
        ```

        return
        ```python
        {
            "logits": {
                "x": np.ndarray,
                "y": np.ndarray
            },
            "action": {
                "x": np.ndarray,
                "y": np.ndarray
            }
            "value": np.ndarray,
            "hidden_state": np.ndarray
        }
        ```
        """

        predict_output_dict = self._model.predict(state_dict, tf.constant(self._epsilon['value']))
        output_dict = tree.map_structure(lambda x: x.numpy(), predict_output_dict)
        return output_dict
