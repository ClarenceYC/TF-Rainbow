from typing import Any, Dict

import tensorflow as tf
import numpy as np

from drill.keys import ACTION, LOGITS
from drill.model.model import Model
from drill.utils import get_hvd

# from algo.utils.converter import to_tensor_as


def filter_by_prefix(inputs_dict, prefix):
    res_dict = {}
    for key, value in inputs_dict.items():
        if key.startswith(prefix):
            res_dict[key] = value
    return res_dict


class DQNModel(Model):
    """Drill 根据 RL 的使用场景提供的一个 Model 的实现
    """

    def __init__(self,
                 network,
                 target_network,
                 learning_rate: float = 1e-3,
                 gamma=0.99,
                 dqn_type='DQN'):
        self._network = network
        self._target_network = target_network
        self.gamma = tf.constant([gamma], dtype=tf.float32)

        self._DQN_type = dqn_type
        self.double_q = False
        if self._DQN_type == 'double_DQN':
            self.double_q = True

        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        hvd = get_hvd("tensorflow")
        hvd.init()

        # self.update_target_network()

    @property
    def network(self):
        return self._network

    @property
    def target_network(self):
        return self._target_network

    def _compute_q_value(self, logits, mask=None):
        """ Future version abstraction
        Compute the q value based on the network's raw output and action mask.
        """
        if mask is not None:
            # the masked q value should be smaller than logits.min()
            min_value = tf.math.reduce_min(logits) - tf.math.reduce_min(logits) - 1.0
            logits = logits + (1 - mask) * min_value
        return logits

    @tf.function
    def predict(self, state_dict: Dict[str, Any], epsilon=0) -> Dict[str, Any]:
        """ `Agent` 根据 `Env` 返回的 state, 做出决策

        Parameters
        ----------
        state_dict : Dict[str, Any]
            包括 state, reward, hidden_state 等信息，例如:
            ```python
            {
                "spatial": np.ndarray,
                "entity": np.ndarray,
                "reward": array,
                "hidden_state": Any,
                ...
            }
            ```

        Returns
        -------
        Dict[str, Any]
            predict_output_dict, 包括 logits, action, value, hidden_state
        """
        predict_output_dict = self._network(state_dict)
        logits = predict_output_dict[LOGITS]

        return {
            ACTION: {"action_x": self.argmax_sample(logits['action_x'], epsilon),
                     "action_y": self.argmax_sample(logits['action_y'], epsilon),
                     "action_dv": self.argmax_sample(logits['action_dv'], epsilon)},
            # 'max_action': {"action_x": self.argmax_sample(logits['action_x'], 0.),
            #                "action_y": self.argmax_sample(logits['action_y'], 0.),
            #                "action_dv": self.argmax_sample(logits['action_dv'], 0.)},
        }

    # @tf.function
    def learn(self, inputs_dict: Dict[str, Any], behavior_info_dict: Dict[str,
    Any]) -> Dict[str, Any]:
        """使用与环境交互获得的 rollout 数据来进行训练

        Parameters
        ----------
        inputs_dict : Dict[str, Any]
            包含 state, reward, hidden_state 等信息，例如:
            ```python
            {
                "spatial": np.ndarray,
                "entity": np.ndarray,
                "reward": array,
                "hidden_state": Any,
                ...
            }
            ```
        behavior_info_dict : Dict[str, Any]
            一个包含 logits, action, value, advantage, decoder_mask 的
            Dict, 是 behavior network 的输出。

        Returns
        -------
        Dict[str, Any]
            训练过程中产生的一些统计数据，比如 loss, entropy, kl 等
        """

        reward = inputs_dict['reward_dqn']
        action_index = behavior_info_dict[ACTION]
        dones = inputs_dict['done_dqn']

        with tf.GradientTape() as tape:
            tape.watch(self._network.trainable_variables)
            # 输入 state, reward, behavior action; 输出 value, new action, new logits（new policy）；
            predict_output_dict = self._network(inputs_dict)
            next_inputs_dict = {'my_units': inputs_dict['nextobs_my_units'],
                                'b_info': inputs_dict['nextobs_b_info']}
            predict_output_dict_target = self._target_network(next_inputs_dict)
            next_result = self._network(next_inputs_dict)

            loss = 0
            td_err_list = []
            for k, v in predict_output_dict[LOGITS].items():
                value = tf.gather(v, tf.reshape(action_index[k], [-1, 1]), axis=1, batch_dims=1)
                value = tf.reshape(value, [-1])
                if self._target_network is not None:
                    target_v = predict_output_dict_target[LOGITS][k]
                else:
                    target_v = next_result[LOGITS][k]
                if self.double_q:
                    target_value = tf.gather(target_v, tf.argmax(next_result[LOGITS][k], axis=1), batch_dims=1)
                    # target_value = target_v.numpy()[np.arange(len(target_v)), tf.argmax(next_result[LOGITS][k], axis=1)]
                else:
                    target_value = tf.reduce_max(target_v, axis=1, keepdims=True)
                target = reward + self.gamma * (1. - dones) * target_value
                td_err = tf.square(tf.stop_gradient(target) - value)
                td_err_list.append(td_err)
                loss += tf.reduce_mean(td_err * inputs_dict.get('weight', 1.0))

            td_err = sum(td_err_list)

        hvd = get_hvd("tensorflow")
        tape = hvd.DistributedGradientTape(tape)
        # 反向传播，计算所有 weight 的 gradient
        gradients = tape.gradient(loss, self._network.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 40.0)
        # 更新 weight
        self._optimizer.apply_gradients(zip(gradients, self._network.trainable_variables))

        summary = {
            "loss": loss,
            "weight": td_err
        }
        return summary

    def argmax_sample(self, logits, epsilon: float) -> tf.Tensor:
        action = tf.argmax(logits, axis=-1,
                           output_type=tf.int32)  # todo 返回 vector 中的最大值的索引号, axis对吗
        batch_size = tf.shape(logits)[0]
        num_actions = tf.shape(logits)[-1]
        random_actions = tf.random.uniform(tf.stack([batch_size]), minval=0,
                                           maxval=num_actions, dtype=tf.int32)
        chose_random = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=1,
                                         dtype=tf.float32) < epsilon
        action = tf.where(chose_random, random_actions, action)
        action = tf.stop_gradient(action)
        return action
