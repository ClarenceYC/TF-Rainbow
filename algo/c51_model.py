from typing import Any, Dict

import tensorflow as tf
import numpy as np

from drill.keys import ACTION, LOGITS
from drill.model.model import Model
from drill.utils import get_hvd

from algo.dqn_model import DQNModel

class C51Model(DQNModel):
    def __init__(
            self,
            network,
            target_network,
            learning_rate: float = 1e-3,
            gamma=0.99,
            num_atoms: int = 51,
            v_min: float = -10.0,
            v_max: float = 10.0,
            **kwargs: Any,
    ):
        super(C51Model, self).__init__(network, target_network, learning_rate, gamma, dqn_type='double_DQN')
        self._num_atoms = num_atoms
        self._v_min = v_min
        self._v_max = v_max

        self.support = tf.linspace(self._v_min, self._v_max, self._num_atoms)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

    def _target_dist(self, logits, act):
        """
        Future abstract version
        :param logits:
        :param act:
        :return:
        """
        pass

    def _compute_q_value(self, logits):
        return logits*self.support

    def predict(self, state_dict: Dict[str, Any], epsilon) -> Dict[str, Any]:
        predict_output_dict = self._network(state_dict)
        logits = predict_output_dict[LOGITS]

        return {
            ACTION: {
                "action_x": self.argmax_sample(tf.math.reduce_sum(self._compute_q_value(logits['action_x']), axis=2), epsilon),
                "action_y": self.argmax_sample(tf.math.reduce_sum(self._compute_q_value(logits['action_y']), axis=2), epsilon),
                "action_dv": self.argmax_sample(tf.math.reduce_sum(self._compute_q_value(logits['action_dv']), axis=2), epsilon)
            },
        }

    def learn(self, inputs_dict: Dict[str, Any], behavior_info_dict: Dict[str,
    Any]) -> Dict[str, Any]:
        reward = inputs_dict['reward_dqn']
        action_index = behavior_info_dict[ACTION]
        dones = inputs_dict['done_dqn']

        with tf.GradientTape() as tape:
            # 输入 state, reward, behavior action; 输出 value, new action, new logits（new policy）；
            predict_output_dict = self._network(inputs_dict)
            next_inputs_dict = {'my_units': inputs_dict['nextobs_my_units'],
                                'b_info': inputs_dict['nextobs_b_info']}
            predict_output_dict_target = self._target_network(next_inputs_dict)
            next_result = self._network(next_inputs_dict)

            loss = 0
            CE_list = []
            for k, v in predict_output_dict[LOGITS].items():
                curr_dist = tf.gather(v, tf.reshape(action_index[k], [-1, 1]), axis=1, batch_dims=1)
                curr_dist = tf.reshape(curr_dist, [curr_dist.shape[0], -1])
                if self._target_network is not None:
                    target_dist = predict_output_dict_target[LOGITS][k]
                else:
                    target_dist = next_result[LOGITS][k]
                if self.double_q:
                    next_dist = tf.gather(
                        target_dist,
                        tf.argmax(tf.math.reduce_sum(next_result[LOGITS][k]*self.support, axis=-1), axis=1),
                        batch_dims=1
                    )
                    # target_value = target_v.numpy()[np.arange(len(target_v)), tf.argmax(next_result[LOGITS][k], axis=1)]
                else:
                    next_dist = tf.gather(
                        target_dist,
                        tf.argmax(tf.math.reduce_sum(target_dist[LOGITS][k]*self.support, axis=-1), axis=1),
                        batch_dims=1
                    )

                target_support = tf.reshape(reward, [-1, 1]) + self.gamma * (1. - dones) * tf.concat([tf.reshape(self.support, [1, -1])]*reward.shape[0], axis=0)

                target_dist = (
                    1 - tf.abs(tf.reshape(target_support, [-1, 1, self._num_atoms]) - tf.reshape(self.support, [1, -1, 1])) / self.delta_z
                )
                target_dist = tf.clip_by_value(target_dist, clip_value_min=0, clip_value_max=1) * tf.reshape(next_dist, [-1, self._num_atoms, 1])
                target_dist = tf.reduce_sum(target_dist, axis=-1)
                cross_entropy = -tf.reduce_sum(target_dist * tf.math.log(curr_dist + 1e-8), axis=1)
                CE_list.append(cross_entropy)
                loss += tf.math.reduce_mean(cross_entropy)

            CE_loss = sum(CE_list)

            hvd = get_hvd("tensorflow")
            tape = hvd.DistributedGradientTape(tape)
            # 反向传播，计算所有 weight 的 gradient
            gradients = tape.gradient(loss, self._network.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, 40.0)
            # 更新 weight
            self._optimizer.apply_gradients(zip(gradients, self._network.trainable_variables))

            summary = {
                "loss": loss,
                "weight": CE_loss
            }
            return summary
