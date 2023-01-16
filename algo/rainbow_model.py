from typing import Any, Dict

import tensorflow as tf
import numpy as np

from drill.keys import ACTION, LOGITS
from drill.model.model import Model
from drill.utils import get_hvd

from algo.c51_model import C51Model
from algo.dqn_net import sample_noise


class RainbowModel(C51Model):
    @tf.function
    def learn(self, inputs_dict: Dict[str, Any], behavior_info_dict: Dict[str,
    Any]) -> Dict[str, Any]:
        if hasattr(self._network, 'train'):
            self._network.train()

        if hasattr(self._target_network, 'train'):
            self._target_network.train()

        sample_noise(self._network)
        if self._target_network is not None and sample_noise(self._target_network):
            self._target_network.train()
        return super().learn(inputs_dict=inputs_dict, behavior_info_dict=behavior_info_dict)

    @tf.function
    def predict(self, state_dict: Dict[str, Any], epsilon) -> Dict[str, Any]:
        if hasattr(self._network, 'eval'):
            self._network.eval()

        if hasattr(self._target_network, 'eval'):
            self._target_network.eval()

        return super().predict(state_dict=state_dict, epsilon=epsilon)

