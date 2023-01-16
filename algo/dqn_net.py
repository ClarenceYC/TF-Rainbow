from typing import Dict

import tensorflow as tf
from tensorflow import keras
import numpy as np

from drill.keys import ACTION, ADVANTAGE, DECODER_MASK, LOGITS, VALUE


class DQNNetwork(keras.Model):

    def __init__(self, config={}):
        super(DQNNetwork, self).__init__()
        self._config = config

        self.self_info_encoder = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
        ])

        self.target_info_encoder = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
        ])

        self.aggregator = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(512, activation='relu'),
        ])

        self.action_x_decoder = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(10, activation='relu'),
        ])

        self.action_y_decoder = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(10, activation='relu'),
        ])

        self.action_dv_decoder = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(10, activation='relu'),
        ])

    def call(self, inputs: Dict[str, tf.Tensor]):
        h_1 = self.self_info_encoder(
            tf.reshape(inputs['my_units'], [inputs['my_units'].shape[0], -1]))
        h_2 = self.target_info_encoder(inputs['b_info'])

        h = tf.concat([h_1, h_2], axis=1)
        h = self.aggregator(h)
        action_x = self.action_x_decoder(h)
        action_y = self.action_y_decoder(h)
        action_dv = self.action_dv_decoder(h)

        predict_output_dict = {
            LOGITS: {"action_x": action_x, "action_y": action_y, "action_dv": action_dv},
        }
        return predict_output_dict


class DuelingDQNNetwork(keras.Model):
    def __init__(self, config={}):
        super(DuelingDQNNetwork, self).__init__()
        self.config = config

        self.self_info_encoder = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
        ])

        self.target_info_encoder = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
        ])

        self.aggregator = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(512, activation='relu'),
        ])

        self.action_x_value = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(1, activation='relu'),
        ])
        self.action_x_advantage = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(10, activation='relu'),
        ])

        self.action_y_value = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(1, activation='relu'),
        ])
        self.action_y_advantage = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(10, activation='relu'),
        ])

        self.action_dv_value = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(1, activation='relu'),
        ])
        self.action_dv_advantage = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(10, activation='relu'),
        ])

    def call(self, inputs: Dict[str, tf.Tensor]):
        h_1 = self.self_info_encoder(
            tf.reshape(inputs['my_units'], [inputs['my_units'].shape[0], -1]))
        h_2 = self.target_info_encoder(inputs['b_info'])
        h = tf.concat([h_1, h_2], axis=1)
        h = self.aggregator(h)

        action_x_value = self.action_x_value(h)
        action_x_advantage = self.action_x_advantage(h)
        mean_x = tf.reduce_mean(action_x_advantage, axis=-1, keepdims=True)
        action_x = action_x_value + action_x_advantage - mean_x

        action_y_value = self.action_y_value(h)
        action_y_advantage = self.action_y_advantage(h)
        mean_y = tf.reduce_mean(action_y_advantage, axis=-1, keepdims=True)
        action_y = action_y_value + action_y_advantage - mean_y

        action_dv_value = self.action_dv_value(h)
        action_dv_advantage = self.action_dv_advantage(h)
        mean_dv = tf.reduce_mean(action_dv_advantage, axis=-1, keepdims=True)
        action_dv = action_dv_value + action_dv_advantage - mean_dv

        predict_output_dict = {
            LOGITS: {"action_x": action_x, "action_y": action_y, "action_dv": action_dv},
        }
        return predict_output_dict


class C51Network(DQNNetwork):
    def __init__(self, num_atoms=51, config={}):
        self._num_atoms = num_atoms

        super(DQNNetwork, self).__init__()
        self._config = config

        self.self_info_encoder = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
        ])

        self.target_info_encoder = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
        ])

        self.aggregator = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(512, activation='relu'),
        ])

        self.action_x_decoder = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(10* self._num_atoms, activation='relu'),
        ])

        self.action_y_decoder = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(10 * self._num_atoms, activation='relu'),
        ])

        self.action_dv_decoder = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(10 * self._num_atoms, activation='relu'),
        ])

    def call(self, inputs: Dict[str, tf.Tensor]):
        logits_dict = super(C51Network, self).call(inputs)
        for action in logits_dict[LOGITS]:
            logits_dict[LOGITS][action] = tf.reshape(
                tf.keras.activations.softmax(tf.reshape(logits_dict[LOGITS][action], shape=[-1, self._num_atoms])),
                shape=[-1, 10, self._num_atoms]
            )

        return logits_dict


class RainbowNetwork(C51Network):
    def __init__(self, num_atoms=51, config={}):
        super(RainbowNetwork, self).__init__(num_atoms=num_atoms, config=config)

        self.action_x_decoder = keras.Sequential([
            NoisyLinear(in_features=512, out_features=512), keras.layers.ReLU(),
            NoisyLinear(in_features=512, out_features=256), keras.layers.ReLU(),
            NoisyLinear(in_features=256, out_features=10 * self._num_atoms), keras.layers.ReLU()
        ])

        self.action_y_decoder = keras.Sequential([
            NoisyLinear(in_features=512, out_features=512), keras.layers.ReLU(),
            NoisyLinear(in_features=512, out_features=256), keras.layers.ReLU(),
            NoisyLinear(in_features=256, out_features=10 * self._num_atoms), keras.layers.ReLU()
        ])

        self.action_dv_decoder = keras.Sequential([
            NoisyLinear(in_features=512, out_features=512), keras.layers.ReLU(),
            NoisyLinear(in_features=512, out_features=256), keras.layers.ReLU(),
            NoisyLinear(in_features=256, out_features=10 * self._num_atoms), keras.layers.ReLU()
        ])

        self.training = True

    def train(self):
        for m in self.submodules:
            if isinstance(m, NoisyLinear):
                m.train()
        self.training = True

    def eval(self):
        for m in self.submodules:
            if isinstance(m, NoisyLinear):
                m.eval()
        self.training = False


class NoisyLinear(keras.Model):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            noisy_std: float = 0.5,
            config={}
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma = noisy_std
        bound = 1 / np.sqrt(self.in_features)

        # initializer
        constant_initializer = tf.constant_initializer(value=self.sigma / np.sqrt(self.in_features))
        uniform_initializer = tf.random_uniform_initializer(minval=-bound, maxval=bound)

        # Learnable parameters.
        self.mu_W = tf.Variable(uniform_initializer(shape=[self.out_features, self.in_features], dtype=tf.float32))
        self.sigma_W = tf.Variable(uniform_initializer(shape=[self.out_features, self.in_features], dtype=tf.float32))
        self.mu_bias = tf.Variable(constant_initializer(shape=[out_features], dtype=tf.float32))
        self.sigma_bias = tf.Variable(constant_initializer(shape=[out_features], dtype=tf.float32))

        # Factorized noise parameters.
        self.eps_p = tf.constant(tf.random.normal([in_features]), shape=[in_features])
        self.eps_q = tf.constant(tf.random.normal([out_features]), shape=[out_features])

        self.training = True

        # self.reset()
        self.sample()

    def reset(self) -> None:
        bound = 1 / np.sqrt(self.in_features)
        self.mu_W.assign(tf.random.uniform([self.out_features, self.in_features], minval=-bound, maxval=bound))
        self.mu_bias.assign(tf.random.uniform([self.out_features, self.in_features], minval=-bound, maxval=bound))
        self.sigma_W.assign(self.sigma / np.sqrt(self.in_features))
        self.sigma_bias.assign(self.sigma / np.sqrt(self.in_features))

    def f(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.random.normal(x.shape)
        return tf.math.sign(x) * tf.math.sqrt(tf.math.abs(x))

    def sample(self) -> None:
        self.eps_p = self.f(self.eps_p)  # type: ignore
        self.eps_q = self.f(self.eps_q)  # type: ignore

    def call(self, x: tf.Tensor):
        if self.training:
            weight = self.mu_W + self.sigma_W * (
                tf.experimental.numpy.outer(self.eps_q, self.eps_p)
            )
            bias = self.mu_bias + self.sigma_bias * self.eps_q  # type: ignore
        else:
            weight = self.mu_W
            bias = self.mu_bias

        return tf.matmul(x, tf.transpose(weight)) + bias

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


def sample_noise(model: tf.Module) -> bool:
    """Sample the random noises of NoisyLinear modules in the model.

    :param model: a PyTorch module which may have NoisyLinear submodules.
    :returns: True if model has at least one NoisyLinear submodule;
        otherwise, False.
    """
    done = False
    for m in model.submodules:
        if isinstance(m, NoisyLinear):
            m.sample()
            done = True
    return done
