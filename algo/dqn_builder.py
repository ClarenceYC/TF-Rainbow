from copy import deepcopy
from typing import Dict

import numpy as np

import drill
from drill.builder import BPBuilder
from drill.utils import construct


class DQNBuilder(BPBuilder):
    def __init__(self,
                 agents: Dict[str, Dict[str, str]],
                 models,
                 env,
                 pipeline,
                 backend='tensorflow'):
        super().__init__(agents, models, env, pipeline, backend)

    def build_model(self, model_name: str, show_graph=False):
        model_config = self._models[model_name]

        if "save" in model_config:
            self._save_params[model_name] = dict()
            # 训练云定制化
            if drill.local_run:
                self._save_params[model_name]['path'] = 'models'
            else:
                self._save_params[model_name]['path'] = '/job/model'
            self._save_params[model_name]['interval'] = model_config["save"].get('interval', 100)
            self._save_params[model_name]['mode'] = model_config["save"].get('mode', 'npz')
            del model_config["save"]

        network_config = model_config["params"]["network"]
        network = construct(network_config)
        target_network = construct(network_config)
        pipeline = self.build_pipeline()
        agent_name = self._get_agent_name(model_name)

        if self.backend == "tensorflow":
            fake_state = pipeline.get_fake_state(agent_name, batch_size=1)

            hidden_state_dict = self.get_initial_state(agent_name)
            for name, hidden_state in hidden_state_dict.items():
                fake_state[name] = hidden_state[np.newaxis, ...]

            network(fake_state)
            target_network(fake_state)

        model_config_copy = deepcopy(model_config)
        model_config_copy["params"]["network"] = network
        model_config_copy["params"]["target_network"] = target_network

        if "load" in model_config:

            from drill.utils import load_model
            if drill.local_run:
                model_file = model_config["load"]["model_file"]
            else:
                model_file = f'/job/project/{model_config["load"]["model_file"]}'
            load_model(network, model_file, self.backend,
                       model_config["load"].get("mode", 'npz'))
            if "step" in model_config["load"]:
                self._learn_step = model_config["load"]["learn_step"]

        model = construct(model_config_copy)
        # 注意，由于drill目前会对flowmodel多次初始化，如果保存模型，会生成三个同样结构的
        if show_graph and self.backend == "tensorflow":
            import tensorflow as tf
            from tensorflow.python.ops import summary_ops_v2
            graph_writer = tf.summary.create_file_writer(logdir='./nn_graph')
            with graph_writer.as_default():
                graph_test = model.predict.get_concrete_function(fake_state).graph
                summary_ops_v2.graph(graph_test.as_graph_def())
        return model

    def get_initial_state(self, agent_name):
        return {}
