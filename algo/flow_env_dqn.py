from __future__ import annotations

import copy
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Union

import numpy as np

from drill.keys import (ACTION, ADVANTAGE, HIDDEN_STATE, DECODER_MASK, DONE, LOGITS, REWARD)

from drill.pipeline import ActionData

if TYPE_CHECKING:
    from drill.builder import Builder


class FlowEnvDQN:

    def __init__(self, environment_description):
        builder: Builder = environment_description.environment_creator_user_args['builder']
        env_id = environment_description.node_id * environment_description.num_envs_on_this_node \
            + environment_description.environment_id_on_this_node
        env_extra_info = {'node_id': environment_description.node_id,
                          'num_envs_on_this_node': environment_description.num_envs_on_this_node,
                          'environment_id_on_this_node': environment_description.environment_id_on_this_node}
        env_extra_info.update(environment_description.environment_creator_user_args['extra_info'])
        self._builder = builder
        self._env = builder.build_env(env_id, env_extra_info)
        self._pipeline = builder.build_pipeline()

        self.flow_env_config = {}
        self._episode_done = None
        self._obs_data = None

        # 上个 action, logits, value
        # <key = agent name, value = {logits, action, value}>
        self._last_hidden_state_dict = defaultdict(dict)
        # 当前剩余 agents
        self._agent_names = []
        # mask history {agent_name: [{decoder_name: mask}]}
        # self._dynamic_mask_history = defaultdict(list)
        self._last_agent_to_reward = {}

    @property
    def agent_names(self) -> List[str]:
        """ 返回环境中当前剩余的 agents 的名字

        Returns
        -------
        List[str]
            agent_names
        """
        return self._agent_names

    @property
    def env(self):
        return self._env

    def reset(self):
        """ 重置状态，开始一个新的 episode
        """
        self._obs_data = self._env.reset()
        self._episode_done = False
        self._last_hidden_state_dict = defaultdict(dict)
        self._agent_names = []
        self._pipeline.reset()

    def _get_hidden_state(self, agent_name):
        if HIDDEN_STATE in self._last_hidden_state_dict[agent_name]:
            predict_output_dict = self._last_hidden_state_dict[agent_name]
            hidden_state_dict = {HIDDEN_STATE: predict_output_dict[HIDDEN_STATE]}
            return hidden_state_dict
        return self._builder.get_initial_state(agent_name)

    def observe(self) -> Dict[str, Dict[str, Union[Dict, str]]]:
        """ 获取 observation

        Returns
        -------
        Dict[str, Dict[str, Union[Dict, str]]]
            需要让 model inference 的 agent 的 observation。包括网络需要的输入以及对应的 model，
            另外还包括 reward 和 done

        Examples
        --------
        return
        ```python
        observe_return = {
            "red_agent":
                {"obs":
                    {"spatial": np.ndarray,
                        "entity": np.ndarray,
                        "reward": array,
                        "hidden_state": Any,
                        ...
                    },
                "model": "battle_model"}}
            "blue_agent": ...
        }
        ```
        """
        # 1. 调用 pipeline 将环境返回的状态信息(observation)处理成网络可识别
        # 的 state，reward (环境可能不会返回reward)
        # example state_dict format {agent_name: {fs.name: processed_obs}}
        # example reward_dict format {agent_name: reward}
        agent_to_state, agent_to_reward = self._pipeline.pre_process(self._obs_data,
                                                                     self._episode_done)
        self._last_agent_to_reward.update(agent_to_reward)

        episode_done = self._episode_done
        if episode_done:
            self.reset()
            agent_to_state, _ = self._pipeline.pre_process(self._obs_data, self._episode_done)

        observe_return = {}
        for agent_name, agent_state_dict in agent_to_state.items():
            # TODO: recording each individual reward target
            reward = self._last_agent_to_reward[agent_name]
            if isinstance(reward, dict):
                agent_state_dict[REWARD] = np.array(sum(reward.values()), dtype=np.float32)
            else:
                agent_state_dict[REWARD] = np.array(reward, dtype=np.float32)
            agent_state_dict[DONE] = np.array(episode_done, dtype=np.float32)
            hidden_state_dict = self._get_hidden_state(agent_name)
            agent_state_dict.update(hidden_state_dict)

            observe_return[agent_name] = {
                'obs': agent_state_dict,
                'model': self._builder.get_model_name(agent_name)
            }

        self._agent_names = [agent_name for agent_name in observe_return.keys()]

        return observe_return

    def step(self, agent_name: str, predict_output: Dict[str,
    Any]) -> Dict[str, Dict[str, np.ndarray]]:
        """ 根据 model inference 的结果和环境进行交互。
        model 根据 observation 做 inference 得到的结果通过参数 `predict_output` 返回。
        注意：多智能体时，`observe` 可能同时返回所有 agent 的 observation 给 model 做
        inference，但是 `step` 的参数 `predict_output` 只包含参数 `agent_name` 对应的
        inference 结果。且不保证不同 agent step 调用的顺序。

        Parameters
        ----------
        agent_name : str
            agent 的名字
        predict_output : Dict[str, Any]
            `agent_name` 对应的 model inference 的结果
            包含该智能体的 action, logits, value, hidden_state。

        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            decoder_mask。复杂一点的场景，action 是有多头的（multi-head）， 而 action 会作为样本收集起来参与训练，
            但并不一定每一个 head 都是 “有效的”。举个例子，假设动作空间为

            * meta: 移动/攻击
            * postion: 目标位置
            * target: 目标单位

            当 meta 为移动的时候，postion 这个 head 是有效的，而 target 无效，所以 target 这个 head 就不应该参与
            loss 计算。decoder mask 的作用就是讲 “无效的” head mask 掉，不参与 loss 计算。

        """
        if HIDDEN_STATE in predict_output:
            self._last_hidden_state_dict[agent_name][HIDDEN_STATE] = copy.deepcopy(
                predict_output[HIDDEN_STATE])

        data = ActionData(action=copy.deepcopy(predict_output[ACTION]),
                          predict_output=copy.deepcopy(predict_output))
        action_dict, action_mask, decoder_mask = self._pipeline.post_process(agent_name, data)

        if action_dict is not None:
            self._obs_data, self._episode_done = self._env.step(action_dict)
        #     for agent_name, mask in action_mask.items():
        #         self._dynamic_mask_history[agent_name].append(mask)

        # FIXME: change this return once Flow implement a newer version
        return {DECODER_MASK: decoder_mask}

    def enhance_fragment(self, agent_name: str, fragments: List[Dict]):
        """ 对不断 `observe` 和 `step` 收集的数据进行处理

        什么时候调用这个方法？
        一次 `observe` 和 `step` 收集的数据记为一个 fragment，当收集到的数据达到
        `fragment_size` （一个配置参数）时调用此方法

        注意: fragments 只能原地修改，这个方法不接受返回值，这是由 flow 决定的

        Parameters
        ----------
        agent_name : str
            agent 的名字
        fragments : List[Dict]
            长度为 `fragment_size`，每一个元素都是 3 元组，分别对应 `observe` 的
            返回值（准确的说是 `observe_return[agent_name]["obs"]`，`step` 的参数
            `predict_output` 和 `step` 的返回值。
        """

        # 对于多智能体竞争环境存在一种情况：
        # 采集 fragment 的顺序：先依次执行FlowEnv.step() 和 FlowEnv.observe() 得
        # 到 state, reward, 然后执行 FlowModel.predict() 得到 action, value，然后采集
        # 一个 fragment(state, reward, action, value)，然后执行下一个 FlowEnv.step()。
        # 设 fragment_size=64+1 = 65，第 64 次执行 FlowEnv.step() 时，已
        # 经采集了 64 个fragment（若某个 agent 在第 64 次 step done了，返回的
        # obs, reward, done 不为空），开始采集第 65 个fragment，然后会执行第 65 个
        # FlowEnv.step(), 再执行 FlowEnv.enhance_fragment()，此时进入 enhance_fragment()
        # 后会遇到问题: 此时的 self._env_info 已经不包含刚刚 done 的 agent 了，因此 last_reward
        # = None。因此额外使用 self._final_reward 维护所有 done 的 agent 的最后一个 reward。

        self.n_step = 1
        self.discount = 0.99

        rewards, nextobs_my_units, nextobs_b_info, dones = self._get_rewards_and_nextobs_and_dones(agent_name, fragments)

        reward_len = len(rewards)
        rewards_nstep = []
        for i in range(reward_len):
            res = 0
            for j in range(self.n_step):
                if i+j >= reward_len:
                    break
                res += rewards[i+j] * (self.discount ** j)
                if dones[i+j] > 0.:
                    break
            rewards_nstep.append(res)

        fragments_len = len(fragments)
        rewards_len = len(rewards_nstep)
        for i in range(rewards_len):
            fragments[i][0].update({"reward_dqn": np.asarray(rewards_nstep[i], dtype=np.float32)})
            fragments[i][0].update({"nextobs_my_units": nextobs_my_units[i]})
            fragments[i][0].update({"nextobs_b_info": nextobs_b_info[i]})
            fragments[i][0].update({"done_dqn": np.asarray(dones[i], dtype=np.float32)})
            fragments[i].append({ADVANTAGE: np.asarray(1., dtype=np.float32)})  # DQN没有critic，仅用于占位
        for _ in range((self.n_step + 1)):
            fragments.pop(-1)



    def _get_rewards_and_nextobs_and_dones(self, agent_name, fragments):
        """
        Args:
            agent_name: agent的名字
            last_reward: 最后一个step对应的reward
            done: episode是否结束
            fragments: n个连续的step收集到的样本数据
        我们可以简单把fragments视作n个连续的step，每个step包含(s, r, a, v)，其实并不是这样，但这样方便理解。
        fragments: [(s_0, r_0, a_0, v_0), (s_1, r_1, a_1, v_1), ...]
        比较难于理解的是r_0并不是在s_0下执行动作a_0后得到的reward，r_1才是。这部分逻辑可以参考observe方法。
        r_0是没有意义的，所以在下面的代码中过滤掉了r_0(见代码if i > 0)
        """
        rewards = []
        nextobs_my_units = []
        nextobs_b_info = []
        dones = []
        for i in range(len(fragments)):
            flow_state, flow_action_dict, _ = fragments[i]
            if i > 0:
                rewards.append(copy.deepcopy(flow_state['reward']))
                dones.append(copy.deepcopy(flow_state['done']))
                nextobs_1 = copy.deepcopy(flow_state['my_units'])
                nextobs_2 = copy.deepcopy(flow_state['b_info'])
                for j in range(self.n_step):
                    if i+j >= len(fragments):
                        break
                    flow_state, flow_action_dict, _ = fragments[i+j]
                    if flow_state['done'] > 0.:
                        break
                    nextobs_1 = copy.deepcopy(flow_state['my_units'])
                    nextobs_2 = copy.deepcopy(flow_state['b_info'])
                # nextobs_my_units.append({'my_units': nextobs_1})
                # nextobs_b_info.append({'b_info': nextobs_2})
                nextobs_my_units.append(nextobs_1)
                nextobs_b_info.append(nextobs_2)

        return rewards, nextobs_my_units, nextobs_b_info, dones

    def render(self, **kwargs):
        if hasattr(self._env, 'render'):
            return self._env.render(**kwargs)  # type: ignore
        else:
            raise RuntimeError("self._env does not have a render method!")