#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time
import random
from agent_ppo.feature.definition import (
    sample_process,
    build_frame,
    FrameCollector,
    NONE_ACTION,
    lineup_iterator_roundrobin_camp_heroes,
)
from agent_ppo.conf.conf import GameConfig
from tools.env_conf_manager import EnvConfManager
from tools.model_pool_utils import get_valid_model_pool
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


TERMINAL_WIN_REWARD = 8.0
TERMINAL_LOSS_REWARD = -6.0
TERMINAL_TIMEOUT_REWARD = -2.5

MATCHUP_LINEUPS = {
    "112v112": [112, 112],
    "112v133": [112, 133],
    "133v112": [133, 112],
    "133v133": [133, 133],
    "luban_mirror": [112, 112],
    "luban_vs_direnjie": [112, 133],
    "direnjie_vs_luban": [133, 112],
    "direnjie_mirror": [133, 133],
}

EVAL_MATCHUP_KEYS = ("112v112", "112v133", "133v112", "133v133")


def fixed_lineup_iterator(lineup):
    while True:
        yield list(lineup)


def cycle_lineup_iterator(lineups):
    while True:
        for lineup in lineups:
            yield list(lineup)


def parse_weighted_matchups(spec):
    lineups = []
    for raw_item in str(spec or "").split(","):
        item = raw_item.strip().lower().replace("-", "_")
        if not item:
            continue
        if ":" in item:
            name, raw_weight = item.split(":", 1)
        else:
            name, raw_weight = item, "1"
        name = name.strip()
        try:
            weight = int(float(raw_weight.strip()))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid matchup weight item: {raw_item!r}") from None
        if name not in MATCHUP_LINEUPS:
            valid = ", ".join(sorted(MATCHUP_LINEUPS))
            raise ValueError(f"Invalid matchup name in weighted schedule: {name!r}. Valid options: {valid}")
        if weight <= 0:
            continue
        lineups.extend([MATCHUP_LINEUPS[name]] * weight)
    if not lineups:
        raise ValueError(f"Empty weighted matchup schedule: {spec!r}")
    return lineups


def describe_lineups(lineups):
    return "[" + ", ".join(f"{lineup[0]}v{lineup[1]}" for lineup in lineups) + "]"


def eval_matchup_key_iterator():
    while True:
        for key in EVAL_MATCHUP_KEYS:
            yield key


def lineup_for_model_matchup(matchup_key, monitor_side):
    own_hero, opponent_hero = MATCHUP_LINEUPS[matchup_key]
    if int(monitor_side) == 0:
        return [own_hero, opponent_hero]
    return [opponent_hero, own_hero]


def matchup_key_from_usr_conf(usr_conf, monitor_side):
    blue_hero_ids, red_hero_ids = EnvConfManager.extract_hero_ids_from_usr_conf(usr_conf)
    if not blue_hero_ids or not red_hero_ids:
        return "unknown"
    blue_hero = int(blue_hero_ids[0])
    red_hero = int(red_hero_ids[0])
    if int(monitor_side) == 0:
        return f"{blue_hero}v{red_hero}"
    return f"{red_hero}v{blue_hero}"


def set_usr_conf_lineup(usr_conf, lineup):
    lineups = usr_conf.get("lineups")
    if not isinstance(lineups, dict):
        return False
    updated = True
    for camp_key, hero_id in (("blue_camp", lineup[0]), ("red_camp", lineup[1])):
        camp = lineups.get(camp_key)
        if isinstance(camp, list) and camp and isinstance(camp[0], dict):
            camp[0]["hero_id"] = int(hero_id)
        else:
            updated = False
    return updated


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _same_id(left, right):
    if left is None or right is None:
        return False
    return str(left) == str(right)


def _hp_ratio(unit):
    if not unit:
        return 0.0
    try:
        hp = float(unit.get("hp", 0.0))
        max_hp = float(unit.get("max_hp", 0.0))
    except (TypeError, ValueError):
        return 0.0
    if max_hp <= 0.0:
        return 0.0
    return max(0.0, min(1.0, hp / max_hp))


def _find_own_hero(frame_state, player_id, fallback_camp=None):
    for hero in frame_state.get("hero_states", []) or []:
        if _same_id(hero.get("player_id"), player_id) or _same_id(hero.get("runtime_id"), player_id):
            return hero
    for hero in frame_state.get("hero_states", []) or []:
        if fallback_camp is not None and _safe_int(hero.get("camp"), -1) == _safe_int(fallback_camp, -2):
            return hero
    return None


def _tower_hp_pair(frame_state, agent_camp):
    own_tower_hp = 0.0
    enemy_tower_hp = 0.0
    agent_camp = _safe_int(agent_camp, -1)
    for npc in frame_state.get("npc_states", []) or []:
        config_id = _safe_int(npc.get("config_id"), 0)
        sub_type = _safe_int(npc.get("sub_type"), 0)
        if sub_type != 21 and config_id not in {1111, 1112}:
            continue
        hp = max(0.0, _safe_float(npc.get("hp"), 0.0))
        if _safe_int(npc.get("camp"), -2) == agent_camp:
            own_tower_hp = max(own_tower_hp, hp)
        else:
            enemy_tower_hp = max(enemy_tower_hp, hp)
    return own_tower_hp, enemy_tower_hp


def _episode_win_score(frame_state, observation, monitor_side, agent_camp, truncated):
    if truncated:
        return 0.0
    obs = observation.get(str(monitor_side), {}) if isinstance(observation, dict) else {}
    win_value = obs.get("win")
    if win_value is not None:
        return 1.0 if _safe_float(win_value) > 0.0 else 0.0
    own_tower_hp, enemy_tower_hp = _tower_hp_pair(frame_state, agent_camp)
    if enemy_tower_hp <= 0.0 and own_tower_hp > 0.0:
        return 1.0
    if own_tower_hp <= 0.0 and enemy_tower_hp > 0.0:
        return 0.0
    return 0.0


def build_eval_matchup_monitor_data(matchup_key, observation, monitor_side, frame_no, truncated, agent_camp):
    if matchup_key not in EVAL_MATCHUP_KEYS:
        return {}
    obs = observation.get(str(monitor_side), {})
    frame_state = obs.get("frame_state", {})
    agent_camp = obs.get("camp", obs.get("player_camp", agent_camp))
    own_hero = _find_own_hero(frame_state, obs.get("player_id"), agent_camp) or {}
    own_tower_hp, enemy_tower_hp = _tower_hp_pair(frame_state, agent_camp)
    frame_base = max(1.0, _safe_float(frame_no, 1.0))
    prefix = f"eval_{matchup_key}_"
    return {
        prefix + "win": round(_episode_win_score(frame_state, observation, monitor_side, agent_camp, truncated), 4),
        prefix + "self_tower_hp": round(own_tower_hp, 2),
        prefix + "enemy_tower_hp": round(enemy_tower_hp, 2),
        prefix + "frame": round(frame_base, 2),
        prefix + "timeout": 1.0 if truncated else 0.0,
        prefix + "money_per_frame": round(_safe_float(own_hero.get("money"), 0.0) / frame_base, 6),
        prefix + "kill": round(_safe_float(own_hero.get("kill_cnt"), 0.0), 4),
        prefix + "death": round(_safe_float(own_hero.get("dead_cnt"), 0.0), 4),
        prefix + "hurt_to_hero": round(_safe_float(own_hero.get("total_hurt_to_hero"), 0.0) / frame_base, 6),
        prefix + "hurt_by_hero": round(_safe_float(own_hero.get("total_be_hurt_by_hero"), 0.0) / frame_base, 6),
    }


def _terminal_reward_bonus(frame_state, agent_camp, truncated):
    if truncated:
        return TERMINAL_TIMEOUT_REWARD
    own_tower_hp = 0.0
    enemy_tower_hp = 0.0
    for npc in frame_state.get("npc_states", []) or []:
        config_id = npc.get("config_id")
        sub_type = npc.get("sub_type")
        is_tower = sub_type == 21 or config_id in {1111, 1112}
        is_crystal = sub_type == 24 or config_id in {1113, 1114}
        if not is_tower and not is_crystal:
            continue
        hp = _hp_ratio(npc)
        if npc.get("camp") == agent_camp:
            own_tower_hp = max(own_tower_hp, hp)
        else:
            enemy_tower_hp = max(enemy_tower_hp, hp)
    if enemy_tower_hp <= 0.0 and own_tower_hp > 0.0:
        return TERMINAL_WIN_REWARD
    if own_tower_hp <= 0.0 and enemy_tower_hp > 0.0:
        return TERMINAL_LOSS_REWARD
    return 0.0


def _add_terminal_reward(reward, bonus):
    if bonus == 0.0 or not isinstance(reward, dict):
        return reward
    updated = dict(reward)
    reward_vector = list(updated.get("reward_vector", []))
    if reward_vector:
        reward_vector[0] = reward_vector[0] + bonus
        if len(reward_vector) > 1:
            reward_vector[1] = reward_vector[1] + 0.75 * bonus
        updated["reward_vector"] = reward_vector
    updated["reward_sum"] = float(updated.get("reward_sum", 0.0)) + bonus
    return updated


def build_training_lineup_iterator(logger=None):
    matchup = os.environ.get("R5_TRAIN_MATCHUP", GameConfig.TRAINING_MATCHUP)
    matchup = str(matchup).strip().lower().replace("-", "_")
    if matchup in ("all", "roundrobin", "round_robin", "mixed", "four"):
        if logger:
            logger.info("R5 training matchup schedule: all four matchups round-robin")
        return lineup_iterator_roundrobin_camp_heroes([112, 133])
    if matchup in ("custom", "custom_weighted", "weighted", "weighted_matchups"):
        spec = os.environ.get("R5_TRAIN_MATCHUP_WEIGHTS", GameConfig.TRAINING_MATCHUP_WEIGHTS)
        lineups = parse_weighted_matchups(spec)
        if logger:
            logger.info(
                "R6 training matchup schedule: custom weighted cycle "
                f"{describe_lineups(lineups)} from R5_TRAIN_MATCHUP_WEIGHTS={spec!r}"
            )
        return cycle_lineup_iterator(lineups)
    if matchup in ("champion_direnjie_focus", "direnjie_luban_focus", "133_112_focus"):
        lineups = [
            [133, 112],
            [112, 133],
            [133, 112],
            [112, 133],
            [133, 133],
            [112, 112],
        ]
        if logger:
            logger.info(
                "R6 training matchup schedule: champion direnjie-luban focus cycle "
                "[133v112, 112v133, 133v112, 112v133, 133v133, 112v112]"
            )
        return cycle_lineup_iterator(lineups)
    if matchup in ("direnjie_focus", "direnjie_remedial"):
        lineups = [
            [133, 112],
            [133, 133],
            [133, 112],
            [133, 133],
            [112, 133],
            [112, 112],
        ]
        if logger:
            logger.info(
                "R5 training matchup schedule: direnjie focus cycle "
                "[133v112, 133v133, 133v112, 133v133, 112v133, 112v112]"
            )
        return cycle_lineup_iterator(lineups)
    if matchup not in MATCHUP_LINEUPS:
        valid = ", ".join(
            [
                "all",
                "custom_weighted",
                "champion_direnjie_focus",
                "direnjie_focus",
                "112v112",
                "112v133",
                "133v112",
                "133v133",
            ]
        )
        raise ValueError(f"Invalid R5_TRAIN_MATCHUP={matchup!r}. Valid options: {valid}")
    lineup = MATCHUP_LINEUPS[matchup]
    if logger:
        logger.info(f"R5 training matchup schedule: fixed {matchup}, blue={lineup[0]}, red={lineup[1]}")
    return fixed_lineup_iterator(lineup)


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    # Whether the agent is training, corresponding to do_predicts
    # 智能体是否进行训练
    do_learns = [True, True]
    last_save_model_time = time.time()

    if logger:
        logger.info(
            "R5.6.1 platform longtrain config: "
            f"phase={GameConfig.TRAINING_PHASE}, matchup={GameConfig.TRAINING_MATCHUP}, "
            f"tower_edge_poke={GameConfig.REWARD_WEIGHT_DICT.get('tower_edge_poke')}, "
            f"hero_skill_pressure={GameConfig.REWARD_WEIGHT_DICT.get('hero_skill_pressure')}, "
            f"matchup_strategy={GameConfig.REWARD_WEIGHT_DICT.get('matchup_strategy')}, "
            f"action_priority={GameConfig.ACTION_PRIORITY_ENABLED}, "
            f"priority_train_scale={GameConfig.ACTION_PRIORITY_TRAIN_SCALE}, "
            f"priority_eval_scale={GameConfig.ACTION_PRIORITY_EVAL_SCALE}"
        )

    # Create environment configuration manager instance
    # 创建对局配置管理器实例
    env_conf_manager = EnvConfManager(
        config_path="agent_ppo/conf/train_env_conf.toml",
        logger=logger,
    )

    # Lineup iterator (112:Luban, 133:DiRenjie)
    # 阵容迭代器 (112:鲁班， 133:狄仁杰)
    lineup_iterator = build_training_lineup_iterator(logger)

    # Create EpisodeRunner instance
    # 创建 EpisodeRunner 实例
    episode_runner = EpisodeRunner(
        env=envs[0],
        agents=agents,
        logger=logger,
        monitor=monitor,
        env_conf_manager=env_conf_manager,
        lineup_iterator=lineup_iterator,
    )

    while True:
        # Run episodes and collect data
        # 运行对局并收集数据
        for g_data in episode_runner.run_episodes():
            for index, (d_learn, agent) in enumerate(zip(do_learns, agents)):
                if d_learn and len(g_data[index]) > 0:
                    # The learner trains in a while true loop, here learn actually sends samples
                    # learner 采用 while true 训练，此处 learn 实际为发送样本
                    agent.send_sample_data(g_data[index])
            g_data.clear()

            now = time.time()
            if now - last_save_model_time > GameConfig.MODEL_SAVE_INTERVAL:
                agents[0].save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agents, logger, monitor, env_conf_manager, lineup_iterator):
        self.env = env
        self.agents = agents
        self.logger = logger
        self.monitor = monitor
        self.env_conf_manager = env_conf_manager
        self.lineup_iterator = lineup_iterator
        self.agent_num = len(agents)
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.eval_matchup_keys = eval_matchup_key_iterator()

    def _call_init_config(self, usr_conf):
        """Call init_config on both agents to get summoner skill selections,
        then inject the results into usr_conf.
        调用双方 agent 的 init_config 获取召唤师技能选择，并注入 usr_conf。
        """
        blue_hero_ids, red_hero_ids = EnvConfManager.extract_hero_ids_from_usr_conf(usr_conf)

        # camp_keys[i] is the camp key for agents[i] based on monitor_side
        # monitor_side 的 agent 对应 blue/red 取决于 monitor_side 配置
        monitor_side = self.env_conf_manager.get_monitor_side()
        camp_keys = ["blue_camp", "red_camp"]

        for agent_idx, agent in enumerate(self.agents):
            # Determine which camp this agent controls
            # 确定该 agent 控制哪个阵营
            if agent_idx == 0:
                my_hero_ids = blue_hero_ids
                opponent_hero_ids = red_hero_ids
                camp_key = camp_keys[0]
            else:
                my_hero_ids = red_hero_ids
                opponent_hero_ids = blue_hero_ids
                camp_key = camp_keys[1]

            config_data = {
                "my_camp": camp_key,
                "my_heroes": my_hero_ids,
                "opponent_heroes": opponent_hero_ids,
            }

            select_skills = agent.init_config(config_data)
            EnvConfManager.inject_select_skills(usr_conf, camp_key, select_skills)
            self.logger.info(
                f"Agent[{agent_idx}] init_config: camp={camp_key}, select_skills={select_skills}"
            )

    def run_episodes(self):
        # Single environment process
        # 单局流程
        while True:
            # Retrieving training metrics
            # 获取训练中的指标
            training_metrics = get_training_metrics()
            if training_metrics:
                for key, value in training_metrics.items():
                    if key == "env":
                        for env_key, env_value in value.items():
                            self.logger.info(f"training_metrics {key} {env_key} is {env_value}")
                    else:
                        self.logger.info(f"training_metrics {key} is {value}")

            # Update environment configuration
            # Can use a list of length 2 to pass in the lineup id of the current game
            # 更新对局配置, 可以用长度为2的列表传入当前对局的阵容id
            lineup = next(self.lineup_iterator)
            usr_conf, is_eval, monitor_side = self.env_conf_manager.update_config(lineup)
            if is_eval:
                eval_matchup_key = next(self.eval_matchup_keys)
                eval_lineup = lineup_for_model_matchup(eval_matchup_key, monitor_side)
                if set_usr_conf_lineup(usr_conf, eval_lineup):
                    lineup = eval_lineup
                else:
                    eval_matchup_key = matchup_key_from_usr_conf(usr_conf, monitor_side)
            else:
                eval_matchup_key = matchup_key_from_usr_conf(usr_conf, monitor_side)

            # Call init_config on agents to get summoner skill selections
            # 调用 agent 的 init_config 获取召唤师技能选择，注入 usr_conf
            self._call_init_config(usr_conf)

            # Start a new environment
            # 启动新对局，返回初始环境状态

            env_obs = self.env.reset(usr_conf=usr_conf)
            # Disaster recovery
            # 容灾
            if handle_disaster_recovery(env_obs, self.logger):
                break

            observation = env_obs["observation"]
            extra_info = env_obs["extra_info"]

            # Reset agents
            # 重置智能体
            self.reset_agents(observation)

            # Reset environment frame collector
            # 重置环境帧收集器
            frame_collector = FrameCollector(self.agent_num)

            # Game variables
            # 对局变量
            self.episode_cnt += 1
            frame_no = 0
            reward_sum_list = [0] * self.agent_num
            is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
            self.logger.info(
                f"Episode {self.episode_cnt} start, eval:{is_eval}, "
                f"matchup:{eval_matchup_key}, usr_conf is {usr_conf}"
            )

            # Reward initialization
            # 回报初始化
            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                if do_sample:
                    reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                    observation[str(i)]["reward"] = reward
                    reward_sum_list[i] += reward["reward_sum"]

            while True:
                # Initialize the default actions. If the agent does not make a decision, env.step uses the default action.
                # 初始化默认的actions，如果智能体不进行决策，则env.step使用默认action
                actions = [NONE_ACTION] * self.agent_num

                for index, (do_predict, do_sample, agent) in enumerate(
                    zip(self.do_predicts, self.do_samples, self.agents)
                ):
                    if do_predict:
                        if not is_eval:
                            actions[index] = agent.predict(observation[str(index)])
                        else:
                            actions[index] = agent.exploit(observation[str(index)])
                        if getattr(agent, "reward_manager", None) is not None:
                            agent.reward_manager.record_action(actions[index])

                        # Only sample when do_sample=True and is_eval=False
                        # 评估对局数据不采样，不是训练中最新模型产生的数据不采样
                        if not is_eval and do_sample:
                            frame = build_frame(agent, observation[str(index)])
                            frame_collector.save_frame(frame, agent_id=index)

                # Step forward
                # 推进环境到下一帧，得到新的状态
                env_reward, env_obs = self.env.step(actions)
                # Disaster recovery
                # 容灾
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                frame_no = env_obs["frame_no"]
                observation = env_obs["observation"]
                extra_info = env_obs["extra_info"]
                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]

                # Reward generation
                # 计算回报，作为当前环境状态observation的一部分
                for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                    if do_sample:
                        reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                        observation[str(i)]["reward"] = reward
                        reward_sum_list[i] += reward["reward_sum"]

                # Normal end or timeout exit, run train_test will exit early
                # 正常结束或超时退出，运行train_test时会提前退出
                is_gameover = terminated or truncated or (is_train_test and frame_no >= 1000)
                if is_gameover:
                    for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                        if not is_eval and do_sample and not is_train_test:
                            bonus = _terminal_reward_bonus(
                                observation[str(i)]["frame_state"],
                                getattr(agent, "hero_camp", None),
                                truncated,
                            )
                            if bonus:
                                observation[str(i)]["reward"] = _add_terminal_reward(
                                    observation[str(i)]["reward"],
                                    bonus,
                                )
                                reward_sum_list[i] += bonus
                    if self.logger:
                        for i, agent in enumerate(self.agents):
                            reward_manager = getattr(agent, "reward_manager", None)
                            if reward_manager and hasattr(reward_manager, "get_episode_stats"):
                                self.logger.info(
                                    f"R2_3_LANE_SUMMARY episode={self.episode_cnt} agent={i} "
                                    f"frame_no={frame_no} truncated={truncated} stats={reward_manager.get_episode_stats()}"
                                )
                    self.logger.info(
                        f"episode_{self.episode_cnt} terminated in fno_{frame_no}, truncated:{truncated}, eval:{is_eval}, reward_sum:{reward_sum_list[monitor_side]}"
                    )
                    # Reward for saving the last state of the environment
                    # 保存环境最后状态的reward
                    for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                        if not is_eval and do_sample:
                            frame_collector.save_last_frame(
                                agent_id=i,
                                reward=observation[str(i)]["reward"],
                            )

                    now = time.time()
                    should_report_monitor = is_eval or (now - self.last_report_monitor_time >= 60)
                    if should_report_monitor:
                        monitor_data = {"episode_cnt": self.episode_cnt}
                        if self.monitor:
                            if is_eval:
                                monitor_data["reward"] = round(reward_sum_list[monitor_side], 2)
                                monitor_data.update(
                                    build_eval_matchup_monitor_data(
                                        eval_matchup_key,
                                        observation,
                                        monitor_side,
                                        frame_no,
                                        truncated,
                                        getattr(self.agents[monitor_side], "hero_camp", None),
                                    )
                                )
                            self.monitor.put_data({os.getpid(): monitor_data})
                            if not is_eval:
                                self.last_report_monitor_time = now

                    # Sample process
                    # 进行样本处理，准备训练
                    if len(frame_collector) > 0 and not is_eval:
                        list_agents_samples = sample_process(frame_collector)
                        yield list_agents_samples
                    break

    def reset_agents(self, observation):
        opponent_agent = self.env_conf_manager.get_opponent_agent()
        monitor_side = self.env_conf_manager.get_monitor_side()
        is_train_test = os.environ.get("is_train_test", "False").lower() == "true"

        # The 'do_predicts' specifies which agents are to perform model predictions.
        # do_predicts 指定哪些智能体要进行模型预测
        # The 'do_samples' specifies which agents are to perform training sampling.
        # do_samples 指定哪些智能体要进行训练采样
        self.do_predicts = [True, True]
        self.do_samples = [True, True]

        # Load model according to the configuration
        # 根据对局配置加载模型
        for i, agent in enumerate(self.agents):
            # Report the latest model in the training camp to the monitor
            # 训练中最新模型所在阵营上报监控
            if i == monitor_side:
                # monitor_side uses the latest model
                # monitor_side 使用最新模型
                agent.load_model(id="latest")
            else:
                if opponent_agent == "common_ai":
                    # common_ai does not need to load a model, no need to predict
                    # 如果对手是 common_ai 则不需要加载模型, 也不需要进行预测
                    self.do_predicts[i] = False
                    self.do_samples[i] = False
                elif opponent_agent == "selfplay":
                    # Training model, "latest" - latest model, "random" - random model from the model pool
                    # 加载训练过的模型，可以选择最新模型，也可以选择随机模型 "latest" - 最新模型, "random" - 模型池中随机模型
                    agent.load_model(id="latest")
                else:
                    # Opponent model, model_id is checked from kaiwu.json
                    # 选择kaiwu.json中设置的对手模型, model_id 即 opponent_agent，必须设置正确否则报错
                    eval_candidate_model = get_valid_model_pool(self.logger)
                    if int(opponent_agent) not in eval_candidate_model:
                        raise Exception(f"opponent_agent model_id {opponent_agent} not in {eval_candidate_model}")
                    else:
                        if is_train_test:
                            # Run train_test, cannot get opponent agent, so replace with latest model
                            # 运行 train_test 时, 无法获取到对手模型，因此将替换为最新模型
                            self.logger.info(f"Run train_test, cannot get opponent agent, so replace with latest model")
                            agent.load_model(id="latest")
                        else:
                            agent.load_opponent_agent(id=opponent_agent)
                        self.do_samples[i] = False
            # Reset agent
            # 重置agent
            agent.reset(observation[str(i)])
