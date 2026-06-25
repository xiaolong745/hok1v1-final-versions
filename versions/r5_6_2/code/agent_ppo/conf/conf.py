#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import os


def _env_choice(name, default):
    value = os.environ.get(name, default)
    return str(value).strip().lower() or default


def _env_int(name, default):
    value = os.environ.get(name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _env_bool(name, default):
    value = str(os.environ.get(name, default)).strip().lower()
    return value not in {"0", "false", "off", "no"}


def _env_float(name, default):
    value = os.environ.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _merge_reward_weights(base_weights, override_weights):
    weights = dict(base_weights)
    weights.update(override_weights)
    return weights


class GameConfig:
    # R5.6.1 keeps the frozen R5.3 observation/model contract and trains the
    # tactical priority layer across all four lineups by default.
    TRAINING_PHASE = _env_choice("R5_TRAINING_PHASE", "tactic")
    # Training matchup schedule. Override with env var R5_TRAIN_MATCHUP.
    # Options: all, 112v112, 112v133, 133v112, 133v133, direnjie_focus.
    TRAINING_MATCHUP = _env_choice("R5_TRAIN_MATCHUP", "all")
    # Run a fixed common_ai evaluation probe every N started episodes so the
    # selfplay policy still has a stable external health check.
    COMMON_AI_BASELINE_ENABLED = _env_bool("R5_COMMON_AI_BASELINE", "false")
    COMMON_AI_BASELINE_INTERVAL = max(4, _env_int("R5_COMMON_AI_BASELINE_INTERVAL", 40))
    FIXED_EVAL_WINDOW = max(4, _env_int("R5_FIXED_EVAL_WINDOW", 24))
    ACTION_PRIORITY_ENABLED = _env_bool("R5_ACTION_PRIORITY_ENABLED", "true")
    ACTION_PRIORITY_TRAIN_SCALE = _env_float("R5_ACTION_PRIORITY_TRAIN_SCALE", 0.12)
    ACTION_PRIORITY_EVAL_SCALE = _env_float("R5_ACTION_PRIORITY_EVAL_SCALE", 0.30)
    # Set the weight of each reward item and use it in reward_manager
    # 设置各个回报项的权重，在reward_manager中使用
    BASE_REWARD_WEIGHT_DICT = {
        "tower_hp_point": 8.5,
        "own_tower_hp_point": 6.5,
        "hp_trade": 0.35,
        "economy": 0.8,
        "exp": 0.0,
        "kill": 1.0,
        "death": 4.5,
        "safe_tower_pressure": 0.55,
        "tower_edge_poke": 0.35,
        "tower_window_conversion": 1.85,
        "missed_tower_window": 1.1,
        "late_tower_urgency": 1.35,
        "unsafe_tower_dive": 0.8,
        "low_hp_retreat": 0.28,
        "lane_pressure": 0.2,
        "lane_clear": 0.9,
        "over_wave_fight": 1.1,
        "neutral_resource": 0.75,
        "recovery_supply": 1.15,
        "recall_discipline": 0.9,
        "attack_move_rhythm": 0.35,
        "spacing_control": 0.35,
        "post_skill_attack": 0.35,
        "skill_discipline": 0.5,
        "skill_tower_waste": 0.9,
        "passive_followup": 0.5,
        "hero_skill_pressure": 0.15,
        "matchup_strategy": 0.0,
        "post_kill_conversion": 1.35,
        "grass_reposition": 0.18,
        "forward": 0.0,
    }
    PHASE_REWARD_OVERRIDES = {
        # Basic operation stage: keep matchup rewards mostly quiet so the model
        # first stabilizes attack, lane, tower, movement, and skill basics.
        "base": {
            "tower_edge_poke": 0.35,
            "hero_skill_pressure": 0.15,
            "matchup_strategy": 0.0,
            "recovery_supply": 0.95,
            "recall_discipline": 0.45,
        },
        # R5.5 tactical selfplay stage: keep the frozen schema, but tighten
        # low-HP trade, tower-pressure, grass-reset, and objective conversion
        # behavior so selfplay does not collapse into passive retreats.
        "tactic": {
            "tower_edge_poke": 0.9,
            "hero_skill_pressure": 0.62,
            "matchup_strategy": 1.2,
            "passive_followup": 0.78,
            "post_skill_attack": 0.48,
            "spacing_control": 0.52,
            "skill_discipline": 0.62,
            "skill_tower_waste": 0.78,
            "recovery_supply": 0.6,
            "recall_discipline": 0.18,
            "low_hp_retreat": 0.14,
            "neutral_resource": 0.4,
            "lane_clear": 0.8,
        },
        # DirenJie remedial stage: keep the overall tactical structure but bias
        # the optimization budget toward DiRenJie timing, spacing, and tower
        # conversion so the four-cell evaluation matrix stays balanced.
        "direnjie_remedial": {
            "tower_edge_poke": 0.88,
            "hero_skill_pressure": 0.82,
            "matchup_strategy": 1.55,
            "passive_followup": 0.62,
            "post_skill_attack": 0.58,
            "spacing_control": 0.68,
            "skill_discipline": 0.78,
            "skill_tower_waste": 0.90,
            "recovery_supply": 0.55,
            "recall_discipline": 0.14,
            "low_hp_retreat": 0.10,
            "neutral_resource": 0.32,
            "lane_clear": 0.76,
        },
    }
    REWARD_WEIGHT_DICT = _merge_reward_weights(
        BASE_REWARD_WEIGHT_DICT,
        PHASE_REWARD_OVERRIDES.get(TRAINING_PHASE, PHASE_REWARD_OVERRIDES["tactic"]),
    )

    MATCHUP_PROFILE_SCALES = {
        "default": {},
        "112v112": {
            "passive_followup": 1.20,
            "matchup_strategy": 1.05,
            "tower_edge_poke": 1.05,
        },
        "112v133": {
            "tower_edge_poke": 1.20,
            "matchup_strategy": 1.15,
            "spacing_control": 1.15,
            "unsafe_tower_dive": 1.10,
        },
        "133v112": {
            "hero_skill_pressure": 1.30,
            "matchup_strategy": 1.20,
            "spacing_control": 1.10,
            "tower_edge_poke": 1.05,
        },
        "133v133": {
            "hero_skill_pressure": 1.25,
            "matchup_strategy": 1.25,
            "skill_discipline": 1.15,
            "spacing_control": 1.15,
            "tower_edge_poke": 0.90,
        },
    }

    DIRENJIE_REMEDIAL_MATCHUP_SCALES = {
        "default": {},
        "112v112": {
            "passive_followup": 0.92,
            "tower_edge_poke": 0.92,
            "matchup_strategy": 0.90,
        },
        "112v133": {
            "tower_edge_poke": 0.96,
            "matchup_strategy": 0.96,
            "spacing_control": 1.00,
        },
        "133v112": {
            "hero_skill_pressure": 1.55,
            "matchup_strategy": 1.45,
            "spacing_control": 1.25,
            "post_skill_attack": 1.20,
            "skill_discipline": 1.22,
            "tower_edge_poke": 1.10,
        },
        "133v133": {
            "hero_skill_pressure": 1.50,
            "matchup_strategy": 1.50,
            "skill_discipline": 1.30,
            "spacing_control": 1.25,
            "post_skill_attack": 1.15,
            "tower_edge_poke": 1.00,
        },
    }

    @classmethod
    def matchup_profile_scales(cls, matchup_key):
        phase = str(getattr(cls, "TRAINING_PHASE", "tactic")).strip().lower()
        if phase == "direnjie_remedial":
            table = cls.DIRENJIE_REMEDIAL_MATCHUP_SCALES
        else:
            table = cls.MATCHUP_PROFILE_SCALES
        scales = dict(table.get("default", {}))
        scales.update(table.get(matchup_key, {}))
        return scales
    # Time decay factor, used in reward_manager
    # 时间衰减因子，在reward_manager中使用
    TIME_SCALE_ARG = 0
    # Model save interval configuration, used in workflow
    # 模型保存间隔配置，在workflow中使用
    MODEL_SAVE_INTERVAL = 1800


# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
class DimConfig:
    BASE_FEATURE_DIM = 208
    HERO_UNIT_FEATURE_DIM = 32
    HERO_UNIT_COUNT = 1
    SOLDIER_UNIT_FEATURE_DIM = 20
    SOLDIER_UNIT_COUNT = 4
    ORGAN_UNIT_FEATURE_DIM = 20
    ORGAN_UNIT_COUNT = 2
    LOCAL_MAP_GRID_SIZE = 8
    LOCAL_MAP_CHANNELS = 6
    LOCAL_MAP_FEATURE_DIM = LOCAL_MAP_GRID_SIZE * LOCAL_MAP_GRID_SIZE * LOCAL_MAP_CHANNELS
    UNIT_GROUP_FEATURE_DIM = (
        HERO_UNIT_FEATURE_DIM * HERO_UNIT_COUNT * 2
        + SOLDIER_UNIT_FEATURE_DIM * SOLDIER_UNIT_COUNT * 2
        + ORGAN_UNIT_FEATURE_DIM * ORGAN_UNIT_COUNT * 2
    )
    DIM_OF_FEATURE = [BASE_FEATURE_DIM + UNIT_GROUP_FEATURE_DIM + LOCAL_MAP_FEATURE_DIM]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    ACTION_PRIORITY_ENABLED = GameConfig.ACTION_PRIORITY_ENABLED
    ACTION_PRIORITY_TRAIN_SCALE = GameConfig.ACTION_PRIORITY_TRAIN_SCALE
    ACTION_PRIORITY_EVAL_SCALE = GameConfig.ACTION_PRIORITY_EVAL_SCALE
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DATA_SPLIT_SHAPE = [
        DimConfig.DIM_OF_FEATURE[0] + 85,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        LSTM_UNIT_SIZE,
        LSTM_UNIT_SIZE,
    ]
    SERI_VEC_SPLIT_SHAPE = [(DimConfig.DIM_OF_FEATURE[0],), (85,)]
    INIT_LEARNING_RATE_START = 1e-3
    TARGET_LR = 1e-4
    TARGET_STEP = 5000
    BETA_START = 0.025
    LOG_EPSILON = 1e-6
    VALUE_LOSS_COEF = 0.5
    ADVANTAGE_NORM_EPS = 1e-8
    POLICY_RATIO_MAX = 3.0
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001

    TARGET_EMBED_DIM = 32

    data_shapes = [
        [(DimConfig.DIM_OF_FEATURE[0] + 85) * 16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [144],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [512],
        [512],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
