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


def _env_text(name, default):
    value = os.environ.get(name, default)
    value = str(value).strip()
    return value if value else default


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
    # R8 is a new observation/value contract. Keep the old R5 env knobs as
    # fallbacks so existing platform scripts can still override schedules.
    TRAINING_PHASE = _env_choice("R8_TRAINING_PHASE", os.environ.get("R5_TRAINING_PHASE", "r8"))
    # Training matchup schedule. Override with env var R5_TRAIN_MATCHUP.
    # Options: all, 112v112, 112v133, 133v112, 133v133,
    # champion_direnjie_focus, direnjie_focus, custom_weighted.
    TRAINING_MATCHUP = _env_choice("R5_TRAIN_MATCHUP", "custom_weighted")
    # Used only when TRAINING_MATCHUP is custom_weighted. Format:
    # "133v112:4,112v133:2,133v133:2,112v112:1".
    TRAINING_MATCHUP_WEIGHTS = _env_text(
        "R5_TRAIN_MATCHUP_WEIGHTS",
        "133v112:4,112v133:2,133v133:2,112v112:1",
    )
    # Run a fixed common_ai evaluation probe every N started episodes so the
    # selfplay policy still has a stable external health check.
    COMMON_AI_BASELINE_ENABLED = _env_bool("R5_COMMON_AI_BASELINE", "false")
    COMMON_AI_BASELINE_INTERVAL = max(4, _env_int("R5_COMMON_AI_BASELINE_INTERVAL", 40))
    FIXED_EVAL_WINDOW = max(4, _env_int("R5_FIXED_EVAL_WINDOW", 24))
    ACTION_PRIORITY_ENABLED = _env_bool("R8_ACTION_PRIORITY_ENABLED", "true")
    ACTION_PRIORITY_TRAIN_SCALE = _env_float("R8_ACTION_PRIORITY_TRAIN_SCALE", 0.08)
    ACTION_PRIORITY_EVAL_SCALE = _env_float("R8_ACTION_PRIORITY_EVAL_SCALE", 0.20)
    VALUE_HEAD_NAMES = ("total", "objective", "lane", "combat", "survival", "resource")
    VALUE_HEAD_LOSS_WEIGHTS = (1.00, 0.30, 0.20, 0.20, 0.25, 0.20)
    VALUE_HEAD_CLIPS = {
        "objective": (-3.0, 4.0),
        "lane": (-1.5, 2.0),
        "combat": (-2.0, 2.5),
        "survival": (-4.0, 1.5),
        "resource": (-1.0, 1.2),
    }
    REWARD_HEAD_GROUPS = {
        "objective": (
            "tower_hp_point",
            "own_tower_hp_point",
            "safe_tower_pressure",
            "tower_edge_poke",
            "tower_window_conversion",
            "missed_tower_window",
            "late_tower_urgency",
            "tower_defense_pressure",
            "post_kill_conversion",
            "forward",
        ),
        "lane": (
            "economy",
            "exp",
            "lane_pressure",
            "lane_clear",
            "over_wave_fight",
            "neutral_resource",
        ),
        "combat": (
            "hp_trade",
            "kill",
            "attack_move_rhythm",
            "spacing_control",
            "post_skill_attack",
            "skill_discipline",
            "skill_tower_waste",
            "passive_followup",
            "hero_skill_pressure",
            "matchup_strategy",
        ),
        "survival": (
            "death",
            "unsafe_tower_dive",
            "low_hp_retreat",
            "recovery_supply",
            "recall_discipline",
            "grass_reposition",
        ),
        "resource": (
            "ep_discipline",
        ),
    }
    REWARD_HEAD_BY_ITEM = {
        reward_name: head_name
        for head_name, reward_names in REWARD_HEAD_GROUPS.items()
        for reward_name in reward_names
    }
    # Set the weight of each reward item and use it in reward_manager
    # 设置各个回报项的权重，在reward_manager中使用
    BASE_REWARD_WEIGHT_DICT = {
        "tower_hp_point": 8.5,
        "own_tower_hp_point": 6.5,
        "hp_trade": 0.45,
        "economy": 0.8,
        "exp": 0.0,
        "kill": 4.5,
        "death": 4.5,
        "safe_tower_pressure": 0.55,
        "tower_edge_poke": 0.35,
        "tower_window_conversion": 1.85,
        "missed_tower_window": 1.1,
        "late_tower_urgency": 1.35,
        "tower_defense_pressure": 0.85,
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
        "ep_discipline": 0.55,
        "skill_tower_waste": 0.9,
        "passive_followup": 0.5,
        "hero_skill_pressure": 0.15,
        "matchup_strategy": 0.0,
        "post_kill_conversion": 1.35,
        "grass_reposition": 0.18,
        "forward": 0.08,
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
            "forward": 0.10,
        },
        # R8 is trained from scratch with explicit objective/lane/combat heads.
        # Keep forward as a weak anti-idle signal so it cannot overpower tower,
        # lane, survival, or resource discipline rewards.
        "r8": {
            "tower_window_conversion": 2.10,
            "missed_tower_window": 1.35,
            "late_tower_urgency": 1.70,
            "tower_defense_pressure": 1.15,
            "post_kill_conversion": 1.70,
            "safe_tower_pressure": 0.72,
            "tower_edge_poke": 0.78,
            "hp_trade": 0.58,
            "kill": 4.5,
            "hero_skill_pressure": 0.68,
            "matchup_strategy": 0.0,
            "passive_followup": 0.90,
            "post_skill_attack": 0.55,
            "spacing_control": 0.50,
            "skill_discipline": 0.60,
            "ep_discipline": 0.68,
            "skill_tower_waste": 0.76,
            "recovery_supply": 0.68,
            "recall_discipline": 0.34,
            "low_hp_retreat": 0.16,
            "neutral_resource": 0.42,
            "lane_clear": 0.78,
            "forward": 0.10,
            "grass_reposition": 0.20,
        },
        # R5.5 tactical selfplay stage: keep the frozen schema, but tighten
        # low-HP trade, tower-pressure, grass-reset, and objective conversion
        # behavior so selfplay does not collapse into passive retreats.
        "tactic": {
            "tower_edge_poke": 0.9,
            "hp_trade": 0.75,
            "kill": 4.5,
            "hero_skill_pressure": 0.78,
            "matchup_strategy": 1.2,
            "passive_followup": 0.78,
            "post_skill_attack": 0.48,
            "spacing_control": 0.52,
            "skill_discipline": 0.74,
            "ep_discipline": 0.78,
            "skill_tower_waste": 0.78,
            "recovery_supply": 0.6,
            "recall_discipline": 0.18,
            "low_hp_retreat": 0.14,
            "neutral_resource": 0.32,
            "lane_clear": 0.68,
            "forward": 0.06,
            "grass_reposition": 0.22,
        },
        # DirenJie remedial stage: keep the overall tactical structure but bias
        # the optimization budget toward DiRenJie timing, spacing, and tower
        # conversion so the four-cell evaluation matrix stays balanced.
        "direnjie_remedial": {
            "tower_edge_poke": 0.88,
            "hp_trade": 0.72,
            "kill": 4.5,
            "hero_skill_pressure": 0.82,
            "matchup_strategy": 1.55,
            "passive_followup": 0.62,
            "post_skill_attack": 0.58,
            "spacing_control": 0.68,
            "skill_discipline": 0.78,
            "ep_discipline": 0.82,
            "skill_tower_waste": 0.90,
            "recovery_supply": 0.55,
            "recall_discipline": 0.14,
            "low_hp_retreat": 0.10,
            "neutral_resource": 0.32,
            "lane_clear": 0.76,
            "forward": 0.05,
        },
    }
    REWARD_WEIGHT_DICT = _merge_reward_weights(
        BASE_REWARD_WEIGHT_DICT,
        PHASE_REWARD_OVERRIDES.get(TRAINING_PHASE, PHASE_REWARD_OVERRIDES["r8"]),
    )

    MATCHUP_PROFILE_SCALES = {
        "default": {},
        "112v112": {
            "passive_followup": 1.20,
            "matchup_strategy": 1.05,
            "tower_edge_poke": 1.05,
        },
        "112v133": {
            "hero_skill_pressure": 1.34,
            "passive_followup": 1.36,
            "tower_edge_poke": 1.36,
            "matchup_strategy": 1.30,
            "spacing_control": 1.15,
            "unsafe_tower_dive": 1.10,
            "lane_clear": 0.84,
        },
        "133v112": {
            "hero_skill_pressure": 1.48,
            "matchup_strategy": 1.62,
            "spacing_control": 1.22,
            "post_skill_attack": 1.18,
            "skill_discipline": 1.18,
            "ep_discipline": 1.12,
            "hp_trade": 1.12,
            "tower_edge_poke": 1.08,
            "lane_clear": 0.86,
            "recovery_supply": 0.82,
        },
        "133v133": {
            "hero_skill_pressure": 1.25,
            "matchup_strategy": 1.25,
            "skill_discipline": 1.15,
            "ep_discipline": 1.10,
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
            "ep_discipline": 1.16,
            "tower_edge_poke": 1.10,
        },
        "133v133": {
            "hero_skill_pressure": 1.50,
            "matchup_strategy": 1.50,
            "skill_discipline": 1.30,
            "ep_discipline": 1.18,
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
    R8_DENSE_DIM = 256
    R8_DISCRETE_DIM = 640
    R8_UNIT_SLOT_COUNT = 16
    R8_UNIT_SLOT_DIM = 40
    R8_UNIT_FEATURE_DIM = R8_UNIT_SLOT_COUNT * R8_UNIT_SLOT_DIM
    R8_LANE_PROGRESS_BINS = 16
    R8_LANE_LATERAL_BINS = 4
    R8_LANE_GRID_CHANNELS = 8
    R8_LANE_GRID_DIM = R8_LANE_PROGRESS_BINS * R8_LANE_LATERAL_BINS * R8_LANE_GRID_CHANNELS
    R8_TEMPORAL_DIM = 160

    # Keep legacy names as aliases for old helpers that may still be imported.
    BASE_FEATURE_DIM = R8_DENSE_DIM
    HERO_UNIT_FEATURE_DIM = R8_UNIT_SLOT_DIM
    HERO_UNIT_COUNT = 2
    SOLDIER_UNIT_FEATURE_DIM = R8_UNIT_SLOT_DIM
    SOLDIER_UNIT_COUNT = 8
    ORGAN_UNIT_FEATURE_DIM = R8_UNIT_SLOT_DIM
    ORGAN_UNIT_COUNT = 4
    LOCAL_MAP_GRID_SIZE = R8_LANE_PROGRESS_BINS
    LOCAL_MAP_CHANNELS = R8_LANE_GRID_CHANNELS
    LOCAL_MAP_FEATURE_DIM = R8_LANE_GRID_DIM
    UNIT_GROUP_FEATURE_DIM = R8_UNIT_FEATURE_DIM
    DIM_OF_FEATURE = [
        R8_DENSE_DIM
        + R8_DISCRETE_DIM
        + R8_UNIT_FEATURE_DIM
        + R8_LANE_GRID_DIM
        + R8_TEMPORAL_DIM
    ]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    ACTION_PRIORITY_ENABLED = GameConfig.ACTION_PRIORITY_ENABLED
    ACTION_PRIORITY_TRAIN_SCALE = GameConfig.ACTION_PRIORITY_TRAIN_SCALE
    ACTION_PRIORITY_EVAL_SCALE = GameConfig.ACTION_PRIORITY_EVAL_SCALE
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    VALUE_HEAD_NAMES = GameConfig.VALUE_HEAD_NAMES
    VALUE_HEAD_DIM = len(VALUE_HEAD_NAMES)
    VALUE_HEAD_LOSS_WEIGHTS = GameConfig.VALUE_HEAD_LOSS_WEIGHTS
    DATA_SPLIT_SHAPE = [
        DimConfig.DIM_OF_FEATURE[0] + 85,
        VALUE_HEAD_DIM,
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
    INIT_LEARNING_RATE_START = _env_float("R8_INIT_LR", os.environ.get("R5_INIT_LR", 8e-5))
    TARGET_LR = _env_float("R8_TARGET_LR", os.environ.get("R5_TARGET_LR", 8e-5))
    TARGET_STEP = _env_int("R8_TARGET_STEP", os.environ.get("R5_TARGET_STEP", 1))
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
        [VALUE_HEAD_DIM * 16],
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
