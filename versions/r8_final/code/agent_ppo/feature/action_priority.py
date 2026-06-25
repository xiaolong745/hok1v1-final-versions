#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Action-priority logits prior for the frozen R5 tactical schema.

The prior never creates actions directly. It only adds small task-conditioned
logit offsets before the existing legal_action/sub_action_mask filtering.
"""

import numpy as np

from agent_ppo.conf.conf import DimConfig


TACTICAL_FEATURE_DIM = 200
# Tactical features sit after hero(3) + organ(7) inside the frozen 208-dim
# base block, not at the tail of the full 896-dim feature vector.
TACTICAL_START = DimConfig.BASE_FEATURE_DIM - TACTICAL_FEATURE_DIM

BASE_HERO_ALIVE = 0
BASE_HERO_LOC_X = 1
BASE_HERO_LOC_Z = 2
BASE_ENEMY_TOWER_ALIVE = 3
BASE_ENEMY_TOWER_REL_X = 7
BASE_ENEMY_TOWER_REL_Z = 8
BASE_ENEMY_TOWER_HP = 9
OWN_CAKE_X = -15220.0
OWN_CAKE_Z = -15120.0
HERO_LOC_MIN = -60000.0
HERO_LOC_RANGE = 120000.0

BUTTON_INVALID = 0
BUTTON_NONE = 1
BUTTON_MOVE = 2
BUTTON_ATTACK = 3
BUTTON_SKILL_1 = 4
BUTTON_SKILL_2 = 5
BUTTON_SKILL_3 = 6
BUTTON_HEAL = 7
BUTTON_SUMMONER = 8
BUTTON_RECALL = 9

TARGET_NONE = 0
TARGET_ENEMY = 1
TARGET_SELF = 2
TARGET_SOLDIER_MIN = 3
TARGET_SOLDIER_MAX = 6
TARGET_TOWER = 7
TARGET_MONSTER = 8

R8_DENSE_DIM = int(DimConfig.R8_DENSE_DIM)
R8_DISCRETE_DIM = int(DimConfig.R8_DISCRETE_DIM)
R8_UNIT_FEATURE_DIM = int(DimConfig.R8_UNIT_FEATURE_DIM)
R8_LANE_GRID_DIM = int(DimConfig.R8_LANE_GRID_DIM)
R8_TEMPORAL_DIM = int(DimConfig.R8_TEMPORAL_DIM)
R8_TOTAL_DIM = int(DimConfig.DIM_OF_FEATURE[0])
R8_DISCRETE_START = R8_DENSE_DIM
R8_TEMPORAL_START = R8_DENSE_DIM + R8_DISCRETE_DIM + R8_UNIT_FEATURE_DIM + R8_LANE_GRID_DIM

R8_MAIN_TASK_COUNT = 10
R8_TASK_POST_KILL_PUSH = 0
R8_TASK_TOWER_HIT = 1
R8_TASK_TOWER_STAGE = 2
R8_TASK_OWN_TOWER_DEFEND = 3
R8_TASK_KILL_COMMIT = 4
R8_TASK_FIGHT_POKE = 5
R8_TASK_LANE_CLEAR = 6
R8_TASK_RECOVER = 7
R8_TASK_FARM_NEUTRAL = 8
R8_TASK_IDLE_RESET = 9

# R8 temporal layout: 33 base + 10 task_scores + 28 tactical_summary + 10 main_task_onehot + ...
# Base includes tower HP stages/defense/out-pressure and explicit grass/vision flags.
# Tactical summary includes: hp_advantage(67), hp_ratio_level(68), in_grass(69), purposeful_grass(70)
R8_TEMPORAL_HP_ADVANTAGE = R8_TEMPORAL_START + 67
R8_TEMPORAL_HP_RATIO_LEVEL = R8_TEMPORAL_START + 68
R8_TEMPORAL_IN_GRASS = R8_TEMPORAL_START + 69
R8_TEMPORAL_PURPOSEFUL_GRASS = R8_TEMPORAL_START + 70
R8_TEMPORAL_TASK_OFFSET = R8_TEMPORAL_START + 71


class TacticalIndex:
    OWN_HP = 0
    ENEMY_HP = 1
    DIST_OWN_TOWER = 8
    DIST_ENEMY_TOWER = 9
    MOVE_DELTA = 29
    SHOULD_CLEAR_WAVE = 36
    CAN_PRESSURE_TOWER = 39
    MUST_HIT_TOWER_NOW = 40
    CAN_FIGHT = 42
    SHOULD_KITE_BACK = 43
    KILL_WINDOW = 44
    SHOULD_TAKE_CAKE = 46
    SHOULD_RECALL = 47
    LOW_HP_STALL = 48
    DANGER_LEVEL = 50
    PASSIVE_FOLLOWUP = 51
    OWN_IS_LUBAN = 54
    OWN_IS_DIRENJIE = 55
    ENEMY_IS_LUBAN = 56
    ENEMY_IS_DIRENJIE = 57
    DIRENJIE_SKILL1_POKE = 61
    DIRENJIE_SKILL2_ESCAPE = 62
    DIRENJIE_ULT_KILL = 64
    DIRENJIE_FOLLOWUP = 66
    ENEMY_DIRENJIE_THREAT = 67
    LUBAN_VS_DIRENJIE = 69
    DIRENJIE_VS_LUBAN = 70
    LUBAN_INTO_DIRENJIE_SAFE_PASSIVE = 73
    DIRENJIE_INTO_LUBAN_PUNISH = 74
    LUBAN_PASSIVE_TOWER = 154
    LUBAN_PASSIVE_HERO = 155
    LUBAN_FOLLOWUP = 156
    LUBAN_SKILL2_SAVE = 157
    LUBAN_SKILL2_EXECUTE = 158
    LUBAN_SKILL_WASTE = 159
    LUBAN_TOWER_IGNORE_ENEMY = 160
    ENEMY_MISSING = 185
    ENEMY_MISSING_RATIO = 186
    ENEMY_CLOSE_EXTENDED = 187
    COMBO_START = 188
    COMBO_CHAIN = 189
    COMBO_ABORT = 190
    DIRENJIE_SKILL2_COMBO = 191
    LUBAN_SKILL2_COMBO = 192
    EP_RATIO = 193
    EP_CONSERVE = 194
    EP_HOLD_SKILLS = 195
    EP_RESUME_READY = 196
    EP_CRITICAL = 197
    EP_RECOVER = 198
    TOWER_EDGE_OVERSTEP = 178
    TOWER_WAVE_LOST_RESET = 181
    # R8-c6: HP advantage and ratio level (corrected indices for 200-dim tactical vector)
    HP_ADVANTAGE = 76
    HP_RATIO_LEVEL = 77


def _clip01_np(value):
    return np.clip(value, 0.0, 1.0)


def _feature_tactical_np(feature):
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    end = TACTICAL_START + TACTICAL_FEATURE_DIM
    if feature.shape[0] < end:
        return None
    return feature[TACTICAL_START:end]


def _normalize_dir_np(x, z):
    norm = float(np.sqrt(x * x + z * z))
    if norm <= 1e-6:
        return 0.0, 0.0
    return float(x / norm), float(z / norm)


def _r8_main_task_index_np(feature):
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    if feature.shape[0] < R8_TOTAL_DIM:
        return None
    # R8 temporal layout: 33 base temporal values + 10 task scores +
    # 28 tactical summary values, then the 10-way main-task one-hot.
    task_offset = R8_TEMPORAL_TASK_OFFSET
    task_slice = feature[task_offset : task_offset + R8_MAIN_TASK_COUNT]
    if task_slice.shape[0] != R8_MAIN_TASK_COUNT:
        return None
    return int(np.argmax(task_slice))


def _base_enemy_tower_direction_np(feature):
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    if feature.shape[0] <= BASE_ENEMY_TOWER_HP:
        return 0.0, 0.0
    tower_known = max(
        float(_clip01_np(feature[BASE_ENEMY_TOWER_ALIVE])),
        float(_clip01_np(feature[BASE_ENEMY_TOWER_HP])),
    )
    if tower_known <= 0.0:
        return 0.0, 0.0
    x = float(np.clip((feature[BASE_ENEMY_TOWER_REL_X] - 0.5) * 2.0, -1.0, 1.0))
    z = float(np.clip((feature[BASE_ENEMY_TOWER_REL_Z] - 0.5) * 2.0, -1.0, 1.0))
    return _normalize_dir_np(x, z)


def _base_own_cake_direction_np(feature):
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    if feature.shape[0] <= BASE_HERO_LOC_Z or float(_clip01_np(feature[BASE_HERO_ALIVE])) <= 0.0:
        return 0.0, 0.0
    hero_x = float(feature[BASE_HERO_LOC_X]) * HERO_LOC_RANGE + HERO_LOC_MIN
    hero_z = float(feature[BASE_HERO_LOC_Z]) * HERO_LOC_RANGE + HERO_LOC_MIN
    return _normalize_dir_np(OWN_CAKE_X - hero_x, OWN_CAKE_Z - hero_z)


def _movement_direction_np(feature, tactical, short_reset, tower_pressure, commit_kill):
    enemy_tower_x, enemy_tower_z = _base_enemy_tower_direction_np(feature)
    own_tower_x, own_tower_z = -enemy_tower_x, -enemy_tower_z
    cake_x, cake_z = _base_own_cake_direction_np(feature)
    if abs(cake_x) + abs(cake_z) <= 1e-6:
        cake_x, cake_z = own_tower_x, own_tower_z

    should_recall = float(_clip01_np(tactical[TacticalIndex.SHOULD_RECALL]))
    should_take_cake = float(_clip01_np(tactical[TacticalIndex.SHOULD_TAKE_CAKE]))
    low_hp_stall = float(_clip01_np(tactical[TacticalIndex.LOW_HP_STALL]))
    tower_overstep = float(_clip01_np(tactical[TacticalIndex.TOWER_EDGE_OVERSTEP]))
    tower_reset = float(_clip01_np(tactical[TacticalIndex.TOWER_WAVE_LOST_RESET]))
    move_delta = float(_clip01_np(tactical[TacticalIndex.MOVE_DELTA]))
    stuck_escape = float(np.clip((0.08 - move_delta) / 0.08, 0.0, 1.0))

    escape_drive = float(_clip01_np(max(short_reset, should_recall, low_hp_stall, tower_reset, tower_overstep)))
    cake_drive = float(_clip01_np(should_take_cake))
    pressure_drive = float(_clip01_np(tower_pressure * (1.0 - max(escape_drive, commit_kill)) * (1.0 - tower_overstep)))

    x = cake_drive * cake_x + escape_drive * own_tower_x + 0.22 * pressure_drive * enemy_tower_x
    z = cake_drive * cake_z + escape_drive * own_tower_z + 0.22 * pressure_drive * enemy_tower_z

    perp_x, perp_z = -enemy_tower_z, enemy_tower_x
    side = -1.0 if cake_x * perp_x + cake_z * perp_z < 0.0 else 1.0
    lateral_drive = 0.45 * stuck_escape * escape_drive
    x += lateral_drive * side * perp_x
    z += lateral_drive * side * perp_z

    dir_x, dir_z = _normalize_dir_np(x, z)
    priority = float(_clip01_np(max(escape_drive, cake_drive, 0.30 * pressure_drive)))
    if abs(dir_x) + abs(dir_z) <= 1e-6:
        priority = 0.0
    return dir_x, dir_z, priority


def _task_scores_np(tactical, feature):
    own_hp = _clip01_np(tactical[TacticalIndex.OWN_HP])
    enemy_hp = _clip01_np(tactical[TacticalIndex.ENEMY_HP])
    should_clear = _clip01_np(tactical[TacticalIndex.SHOULD_CLEAR_WAVE])
    can_tower = _clip01_np(tactical[TacticalIndex.CAN_PRESSURE_TOWER])
    must_hit_tower = _clip01_np(tactical[TacticalIndex.MUST_HIT_TOWER_NOW])
    can_fight = _clip01_np(tactical[TacticalIndex.CAN_FIGHT])
    should_kite = _clip01_np(tactical[TacticalIndex.SHOULD_KITE_BACK])
    kill_window = _clip01_np(tactical[TacticalIndex.KILL_WINDOW])
    should_take_cake = _clip01_np(tactical[TacticalIndex.SHOULD_TAKE_CAKE])
    should_recall = _clip01_np(tactical[TacticalIndex.SHOULD_RECALL])
    low_hp_stall = _clip01_np(tactical[TacticalIndex.LOW_HP_STALL])
    danger = _clip01_np(tactical[TacticalIndex.DANGER_LEVEL])
    passive_followup = _clip01_np(tactical[TacticalIndex.PASSIVE_FOLLOWUP])
    tower_overstep = _clip01_np(tactical[TacticalIndex.TOWER_EDGE_OVERSTEP])
    tower_reset = _clip01_np(tactical[TacticalIndex.TOWER_WAVE_LOST_RESET])
    own_is_luban = _clip01_np(tactical[TacticalIndex.OWN_IS_LUBAN])
    own_is_direnjie = _clip01_np(tactical[TacticalIndex.OWN_IS_DIRENJIE])
    enemy_is_luban = _clip01_np(tactical[TacticalIndex.ENEMY_IS_LUBAN])
    enemy_is_direnjie = _clip01_np(tactical[TacticalIndex.ENEMY_IS_DIRENJIE])
    luban_vs_direnjie = own_is_luban * enemy_is_direnjie * _clip01_np(tactical[TacticalIndex.LUBAN_VS_DIRENJIE])
    enemy_direnjie_threat = enemy_is_direnjie * _clip01_np(tactical[TacticalIndex.ENEMY_DIRENJIE_THREAT])
    luban_safe_passive = own_is_luban * enemy_is_direnjie * _clip01_np(tactical[TacticalIndex.LUBAN_INTO_DIRENJIE_SAFE_PASSIVE])
    luban_passive_tower = own_is_luban * _clip01_np(tactical[TacticalIndex.LUBAN_PASSIVE_TOWER])
    luban_passive_hero = own_is_luban * _clip01_np(tactical[TacticalIndex.LUBAN_PASSIVE_HERO])
    luban_followup = own_is_luban * _clip01_np(tactical[TacticalIndex.LUBAN_FOLLOWUP])
    luban_skill2_save = own_is_luban * _clip01_np(tactical[TacticalIndex.LUBAN_SKILL2_SAVE])
    luban_skill2_execute = own_is_luban * _clip01_np(tactical[TacticalIndex.LUBAN_SKILL2_EXECUTE])
    luban_skill_waste = own_is_luban * _clip01_np(tactical[TacticalIndex.LUBAN_SKILL_WASTE])
    luban_tower_ignore_enemy = own_is_luban * _clip01_np(tactical[TacticalIndex.LUBAN_TOWER_IGNORE_ENEMY])
    enemy_missing = _clip01_np(tactical[TacticalIndex.ENEMY_MISSING])
    enemy_missing_ratio = _clip01_np(tactical[TacticalIndex.ENEMY_MISSING_RATIO])
    enemy_close_extended = _clip01_np(tactical[TacticalIndex.ENEMY_CLOSE_EXTENDED])
    combo_start = _clip01_np(tactical[TacticalIndex.COMBO_START])
    combo_chain = _clip01_np(tactical[TacticalIndex.COMBO_CHAIN])
    combo_abort = _clip01_np(tactical[TacticalIndex.COMBO_ABORT])
    direnjie_skill2_combo = own_is_direnjie * _clip01_np(tactical[TacticalIndex.DIRENJIE_SKILL2_COMBO])
    luban_skill2_combo = own_is_luban * _clip01_np(tactical[TacticalIndex.LUBAN_SKILL2_COMBO])
    direnjie_vs_luban = own_is_direnjie * enemy_is_luban * _clip01_np(tactical[TacticalIndex.DIRENJIE_VS_LUBAN])
    direnjie_punish_luban = own_is_direnjie * enemy_is_luban * _clip01_np(tactical[TacticalIndex.DIRENJIE_INTO_LUBAN_PUNISH])
    skill1_poke = own_is_direnjie * _clip01_np(tactical[TacticalIndex.DIRENJIE_SKILL1_POKE])
    skill2_escape = own_is_direnjie * _clip01_np(tactical[TacticalIndex.DIRENJIE_SKILL2_ESCAPE])
    ult_kill = own_is_direnjie * _clip01_np(tactical[TacticalIndex.DIRENJIE_ULT_KILL])
    direnjie_followup = own_is_direnjie * _clip01_np(tactical[TacticalIndex.DIRENJIE_FOLLOWUP])
    ep_conserve = _clip01_np(tactical[TacticalIndex.EP_CONSERVE])
    ep_hold_skills = _clip01_np(tactical[TacticalIndex.EP_HOLD_SKILLS])
    ep_resume_ready = _clip01_np(tactical[TacticalIndex.EP_RESUME_READY])
    ep_critical = _clip01_np(tactical[TacticalIndex.EP_CRITICAL])
    raw_ep_throttle = max(ep_conserve, ep_hold_skills, ep_critical)

    enemy_weaker = 1.0 if enemy_hp > 0.0 and enemy_hp + 0.08 <= own_hp else 0.0
    low_hp = 1.0 if own_hp < 0.48 else 0.0
    low_hp_commit = low_hp * enemy_weaker * can_fight

    commit_kill = max(kill_window, ult_kill, luban_skill2_execute, low_hp_commit)
    short_reset = max(
        should_recall,
        should_take_cake,
        low_hp_stall,
        tower_reset,
        tower_overstep * max(danger, 0.35),
        luban_skill2_save * max(danger, 0.45),
        should_kite * max(danger, 0.55 * skill2_escape),
    )
    short_reset *= 1.0 - 0.85 * commit_kill
    luban_lane_deny = own_is_luban * max(luban_passive_hero, luban_followup, luban_vs_direnjie * enemy_close_extended)
    tower_hit_intent = _clip01_np((must_hit_tower - 0.55) / 0.45)
    tower_hit = max(tower_hit_intent, luban_passive_tower, luban_tower_ignore_enemy)
    tower_stage = _clip01_np(can_tower * (1.0 - tower_hit))
    tower_pressure = max(tower_stage, tower_hit)
    hard_tower_objective = 1.0 if tower_hit > 0.86 and must_hit_tower > 0.72 and should_take_cake < 0.10 and should_recall < 0.10 else 0.0
    commit_kill *= 1.0 - 0.85 * hard_tower_objective
    short_reset *= 1.0 - 0.92 * hard_tower_objective
    ep_relief = max(0.55 * hard_tower_objective, 0.75 if enemy_hp <= 0.24 else 0.35 * commit_kill)
    ep_skill_throttle = _clip01_np(raw_ep_throttle * (1.0 - ep_relief))
    luban_return_rocket = own_is_luban * enemy_missing * max(hard_tower_objective, 0.55 * tower_stage, 0.65 * must_hit_tower) * (1.0 - combo_abort)
    objective_or_lane = max(tower_pressure, should_clear)
    direnjie_luban_pressure = max(
        direnjie_punish_luban,
        skill1_poke * (1.0 - 0.60 * low_hp_stall),
        ult_kill,
        direnjie_followup,
    )
    active_pressure_claim = max(
        objective_or_lane,
        can_fight,
        passive_followup,
        combo_start,
        combo_chain,
        luban_passive_hero,
        luban_followup,
        luban_lane_deny,
        direnjie_luban_pressure,
    )
    short_reset *= 1.0 - 0.70 * active_pressure_claim
    clear_wave = should_clear * (1.0 - 0.70 * commit_kill) * (1.0 - 0.25 * short_reset)
    tower_pressure *= (1.0 - (0.45 * (1.0 - hard_tower_objective)) * commit_kill) * (
        1.0 - (0.15 * (1.0 - hard_tower_objective)) * short_reset
    )
    hero_contest = max(
        can_fight,
        passive_followup,
        luban_passive_hero,
        luban_followup,
        luban_lane_deny,
        direnjie_followup,
        direnjie_luban_pressure,
        combo_start,
        combo_chain,
    )
    hero_contest *= 1.0 - 0.50 * combo_abort
    hero_contest *= 1.0 - 0.35 * short_reset
    hero_contest *= 1.0 - 0.18 * clear_wave
    if hero_contest > 0.62 and clear_wave > 0.0:
        clear_wave *= 0.52 if max(luban_lane_deny, direnjie_luban_pressure) > 0.35 else 0.68
    move_dir_x, move_dir_z, move_priority = _movement_direction_np(
        feature,
        tactical,
        float(_clip01_np(short_reset)),
        float(_clip01_np(tower_pressure)),
        float(_clip01_np(commit_kill)),
    )

    return {
        "clear_wave": _clip01_np(clear_wave),
        "tower_pressure": _clip01_np(tower_pressure),
        "hero_contest": _clip01_np(hero_contest),
        "commit_kill": _clip01_np(commit_kill),
        "short_reset": _clip01_np(short_reset),
        "take_cake": _clip01_np(should_take_cake),
        "low_hp_stall": _clip01_np(low_hp_stall),
        "move_dir_x": move_dir_x,
        "move_dir_z": move_dir_z,
        "move_priority": _clip01_np(move_priority),
        "hard_tower_objective": _clip01_np(hard_tower_objective),
        "tower_stage": _clip01_np(tower_stage),
        "tower_hit": _clip01_np(tower_hit),
        "luban_return_rocket": _clip01_np(luban_return_rocket),
        "luban_passive_tower": _clip01_np(luban_passive_tower),
        "luban_passive_hero": _clip01_np(luban_passive_hero),
        "luban_followup": _clip01_np(luban_followup),
        "luban_skill2_save": _clip01_np(luban_skill2_save),
        "luban_skill2_execute": _clip01_np(luban_skill2_execute),
        "luban_skill_waste": _clip01_np(luban_skill_waste),
        "luban_tower_ignore_enemy": _clip01_np(luban_tower_ignore_enemy),
        "luban_lane_deny": _clip01_np(luban_lane_deny),
        "enemy_missing": _clip01_np(enemy_missing),
        "enemy_missing_ratio": _clip01_np(enemy_missing_ratio),
        "combo_start": _clip01_np(combo_start),
        "combo_chain": _clip01_np(combo_chain),
        "combo_abort": _clip01_np(combo_abort),
        "direnjie_skill2_combo": _clip01_np(direnjie_skill2_combo),
        "luban_skill2_combo": _clip01_np(luban_skill2_combo),
        "luban_vs_direnjie": _clip01_np(luban_vs_direnjie),
        "enemy_direnjie_threat": _clip01_np(enemy_direnjie_threat),
        "luban_safe_passive": _clip01_np(luban_safe_passive),
        "direnjie_skill1_poke": _clip01_np(skill1_poke),
        "direnjie_skill2_escape": skill2_escape,
        "direnjie_ult_kill": ult_kill,
        "direnjie_vs_luban": _clip01_np(direnjie_vs_luban),
        "direnjie_punish_luban": _clip01_np(direnjie_punish_luban),
        "direnjie_luban_pressure": _clip01_np(direnjie_luban_pressure),
        "ep_skill_throttle": _clip01_np(ep_skill_throttle),
        "ep_conserve": _clip01_np(ep_conserve),
        "ep_resume_ready": _clip01_np(ep_resume_ready),
        "ep_critical": _clip01_np(ep_critical),
    }


def build_numpy_action_priority_bias(feature, label_size_list, scale):
    if scale <= 0.0:
        return None
    if int(DimConfig.DIM_OF_FEATURE[0]) == R8_TOTAL_DIM:
        task_index = _r8_main_task_index_np(feature)
        if task_index is None:
            return None
        button = np.zeros(label_size_list[0], dtype=np.float32)
        move_x = np.zeros(label_size_list[1], dtype=np.float32)
        move_z = np.zeros(label_size_list[2], dtype=np.float32)
        skill_x = np.zeros(label_size_list[3], dtype=np.float32)
        skill_z = np.zeros(label_size_list[4], dtype=np.float32)
        target = np.zeros(label_size_list[5], dtype=np.float32)
        button[BUTTON_INVALID] -= 1.5
        button[BUTTON_NONE] -= 0.35

        if task_index == R8_TASK_POST_KILL_PUSH or task_index == R8_TASK_LANE_CLEAR or task_index == R8_TASK_OWN_TOWER_DEFEND:
            button[BUTTON_ATTACK] += 0.65
            button[BUTTON_SKILL_1] += 0.22
            button[BUTTON_SKILL_3] += 0.16
            target[TARGET_SOLDIER_MIN : TARGET_SOLDIER_MAX + 1] += 0.95
            target[TARGET_ENEMY] -= 0.28
            target[TARGET_TOWER] -= 0.40
        elif task_index == R8_TASK_TOWER_HIT or task_index == R8_TASK_TOWER_STAGE:
            button[BUTTON_ATTACK] += 0.72 if task_index == R8_TASK_TOWER_HIT else 0.18
            button[BUTTON_MOVE] += 0.30 if task_index == R8_TASK_TOWER_STAGE else -0.05
            button[BUTTON_SKILL_1] += 0.12
            button[BUTTON_SKILL_3] += 0.10
            button[BUTTON_RECALL] -= 0.55
            button[BUTTON_HEAL] -= 0.15
            target[TARGET_TOWER] += 1.05 if task_index == R8_TASK_TOWER_HIT else 0.32
            target[TARGET_SOLDIER_MIN : TARGET_SOLDIER_MAX + 1] += 0.20 if task_index == R8_TASK_TOWER_STAGE else -0.25
            target[TARGET_ENEMY] -= 0.18
        elif task_index == R8_TASK_KILL_COMMIT or task_index == R8_TASK_FIGHT_POKE:
            button[BUTTON_ATTACK] += 0.70
            button[BUTTON_SKILL_1] += 0.38
            button[BUTTON_SKILL_2] += 0.20
            button[BUTTON_SKILL_3] += 0.54 if task_index == R8_TASK_KILL_COMMIT else 0.28
            button[BUTTON_SUMMONER] += 0.18 if task_index == R8_TASK_KILL_COMMIT else 0.06
            button[BUTTON_RECALL] -= 0.45
            target[TARGET_ENEMY] += 1.15 if task_index == R8_TASK_KILL_COMMIT else 0.82
            target[TARGET_SOLDIER_MIN : TARGET_SOLDIER_MAX + 1] -= 0.35
            target[TARGET_TOWER] -= 0.18
        elif task_index == R8_TASK_RECOVER:
            button[BUTTON_MOVE] += 0.85
            button[BUTTON_HEAL] += 0.20
            button[BUTTON_RECALL] += 0.18
            button[BUTTON_ATTACK] -= 0.30
            button[BUTTON_SKILL_1] -= 0.15
            button[BUTTON_SKILL_3] -= 0.18
            target[TARGET_NONE] += 0.35
            target[TARGET_SELF] += 0.42
            target[TARGET_ENEMY] -= 0.24
            target[TARGET_TOWER] -= 0.22
        elif task_index == R8_TASK_FARM_NEUTRAL:
            button[BUTTON_ATTACK] += 0.25
            button[BUTTON_MOVE] += 0.18
            target[TARGET_MONSTER] += 0.90
            target[TARGET_ENEMY] -= 0.25
            target[TARGET_TOWER] -= 0.18

        # R8-c6: HP advantage chase/retreat modulation
        feature_f = np.asarray(feature, dtype=np.float32).reshape(-1)
        if feature_f.shape[0] > R8_TEMPORAL_HP_RATIO_LEVEL:
            hp_adv = float(np.clip(feature_f[R8_TEMPORAL_HP_ADVANTAGE], -1.0, 1.0))
            hp_lvl = float(np.clip(feature_f[R8_TEMPORAL_HP_RATIO_LEVEL], -1.0, 1.0))
            # Use both continuous HP advantage and coarse ratio level.
            # Positive = we have HP advantage -> chase harder.
            # Negative = enemy has HP advantage -> retreat/poke safer.
            chase_boost = max(0.0, hp_lvl) * 0.28 + max(0.0, hp_adv) * 0.18
            retreat_boost = max(0.0, -hp_lvl) * 0.22 + max(0.0, -hp_adv) * 0.14
            # Kill commit / fight poke tasks: boost attack if we have HP advantage
            if task_index in (R8_TASK_KILL_COMMIT, R8_TASK_FIGHT_POKE):
                button[BUTTON_ATTACK] += chase_boost
                button[BUTTON_SKILL_1] += chase_boost * 0.5
                target[TARGET_ENEMY] += chase_boost * 0.8
                button[BUTTON_RECALL] -= retreat_boost * 0.3
            # Recover task: boost retreat if enemy has HP advantage
            elif task_index == R8_TASK_RECOVER:
                button[BUTTON_MOVE] += retreat_boost * 0.6
                button[BUTTON_HEAL] += retreat_boost * 0.3
                target[TARGET_ENEMY] -= retreat_boost * 0.4
            # Tower tasks: slight poke advantage when HP lead
            elif task_index in (R8_TASK_TOWER_HIT, R8_TASK_TOWER_STAGE):
                button[BUTTON_ATTACK] += chase_boost * 0.3
                button[BUTTON_RECALL] -= retreat_boost * 0.2

        parts = [button, move_x, move_z, skill_x, skill_z, target]
        return np.concatenate(parts, axis=0) * float(scale)
    if int(DimConfig.DIM_OF_FEATURE[0]) != 896:
        return None
    tactical = _feature_tactical_np(feature)
    if tactical is None:
        return None

    task = _task_scores_np(tactical, feature)
    button = np.zeros(label_size_list[0], dtype=np.float32)
    move_x = np.zeros(label_size_list[1], dtype=np.float32)
    move_z = np.zeros(label_size_list[2], dtype=np.float32)
    skill_x = np.zeros(label_size_list[3], dtype=np.float32)
    skill_z = np.zeros(label_size_list[4], dtype=np.float32)
    target = np.zeros(label_size_list[5], dtype=np.float32)

    _fill_numpy_bias(button, move_x, move_z, skill_x, skill_z, target, task)
    parts = [button, move_x, move_z, skill_x, skill_z, target]
    return np.concatenate(parts, axis=0) * float(scale)


def apply_numpy_action_priority(logits, feature, label_size_list, scale):
    bias = build_numpy_action_priority_bias(feature, label_size_list, scale)
    if bias is None:
        return logits
    return np.asarray(logits, dtype=np.float32) + bias


def _fill_numpy_axis_bias(axis, direction, strength):
    if axis.shape[0] <= 1 or strength <= 0.0 or abs(direction) <= 1e-6:
        return
    coords = np.linspace(-1.0, 1.0, axis.shape[0], dtype=np.float32)
    axis += coords * float(direction) * float(strength)


def _fill_numpy_move_bias(move_x, move_z, task):
    strength = 0.70 * float(task["move_priority"])
    _fill_numpy_axis_bias(move_x, task["move_dir_x"], strength)
    _fill_numpy_axis_bias(move_z, task["move_dir_z"], strength)


def _fill_numpy_bias(button, move_x, move_z, skill_x, skill_z, target, task):
    any_task = max(task["clear_wave"], task["tower_pressure"], task["hero_contest"], task["commit_kill"], task["short_reset"])
    button[BUTTON_INVALID] -= 2.0 * any_task
    button[BUTTON_NONE] -= 0.45 * any_task

    combo_start = task["combo_start"] * (1.0 - task["combo_abort"])
    combo_chain = task["combo_chain"] * (1.0 - task["combo_abort"])
    combo_drive = max(combo_start, combo_chain)

    clear_wave = task["clear_wave"]
    button[BUTTON_ATTACK] += 1.00 * clear_wave
    button[BUTTON_SKILL_1] += 0.25 * clear_wave
    button[BUTTON_SKILL_3] += 0.15 * clear_wave
    target[TARGET_SOLDIER_MIN : TARGET_SOLDIER_MAX + 1] += 1.15 * clear_wave
    target[TARGET_ENEMY] -= 0.45 * clear_wave
    target[TARGET_TOWER] -= 0.55 * clear_wave

    tower_pressure = task["tower_pressure"]
    tower_stage = task["tower_stage"]
    tower_hit = task["tower_hit"]
    hard_tower = task["hard_tower_objective"]
    button[BUTTON_MOVE] += 0.55 * tower_stage
    button[BUTTON_ATTACK] += 0.12 * tower_stage + 1.05 * tower_hit + 0.45 * hard_tower
    button[BUTTON_SKILL_1] += 0.06 * tower_stage + 0.22 * tower_hit
    button[BUTTON_SKILL_3] += 0.05 * tower_stage + 0.18 * tower_hit
    button[BUTTON_MOVE] -= 0.22 * hard_tower
    button[BUTTON_RECALL] -= 0.95 * hard_tower
    button[BUTTON_HEAL] -= 0.28 * hard_tower
    target[TARGET_TOWER] += 0.08 * tower_stage + 1.35 * tower_hit + 0.90 * hard_tower
    target[TARGET_ENEMY] -= 0.42 * hard_tower
    target[TARGET_SELF] -= 0.35 * hard_tower
    target[TARGET_NONE] -= 0.28 * hard_tower
    target[TARGET_SOLDIER_MIN : TARGET_SOLDIER_MAX + 1] += 0.35 * tower_stage
    target[TARGET_SOLDIER_MIN : TARGET_SOLDIER_MAX + 1] -= 0.55 * tower_hit + 0.35 * hard_tower

    hero_contest = task["hero_contest"]
    button[BUTTON_ATTACK] += 0.85 * hero_contest
    button[BUTTON_SKILL_1] += 0.50 * hero_contest + 0.42 * combo_start + 0.18 * combo_chain
    button[BUTTON_SKILL_2] += 0.28 * combo_drive + 0.38 * task["direnjie_skill2_combo"] + 0.30 * task["luban_skill2_combo"]
    button[BUTTON_SKILL_3] += 0.38 * hero_contest + 0.26 * combo_start + 0.18 * combo_chain
    button[BUTTON_MOVE] -= 0.10 * hero_contest
    target[TARGET_ENEMY] += 1.28 * hero_contest + 0.55 * combo_drive
    target[TARGET_SOLDIER_MIN : TARGET_SOLDIER_MAX + 1] -= 0.62 * hero_contest + 0.28 * combo_drive

    commit_kill = task["commit_kill"]
    button[BUTTON_ATTACK] += 0.85 * commit_kill
    button[BUTTON_SKILL_1] += 0.42 * commit_kill
    button[BUTTON_SKILL_3] += 0.95 * commit_kill
    button[BUTTON_SUMMONER] += 0.25 * commit_kill
    button[BUTTON_HEAL] -= 0.18 * commit_kill
    button[BUTTON_MOVE] -= 0.45 * commit_kill
    button[BUTTON_RECALL] -= 0.70 * commit_kill
    target[TARGET_ENEMY] += 1.55 * commit_kill
    target[TARGET_SOLDIER_MIN : TARGET_SOLDIER_MAX + 1] -= 0.70 * commit_kill
    target[TARGET_TOWER] -= 0.45 * commit_kill * (1.0 - hard_tower)

    short_reset = task["short_reset"]
    button[BUTTON_MOVE] += 1.00 * short_reset
    button[BUTTON_SKILL_2] += 0.45 * max(short_reset, task["direnjie_skill2_escape"]) * (1.0 - combo_drive)
    button[BUTTON_HEAL] += 0.25 * short_reset
    button[BUTTON_RECALL] += 0.25 * short_reset
    button[BUTTON_ATTACK] -= 0.38 * short_reset
    button[BUTTON_SKILL_1] -= 0.18 * short_reset
    button[BUTTON_SKILL_3] -= 0.22 * short_reset
    button[BUTTON_SUMMONER] -= 0.40 * short_reset
    target[TARGET_NONE] += 0.45 * short_reset
    target[TARGET_SELF] += 0.50 * short_reset
    target[TARGET_ENEMY] -= 0.28 * short_reset
    target[TARGET_TOWER] -= 0.25 * short_reset * (1.0 - hard_tower)
    target[slice(TARGET_SOLDIER_MIN, TARGET_SOLDIER_MAX + 1)] -= 0.18 * short_reset

    take_cake = task["take_cake"]
    button[BUTTON_MOVE] += 0.55 * take_cake
    button[BUTTON_HEAL] += 0.10 * take_cake
    button[BUTTON_SUMMONER] -= 0.25 * take_cake
    button[BUTTON_RECALL] -= 0.30 * take_cake
    button[BUTTON_ATTACK] -= 0.25 * take_cake
    target[TARGET_NONE] += 0.25 * take_cake
    target[TARGET_SELF] += 0.20 * take_cake
    target[TARGET_ENEMY] -= 0.25 * take_cake
    target[TARGET_TOWER] -= 0.20 * take_cake
    target[slice(TARGET_SOLDIER_MIN, TARGET_SOLDIER_MAX + 1)] -= 0.20 * take_cake
    _fill_numpy_move_bias(move_x, move_z, task)
    luban_return_rocket = task["luban_return_rocket"]
    button[BUTTON_SKILL_2] += 0.42 * luban_return_rocket
    _fill_numpy_axis_bias(skill_x, task["move_dir_x"], 0.55 * luban_return_rocket)
    _fill_numpy_axis_bias(skill_z, task["move_dir_z"], 0.55 * luban_return_rocket)

    luban_tower = max(task["luban_passive_tower"], task["luban_tower_ignore_enemy"])
    luban_hero = max(task["luban_passive_hero"], task["luban_followup"], task["luban_lane_deny"])
    luban_skill2 = max(task["luban_skill2_save"], task["luban_skill2_execute"])
    luban_skill2 = max(luban_skill2, task["luban_skill2_combo"])
    button[BUTTON_ATTACK] += 0.38 * max(luban_tower, luban_hero) + 0.20 * task["luban_lane_deny"]
    button[BUTTON_SKILL_1] += 0.18 * max(luban_tower, luban_hero) + 0.14 * task["luban_lane_deny"]
    button[BUTTON_SKILL_2] += 0.58 * luban_skill2
    target[TARGET_TOWER] += 0.58 * luban_tower
    target[TARGET_ENEMY] += 0.52 * luban_hero + 0.68 * task["luban_skill2_execute"] + 0.28 * task["luban_skill2_save"] + 0.22 * task["luban_lane_deny"]
    target[TARGET_SOLDIER_MIN : TARGET_SOLDIER_MAX + 1] -= 0.25 * max(luban_tower, luban_hero) + 0.18 * task["luban_lane_deny"]
    skill_waste = task["luban_skill_waste"]
    button[BUTTON_SKILL_1] -= 0.20 * skill_waste
    button[BUTTON_SKILL_2] -= 0.24 * skill_waste
    button[BUTTON_SKILL_3] -= 0.16 * skill_waste

    button[BUTTON_SKILL_1] += 0.25 * task["direnjie_skill1_poke"]
    button[BUTTON_SKILL_3] += 0.55 * task["direnjie_ult_kill"]
    button[BUTTON_SUMMONER] += 0.16 * task["direnjie_ult_kill"]
    target[TARGET_ENEMY] += 0.65 * task["direnjie_ult_kill"]

    luban_pressure = task["luban_vs_direnjie"] * (1.0 - 0.65 * task["enemy_direnjie_threat"])
    button[BUTTON_ATTACK] += 0.22 * luban_pressure
    button[BUTTON_SKILL_1] += 0.14 * luban_pressure
    button[BUTTON_SUMMONER] += 0.12 * luban_pressure
    target[TARGET_ENEMY] += 0.28 * luban_pressure
    target[TARGET_SOLDIER_MIN : TARGET_SOLDIER_MAX + 1] -= 0.18 * luban_pressure
    button[BUTTON_MOVE] += 0.16 * task["enemy_direnjie_threat"] * task["luban_vs_direnjie"]

    direnjie_luban_poke = max(task["direnjie_vs_luban"], task["direnjie_punish_luban"])
    direnjie_luban_pressure = task["direnjie_luban_pressure"]
    button[BUTTON_SKILL_1] += 0.30 * direnjie_luban_poke + 0.12 * direnjie_luban_pressure
    button[BUTTON_ATTACK] += 0.16 * max(task["direnjie_punish_luban"], task["direnjie_ult_kill"]) - 0.04 * task["direnjie_vs_luban"] * (
        1.0 - direnjie_luban_pressure
    )
    button[BUTTON_SKILL_3] += 0.18 * task["direnjie_ult_kill"]
    target[TARGET_ENEMY] += 0.42 * task["direnjie_punish_luban"] + 0.32 * task["direnjie_skill1_poke"] + 0.45 * task["direnjie_ult_kill"]
    target[TARGET_SOLDIER_MIN : TARGET_SOLDIER_MAX + 1] -= 0.26 * direnjie_luban_pressure

    vision_probe = min(task["enemy_missing"] * (0.55 + 0.45 * task["enemy_missing_ratio"]), 1.0)
    button[BUTTON_MOVE] += 0.12 * vision_probe
    button[BUTTON_SKILL_1] += 0.06 * vision_probe * task["luban_vs_direnjie"]
    target[TARGET_NONE] += 0.06 * vision_probe

    ep_skill_throttle = task["ep_skill_throttle"]
    ep_critical = task["ep_critical"]
    ep_resume_gap = task["ep_conserve"] * (1.0 - task["ep_resume_ready"])
    button[BUTTON_ATTACK] += 0.35 * ep_skill_throttle
    button[BUTTON_MOVE] += 0.18 * ep_resume_gap
    button[BUTTON_SKILL_1] -= 0.72 * ep_skill_throttle + 0.35 * ep_critical
    button[BUTTON_SKILL_2] -= 0.56 * ep_skill_throttle + 0.30 * ep_critical
    button[BUTTON_SKILL_3] -= 0.85 * ep_skill_throttle + 0.45 * ep_critical


def build_torch_action_priority_bias(feature_vec, label_size_list, scale):
    if scale <= 0.0:
        return None
    if int(DimConfig.DIM_OF_FEATURE[0]) == R8_TOTAL_DIM:
        import torch

        if feature_vec.shape[-1] < R8_TOTAL_DIM:
            return None
        dtype = feature_vec.dtype
        device = feature_vec.device
        batch = feature_vec.shape[0]
        button = torch.zeros((batch, label_size_list[0]), dtype=dtype, device=device)
        move_x = torch.zeros((batch, label_size_list[1]), dtype=dtype, device=device)
        move_z = torch.zeros((batch, label_size_list[2]), dtype=dtype, device=device)
        skill_x = torch.zeros((batch, label_size_list[3]), dtype=dtype, device=device)
        skill_z = torch.zeros((batch, label_size_list[4]), dtype=dtype, device=device)
        target = torch.zeros((batch, label_size_list[5]), dtype=dtype, device=device)

        task_offset = R8_TEMPORAL_TASK_OFFSET
        task_slice = feature_vec[:, task_offset : task_offset + R8_MAIN_TASK_COUNT]
        if task_slice.shape[1] != R8_MAIN_TASK_COUNT:
            return None
        task_index = torch.argmax(task_slice, dim=1)
        task_mask = lambda idx: (task_index == idx).to(dtype)

        button[:, BUTTON_INVALID] -= 1.5
        button[:, BUTTON_NONE] -= 0.35

        clear_wave = torch.maximum(
            torch.maximum(task_mask(R8_TASK_POST_KILL_PUSH), task_mask(R8_TASK_LANE_CLEAR)),
            task_mask(R8_TASK_OWN_TOWER_DEFEND),
        )
        tower_hit = task_mask(R8_TASK_TOWER_HIT)
        tower_stage = task_mask(R8_TASK_TOWER_STAGE)
        hero_contest = torch.maximum(task_mask(R8_TASK_KILL_COMMIT), task_mask(R8_TASK_FIGHT_POKE))
        commit_kill = task_mask(R8_TASK_KILL_COMMIT)
        recover = task_mask(R8_TASK_RECOVER)
        farm_neutral = task_mask(R8_TASK_FARM_NEUTRAL)

        button[:, BUTTON_ATTACK] += 0.65 * clear_wave
        button[:, BUTTON_SKILL_1] += 0.22 * clear_wave
        button[:, BUTTON_SKILL_3] += 0.16 * clear_wave
        target[:, slice(TARGET_SOLDIER_MIN, TARGET_SOLDIER_MAX + 1)] += 0.95 * clear_wave.unsqueeze(1)
        target[:, TARGET_ENEMY] -= 0.28 * clear_wave
        target[:, TARGET_TOWER] -= 0.40 * clear_wave

        button[:, BUTTON_ATTACK] += 0.18 * tower_stage + 0.72 * tower_hit
        button[:, BUTTON_MOVE] += 0.30 * tower_stage - 0.05 * tower_hit
        button[:, BUTTON_SKILL_1] += 0.12 * (tower_stage + tower_hit)
        button[:, BUTTON_SKILL_3] += 0.10 * (tower_stage + tower_hit)
        button[:, BUTTON_RECALL] -= 0.55 * (tower_stage + tower_hit)
        button[:, BUTTON_HEAL] -= 0.15 * (tower_stage + tower_hit)
        target[:, TARGET_TOWER] += 0.32 * tower_stage + 1.05 * tower_hit
        target[:, slice(TARGET_SOLDIER_MIN, TARGET_SOLDIER_MAX + 1)] += 0.20 * tower_stage.unsqueeze(1) - 0.25 * tower_hit.unsqueeze(1)
        target[:, TARGET_ENEMY] -= 0.18 * (tower_stage + tower_hit)

        button[:, BUTTON_ATTACK] += 0.70 * hero_contest
        button[:, BUTTON_SKILL_1] += 0.38 * hero_contest
        button[:, BUTTON_SKILL_2] += 0.20 * hero_contest
        button[:, BUTTON_SKILL_3] += 0.28 * hero_contest + 0.26 * commit_kill
        button[:, BUTTON_SUMMONER] += 0.06 * hero_contest + 0.12 * commit_kill
        button[:, BUTTON_RECALL] -= 0.45 * hero_contest
        target[:, TARGET_ENEMY] += 0.82 * hero_contest + 0.33 * commit_kill
        target[:, slice(TARGET_SOLDIER_MIN, TARGET_SOLDIER_MAX + 1)] -= 0.35 * hero_contest.unsqueeze(1)
        target[:, TARGET_TOWER] -= 0.18 * hero_contest

        button[:, BUTTON_MOVE] += 0.85 * recover
        button[:, BUTTON_HEAL] += 0.20 * recover
        button[:, BUTTON_RECALL] += 0.18 * recover
        button[:, BUTTON_ATTACK] -= 0.30 * recover
        button[:, BUTTON_SKILL_1] -= 0.15 * recover
        button[:, BUTTON_SKILL_3] -= 0.18 * recover
        target[:, TARGET_NONE] += 0.35 * recover
        target[:, TARGET_SELF] += 0.42 * recover
        target[:, TARGET_ENEMY] -= 0.24 * recover
        target[:, TARGET_TOWER] -= 0.22 * recover

        button[:, BUTTON_ATTACK] += 0.25 * farm_neutral
        button[:, BUTTON_MOVE] += 0.18 * farm_neutral
        target[:, TARGET_MONSTER] += 0.90 * farm_neutral
        target[:, TARGET_ENEMY] -= 0.25 * farm_neutral
        target[:, TARGET_TOWER] -= 0.18 * farm_neutral

        # R8-c6: HP advantage chase/retreat modulation (torch path)
        hp_adv = torch.clamp(feature_vec[:, R8_TEMPORAL_HP_ADVANTAGE], -1.0, 1.0)
        hp_lvl = torch.clamp(feature_vec[:, R8_TEMPORAL_HP_RATIO_LEVEL], -1.0, 1.0)
        chase_boost = torch.clamp(hp_lvl, 0.0, 1.0) * 0.28 + torch.clamp(hp_adv, 0.0, 1.0) * 0.18
        retreat_boost = torch.clamp(-hp_lvl, 0.0, 1.0) * 0.22 + torch.clamp(-hp_adv, 0.0, 1.0) * 0.14
        fight_mask = torch.maximum(commit_kill, task_mask(R8_TASK_FIGHT_POKE))
        button[:, BUTTON_ATTACK] += chase_boost * fight_mask
        button[:, BUTTON_SKILL_1] += chase_boost * 0.5 * fight_mask
        target[:, TARGET_ENEMY] += chase_boost * 0.8 * fight_mask
        button[:, BUTTON_RECALL] -= retreat_boost * 0.3 * fight_mask
        button[:, BUTTON_MOVE] += retreat_boost * 0.6 * recover
        button[:, BUTTON_HEAL] += retreat_boost * 0.3 * recover
        target[:, TARGET_ENEMY] -= retreat_boost * 0.4 * recover
        tower_mask = tower_stage + tower_hit
        button[:, BUTTON_ATTACK] += chase_boost * 0.3 * tower_mask
        button[:, BUTTON_RECALL] -= retreat_boost * 0.2 * tower_mask

        return [part * float(scale) for part in (button, move_x, move_z, skill_x, skill_z, target)]
    if int(DimConfig.DIM_OF_FEATURE[0]) != 896:
        return None
    import torch

    if feature_vec.shape[-1] < TACTICAL_START + TACTICAL_FEATURE_DIM:
        return None

    tactical = feature_vec[:, TACTICAL_START : TACTICAL_START + TACTICAL_FEATURE_DIM]
    dtype = feature_vec.dtype
    device = feature_vec.device
    batch = feature_vec.shape[0]
    button = torch.zeros((batch, label_size_list[0]), dtype=dtype, device=device)
    move_x = torch.zeros((batch, label_size_list[1]), dtype=dtype, device=device)
    move_z = torch.zeros((batch, label_size_list[2]), dtype=dtype, device=device)
    skill_x = torch.zeros((batch, label_size_list[3]), dtype=dtype, device=device)
    skill_z = torch.zeros((batch, label_size_list[4]), dtype=dtype, device=device)
    target = torch.zeros((batch, label_size_list[5]), dtype=dtype, device=device)

    def g(index):
        return torch.clamp(tactical[:, index], 0.0, 1.0)

    def normalize_dir(x, z):
        norm = torch.sqrt(x * x + z * z)
        safe_norm = torch.clamp(norm, min=1e-6)
        return torch.where(norm > 1e-6, x / safe_norm, torch.zeros_like(x)), torch.where(
            norm > 1e-6,
            z / safe_norm,
            torch.zeros_like(z),
        )

    def base_enemy_tower_direction():
        if feature_vec.shape[-1] <= BASE_ENEMY_TOWER_HP:
            zeros = torch.zeros((batch,), dtype=dtype, device=device)
            return zeros, zeros
        tower_known = torch.maximum(
            torch.clamp(feature_vec[:, BASE_ENEMY_TOWER_ALIVE], 0.0, 1.0),
            torch.clamp(feature_vec[:, BASE_ENEMY_TOWER_HP], 0.0, 1.0),
        )
        x = torch.clamp((feature_vec[:, BASE_ENEMY_TOWER_REL_X] - 0.5) * 2.0, -1.0, 1.0)
        z = torch.clamp((feature_vec[:, BASE_ENEMY_TOWER_REL_Z] - 0.5) * 2.0, -1.0, 1.0)
        x, z = normalize_dir(x, z)
        known = (tower_known > 0.0).to(dtype)
        return x * known, z * known

    def base_own_cake_direction():
        if feature_vec.shape[-1] <= BASE_HERO_LOC_Z:
            zeros = torch.zeros((batch,), dtype=dtype, device=device)
            return zeros, zeros
        hero_alive = torch.clamp(feature_vec[:, BASE_HERO_ALIVE], 0.0, 1.0)
        hero_x = feature_vec[:, BASE_HERO_LOC_X] * HERO_LOC_RANGE + HERO_LOC_MIN
        hero_z = feature_vec[:, BASE_HERO_LOC_Z] * HERO_LOC_RANGE + HERO_LOC_MIN
        x, z = normalize_dir(OWN_CAKE_X - hero_x, OWN_CAKE_Z - hero_z)
        alive = (hero_alive > 0.0).to(dtype)
        return x * alive, z * alive

    own_hp = g(TacticalIndex.OWN_HP)
    enemy_hp = g(TacticalIndex.ENEMY_HP)
    should_clear = g(TacticalIndex.SHOULD_CLEAR_WAVE)
    can_tower = g(TacticalIndex.CAN_PRESSURE_TOWER)
    must_hit_tower = g(TacticalIndex.MUST_HIT_TOWER_NOW)
    can_fight = g(TacticalIndex.CAN_FIGHT)
    should_kite = g(TacticalIndex.SHOULD_KITE_BACK)
    kill_window = g(TacticalIndex.KILL_WINDOW)
    should_take_cake = g(TacticalIndex.SHOULD_TAKE_CAKE)
    should_recall = g(TacticalIndex.SHOULD_RECALL)
    low_hp_stall = g(TacticalIndex.LOW_HP_STALL)
    danger = g(TacticalIndex.DANGER_LEVEL)
    passive_followup = g(TacticalIndex.PASSIVE_FOLLOWUP)
    tower_overstep = g(TacticalIndex.TOWER_EDGE_OVERSTEP)
    tower_reset = g(TacticalIndex.TOWER_WAVE_LOST_RESET)
    move_delta = g(TacticalIndex.MOVE_DELTA)
    own_is_luban = g(TacticalIndex.OWN_IS_LUBAN)
    own_is_direnjie = g(TacticalIndex.OWN_IS_DIRENJIE)
    enemy_is_luban = g(TacticalIndex.ENEMY_IS_LUBAN)
    enemy_is_direnjie = g(TacticalIndex.ENEMY_IS_DIRENJIE)
    luban_vs_direnjie = own_is_luban * enemy_is_direnjie * g(TacticalIndex.LUBAN_VS_DIRENJIE)
    enemy_direnjie_threat = enemy_is_direnjie * g(TacticalIndex.ENEMY_DIRENJIE_THREAT)
    luban_safe_passive = own_is_luban * enemy_is_direnjie * g(TacticalIndex.LUBAN_INTO_DIRENJIE_SAFE_PASSIVE)
    luban_passive_tower = own_is_luban * g(TacticalIndex.LUBAN_PASSIVE_TOWER)
    luban_passive_hero = own_is_luban * g(TacticalIndex.LUBAN_PASSIVE_HERO)
    luban_followup = own_is_luban * g(TacticalIndex.LUBAN_FOLLOWUP)
    luban_skill2_save = own_is_luban * g(TacticalIndex.LUBAN_SKILL2_SAVE)
    luban_skill2_execute = own_is_luban * g(TacticalIndex.LUBAN_SKILL2_EXECUTE)
    luban_skill_waste = own_is_luban * g(TacticalIndex.LUBAN_SKILL_WASTE)
    luban_tower_ignore_enemy = own_is_luban * g(TacticalIndex.LUBAN_TOWER_IGNORE_ENEMY)
    enemy_missing = g(TacticalIndex.ENEMY_MISSING)
    enemy_missing_ratio = g(TacticalIndex.ENEMY_MISSING_RATIO)
    enemy_close_extended = g(TacticalIndex.ENEMY_CLOSE_EXTENDED)
    combo_start = g(TacticalIndex.COMBO_START)
    combo_chain = g(TacticalIndex.COMBO_CHAIN)
    combo_abort = g(TacticalIndex.COMBO_ABORT)
    direnjie_skill2_combo = own_is_direnjie * g(TacticalIndex.DIRENJIE_SKILL2_COMBO)
    luban_skill2_combo = own_is_luban * g(TacticalIndex.LUBAN_SKILL2_COMBO)
    direnjie_vs_luban = own_is_direnjie * enemy_is_luban * g(TacticalIndex.DIRENJIE_VS_LUBAN)
    direnjie_punish_luban = own_is_direnjie * enemy_is_luban * g(TacticalIndex.DIRENJIE_INTO_LUBAN_PUNISH)
    skill1_poke = own_is_direnjie * g(TacticalIndex.DIRENJIE_SKILL1_POKE)
    skill2_escape = own_is_direnjie * g(TacticalIndex.DIRENJIE_SKILL2_ESCAPE)
    ult_kill = own_is_direnjie * g(TacticalIndex.DIRENJIE_ULT_KILL)
    direnjie_followup = own_is_direnjie * g(TacticalIndex.DIRENJIE_FOLLOWUP)
    ep_conserve = g(TacticalIndex.EP_CONSERVE)
    ep_hold_skills = g(TacticalIndex.EP_HOLD_SKILLS)
    ep_resume_ready = g(TacticalIndex.EP_RESUME_READY)
    ep_critical = g(TacticalIndex.EP_CRITICAL)
    raw_ep_throttle = torch.maximum(torch.maximum(ep_conserve, ep_hold_skills), ep_critical)

    enemy_weaker = ((enemy_hp > 0.0) & (enemy_hp + 0.08 <= own_hp)).to(dtype)
    low_hp = (own_hp < 0.48).to(dtype)
    low_hp_commit = low_hp * enemy_weaker * can_fight

    commit_kill = torch.maximum(
        torch.maximum(kill_window, ult_kill),
        torch.maximum(luban_skill2_execute, low_hp_commit),
    )
    short_reset = torch.maximum(
        torch.maximum(should_recall, should_take_cake),
        torch.maximum(
            torch.maximum(low_hp_stall, torch.maximum(tower_reset, tower_overstep * torch.maximum(danger, danger.new_full(danger.shape, 0.35)))),
            torch.maximum(
                luban_skill2_save * torch.maximum(danger, danger.new_full(danger.shape, 0.45)),
                should_kite * torch.maximum(danger, 0.55 * skill2_escape),
            ),
        ),
    )
    short_reset = short_reset * (1.0 - 0.85 * commit_kill)
    luban_lane_deny = own_is_luban * torch.maximum(
        torch.maximum(luban_passive_hero, luban_followup),
        luban_vs_direnjie * enemy_close_extended,
    )
    tower_hit_intent = torch.clamp((must_hit_tower - 0.55) / 0.45, 0.0, 1.0)
    tower_hit = torch.maximum(
        tower_hit_intent,
        torch.maximum(luban_passive_tower, luban_tower_ignore_enemy),
    )
    tower_stage = torch.clamp(can_tower * (1.0 - tower_hit), 0.0, 1.0)
    tower_pressure = torch.maximum(tower_stage, tower_hit)
    hard_tower_objective = (
        (tower_hit > 0.86)
        & (must_hit_tower > 0.72)
        & (should_take_cake < 0.10)
        & (should_recall < 0.10)
    ).to(dtype)
    commit_kill = commit_kill * (1.0 - 0.85 * hard_tower_objective)
    short_reset = short_reset * (1.0 - 0.92 * hard_tower_objective)
    ep_relief = torch.maximum(
        0.55 * hard_tower_objective,
        torch.where(enemy_hp <= 0.24, enemy_hp.new_full(enemy_hp.shape, 0.75), 0.35 * commit_kill),
    )
    ep_skill_throttle = torch.clamp(raw_ep_throttle * (1.0 - ep_relief), 0.0, 1.0)
    luban_return_rocket = own_is_luban * enemy_missing * torch.maximum(
        torch.maximum(hard_tower_objective, 0.55 * tower_stage),
        0.65 * must_hit_tower,
    ) * (1.0 - combo_abort)
    direnjie_luban_pressure = torch.maximum(
        torch.maximum(direnjie_punish_luban, skill1_poke * (1.0 - 0.60 * low_hp_stall)),
        torch.maximum(ult_kill, direnjie_followup),
    )
    objective_or_lane = torch.maximum(tower_pressure, should_clear)
    active_pressure_claim = torch.maximum(
        torch.maximum(torch.maximum(torch.maximum(objective_or_lane, can_fight), passive_followup), torch.maximum(combo_start, combo_chain)),
        torch.maximum(
            torch.maximum(torch.maximum(luban_passive_hero, luban_followup), luban_lane_deny),
            direnjie_luban_pressure,
        ),
    )
    short_reset = short_reset * (1.0 - 0.70 * active_pressure_claim)
    clear_wave = should_clear * (1.0 - 0.70 * commit_kill) * (1.0 - 0.25 * short_reset)
    tower_pressure = tower_pressure * (1.0 - (0.45 * (1.0 - hard_tower_objective)) * commit_kill) * (
        1.0 - (0.15 * (1.0 - hard_tower_objective)) * short_reset
    )
    hero_contest = torch.maximum(
        torch.maximum(torch.maximum(can_fight, passive_followup), torch.maximum(combo_start, combo_chain)),
        torch.maximum(
            torch.maximum(torch.maximum(luban_passive_hero, luban_followup), luban_lane_deny),
            torch.maximum(direnjie_followup, direnjie_luban_pressure),
        ),
    )
    hero_contest = hero_contest * (1.0 - 0.50 * combo_abort)
    hero_contest = hero_contest * (1.0 - 0.35 * short_reset)
    hero_contest = hero_contest * (1.0 - 0.18 * clear_wave)
    clear_wave = torch.where(
        hero_contest > 0.62,
        clear_wave
        * torch.where(
            torch.maximum(luban_lane_deny, direnjie_luban_pressure) > 0.35,
            clear_wave.new_full(clear_wave.shape, 0.52),
            clear_wave.new_full(clear_wave.shape, 0.68),
        ),
        clear_wave,
    )

    enemy_tower_x, enemy_tower_z = base_enemy_tower_direction()
    own_tower_x, own_tower_z = -enemy_tower_x, -enemy_tower_z
    cake_x, cake_z = base_own_cake_direction()
    cake_missing = ((torch.abs(cake_x) + torch.abs(cake_z)) <= 1e-6).to(dtype)
    cake_x = cake_x * (1.0 - cake_missing) + own_tower_x * cake_missing
    cake_z = cake_z * (1.0 - cake_missing) + own_tower_z * cake_missing

    stuck_escape = torch.clamp((0.08 - move_delta) / 0.08, 0.0, 1.0)
    escape_drive = torch.clamp(
        torch.maximum(torch.maximum(short_reset, should_recall), torch.maximum(low_hp_stall, torch.maximum(tower_reset, tower_overstep))),
        0.0,
        1.0,
    )
    cake_drive = torch.clamp(should_take_cake, 0.0, 1.0)
    pressure_drive = torch.clamp(tower_pressure * (1.0 - torch.maximum(escape_drive, commit_kill)) * (1.0 - tower_overstep), 0.0, 1.0)
    move_dir_x = cake_drive * cake_x + escape_drive * own_tower_x + 0.22 * pressure_drive * enemy_tower_x
    move_dir_z = cake_drive * cake_z + escape_drive * own_tower_z + 0.22 * pressure_drive * enemy_tower_z
    perp_x, perp_z = -enemy_tower_z, enemy_tower_x
    side = torch.where(cake_x * perp_x + cake_z * perp_z < 0.0, -torch.ones_like(cake_x), torch.ones_like(cake_x))
    lateral_drive = 0.45 * stuck_escape * escape_drive
    move_dir_x = move_dir_x + lateral_drive * side * perp_x
    move_dir_z = move_dir_z + lateral_drive * side * perp_z
    move_dir_x, move_dir_z = normalize_dir(move_dir_x, move_dir_z)
    move_priority = torch.clamp(torch.maximum(torch.maximum(escape_drive, cake_drive), 0.30 * pressure_drive), 0.0, 1.0)
    move_missing = ((torch.abs(move_dir_x) + torch.abs(move_dir_z)) <= 1e-6).to(dtype)
    move_priority = move_priority * (1.0 - move_missing)

    task = {
        "clear_wave": torch.clamp(clear_wave, 0.0, 1.0),
        "tower_pressure": torch.clamp(tower_pressure, 0.0, 1.0),
        "hero_contest": torch.clamp(hero_contest, 0.0, 1.0),
        "commit_kill": torch.clamp(commit_kill, 0.0, 1.0),
        "short_reset": torch.clamp(short_reset, 0.0, 1.0),
        "take_cake": torch.clamp(should_take_cake, 0.0, 1.0),
        "low_hp_stall": torch.clamp(low_hp_stall, 0.0, 1.0),
        "move_dir_x": move_dir_x,
        "move_dir_z": move_dir_z,
        "move_priority": move_priority,
        "hard_tower_objective": torch.clamp(hard_tower_objective, 0.0, 1.0),
        "tower_stage": torch.clamp(tower_stage, 0.0, 1.0),
        "tower_hit": torch.clamp(tower_hit, 0.0, 1.0),
        "luban_return_rocket": torch.clamp(luban_return_rocket, 0.0, 1.0),
        "luban_passive_tower": torch.clamp(luban_passive_tower, 0.0, 1.0),
        "luban_passive_hero": torch.clamp(luban_passive_hero, 0.0, 1.0),
        "luban_followup": torch.clamp(luban_followup, 0.0, 1.0),
        "luban_skill2_save": torch.clamp(luban_skill2_save, 0.0, 1.0),
        "luban_skill2_execute": torch.clamp(luban_skill2_execute, 0.0, 1.0),
        "luban_skill_waste": torch.clamp(luban_skill_waste, 0.0, 1.0),
        "luban_tower_ignore_enemy": torch.clamp(luban_tower_ignore_enemy, 0.0, 1.0),
        "luban_lane_deny": torch.clamp(luban_lane_deny, 0.0, 1.0),
        "enemy_missing": torch.clamp(enemy_missing, 0.0, 1.0),
        "enemy_missing_ratio": torch.clamp(enemy_missing_ratio, 0.0, 1.0),
        "combo_start": torch.clamp(combo_start, 0.0, 1.0),
        "combo_chain": torch.clamp(combo_chain, 0.0, 1.0),
        "combo_abort": torch.clamp(combo_abort, 0.0, 1.0),
        "direnjie_skill2_combo": torch.clamp(direnjie_skill2_combo, 0.0, 1.0),
        "luban_skill2_combo": torch.clamp(luban_skill2_combo, 0.0, 1.0),
        "luban_vs_direnjie": torch.clamp(luban_vs_direnjie, 0.0, 1.0),
        "enemy_direnjie_threat": torch.clamp(enemy_direnjie_threat, 0.0, 1.0),
        "luban_safe_passive": torch.clamp(luban_safe_passive, 0.0, 1.0),
        "direnjie_skill1_poke": skill1_poke,
        "direnjie_skill2_escape": skill2_escape,
        "direnjie_ult_kill": ult_kill,
        "direnjie_vs_luban": torch.clamp(direnjie_vs_luban, 0.0, 1.0),
        "direnjie_punish_luban": torch.clamp(direnjie_punish_luban, 0.0, 1.0),
        "direnjie_luban_pressure": torch.clamp(direnjie_luban_pressure, 0.0, 1.0),
        "ep_skill_throttle": torch.clamp(ep_skill_throttle, 0.0, 1.0),
        "ep_conserve": torch.clamp(ep_conserve, 0.0, 1.0),
        "ep_resume_ready": torch.clamp(ep_resume_ready, 0.0, 1.0),
        "ep_critical": torch.clamp(ep_critical, 0.0, 1.0),
    }
    _fill_torch_bias(button, move_x, move_z, skill_x, skill_z, target, task)
    return [part * float(scale) for part in (button, move_x, move_z, skill_x, skill_z, target)]


def _fill_torch_axis_bias(axis, direction, strength):
    if axis.shape[1] <= 1:
        return
    import torch

    coords = torch.linspace(-1.0, 1.0, axis.shape[1], dtype=axis.dtype, device=axis.device)
    axis += strength.unsqueeze(1) * direction.unsqueeze(1) * coords.unsqueeze(0)


def _fill_torch_move_bias(move_x, move_z, task):
    strength = 0.70 * task["move_priority"]
    _fill_torch_axis_bias(move_x, task["move_dir_x"], strength)
    _fill_torch_axis_bias(move_z, task["move_dir_z"], strength)


def _fill_torch_bias(button, move_x, move_z, skill_x, skill_z, target, task):
    import torch

    any_task = torch.maximum(
        torch.maximum(task["clear_wave"], task["tower_pressure"]),
        torch.maximum(torch.maximum(task["hero_contest"], task["commit_kill"]), task["short_reset"]),
    )
    button[:, BUTTON_INVALID] -= 2.0 * any_task
    button[:, BUTTON_NONE] -= 0.45 * any_task

    combo_start = task["combo_start"] * (1.0 - task["combo_abort"])
    combo_chain = task["combo_chain"] * (1.0 - task["combo_abort"])
    combo_drive = torch.maximum(combo_start, combo_chain)

    clear_wave = task["clear_wave"]
    button[:, BUTTON_ATTACK] += 1.00 * clear_wave
    button[:, BUTTON_SKILL_1] += 0.25 * clear_wave
    button[:, BUTTON_SKILL_3] += 0.15 * clear_wave
    target[:, slice(TARGET_SOLDIER_MIN, TARGET_SOLDIER_MAX + 1)] += 1.15 * clear_wave.unsqueeze(1)
    target[:, TARGET_ENEMY] -= 0.45 * clear_wave
    target[:, TARGET_TOWER] -= 0.55 * clear_wave

    tower_pressure = task["tower_pressure"]
    tower_stage = task["tower_stage"]
    tower_hit = task["tower_hit"]
    hard_tower = task["hard_tower_objective"]
    button[:, BUTTON_MOVE] += 0.55 * tower_stage
    button[:, BUTTON_ATTACK] += 0.12 * tower_stage + 1.05 * tower_hit + 0.45 * hard_tower
    button[:, BUTTON_SKILL_1] += 0.06 * tower_stage + 0.22 * tower_hit
    button[:, BUTTON_SKILL_3] += 0.05 * tower_stage + 0.18 * tower_hit
    button[:, BUTTON_MOVE] -= 0.22 * hard_tower
    button[:, BUTTON_RECALL] -= 0.95 * hard_tower
    button[:, BUTTON_HEAL] -= 0.28 * hard_tower
    target[:, TARGET_TOWER] += 0.08 * tower_stage + 1.35 * tower_hit + 0.90 * hard_tower
    target[:, TARGET_ENEMY] -= 0.42 * hard_tower
    target[:, TARGET_SELF] -= 0.35 * hard_tower
    target[:, TARGET_NONE] -= 0.28 * hard_tower
    target[:, slice(TARGET_SOLDIER_MIN, TARGET_SOLDIER_MAX + 1)] += (0.35 * tower_stage).unsqueeze(1)
    target[:, slice(TARGET_SOLDIER_MIN, TARGET_SOLDIER_MAX + 1)] -= (0.55 * tower_hit + 0.35 * hard_tower).unsqueeze(1)

    hero_contest = task["hero_contest"]
    button[:, BUTTON_ATTACK] += 0.85 * hero_contest
    button[:, BUTTON_SKILL_1] += 0.50 * hero_contest + 0.42 * combo_start + 0.18 * combo_chain
    button[:, BUTTON_SKILL_2] += 0.28 * combo_drive + 0.38 * task["direnjie_skill2_combo"] + 0.30 * task["luban_skill2_combo"]
    button[:, BUTTON_SKILL_3] += 0.38 * hero_contest + 0.26 * combo_start + 0.18 * combo_chain
    button[:, BUTTON_MOVE] -= 0.10 * hero_contest
    target[:, TARGET_ENEMY] += 1.28 * hero_contest + 0.55 * combo_drive
    target[:, slice(TARGET_SOLDIER_MIN, TARGET_SOLDIER_MAX + 1)] -= (0.62 * hero_contest + 0.28 * combo_drive).unsqueeze(1)

    commit_kill = task["commit_kill"]
    button[:, BUTTON_ATTACK] += 0.85 * commit_kill
    button[:, BUTTON_SKILL_1] += 0.42 * commit_kill
    button[:, BUTTON_SKILL_3] += 0.95 * commit_kill
    button[:, BUTTON_SUMMONER] += 0.25 * commit_kill
    button[:, BUTTON_HEAL] -= 0.18 * commit_kill
    button[:, BUTTON_MOVE] -= 0.45 * commit_kill
    button[:, BUTTON_RECALL] -= 0.70 * commit_kill
    target[:, TARGET_ENEMY] += 1.55 * commit_kill
    target[:, slice(TARGET_SOLDIER_MIN, TARGET_SOLDIER_MAX + 1)] -= 0.70 * commit_kill.unsqueeze(1)
    target[:, TARGET_TOWER] -= 0.45 * commit_kill * (1.0 - hard_tower)

    short_reset = task["short_reset"]
    button[:, BUTTON_MOVE] += 1.00 * short_reset
    button[:, BUTTON_SKILL_2] += 0.45 * torch.maximum(short_reset, task["direnjie_skill2_escape"]) * (1.0 - combo_drive)
    button[:, BUTTON_HEAL] += 0.25 * short_reset
    button[:, BUTTON_RECALL] += 0.25 * short_reset
    button[:, BUTTON_ATTACK] -= 0.38 * short_reset
    button[:, BUTTON_SKILL_1] -= 0.18 * short_reset
    button[:, BUTTON_SKILL_3] -= 0.22 * short_reset
    button[:, BUTTON_SUMMONER] -= 0.40 * short_reset
    target[:, TARGET_NONE] += 0.45 * short_reset
    target[:, TARGET_SELF] += 0.50 * short_reset
    target[:, TARGET_ENEMY] -= 0.28 * short_reset
    target[:, TARGET_TOWER] -= 0.25 * short_reset * (1.0 - hard_tower)
    target[:, slice(TARGET_SOLDIER_MIN, TARGET_SOLDIER_MAX + 1)] -= 0.18 * short_reset.unsqueeze(1)

    take_cake = task["take_cake"]
    button[:, BUTTON_MOVE] += 0.55 * take_cake
    button[:, BUTTON_HEAL] += 0.10 * take_cake
    button[:, BUTTON_SUMMONER] -= 0.25 * take_cake
    button[:, BUTTON_RECALL] -= 0.30 * take_cake
    button[:, BUTTON_ATTACK] -= 0.25 * take_cake
    target[:, TARGET_NONE] += 0.25 * take_cake
    target[:, TARGET_SELF] += 0.20 * take_cake
    target[:, TARGET_ENEMY] -= 0.25 * take_cake
    target[:, TARGET_TOWER] -= 0.20 * take_cake
    target[:, slice(TARGET_SOLDIER_MIN, TARGET_SOLDIER_MAX + 1)] -= 0.20 * take_cake.unsqueeze(1)
    _fill_torch_move_bias(move_x, move_z, task)
    luban_return_rocket = task["luban_return_rocket"]
    button[:, BUTTON_SKILL_2] += 0.42 * luban_return_rocket
    _fill_torch_axis_bias(skill_x, task["move_dir_x"], 0.55 * luban_return_rocket)
    _fill_torch_axis_bias(skill_z, task["move_dir_z"], 0.55 * luban_return_rocket)

    luban_tower = torch.maximum(task["luban_passive_tower"], task["luban_tower_ignore_enemy"])
    luban_hero = torch.maximum(torch.maximum(task["luban_passive_hero"], task["luban_followup"]), task["luban_lane_deny"])
    luban_skill2 = torch.maximum(torch.maximum(task["luban_skill2_save"], task["luban_skill2_execute"]), task["luban_skill2_combo"])
    luban_window = torch.maximum(luban_tower, luban_hero)
    button[:, BUTTON_ATTACK] += 0.38 * luban_window + 0.20 * task["luban_lane_deny"]
    button[:, BUTTON_SKILL_1] += 0.18 * luban_window + 0.14 * task["luban_lane_deny"]
    button[:, BUTTON_SKILL_2] += 0.58 * luban_skill2
    target[:, TARGET_TOWER] += 0.58 * luban_tower
    target[:, TARGET_ENEMY] += 0.52 * luban_hero + 0.68 * task["luban_skill2_execute"] + 0.28 * task["luban_skill2_save"] + 0.22 * task["luban_lane_deny"]
    target[:, slice(TARGET_SOLDIER_MIN, TARGET_SOLDIER_MAX + 1)] -= (0.25 * luban_window + 0.18 * task["luban_lane_deny"]).unsqueeze(1)
    skill_waste = task["luban_skill_waste"]
    button[:, BUTTON_SKILL_1] -= 0.20 * skill_waste
    button[:, BUTTON_SKILL_2] -= 0.24 * skill_waste
    button[:, BUTTON_SKILL_3] -= 0.16 * skill_waste

    button[:, BUTTON_SKILL_1] += 0.25 * task["direnjie_skill1_poke"]
    button[:, BUTTON_SKILL_3] += 0.55 * task["direnjie_ult_kill"]
    button[:, BUTTON_SUMMONER] += 0.16 * task["direnjie_ult_kill"]
    target[:, TARGET_ENEMY] += 0.65 * task["direnjie_ult_kill"]

    luban_pressure = task["luban_vs_direnjie"] * (1.0 - 0.65 * task["enemy_direnjie_threat"])
    button[:, BUTTON_ATTACK] += 0.22 * luban_pressure
    button[:, BUTTON_SKILL_1] += 0.14 * luban_pressure
    button[:, BUTTON_SUMMONER] += 0.12 * luban_pressure
    target[:, TARGET_ENEMY] += 0.28 * luban_pressure
    target[:, slice(TARGET_SOLDIER_MIN, TARGET_SOLDIER_MAX + 1)] -= 0.18 * luban_pressure.unsqueeze(1)
    button[:, BUTTON_MOVE] += 0.16 * task["enemy_direnjie_threat"] * task["luban_vs_direnjie"]

    direnjie_luban_poke = torch.maximum(task["direnjie_vs_luban"], task["direnjie_punish_luban"])
    direnjie_luban_pressure = task["direnjie_luban_pressure"]
    button[:, BUTTON_SKILL_1] += 0.30 * direnjie_luban_poke + 0.12 * direnjie_luban_pressure
    button[:, BUTTON_ATTACK] += 0.16 * torch.maximum(task["direnjie_punish_luban"], task["direnjie_ult_kill"]) - 0.04 * task[
        "direnjie_vs_luban"
    ] * (1.0 - direnjie_luban_pressure)
    button[:, BUTTON_SKILL_3] += 0.18 * task["direnjie_ult_kill"]
    target[:, TARGET_ENEMY] += (
        0.42 * task["direnjie_punish_luban"] + 0.32 * task["direnjie_skill1_poke"] + 0.45 * task["direnjie_ult_kill"]
    )
    target[:, slice(TARGET_SOLDIER_MIN, TARGET_SOLDIER_MAX + 1)] -= 0.26 * direnjie_luban_pressure.unsqueeze(1)

    vision_probe = torch.clamp(task["enemy_missing"] * (0.55 + 0.45 * task["enemy_missing_ratio"]), 0.0, 1.0)
    button[:, BUTTON_MOVE] += 0.12 * vision_probe
    button[:, BUTTON_SKILL_1] += 0.06 * vision_probe * task["luban_vs_direnjie"]
    target[:, TARGET_NONE] += 0.06 * vision_probe

    ep_skill_throttle = task["ep_skill_throttle"]
    ep_critical = task["ep_critical"]
    ep_resume_gap = task["ep_conserve"] * (1.0 - task["ep_resume_ready"])
    button[:, BUTTON_ATTACK] += 0.35 * ep_skill_throttle
    button[:, BUTTON_MOVE] += 0.18 * ep_resume_gap
    button[:, BUTTON_SKILL_1] -= 0.72 * ep_skill_throttle + 0.35 * ep_critical
    button[:, BUTTON_SKILL_2] -= 0.56 * ep_skill_throttle + 0.30 * ep_critical
    button[:, BUTTON_SKILL_3] -= 0.85 * ep_skill_throttle + 0.45 * ep_critical
