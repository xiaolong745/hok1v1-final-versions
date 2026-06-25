#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
R4 semantic-compression tactical features for HOK1v1.

The first 36 tactical dimensions intentionally stay compatible with R2.2 so a
46-dim R2.2 checkpoint can seed the base policy. Extra game-state information is
compressed into 18 decision-oriented semantics instead of exposing the full
R3/R2.4 raw 38-dim extension directly to PPO.
"""

import math


TACTICAL_FEATURE_DIM = 198
MINION_SUB_TYPE = 11
TOWER_SPRING_SUB_TYPE = 23
TOWER_SUB_TYPE = 21
CRYSTAL_SUB_TYPE = 24
RIVER_SPIRIT_CONFIG_ID = 6827
LANE_SOLDIER_CONFIG_IDS = {6800, 6801, 6802, 6803, 6804, 6805}
TOWER_CONFIG_IDS = {1111, 1112}
CRYSTAL_CONFIG_IDS = {1113, 1114}
SPRING_TOWER_CONFIG_IDS = {44, 46}
CAMP_MID = 0
MAX_FRAME_NO = 20000.0
DEFAULT_ATTACK_RANGE = 8200.0
LUBAN_HERO_ID = 112
DIRENJIE_HERO_ID = 133
LANE_LATERAL_SCALE = 12000.0
TOWER_ATTACK_RANGE_PAD = 1800.0
TOWER_EDGE_SAFE_MARGIN = 250.0
TOWER_EDGE_OUTER_PAD = 1800.0
TOWER_TURTLE_RANGE_PAD = 1200.0
LOW_HP_HOLD_MAX_ENEMY_ADVANTAGE = 0.15
LOW_HP_GRASS_RESET_ENEMY_HP = 0.62
LOW_HP_GRASS_AMBUSH_ENEMY_RANGE_PAD = 2400.0
LOW_HP_RELATIVE_ADVANTAGE = 0.08
POST_KILL_EARLY_END_FRAME = 9000.0
POST_KILL_FINISH_START_FRAME = 12500.0
POST_KILL_FINISH_FULL_FRAME = 15000.0
RECOVERY_HEALTHY_HP = 0.70
SHORT_RESET_EXIT_HP = 0.48
SHORT_RESET_STALL_LIMIT = 5
TOWER_PRESSURE_MIN_HP = 0.42
HARD_RECALL_HP = 0.36
ENEMY_HEALTHY_HP = 0.55
TOWER_CRASH_KILL_ENEMY_HP = 0.34
TOWER_CRASH_KILL_MIN_OWN_HP = 0.14
LOW_HP_TRADE_ENEMY_HP = 0.42
LOW_HP_TRADE_MIN_OWN_HP = 0.14
HERO_IDENTITY_DIM = 6
TACTICAL_SCHEMA_RESERVED_DIM = 64
HEALING_BUFF_IDS = {90015, 10000, 10010, 10014}
SPEED_UP_BUFF_IDS = {11001, 133000, 133001}
SPEED_DOWN_BUFF_IDS = {11002}
CLEANSE_OR_IMMUNE_BUFF_IDS = {11010, 133020}
FRENZY_OR_LIFESTEAL_BUFF_IDS = {80110, 801100, 8011000, 500009}
CONTROL_OR_MARK_BUFF_IDS = {133300, 133310, 912260, 912262, 912263, 911260, 911261, 912330}
KNOWN_BUFF_IDS = (
    HEALING_BUFF_IDS
    | SPEED_UP_BUFF_IDS
    | SPEED_DOWN_BUFF_IDS
    | CLEANSE_OR_IMMUNE_BUFF_IDS
    | FRENZY_OR_LIFESTEAL_BUFF_IDS
    | CONTROL_OR_MARK_BUFF_IDS
)
CAKE_LOCATIONS_BY_CAMP = {
    "PLAYERCAMP_1": {
        "main": (-15220.0, -15120.0),
        "enemy": (15340.0, 15100.0),
    },
    "PLAYERCAMP_2": {
        "main": (15340.0, 15100.0),
        "enemy": (-15220.0, -15120.0),
    },
}


def _safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def _clip(value, low=-1.0, high=1.0):
    return max(low, min(high, _safe_float(value)))


def _clip01(value):
    return _clip(value, 0.0, 1.0)


def _ratio(value, max_value, default=0.0):
    max_value = _safe_float(max_value)
    if max_value <= 0:
        return default
    return _clip01(_safe_float(value) / max_value)


def _phase(value, start, end):
    if end <= start:
        return 1.0 if value >= end else 0.0
    return _clip01((_safe_float(value) - start) / (end - start))


def _signed_ratio(value, scale):
    scale = _safe_float(scale)
    if scale <= 0:
        return 0.0
    return _clip(_safe_float(value) / scale)


def _pos(unit):
    location = unit.get("location", {}) if unit else {}
    return (_safe_float(location.get("x")), _safe_float(location.get("z")))


def _dist(a, b):
    if not a or not b:
        return 99999.0
    ax, az = _pos(a)
    bx, bz = _pos(b)
    return math.hypot(ax - bx, az - bz)


def _hp_ratio(unit):
    if not unit:
        return 0.0
    return _ratio(unit.get("hp"), unit.get("max_hp"))


def _unit_value(unit, key, default=0.0):
    return _safe_float(unit.get(key, default) if unit else default)


def _slot_states(hero):
    if not hero:
        return []
    skill_state = hero.get("skill_state", {})
    if isinstance(skill_state, dict):
        states = skill_state.get("slot_states", [])
    elif isinstance(skill_state, list):
        states = skill_state
    else:
        states = []
    return [state for state in states if isinstance(state, dict)]


class TacticalProcess:
    def __init__(self, camp):
        self.camp = camp
        self.last_pos = None
        self.last_enemy_dist = None
        self.last_hp = None
        self.last_own_tower_dist = None
        self.last_cake_dist = None
        self.low_hp_stall_steps = 0
        self.last_skill_used_times = {}
        self.followup_steps = 0
        self.last_own_kill_cnt = None
        self.last_enemy_dead_cnt = None
        self.post_kill_steps = 0
        self.post_kill_window_steps = 1
        self.post_clear_steps = 0
        self.last_enemy_minion_count = None
        self.recovery_commit_active = False
        self.tower_edge_hold_steps = 0
        self.tower_edge_overstep_steps = 0
        self.enemy_missing_steps = 0

    def reset(self, camp):
        self.camp = camp
        self.last_pos = None
        self.last_enemy_dist = None
        self.last_hp = None
        self.last_own_tower_dist = None
        self.last_cake_dist = None
        self.low_hp_stall_steps = 0
        self.last_skill_used_times = {}
        self.followup_steps = 0
        self.last_own_kill_cnt = None
        self.last_enemy_dead_cnt = None
        self.post_kill_steps = 0
        self.post_kill_window_steps = 1
        self.post_clear_steps = 0
        self.last_enemy_minion_count = None
        self.recovery_commit_active = False
        self.tower_edge_hold_steps = 0
        self.tower_edge_overstep_steps = 0
        self.enemy_missing_steps = 0

    def process_vec_tactical(self, observation):
        frame_state = observation["frame_state"]
        frame_no = _safe_float(frame_state.get("frame_no", 0.0))
        player_id = observation.get("player_id")
        own_hero, enemy_hero = self._find_heroes(frame_state, player_id)
        own_tower, enemy_tower = self._find_towers(frame_state)
        ally_minions, enemy_minions = self._find_minions(frame_state)
        monsters = self._find_monsters(frame_state)
        cakes = self._find_cakes(frame_state)

        own_hp = _hp_ratio(own_hero)
        enemy_hp = _hp_ratio(enemy_hero)
        own_tower_hp = _hp_ratio(own_tower)
        enemy_tower_hp = _hp_ratio(enemy_tower)

        dist_own_tower = _dist(own_hero, own_tower)
        dist_enemy_tower = _dist(own_hero, enemy_tower)
        dist_enemy_hero = _dist(own_hero, enemy_hero)
        attack_range = _safe_float(own_hero.get("attack_range") if own_hero else DEFAULT_ATTACK_RANGE, DEFAULT_ATTACK_RANGE)
        attack_band = _ratio(dist_enemy_hero, max(attack_range, 1.0), default=1.0)
        enemy_in_attack_range = 1.0 if enemy_hero and dist_enemy_hero <= attack_range else 0.0

        ally_near_enemy = self._count_near(ally_minions, enemy_tower, enemy_tower)
        enemy_near_own = self._count_near(enemy_minions, own_tower, own_tower)

        tower_range = _safe_float(enemy_tower.get("attack_range") if enemy_tower else 8800.0, 8800.0)
        in_enemy_tower_range = 1.0 if enemy_tower and dist_enemy_tower <= tower_range else 0.0
        safe_tower_pressure = 1.0 if ally_near_enemy > 0 and own_hp > 0.45 and enemy_tower else 0.0
        retreat_risk = 1.0 if own_hp < 0.35 or (in_enemy_tower_range and ally_near_enemy == 0 and own_hp < 0.65) else 0.0
        no_wave_tower_dive = 1.0 if in_enemy_tower_range and ally_near_enemy == 0 else 0.0

        slots = _slot_states(own_hero)
        enemy_slots = _slot_states(enemy_hero)
        skill_ready_ratio = _ratio(sum(1 for slot in slots if slot.get("usable")), max(len(slots), 1))
        skill_ready_by_slot = self._skill_ready_by_slot(slots)
        enemy_skill_ready_by_slot = self._skill_ready_by_slot(enemy_slots)
        low_enemy_minions = sum(1 for minion in enemy_minions if _hp_ratio(minion) < 0.35)
        move_delta, enemy_dist_delta = self._movement_deltas(own_hero, dist_enemy_hero)
        took_hurt = 1.0 if own_hero and own_hero.get("take_hurt_infos") else 0.0
        in_grass = 1.0 if own_hero and own_hero.get("is_in_grass") else 0.0
        hp_trend = self._hp_trend(own_hp)
        post_kill_state = self._post_kill_state(frame_no, own_hero, enemy_hero)
        post_clear_active = self._post_clear_active(enemy_minions)

        own_config_id = int(_safe_float(own_hero.get("config_id"), -1)) if own_hero else -1
        enemy_config_id = int(_safe_float(enemy_hero.get("config_id"), -1)) if enemy_hero else -1
        own_is_luban = 1.0 if own_config_id == LUBAN_HERO_ID else 0.0
        enemy_is_luban = 1.0 if enemy_config_id == LUBAN_HERO_ID else 0.0
        own_is_direnjie = 1.0 if own_config_id == DIRENJIE_HERO_ID else 0.0
        enemy_is_direnjie = 1.0 if enemy_config_id == DIRENJIE_HERO_ID else 0.0
        own_atk_spd = _ratio(own_hero.get("atk_spd") if own_hero else 0, 20000.0)
        enemy_direnjie_skill_threat = (
            enemy_is_direnjie
            * max(0.60 * enemy_skill_ready_by_slot.get(1, 0.0), enemy_skill_ready_by_slot.get(3, 0.0))
            * (1.0 if enemy_hero and dist_enemy_hero <= attack_range * 1.20 else 0.0)
        )

        lane_state = self._lane_state(
            own_hero=own_hero,
            enemy_hero=enemy_hero,
            own_tower=own_tower,
            enemy_tower=enemy_tower,
            ally_minions=ally_minions,
            enemy_minions=enemy_minions,
            attack_range=attack_range,
            enemy_near_own=enemy_near_own,
            ally_near_enemy=ally_near_enemy,
        )
        neutral_state = self._neutral_state(
            own_hero=own_hero,
            enemy_hero=enemy_hero,
            monsters=monsters,
            attack_range=attack_range,
            own_hp=own_hp,
            lane_clear_priority=lane_state["lane_clear_priority"],
        )
        recovery_state = self._recovery_state(
            own_hero=own_hero,
            enemy_hero=enemy_hero,
            own_tower=own_tower,
            enemy_tower=enemy_tower,
            cakes=cakes,
            own_hp=own_hp,
            hp_trend=hp_trend,
            enemy_near_own=enemy_near_own,
            no_wave_tower_dive=no_wave_tower_dive,
            dist_own_tower=dist_own_tower,
            skill_ready_ratio=skill_ready_ratio,
        )
        tower_state = self._tower_state(
            own_hero=own_hero,
            enemy_hero=enemy_hero,
            enemy_tower=enemy_tower,
            own_hp=own_hp,
            attack_range=attack_range,
            ally_near_enemy=ally_near_enemy,
            no_wave_tower_dive=no_wave_tower_dive,
            enemy_tower_hp=enemy_tower_hp,
            own_is_luban=own_is_luban,
            own_is_direnjie=own_is_direnjie,
        )
        skill_state = self._skill_followup_state(
            own_hero=own_hero,
            slots=slots,
            own_is_luban=own_is_luban,
            own_is_direnjie=own_is_direnjie,
            enemy_in_attack_range=enemy_in_attack_range,
            tower_attack_safe_window=tower_state["tower_attack_safe_window"],
            tower_edge_poke_window=tower_state["tower_edge_poke_window"],
            own_hp=own_hp,
            enemy_hp=enemy_hp,
            dist_enemy_hero=dist_enemy_hero,
            attack_range=attack_range,
            took_hurt=took_hurt,
            hp_trend=hp_trend,
        )
        semantic_state = self._semantic_state(
            own_hp=own_hp,
            enemy_hp=enemy_hp,
            own_hero=own_hero,
            enemy_hero=enemy_hero,
            lane_state=lane_state,
            neutral_state=neutral_state,
            recovery_state=recovery_state,
            tower_state=tower_state,
            skill_state=skill_state,
            enemy_in_attack_range=enemy_in_attack_range,
            took_hurt=took_hurt,
            no_wave_tower_dive=no_wave_tower_dive,
            safe_tower_pressure=safe_tower_pressure,
            retreat_risk=retreat_risk,
            in_grass=in_grass,
            hp_trend=hp_trend,
            post_kill_state=post_kill_state,
        )
        main_task_state = self._main_task_priority(
            lane_state=lane_state,
            recovery_state=recovery_state,
            tower_state=tower_state,
            skill_state=skill_state,
            post_clear_active=post_clear_active,
            own_is_direnjie=own_is_direnjie,
            enemy_in_attack_range=enemy_in_attack_range,
            took_hurt=took_hurt,
            enemy_hp=enemy_hp,
            own_hp=own_hp,
        )
        semantic_state = self._apply_main_task_priority(semantic_state, main_task_state)
        matchup_state = self._matchup_strategy_state(
            own_is_luban=own_is_luban,
            own_is_direnjie=own_is_direnjie,
            enemy_is_luban=enemy_is_luban,
            enemy_is_direnjie=enemy_is_direnjie,
            skill_state=skill_state,
            tower_state=tower_state,
            enemy_skill_ready_by_slot=enemy_skill_ready_by_slot,
            enemy_in_attack_range=enemy_in_attack_range,
            own_hp=own_hp,
            enemy_hp=enemy_hp,
            took_hurt=took_hurt,
            dist_enemy_hero=dist_enemy_hero,
            attack_range=attack_range,
        )
        schema_extra_state = self._schema_lock_extra_features(
            frame_state=frame_state,
            own_hero=own_hero,
            enemy_hero=enemy_hero,
            own_tower=own_tower,
            enemy_tower=enemy_tower,
            ally_minions=ally_minions,
            enemy_minions=enemy_minions,
            monsters=monsters,
            cakes=cakes,
            own_hp=own_hp,
            enemy_hp=enemy_hp,
            dist_own_tower=dist_own_tower,
            dist_enemy_tower=dist_enemy_tower,
            dist_enemy_hero=dist_enemy_hero,
            attack_range=attack_range,
            tower_state=tower_state,
            lane_state=lane_state,
            recovery_state=recovery_state,
            skill_state=skill_state,
            matchup_state=matchup_state,
            skill_ready_by_slot=skill_ready_by_slot,
            enemy_skill_ready_by_slot=enemy_skill_ready_by_slot,
            own_is_luban=own_is_luban,
            own_is_direnjie=own_is_direnjie,
            enemy_is_luban=enemy_is_luban,
            enemy_is_direnjie=enemy_is_direnjie,
            enemy_in_attack_range=enemy_in_attack_range,
            took_hurt=took_hurt,
            hp_trend=hp_trend,
        )

        features = [
            own_hp,
            enemy_hp,
            own_tower_hp,
            enemy_tower_hp,
            _ratio(own_hero.get("money_cnt") if own_hero else 0, 12000.0),
            _signed_ratio(
                _safe_float(own_hero.get("money_cnt") if own_hero else 0)
                - _safe_float(enemy_hero.get("money_cnt") if enemy_hero else 0),
                6000.0,
            ),
            _ratio(own_hero.get("exp") if own_hero else 0, 2200.0),
            _signed_ratio(
                _safe_float(own_hero.get("exp") if own_hero else 0)
                - _safe_float(enemy_hero.get("exp") if enemy_hero else 0),
                2200.0,
            ),
            _ratio(dist_own_tower, 30000.0, default=1.0),
            _ratio(dist_enemy_tower, 30000.0, default=1.0),
            _ratio(len(ally_minions), 8.0),
            _ratio(len(enemy_minions), 8.0),
            _signed_ratio(len(ally_minions) - len(enemy_minions), 8.0),
            _ratio(ally_near_enemy, 4.0),
            _ratio(enemy_near_own, 4.0),
            safe_tower_pressure,
            retreat_risk,
            _ratio(frame_state.get("frame_no", 0), MAX_FRAME_NO),
            _signed_ratio(_unit_value(own_hero, "level") - _unit_value(enemy_hero, "level"), 15.0),
            _signed_ratio(_unit_value(own_hero, "kill_cnt") - _unit_value(enemy_hero, "kill_cnt"), 5.0),
            _signed_ratio(_unit_value(enemy_hero, "dead_cnt") - _unit_value(own_hero, "dead_cnt"), 5.0),
            _signed_ratio(
                _unit_value(own_hero, "total_hurt_to_hero") - _unit_value(own_hero, "total_be_hurt_by_hero"),
                6000.0,
            ),
            skill_ready_ratio,
            1.0 if enemy_hp > 0.0 and enemy_hp < 0.35 else 0.0,
            _ratio(low_enemy_minions, 4.0),
            no_wave_tower_dive,
            _ratio(dist_enemy_hero, 30000.0, default=1.0),
            attack_band,
            enemy_in_attack_range,
            _ratio(move_delta, 6000.0),
            _signed_ratio(enemy_dist_delta, 6000.0),
            took_hurt,
            in_grass,
            skill_ready_by_slot.get(1, 0.0),
            skill_ready_by_slot.get(2, 0.0),
            skill_ready_by_slot.get(3, 0.0),
            semantic_state["should_clear_wave"],
            semantic_state["last_hit_priority"],
            semantic_state["over_wave_danger"],
            semantic_state["can_pressure_tower"],
            semantic_state["must_hit_tower_now"],
            semantic_state["tower_dive_danger"],
            semantic_state["can_fight"],
            semantic_state["should_kite_back"],
            semantic_state["kill_window"],
            semantic_state["should_take_neutral"],
            semantic_state["should_take_cake"],
            semantic_state["should_recall"],
            semantic_state["low_hp_stall"],
            semantic_state["resource_advantage"],
            semantic_state["danger_level"],
            semantic_state["passive_followup"],
            semantic_state["skill_waste_risk"],
            semantic_state["purposeful_grass"],
            own_is_luban,
            own_is_direnjie,
            enemy_is_luban,
            enemy_is_direnjie,
            1.0 if own_is_luban and enemy_is_direnjie else 0.0,
            1.0 if own_is_direnjie and enemy_is_luban else 0.0,
            skill_state["direnjie_stack_pressure"],
            skill_state["direnjie_skill1_poke_window"],
            skill_state["direnjie_skill2_escape_window"],
            skill_state["direnjie_skill2_waste_risk"],
            skill_state["direnjie_ult_kill_window"],
            skill_state["direnjie_ult_waste_risk"],
            skill_state["direnjie_followup_window"],
            enemy_direnjie_skill_threat,
            matchup_state["luban_vs_luban"],
            matchup_state["luban_vs_direnjie"],
            matchup_state["direnjie_vs_luban"],
            matchup_state["direnjie_vs_direnjie"],
            matchup_state["luban_mirror_tower_first"],
            matchup_state["luban_into_direnjie_safe_passive"],
            matchup_state["direnjie_into_luban_punish"],
            matchup_state["direnjie_mirror_cooldown_trade"],
        ]
        features.extend(schema_extra_state)

        if len(features) != TACTICAL_FEATURE_DIM:
            raise ValueError(f"TACTICAL_FEATURE_DIM mismatch: {len(features)} != {TACTICAL_FEATURE_DIM}")
        return [float(_clip(v, -1.0, 1.0)) for v in features]

    def _schema_lock_extra_features(
        self,
        frame_state,
        own_hero,
        enemy_hero,
        own_tower,
        enemy_tower,
        ally_minions,
        enemy_minions,
        monsters,
        cakes,
        own_hp,
        enemy_hp,
        dist_own_tower,
        dist_enemy_tower,
        dist_enemy_hero,
        attack_range,
        tower_state,
        lane_state,
        recovery_state,
        skill_state,
        matchup_state,
        skill_ready_by_slot,
        enemy_skill_ready_by_slot,
        own_is_luban,
        own_is_direnjie,
        enemy_is_luban,
        enemy_is_direnjie,
        enemy_in_attack_range,
        took_hurt,
        hp_trend,
    ):
        own_buff = self._buff_summary(own_hero)
        enemy_buff = self._buff_summary(enemy_hero)
        own_cake_pos, enemy_cake_pos = self._fixed_cake_locations()
        own_fixed_cake_dist = self._point_dist(own_hero, own_cake_pos)
        enemy_fixed_cake_dist = self._point_dist(own_hero, enemy_cake_pos)
        enemy_to_enemy_cake_dist = self._point_dist(enemy_hero, enemy_cake_pos)
        nearest_cake, nearest_cake_dist = self._nearest_cake(cakes, own_hero)
        visible_cake_count = len(cakes)
        river_spirit = self._find_river_spirit(monsters)
        river_spirit_dist = _dist(own_hero, river_spirit)
        river_spirit_hp = _hp_ratio(river_spirit)

        if enemy_hero and enemy_hp > 0.0:
            self.enemy_missing_steps = 0
        else:
            self.enemy_missing_steps = min(self.enemy_missing_steps + 1, 32)

        enemy_tower_range = _safe_float(enemy_tower.get("attack_range") if enemy_tower else 8800.0, 8800.0)
        own_tower_range = _safe_float(own_tower.get("attack_range") if own_tower else 8800.0, 8800.0)
        pressure_outer_range = max(attack_range + TOWER_ATTACK_RANGE_PAD, enemy_tower_range + TOWER_EDGE_OUTER_PAD)
        enemy_under_tower_dist = _dist(enemy_hero, enemy_tower)
        hero_progress = self._lane_progress(own_hero, own_tower, enemy_tower)
        enemy_progress = self._lane_progress(enemy_hero, own_tower, enemy_tower)

        own_attack_target = self._attack_target_id(own_hero)
        enemy_runtime_id = int(_safe_float(enemy_hero.get("runtime_id"), -9999)) if enemy_hero else -9999
        enemy_tower_runtime_id = int(_safe_float(enemy_tower.get("runtime_id"), -9998)) if enemy_tower else -9998
        own_real_cmd_score = self._real_cmd_attack_or_skill(own_hero)
        enemy_real_cmd_score = self._real_cmd_attack_or_skill(enemy_hero)

        real_features = [
            own_buff["healing"],
            enemy_buff["healing"],
            own_buff["recovery_spell"],
            enemy_buff["recovery_spell"],
            own_buff["speed_up"],
            enemy_buff["speed_up"],
            own_buff["speed_down"],
            enemy_buff["speed_down"],
            own_buff["cleanse_or_immune"],
            enemy_buff["cleanse_or_immune"],
            own_buff["frenzy_or_lifesteal"],
            enemy_buff["frenzy_or_lifesteal"],
            own_buff["luban_passive"],
            enemy_buff["luban_passive"],
            own_buff["direnjie_stack"],
            enemy_buff["direnjie_stack"],
            own_buff["control_or_mark"],
            enemy_buff["control_or_mark"],
            own_buff["buff_count"],
            enemy_buff["buff_count"],
            own_buff["mark_layers"],
            enemy_buff["mark_layers"],
            own_buff["unknown_count"],
            enemy_buff["unknown_count"],
            _ratio(own_fixed_cake_dist, 40000.0, default=1.0),
            _ratio(enemy_fixed_cake_dist, 40000.0, default=1.0),
            _ratio(visible_cake_count, 2.0),
            _ratio(nearest_cake_dist, 40000.0, default=1.0),
            1.0 if own_fixed_cake_dist <= 7000.0 else 0.0,
            1.0 if enemy_fixed_cake_dist <= 7000.0 else 0.0,
            1.0 if own_hp < 0.45 and own_fixed_cake_dist <= 18000.0 else 0.0,
            1.0 if enemy_hero and enemy_hp < 0.45 and enemy_to_enemy_cake_dist <= 18000.0 else 0.0,
            1.0 if river_spirit else 0.0,
            1.0 if river_spirit and 0.0 < river_spirit_hp <= 0.30 and river_spirit_dist <= attack_range + 1800.0 else 0.0,
            1.0 if own_hero else 0.0,
            1.0 if enemy_hero and enemy_hp > 0.0 else 0.0,
            1.0 if own_tower else 0.0,
            1.0 if enemy_tower else 0.0,
            1.0 if self.enemy_missing_steps > 0 else 0.0,
            _ratio(self.enemy_missing_steps, 32.0),
            1.0 if enemy_hero and enemy_hp > 0.0 and dist_enemy_hero <= attack_range * 1.25 else 0.0,
            1.0 if enemy_tower and _hp_ratio(enemy_tower) > 0.0 else 0.0,
            _signed_ratio(dist_enemy_tower - enemy_tower_range, 8000.0),
            _signed_ratio(pressure_outer_range - dist_enemy_tower, 8000.0),
            _signed_ratio(dist_own_tower - own_tower_range, 8000.0),
            _ratio(tower_state["ally_wave_tanking_enemy_tower"], 1.0),
            _ratio(enemy_under_tower_dist, max(enemy_tower_range + TOWER_TURTLE_RANGE_PAD, 1.0), default=1.0),
            _clip01((enemy_tower_range - dist_enemy_tower) / max(enemy_tower_range, 1.0)),
            hero_progress,
            enemy_progress,
            _ratio(len(own_hero.get("hit_target_info", []) or []) if own_hero else 0, 6.0),
            _ratio(len(enemy_hero.get("hit_target_info", []) or []) if enemy_hero else 0, 6.0),
            _ratio(len(own_hero.get("take_hurt_infos", []) or []) if own_hero else 0, 6.0),
            _ratio(len(enemy_hero.get("take_hurt_infos", []) or []) if enemy_hero else 0, 6.0),
            1.0 if own_attack_target == enemy_runtime_id else 0.0,
            1.0 if own_attack_target == enemy_tower_runtime_id else 0.0,
            own_real_cmd_score,
            enemy_real_cmd_score,
        ]
        semantic_reserved = self._reserved_skill_semantics(
            own_hp=own_hp,
            enemy_hp=enemy_hp,
            dist_enemy_hero=dist_enemy_hero,
            attack_range=attack_range,
            tower_state=tower_state,
            lane_state=lane_state,
            recovery_state=recovery_state,
            skill_state=skill_state,
            matchup_state=matchup_state,
            skill_ready_by_slot=skill_ready_by_slot,
            enemy_skill_ready_by_slot=enemy_skill_ready_by_slot,
            own_is_luban=own_is_luban,
            own_is_direnjie=own_is_direnjie,
            enemy_is_luban=enemy_is_luban,
            enemy_is_direnjie=enemy_is_direnjie,
            enemy_in_attack_range=enemy_in_attack_range,
            took_hurt=took_hurt,
            hp_trend=hp_trend,
        )
        return real_features + semantic_reserved

    def _reserved_skill_semantics(
        self,
        own_hp,
        enemy_hp,
        dist_enemy_hero,
        attack_range,
        tower_state,
        lane_state,
        recovery_state,
        skill_state,
        matchup_state,
        skill_ready_by_slot,
        enemy_skill_ready_by_slot,
        own_is_luban,
        own_is_direnjie,
        enemy_is_luban,
        enemy_is_direnjie,
        enemy_in_attack_range,
        took_hurt,
        hp_trend,
    ):
        own_skill1 = _clip01(skill_ready_by_slot.get(1, skill_state.get("skill1_cd_ready_ratio", 0.0)))
        own_skill2 = _clip01(skill_ready_by_slot.get(2, skill_state.get("skill2_cd_ready_ratio", 0.0)))
        own_skill3 = _clip01(skill_ready_by_slot.get(3, skill_state.get("skill3_cd_ready_ratio", 0.0)))
        enemy_skill1 = _clip01(enemy_skill_ready_by_slot.get(1, 0.0))
        enemy_skill2 = _clip01(enemy_skill_ready_by_slot.get(2, 0.0))
        enemy_skill3 = _clip01(enemy_skill_ready_by_slot.get(3, 0.0))

        own_core_ready = max(own_skill1, own_skill2, own_skill3)
        enemy_core_ready = max(enemy_skill1, enemy_skill2, enemy_skill3)
        own_core_empty = 1.0 - own_core_ready
        enemy_core_empty = 1.0 - enemy_core_ready
        enemy_close = 1.0 if dist_enemy_hero <= attack_range * 1.20 else 0.0
        enemy_poke_band = 1.0 if dist_enemy_hero <= attack_range * 1.35 else 0.0
        under_pressure = 1.0 if took_hurt or hp_trend < -0.006 or (own_hp < 0.42 and enemy_close) else 0.0
        own_safe_hp = 1.0 if own_hp >= 0.42 else 0.0
        own_healthy = 1.0 if own_hp >= 0.60 else 0.0
        enemy_low = 1.0 if 0.0 < enemy_hp <= 0.35 else 0.0
        enemy_mid_low = 1.0 if 0.0 < enemy_hp <= 0.50 else 0.0

        tower_safe = _clip01(tower_state.get("tower_attack_safe_window", 0.0))
        tower_edge = _clip01(tower_state.get("tower_edge_poke_window", 0.0))
        tower_finish = _clip01(tower_state.get("tower_finish_window", 0.0))
        wave_tanking = _clip01(tower_state.get("ally_wave_tanking_enemy_tower", 0.0))
        tower_overstep = _clip01(tower_state.get("tower_edge_overstep", 0.0))
        tower_reset = _clip01(tower_state.get("tower_wave_lost_reset", 0.0))
        enemy_under_tower = _clip01(tower_state.get("enemy_under_tower", 0.0))
        lane_clear = _clip01(lane_state.get("lane_clear_priority", 0.0))
        last_hit = _clip01(lane_state.get("last_hit_priority", 0.0))
        recall_or_recover = max(
            _clip01(recovery_state.get("safe_recall_window", 0.0)),
            _clip01(recovery_state.get("recovery_need", 0.0)),
        )

        luban_passive = _clip01(skill_state.get("passive_ready", 0.0))
        luban_followup = _clip01(skill_state.get("after_skill_followup_window", 0.0))
        luban_value_window = max(tower_safe, tower_edge, enemy_in_attack_range, lane_clear, last_hit)
        luban_skill_waste = own_is_luban * own_core_ready * (1.0 - luban_value_window)
        luban_skill2_save = own_is_luban * own_skill2 * under_pressure
        luban_skill2_execute = own_is_luban * own_skill2 * enemy_low * enemy_poke_band
        luban_tower_ignore_enemy = own_is_luban * tower_safe * enemy_under_tower * own_safe_hp

        direnjie_skill1 = _clip01(skill_state.get("direnjie_skill1_poke_window", 0.0))
        direnjie_skill2_defense = _clip01(skill_state.get("direnjie_skill2_escape_window", 0.0))
        direnjie_skill2_hold = own_is_direnjie * own_skill2 * (1.0 - under_pressure)
        direnjie_skill3_kill = _clip01(skill_state.get("direnjie_ult_kill_window", 0.0))
        direnjie_skill3_open = own_is_direnjie * own_skill3 * enemy_mid_low * own_healthy * (
            1.0 - 0.55 * enemy_skill2
        )
        direnjie_stack = _clip01(skill_state.get("direnjie_stack_pressure", 0.0))
        direnjie_followup = _clip01(skill_state.get("direnjie_followup_window", 0.0))
        direnjie_kite_cd = own_is_direnjie * own_core_empty * max(enemy_core_ready, under_pressure)

        enemy_direnjie_control_ready = enemy_is_direnjie * max(0.60 * enemy_skill1, enemy_skill3)
        enemy_luban_burst_ready = enemy_is_luban * max(enemy_skill1, enemy_skill2, luban_passive)

        features = [
            own_skill1,
            own_skill2,
            own_skill3,
            enemy_skill1,
            enemy_skill2,
            enemy_skill3,
            own_core_ready,
            own_core_empty,
            enemy_core_ready,
            enemy_core_empty,
            1.0 if own_hp > 0.45 and own_core_ready > 0.5 and enemy_core_empty > 0.5 else 0.0,
            1.0 if own_hp < 0.42 and own_core_empty > 0.5 and enemy_core_ready > 0.5 else 0.0,
            enemy_core_empty * enemy_in_attack_range,
            under_pressure,
            own_safe_hp,
            own_healthy,
            enemy_low,
            enemy_mid_low,
            own_is_luban * luban_passive * tower_safe,
            own_is_luban * luban_passive * enemy_in_attack_range * own_safe_hp,
            own_is_luban * luban_followup * max(tower_safe, enemy_in_attack_range, lane_clear),
            luban_skill2_save,
            luban_skill2_execute,
            luban_skill_waste,
            luban_tower_ignore_enemy,
            own_is_luban * tower_reset,
            direnjie_skill1,
            direnjie_skill2_hold,
            direnjie_skill2_defense,
            direnjie_skill3_kill,
            direnjie_skill3_open,
            direnjie_stack,
            direnjie_followup,
            direnjie_kite_cd,
            _clip01(matchup_state.get("luban_mirror_tower_first", 0.0)),
            own_is_luban * enemy_is_direnjie * enemy_direnjie_control_ready * (1.0 - own_healthy),
            _clip01(matchup_state.get("luban_into_direnjie_safe_passive", 0.0)),
            _clip01(matchup_state.get("direnjie_into_luban_punish", 0.0)),
            own_is_direnjie * enemy_luban_burst_ready * own_skill2,
            own_is_direnjie * enemy_is_direnjie * enemy_skill2 * own_skill3,
            _clip01(matchup_state.get("direnjie_mirror_cooldown_trade", 0.0)),
            tower_edge,
            tower_overstep,
            tower_finish,
            wave_tanking * tower_safe,
            tower_reset,
            lane_clear,
            last_hit,
            recall_or_recover * max(under_pressure, own_core_empty),
        ]
        if len(features) > TACTICAL_SCHEMA_RESERVED_DIM:
            raise ValueError(
                f"TACTICAL_SCHEMA_RESERVED_DIM overflow: {len(features)} > {TACTICAL_SCHEMA_RESERVED_DIM}"
            )
        features.extend([0.0] * (TACTICAL_SCHEMA_RESERVED_DIM - len(features)))
        return features

    def _buff_summary(self, hero):
        buff_ids, mark_layers = self._extract_buff_ids_and_layers(hero)
        buff_id_set = set(buff_ids)
        unknown_ids = [buff_id for buff_id in buff_ids if buff_id not in KNOWN_BUFF_IDS and buff_id // 1000 not in {112, 133}]
        return {
            "healing": 1.0 if buff_id_set & HEALING_BUFF_IDS else 0.0,
            "recovery_spell": 1.0 if buff_id_set & {10000, 10010} else 0.0,
            "speed_up": 1.0 if buff_id_set & SPEED_UP_BUFF_IDS else 0.0,
            "speed_down": 1.0 if buff_id_set & SPEED_DOWN_BUFF_IDS else 0.0,
            "cleanse_or_immune": 1.0 if buff_id_set & CLEANSE_OR_IMMUNE_BUFF_IDS else 0.0,
            "frenzy_or_lifesteal": 1.0 if buff_id_set & FRENZY_OR_LIFESTEAL_BUFF_IDS else 0.0,
            "luban_passive": 1.0 if any(buff_id // 1000 == 112 for buff_id in buff_ids) else 0.0,
            "direnjie_stack": 1.0 if any(buff_id // 1000 == 133 for buff_id in buff_ids) else 0.0,
            "control_or_mark": 1.0 if buff_id_set & CONTROL_OR_MARK_BUFF_IDS else 0.0,
            "buff_count": _ratio(len(buff_ids), 12.0),
            "mark_layers": _ratio(sum(mark_layers), 10.0),
            "unknown_count": _ratio(len(unknown_ids), 8.0),
        }

    def _extract_buff_ids_and_layers(self, hero):
        if not hero:
            return [], []
        buff_state = hero.get("buff_state", {})
        if not isinstance(buff_state, dict):
            return [], []
        ids = []
        layers = []
        containers = []
        for key in ("buff_skills", "buffSkills", "buff_marks", "buffMarks", "buffs"):
            value = buff_state.get(key)
            if isinstance(value, list):
                containers.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, dict):
                containers.append(value)
        for item in containers:
            buff_id = int(_safe_float(item.get("configId", item.get("config_id", item.get("id", 0)))))
            if buff_id > 0:
                ids.append(buff_id)
            layer = _safe_float(item.get("layer", item.get("times", 0)))
            if layer > 0:
                layers.append(layer)
        return ids, layers

    def _fixed_cake_locations(self):
        key = "PLAYERCAMP_2" if self.camp == "PLAYERCAMP_2" or self.camp == 2 or str(self.camp) == "2" else "PLAYERCAMP_1"
        config = CAKE_LOCATIONS_BY_CAMP[key]
        return config["main"], config["enemy"]

    def _point_dist(self, unit, point):
        if not unit or not point:
            return 99999.0
        ux, uz = _pos(unit)
        return math.hypot(ux - point[0], uz - point[1])

    def _find_river_spirit(self, monsters):
        for monster in monsters:
            if int(_safe_float(monster.get("config_id"), 0)) == RIVER_SPIRIT_CONFIG_ID:
                return monster
        return None

    def _attack_target_id(self, hero):
        if not hero:
            return -1
        target = hero.get("attack_target")
        if isinstance(target, dict):
            return int(_safe_float(target.get("runtime_id", target.get("target_id", target.get("id", -1))), -1))
        return int(_safe_float(target, -1))

    def _real_cmd_attack_or_skill(self, hero):
        if not hero:
            return 0.0
        real_cmd = hero.get("real_cmd", {})
        text = str(real_cmd).upper()
        if any(token in text for token in ("ATTACK", "SKILL", "NORMAL_ATTACK")):
            return 1.0
        if isinstance(real_cmd, dict):
            cmd_id = int(_safe_float(real_cmd.get("cmd", real_cmd.get("action", real_cmd.get("button", 0)))))
            return 1.0 if cmd_id in {3, 4, 5, 6} else 0.0
        return 0.0

    def _movement_deltas(self, own_hero, dist_enemy_hero):
        cur_pos = _pos(own_hero) if own_hero else None
        move_delta = 0.0
        enemy_dist_delta = 0.0
        if self.last_pos and cur_pos:
            move_delta = math.hypot(cur_pos[0] - self.last_pos[0], cur_pos[1] - self.last_pos[1])
        if self.last_enemy_dist is not None:
            enemy_dist_delta = dist_enemy_hero - self.last_enemy_dist
        self.last_pos = cur_pos
        self.last_enemy_dist = dist_enemy_hero if dist_enemy_hero < 99999.0 else self.last_enemy_dist
        return move_delta, enemy_dist_delta

    def _hp_trend(self, own_hp):
        if self.last_hp is None:
            trend = 0.0
        else:
            trend = own_hp - self.last_hp
        self.last_hp = own_hp
        return trend

    def _post_kill_state(self, frame_no, own_hero, enemy_hero):
        own_kill_cnt = _unit_value(own_hero, "kill_cnt")
        enemy_dead_cnt = _unit_value(enemy_hero, "dead_cnt")
        kill_event = False
        if self.last_own_kill_cnt is not None and own_kill_cnt > self.last_own_kill_cnt:
            kill_event = True
        if self.last_enemy_dead_cnt is not None and enemy_dead_cnt > self.last_enemy_dead_cnt:
            kill_event = True

        finish_phase = _phase(frame_no, POST_KILL_FINISH_START_FRAME, POST_KILL_FINISH_FULL_FRAME)
        early_reset_phase = 1.0 - _phase(frame_no, POST_KILL_EARLY_END_FRAME, POST_KILL_FINISH_FULL_FRAME)
        window_steps = int(26 + 54 * finish_phase)
        if kill_event:
            self.post_kill_steps = max(self.post_kill_steps, window_steps)
            self.post_kill_window_steps = max(1, window_steps)

        active = 1.0 if self.post_kill_steps > 0 else 0.0
        window_ratio = _ratio(self.post_kill_steps, max(self.post_kill_window_steps, 1), default=0.0)
        if self.post_kill_steps > 0:
            self.post_kill_steps -= 1

        self.last_own_kill_cnt = own_kill_cnt
        self.last_enemy_dead_cnt = enemy_dead_cnt
        return {
            "active": active,
            "window_ratio": window_ratio,
            "finish_phase": finish_phase,
            "early_reset_phase": early_reset_phase,
            "kill_event": 1.0 if kill_event else 0.0,
        }

    def _post_clear_active(self, enemy_minions):
        enemy_minion_count = len(enemy_minions)
        if self.last_enemy_minion_count is not None and self.last_enemy_minion_count > 0 and enemy_minion_count == 0:
            self.post_clear_steps = 6
        elif self.post_clear_steps > 0:
            self.post_clear_steps -= 1
        self.last_enemy_minion_count = enemy_minion_count
        return 1.0 if self.post_clear_steps > 0 else 0.0

    def _skill_ready_by_slot(self, slots):
        ready = {}
        for slot in slots:
            slot_type = int(_safe_float(slot.get("slot_type"), -1))
            ready[slot_type] = 1.0 if slot.get("usable") else 0.0
        return ready

    def _cooldown_ready_ratio(self, slots, slot_type):
        for slot in slots:
            if int(_safe_float(slot.get("slot_type"), -1)) != slot_type:
                continue
            if slot.get("usable"):
                return 1.0
            cooldown = _safe_float(slot.get("cooldown"), 0.0)
            cooldown_max = _safe_float(slot.get("cooldown_max"), 0.0)
            if cooldown_max <= 0.0:
                return 0.0
            return _clip01(1.0 - cooldown / cooldown_max)
        return 0.0

    def _skill_used_now(self, slots):
        used_now = False
        used_times_by_slot = {}
        for slot in slots:
            slot_type = int(_safe_float(slot.get("slot_type"), -1))
            used_times = int(_safe_float(slot.get("usedTimes"), 0))
            used_times_by_slot[slot_type] = used_times
            if _safe_float(slot.get("succUsedInFrame"), 0) > 0:
                used_now = True
            elif used_times > self.last_skill_used_times.get(slot_type, used_times):
                used_now = True
        self.last_skill_used_times = used_times_by_slot
        return used_now

    def _skill_followup_state(
        self,
        own_hero,
        slots,
        own_is_luban,
        own_is_direnjie,
        enemy_in_attack_range,
        tower_attack_safe_window,
        tower_edge_poke_window,
        own_hp,
        enemy_hp,
        dist_enemy_hero,
        attack_range,
        took_hurt,
        hp_trend,
    ):
        if self._skill_used_now(slots):
            self.followup_steps = 3
        else:
            self.followup_steps = max(0, self.followup_steps - 1)

        passive_cooldown = 99999.0
        for passive in own_hero.get("passive_skill", []) if own_hero else []:
            passive_cooldown = min(passive_cooldown, _safe_float(passive.get("cooldown"), 99999.0))
        passive_cooldown_low = 1.0 if passive_cooldown <= 0.0 else _clip01(1.0 - passive_cooldown / 5000.0)
        after_skill_followup_window = 1.0 if self.followup_steps > 0 else 0.0
        passive_ready = 1.0 if own_is_luban and (after_skill_followup_window or passive_cooldown_low > 0.95) else 0.0
        frenzy_good_window = 1.0 if own_hp > 0.25 and (enemy_in_attack_range or tower_attack_safe_window or enemy_hp < 0.45) else 0.0
        skill1_ready = self._cooldown_ready_ratio(slots, 1)
        skill2_ready = self._cooldown_ready_ratio(slots, 2)
        skill3_ready = self._cooldown_ready_ratio(slots, 3)
        enemy_poke_band = 1.0 if dist_enemy_hero <= attack_range * 1.20 else 0.0
        enemy_ult_band = 1.0 if dist_enemy_hero <= attack_range * 1.05 else 0.0
        safe_pressure_band = max(enemy_in_attack_range, tower_attack_safe_window, tower_edge_poke_window)
        enemy_close_threat = 1.0 if dist_enemy_hero <= attack_range * 1.35 else 0.0
        under_pressure = 1.0 if took_hurt or hp_trend < -0.006 or (own_hp < 0.42 and enemy_close_threat) else 0.0
        ult_commit_hint = (
            1.0
            if enemy_hp > 0.0 and own_hp > 0.25 and (enemy_hp < 0.45 or own_hp > enemy_hp + 0.18)
            else 0.0
        )
        direnjie_stack_pressure = own_is_direnjie * safe_pressure_band * (1.0 - 0.60 * under_pressure)
        direnjie_skill1_poke_window = own_is_direnjie * skill1_ready * max(
            enemy_poke_band,
            tower_attack_safe_window,
            tower_edge_poke_window,
        )
        direnjie_skill2_escape_window = own_is_direnjie * skill2_ready * under_pressure
        direnjie_skill2_waste_risk = own_is_direnjie * skill2_ready * (1.0 - under_pressure)
        direnjie_ult_kill_window = own_is_direnjie * skill3_ready * enemy_ult_band * ult_commit_hint
        direnjie_ult_waste_risk = own_is_direnjie * skill3_ready * (1.0 - enemy_ult_band * ult_commit_hint)
        direnjie_followup_window = own_is_direnjie * after_skill_followup_window * max(
            enemy_in_attack_range,
            tower_attack_safe_window,
            1.0 if enemy_hp > 0.0 and enemy_hp < 0.45 else 0.0,
        )

        return {
            "passive_ready": passive_ready,
            "passive_cooldown_low": passive_cooldown_low,
            "skill1_cd_ready_ratio": skill1_ready,
            "skill2_cd_ready_ratio": skill2_ready,
            "skill3_cd_ready_ratio": skill3_ready,
            "after_skill_followup_window": after_skill_followup_window,
            "frenzy_good_window": frenzy_good_window,
            "direnjie_stack_pressure": direnjie_stack_pressure,
            "direnjie_skill1_poke_window": direnjie_skill1_poke_window,
            "direnjie_skill2_escape_window": direnjie_skill2_escape_window,
            "direnjie_skill2_waste_risk": direnjie_skill2_waste_risk,
            "direnjie_ult_kill_window": direnjie_ult_kill_window,
            "direnjie_ult_waste_risk": direnjie_ult_waste_risk,
            "direnjie_followup_window": direnjie_followup_window,
        }

    def _matchup_strategy_state(
        self,
        own_is_luban,
        own_is_direnjie,
        enemy_is_luban,
        enemy_is_direnjie,
        skill_state,
        tower_state,
        enemy_skill_ready_by_slot,
        enemy_in_attack_range,
        own_hp,
        enemy_hp,
        took_hurt,
        dist_enemy_hero,
        attack_range,
    ):
        luban_vs_luban = own_is_luban * enemy_is_luban
        luban_vs_direnjie = own_is_luban * enemy_is_direnjie
        direnjie_vs_luban = own_is_direnjie * enemy_is_luban
        direnjie_vs_direnjie = own_is_direnjie * enemy_is_direnjie

        enemy_close = 1.0 if dist_enemy_hero <= attack_range * 1.18 else 0.0
        enemy_luban_pressure_threat = enemy_is_luban * enemy_close * (1.0 if enemy_hp > 0.20 else 0.35)
        enemy_direnjie_control_threat = (
            enemy_is_direnjie
            * enemy_close
            * max(
                0.55 * enemy_skill_ready_by_slot.get(1, 0.0),
                enemy_skill_ready_by_slot.get(3, 0.0),
            )
        )
        enemy_direnjie_escape_ready = enemy_is_direnjie * enemy_skill_ready_by_slot.get(2, 0.0)

        tower_pressure = max(
            tower_state["tower_attack_safe_window"],
            tower_state["tower_edge_poke_window"],
            0.55 * tower_state["ally_wave_tanking_enemy_tower"],
        )
        luban_passive_pressure = max(skill_state["passive_ready"], 0.65 * skill_state["after_skill_followup_window"])
        luban_mirror_tower_first = luban_vs_luban * max(tower_pressure, luban_passive_pressure) * (
            1.0 - 0.45 * took_hurt
        )
        luban_into_direnjie_safe_passive = luban_vs_direnjie * max(
            tower_pressure * (1.0 - 0.55 * enemy_direnjie_control_threat),
            luban_passive_pressure * (1.0 - 0.40 * enemy_direnjie_control_threat),
        )
        direnjie_into_luban_punish = direnjie_vs_luban * max(
            skill_state["direnjie_skill1_poke_window"],
            skill_state["direnjie_ult_kill_window"],
            skill_state["direnjie_stack_pressure"] * (1.0 - 0.35 * enemy_luban_pressure_threat),
        )
        direnjie_mirror_cooldown_trade = direnjie_vs_direnjie * max(
            skill_state["direnjie_skill1_poke_window"] * (1.0 - 0.35 * enemy_direnjie_control_threat),
            skill_state["direnjie_skill2_escape_window"] * enemy_direnjie_control_threat,
            skill_state["direnjie_ult_kill_window"] * (1.0 - 0.65 * enemy_direnjie_escape_ready),
            skill_state["direnjie_stack_pressure"] * (1.0 - 0.30 * enemy_direnjie_control_threat),
        )

        return {
            "luban_vs_luban": luban_vs_luban,
            "luban_vs_direnjie": luban_vs_direnjie,
            "direnjie_vs_luban": direnjie_vs_luban,
            "direnjie_vs_direnjie": direnjie_vs_direnjie,
            "luban_mirror_tower_first": luban_mirror_tower_first,
            "luban_into_direnjie_safe_passive": luban_into_direnjie_safe_passive,
            "direnjie_into_luban_punish": direnjie_into_luban_punish,
            "direnjie_mirror_cooldown_trade": direnjie_mirror_cooldown_trade,
        }

    def _semantic_state(
        self,
        own_hp,
        enemy_hp,
        own_hero,
        enemy_hero,
        lane_state,
        neutral_state,
        recovery_state,
        tower_state,
        skill_state,
        enemy_in_attack_range,
        took_hurt,
        no_wave_tower_dive,
        safe_tower_pressure,
        retreat_risk,
        in_grass,
        hp_trend,
        post_kill_state,
    ):
        money_gap = _unit_value(own_hero, "money_cnt") - _unit_value(enemy_hero, "money_cnt")
        level_gap = _unit_value(own_hero, "level") - _unit_value(enemy_hero, "level")
        hp_gap = own_hp - enemy_hp
        resource_advantage = _clip(0.65 * hp_gap + 0.20 * _signed_ratio(money_gap, 2500.0) + 0.15 * _signed_ratio(level_gap, 3.0))
        danger_level = _clip01(
            0.35 * retreat_risk
            + 0.25 * recovery_state["enemy_pressure_nearby"]
            + 0.20 * no_wave_tower_dive
            + 0.15 * (1.0 if hp_trend < -0.004 else 0.0)
            + 0.25 * max(0.0, -resource_advantage)
        )

        post_kill_active = post_kill_state["active"]
        post_kill_finish = post_kill_active * post_kill_state["finish_phase"]
        post_kill_reset = post_kill_active * post_kill_state["early_reset_phase"]
        post_kill_clear_wave = (
            post_kill_active
            * max(post_kill_state["early_reset_phase"], 0.55 * post_kill_state["finish_phase"])
            * max(lane_state["enemy_wave_near_self"], lane_state["enemy_front_minion_count"])
        )
        post_kill_tower_plan = post_kill_active * max(post_kill_finish, 0.45 * post_kill_state["window_ratio"]) * max(
            tower_state["tower_attack_safe_window"],
            0.85 * tower_state["tower_edge_poke_window"],
            0.65 * tower_state["ally_wave_tanking_enemy_tower"],
            0.45 * tower_state["tower_in_normal_attack_range"],
        )
        post_kill_recovery_plan = (
            post_kill_active
            * max(post_kill_reset, 0.35)
            * (1.0 if recovery_state["recovery_commit"] else 0.0)
            * (1.0 - max(tower_state["tower_attack_safe_window"], 0.5 * tower_state["ally_wave_tanking_enemy_tower"]))
        )

        should_clear_wave = max(
            lane_state["lane_clear_priority"] * (1.0 - 0.6 * recovery_state["recovery_commit"]),
            lane_state["enemy_wave_near_self"] * (0.6 + 0.4 * lane_state["self_ahead_of_ally_wave"]),
            post_kill_clear_wave,
        )
        last_hit_priority = lane_state["last_hit_window"]
        over_wave_danger = max(
            lane_state["enemy_minion_between_self_enemy"],
            lane_state["self_ahead_of_ally_wave"],
        )
        can_pressure_tower = max(tower_state["tower_attack_safe_window"], tower_state["tower_edge_poke_window"], post_kill_tower_plan)
        must_hit_tower_now = max(
            tower_state["tower_finish_window"],
            safe_tower_pressure + 0.20 * tower_state["ally_wave_tanking_enemy_tower"],
            0.55 * tower_state["tower_edge_poke_window"],
            post_kill_tower_plan,
        )
        tower_dive_danger = max(
            no_wave_tower_dive,
            tower_state["tower_edge_overstep"],
            tower_state["enemy_tower_can_hit_self"] * (1.0 - tower_state["ally_wave_tanking_enemy_tower"]),
        )
        tower_crash_kill_window = (
            1.0
            if tower_state["ally_wave_tanking_enemy_tower"] > 0.0
            and enemy_hp > 0.0
            and enemy_hp <= TOWER_CRASH_KILL_ENEMY_HP
            and own_hp >= TOWER_CRASH_KILL_MIN_OWN_HP
            and enemy_in_attack_range > 0.0
            and (skill_state["skill1_cd_ready_ratio"] > 0.6 or skill_state["skill3_cd_ready_ratio"] > 0.6 or enemy_hp <= 0.24)
            else 0.0
        )
        low_hp_value_trade_window = (
            1.0
            if own_hp >= TOWER_CRASH_KILL_MIN_OWN_HP
            and own_hp < SHORT_RESET_EXIT_HP
            and enemy_hp > 0.0
            and enemy_hp <= 0.42
            and enemy_in_attack_range > 0.0
            and not no_wave_tower_dive
            and (
                tower_state["ally_wave_tanking_enemy_tower"] > 0.0
                or tower_state["tower_edge_poke_window"] > 0.0
                or enemy_hp <= 0.26
            )
            else 0.0
        )
        low_hp_hold_under_tower_window = (
            1.0
            if own_hp < 0.42
            and lane_state["enemy_wave_near_self"] > 0.0
            and enemy_hp > 0.0
            and enemy_hp <= min(0.56, own_hp + LOW_HP_HOLD_MAX_ENEMY_ADVANTAGE)
            and danger_level < 0.52
            else 0.0
        )
        low_hp_grass_ambush_window = (
            1.0
            if own_hp < SHORT_RESET_EXIT_HP
            and in_grass
            and enemy_hp > 0.0
            and enemy_hp <= LOW_HP_GRASS_RESET_ENEMY_HP
            and enemy_in_attack_range > 0.0
            and danger_level < 0.68
            and tower_dive_danger < 0.5
            else 0.0
        )
        enemy_weaker_window = 1.0 if enemy_hp > 0.0 and enemy_hp + LOW_HP_RELATIVE_ADVANTAGE <= own_hp else 0.0
        hero_pressure_interrupt = (
            1.0
            if enemy_in_attack_range > 0.0
            and lane_state["lane_clear_priority"] > 0.0
            and lane_state["own_tower_defend_clear"] == 0.0
            and recovery_state["low_hp_hold_under_tower_window"] == 0.0
            and (
                took_hurt > 0.0
                or enemy_hp >= own_hp + 0.10
                or (own_hp < 0.48 and enemy_hp > 0.28)
            )
            else 0.0
        )
        low_hp_commit_trade_window = max(tower_crash_kill_window, low_hp_value_trade_window)
        should_clear_wave *= 1.0 - 0.55 * hero_pressure_interrupt
        can_fight = _clip01(
            0.40 * enemy_in_attack_range
            + 0.35 * max(0.0, resource_advantage)
            + 0.20 * (1.0 if enemy_hp < 0.42 else 0.0)
            + 0.15 * enemy_weaker_window
            + 0.24 * hero_pressure_interrupt
            + 0.10 * skill_state["frenzy_good_window"]
            + 0.25 * tower_crash_kill_window
            + 0.20 * low_hp_commit_trade_window
            + 0.15 * tower_state["tower_edge_poke_window"]
            - 0.45 * danger_level
            - 0.22 * should_clear_wave
        )
        should_kite_back = _clip01(
            0.45 * danger_level
            + 0.20 * took_hurt
            + 0.20 * recovery_state["enemy_pressure_nearby"]
            + 0.15 * (1.0 if own_hp < 0.42 else 0.0)
            + 0.20 * tower_state["tower_edge_overstep"]
            + 0.25 * tower_state["tower_wave_lost_reset"]
            - 0.18 * low_hp_hold_under_tower_window
            - 0.22 * low_hp_grass_ambush_window
            - 0.25 * low_hp_commit_trade_window
            - 0.12 * enemy_weaker_window
        )
        kill_window = 1.0 if tower_crash_kill_window or low_hp_commit_trade_window or (enemy_hp > 0.0 and enemy_hp < 0.30 and own_hp > 0.28 and danger_level < 0.55) else 0.0
        should_take_neutral = max(
            neutral_state["monster_finish_window"],
            neutral_state["neutral_safe_window"],
            post_kill_active * (1.0 - max(can_pressure_tower, should_clear_wave)) * (1.0 if own_hp > 0.50 else 0.0),
        )
        should_take_cake = max(recovery_state["low_hp_cake_window"], post_kill_recovery_plan * 0.7)
        should_recall = max(recovery_state["safe_recall_window"], post_kill_recovery_plan) * (
            1.0 - max(
                tower_crash_kill_window,
                low_hp_commit_trade_window,
                low_hp_grass_ambush_window,
                low_hp_hold_under_tower_window,
                lane_state["lane_clear_priority"] * lane_state["enemy_wave_near_self"],
            )
        )
        low_hp_stall = _ratio(self.low_hp_stall_steps, 12.0)
        passive_followup = max(skill_state["passive_ready"], skill_state["after_skill_followup_window"])
        skill_waste_risk = 1.0 if danger_level > 0.55 or (not enemy_in_attack_range and not should_clear_wave and not can_pressure_tower) else 0.0
        purposeful_grass = 1.0 if in_grass and (danger_level > 0.45 or can_pressure_tower or enemy_in_attack_range or low_hp_grass_ambush_window > 0.0) else 0.0

        return {
            "should_clear_wave": should_clear_wave,
            "last_hit_priority": last_hit_priority,
            "over_wave_danger": over_wave_danger,
            "can_pressure_tower": can_pressure_tower,
            "must_hit_tower_now": must_hit_tower_now,
            "tower_dive_danger": tower_dive_danger,
            "can_fight": can_fight,
            "should_kite_back": should_kite_back,
            "kill_window": kill_window,
            "should_take_neutral": should_take_neutral,
            "should_take_cake": should_take_cake,
            "should_recall": should_recall,
            "low_hp_stall": low_hp_stall,
            "resource_advantage": resource_advantage,
            "danger_level": danger_level,
            "passive_followup": passive_followup,
            "skill_waste_risk": skill_waste_risk,
            "purposeful_grass": purposeful_grass,
            "low_hp_hold_under_tower_window": low_hp_hold_under_tower_window,
            "low_hp_grass_ambush_window": low_hp_grass_ambush_window,
            "low_hp_commit_trade_window": low_hp_commit_trade_window,
        }

    def _main_task_priority(
        self,
        lane_state,
        recovery_state,
        tower_state,
        skill_state,
        post_clear_active,
        own_is_direnjie,
        enemy_in_attack_range,
        took_hurt,
        enemy_hp,
        own_hp,
    ):
        tower_crash_kill = (
            1.0
            if tower_state["ally_wave_tanking_enemy_tower"] > 0.0
            and 0.0 < enemy_hp <= TOWER_CRASH_KILL_ENEMY_HP
            and own_hp >= TOWER_CRASH_KILL_MIN_OWN_HP
            and enemy_in_attack_range > 0.0
            and (
                skill_state["skill1_cd_ready_ratio"] > 0.6
                or skill_state["skill3_cd_ready_ratio"] > 0.6
                or enemy_hp <= 0.24
            )
            else 0.0
        )
        low_hp_value_trade = (
            1.0
            if own_hp >= LOW_HP_TRADE_MIN_OWN_HP
            and own_hp < SHORT_RESET_EXIT_HP
            and 0.0 < enemy_hp <= LOW_HP_TRADE_ENEMY_HP
            and enemy_in_attack_range > 0.0
            and tower_state["tower_edge_overstep"] == 0.0
            and (
                tower_state["ally_wave_tanking_enemy_tower"] > 0.0
                or tower_state["tower_edge_poke_window"] > 0.0
                or enemy_hp <= 0.26
            )
            else 0.0
        )
        own_tower_defend_clear = _clip01(lane_state.get("own_tower_defend_clear", 0.0))
        hero_interrupts_clear = (
            1.0
            if lane_state["lane_clear_priority"] > 0.0
            and enemy_in_attack_range > 0.0
            and (
                took_hurt > 0.0
                or enemy_hp >= own_hp + 0.10
                or (own_hp < 0.48 and enemy_hp > 0.28)
            )
            and own_tower_defend_clear == 0.0
            and recovery_state["low_hp_hold_under_tower_window"] == 0.0
            else 0.0
        )
        direnjie_ult_follow = (
            1.0
            if own_is_direnjie
            and self.followup_steps > 0
            and skill_state["direnjie_ult_kill_window"] > 0.0
            else 0.0
        )
        task = {
            "clear_wave": 0.0,
            "tower_pressure": 0.0,
            "hero_contest": 0.0,
            "commit_kill": 0.0,
            "short_reset": 0.0,
        }
        if tower_crash_kill or low_hp_value_trade or direnjie_ult_follow:
            task["commit_kill"] = 1.0
        elif recovery_state["safe_recall_window"] > 0.0 or recovery_state["recovery_commit"] > 0.0:
            task["short_reset"] = 1.0
        elif tower_state["tower_wave_lost_reset"] > 0.0:
            task["short_reset"] = 1.0
        elif lane_state["lane_clear_priority"] > 0.0 and own_tower_defend_clear > 0.0:
            task["clear_wave"] = 1.0
        elif tower_state["tower_attack_safe_window"] > 0.0 or tower_state["tower_edge_poke_window"] > 0.0:
            task["tower_pressure"] = 1.0
        elif post_clear_active > 0.0 and enemy_in_attack_range > 0.0:
            task["hero_contest"] = 1.0
        elif hero_interrupts_clear:
            task["hero_contest"] = 1.0
        elif lane_state["lane_clear_priority"] > 0.0:
            task["clear_wave"] = 1.0
        elif enemy_in_attack_range > 0.0 and recovery_state["recovery_commit"] == 0.0:
            task["hero_contest"] = 1.0
        else:
            task["clear_wave"] = 1.0
        return task

    def _apply_main_task_priority(self, semantic_state, main_task_state):
        state = dict(semantic_state)
        if main_task_state["clear_wave"]:
            state["should_clear_wave"] = max(state["should_clear_wave"], 1.0)
            state["can_pressure_tower"] *= 0.35
            state["can_fight"] *= 0.45
            state["should_recall"] *= 0.35
        elif main_task_state["tower_pressure"]:
            state["can_pressure_tower"] = max(state["can_pressure_tower"], 1.0)
            state["must_hit_tower_now"] = max(state["must_hit_tower_now"], 0.85)
            state["should_clear_wave"] *= 0.35
            state["should_recall"] *= 0.25
        elif main_task_state["hero_contest"]:
            state["can_fight"] = max(state["can_fight"], 0.80)
            state["should_clear_wave"] *= 0.45
            state["should_recall"] *= 0.35
        elif main_task_state["commit_kill"]:
            state["can_fight"] = 1.0
            state["kill_window"] = 1.0
            state["should_kite_back"] *= 0.35
            state["should_clear_wave"] *= 0.25
            state["should_recall"] = 0.0
        elif main_task_state["short_reset"]:
            state["should_kite_back"] = max(state["should_kite_back"], 0.85)
            state["should_recall"] = max(state["should_recall"], 0.70)
            state["can_fight"] *= 0.25
            state["can_pressure_tower"] *= 0.25
        state["resource_advantage"] = _clip(state["resource_advantage"])
        return {key: (value if key == "resource_advantage" else _clip01(value)) for key, value in state.items()}

    def _find_heroes(self, frame_state, player_id):
        own_hero, enemy_hero = None, None
        for hero in frame_state.get("hero_states", []):
            is_own = hero.get("runtime_id") == player_id or hero.get("player_id") == player_id or hero.get("camp") == self.camp
            if is_own and own_hero is None:
                own_hero = hero
            elif not is_own and enemy_hero is None:
                enemy_hero = hero
        return own_hero, enemy_hero

    def _find_towers(self, frame_state):
        own_tower, enemy_tower = None, None
        for npc in frame_state.get("npc_states", []):
            config_id = int(_safe_float(npc.get("config_id"), 0))
            if npc.get("sub_type") != TOWER_SUB_TYPE and config_id not in TOWER_CONFIG_IDS:
                continue
            if npc.get("camp") == self.camp:
                own_tower = npc
            else:
                enemy_tower = npc
        return own_tower, enemy_tower

    def _find_minions(self, frame_state):
        ally_minions, enemy_minions = [], []
        for npc in frame_state.get("npc_states", []):
            config_id = int(_safe_float(npc.get("config_id"), 0))
            if npc.get("sub_type") != MINION_SUB_TYPE and config_id not in LANE_SOLDIER_CONFIG_IDS:
                continue
            if npc.get("camp") == self.camp:
                ally_minions.append(npc)
            else:
                enemy_minions.append(npc)
        return ally_minions, enemy_minions

    def _find_monsters(self, frame_state):
        monsters = []
        for npc in frame_state.get("npc_states", []):
            if self._is_monster(npc):
                monsters.append(npc)
        return monsters

    def _is_monster(self, npc):
        config_id = int(_safe_float(npc.get("config_id"), 0))
        if config_id in LANE_SOLDIER_CONFIG_IDS | TOWER_CONFIG_IDS | CRYSTAL_CONFIG_IDS | SPRING_TOWER_CONFIG_IDS:
            return False
        if config_id == RIVER_SPIRIT_CONFIG_ID:
            return True
        actor_type = npc.get("actor_type")
        if isinstance(actor_type, str) and "MONSTER" in actor_type.upper():
            return True
        sub_type = npc.get("sub_type")
        if sub_type in {MINION_SUB_TYPE, TOWER_SPRING_SUB_TYPE, TOWER_SUB_TYPE, CRYSTAL_SUB_TYPE}:
            return False
        max_hp = _safe_float(npc.get("max_hp"), 0.0)
        if max_hp <= 0.0:
            return False
        camp = npc.get("camp")
        return camp in {0, "0", None}

    def _find_cakes(self, frame_state):
        return [cake for cake in frame_state.get("cakes", []) if isinstance(cake, dict)]

    def _count_near(self, units, anchor, tower):
        if not anchor:
            return 0
        tower_range = _safe_float(tower.get("attack_range") if tower else 8800.0, 8800.0)
        return sum(1 for unit in units if _dist(unit, anchor) <= tower_range)

    def _lane_progress(self, unit, own_tower, enemy_tower):
        if not unit or not own_tower or not enemy_tower:
            return 0.0
        ux, uz = _pos(unit)
        ox, oz = _pos(own_tower)
        ex, ez = _pos(enemy_tower)
        vx, vz = ex - ox, ez - oz
        denom = vx * vx + vz * vz
        if denom <= 1.0:
            return 0.0
        return _clip01(((ux - ox) * vx + (uz - oz) * vz) / denom)

    def _lane_lateral_ratio(self, unit, own_tower, enemy_tower):
        if not unit or not own_tower or not enemy_tower:
            return 1.0
        ux, uz = _pos(unit)
        ox, oz = _pos(own_tower)
        ex, ez = _pos(enemy_tower)
        vx, vz = ex - ox, ez - oz
        norm = math.hypot(vx, vz)
        if norm <= 1.0:
            return 1.0
        lateral = abs((ux - ox) * vz - (uz - oz) * vx) / norm
        return _ratio(lateral, LANE_LATERAL_SCALE, default=1.0)

    def _nearest(self, units, anchor):
        if not units or not anchor:
            return None, 99999.0
        best = min(units, key=lambda unit: _dist(unit, anchor))
        return best, _dist(best, anchor)

    def _lane_state(
        self,
        own_hero,
        enemy_hero,
        own_tower,
        enemy_tower,
        ally_minions,
        enemy_minions,
        attack_range,
        enemy_near_own,
        ally_near_enemy,
    ):
        hero_progress = self._lane_progress(own_hero, own_tower, enemy_tower)
        enemy_progress = self._lane_progress(enemy_hero, own_tower, enemy_tower)
        ally_progresses = [self._lane_progress(unit, own_tower, enemy_tower) for unit in ally_minions]
        enemy_progresses = [self._lane_progress(unit, own_tower, enemy_tower) for unit in enemy_minions]
        max_ally_progress = max(ally_progresses) if ally_progresses else hero_progress

        enemy_minion_between = 0.0
        if own_hero and enemy_hero:
            low, high = sorted((hero_progress, enemy_progress))
            for minion in enemy_minions:
                p = self._lane_progress(minion, own_tower, enemy_tower)
                if low <= p <= high and self._lane_lateral_ratio(minion, own_tower, enemy_tower) < 0.55:
                    enemy_minion_between = 1.0
                    break

        self_ahead_of_ally_wave = 1.0 if ally_minions and hero_progress > max_ally_progress + 0.08 else 0.0
        nearest_enemy_minion, nearest_enemy_minion_dist = self._nearest(enemy_minions, own_hero)
        nearest_enemy_minion_hp = _hp_ratio(nearest_enemy_minion)
        last_hit_window = (
            1.0
            if nearest_enemy_minion
            and nearest_enemy_minion_hp > 0.0
            and nearest_enemy_minion_hp < 0.35
            and nearest_enemy_minion_dist <= attack_range + 1800.0
            else 0.0
        )

        enemy_front_count = sum(1 for p in enemy_progresses if p >= hero_progress - 0.03)
        ally_front_count = sum(1 for p in ally_progresses if p >= hero_progress - 0.03)
        enemy_wave_near_self = 1.0 if nearest_enemy_minion_dist <= attack_range + 3500.0 else 0.0
        own_tower_low_hp = 1.0 if own_tower and 0.0 < _hp_ratio(own_tower) <= 0.35 else 0.0
        own_tower_defend_clear = 1.0 if enemy_near_own > 0 or (own_tower_low_hp and enemy_wave_near_self) else 0.0
        lane_clear_priority = 1.0 if enemy_wave_near_self and (
            own_tower_defend_clear
            or nearest_enemy_minion_dist <= attack_range + 1200.0
            or self_ahead_of_ally_wave
            or enemy_minion_between
        ) else 0.0
        lane_advantage = _clip((ally_front_count - enemy_front_count) / 6.0)

        return {
            "enemy_minion_between_self_enemy": enemy_minion_between,
            "self_ahead_of_ally_wave": self_ahead_of_ally_wave,
            "nearest_enemy_minion_dist": _ratio(nearest_enemy_minion_dist, 30000.0, default=1.0),
            "nearest_enemy_minion_hp": nearest_enemy_minion_hp,
            "last_hit_window": last_hit_window,
            "enemy_front_minion_count": _ratio(enemy_front_count, 6.0),
            "ally_front_minion_count": _ratio(ally_front_count, 6.0),
            "enemy_wave_near_self": enemy_wave_near_self,
            "lane_clear_priority": lane_clear_priority,
            "own_tower_defend_clear": own_tower_defend_clear,
            "own_tower_low_hp": own_tower_low_hp,
            "lane_advantage": lane_advantage,
            "ally_near_enemy": ally_near_enemy,
        }

    def _neutral_state(self, own_hero, enemy_hero, monsters, attack_range, own_hp, lane_clear_priority):
        nearest_monster, nearest_monster_dist = self._nearest(monsters, own_hero)
        monster_visible = 1.0 if nearest_monster else 0.0
        nearest_monster_hp = _hp_ratio(nearest_monster)
        monster_in_attack_range = 1.0 if nearest_monster and nearest_monster_dist <= attack_range + 1600.0 else 0.0
        enemy_close = enemy_hero and _dist(own_hero, enemy_hero) <= attack_range * 1.25
        monster_finish_window = (
            1.0
            if nearest_monster
            and 0.0 < nearest_monster_hp <= 0.28
            and nearest_monster_dist <= attack_range + 1800.0
            and own_hp > 0.45
            and not enemy_close
            else 0.0
        )
        neutral_safe_window = (
            1.0
            if monster_visible
            and own_hp > 0.48
            and (not lane_clear_priority or monster_finish_window)
            and not enemy_close
            and nearest_monster_dist <= 18000.0
            else 0.0
        )
        return {
            "monster_visible": monster_visible,
            "nearest_monster_dist": _ratio(nearest_monster_dist, 30000.0, default=1.0),
            "nearest_monster_hp": nearest_monster_hp,
            "monster_in_attack_range": monster_in_attack_range,
            "neutral_safe_window": neutral_safe_window,
            "monster_finish_window": monster_finish_window,
        }

    def _cake_pos(self, cake):
        collider = cake.get("collider", {}) if cake else {}
        location = collider.get("location", {}) if isinstance(collider, dict) else {}
        return (_safe_float(location.get("x")), _safe_float(location.get("z")))

    def _cake_dist(self, cake, unit):
        if not cake or not unit:
            return 99999.0
        cx, cz = self._cake_pos(cake)
        ux, uz = _pos(unit)
        return math.hypot(cx - ux, cz - uz)

    def _nearest_cake(self, cakes, own_hero):
        if not cakes or not own_hero:
            return None, 99999.0
        best = min(cakes, key=lambda cake: self._cake_dist(cake, own_hero))
        return best, self._cake_dist(best, own_hero)

    def _recovery_state(
        self,
        own_hero,
        enemy_hero,
        own_tower,
        enemy_tower,
        cakes,
        own_hp,
        hp_trend,
        enemy_near_own,
        no_wave_tower_dive,
        dist_own_tower,
        skill_ready_ratio,
    ):
        nearest_cake, nearest_cake_dist = self._nearest_cake(cakes, own_hero)
        cake_visible = 1.0 if nearest_cake else 0.0
        cake_on_own_side = 0.0
        if nearest_cake and own_tower and enemy_tower:
            cake_progress = self._lane_progress({"location": {"x": self._cake_pos(nearest_cake)[0], "z": self._cake_pos(nearest_cake)[1]}}, own_tower, enemy_tower)
            cake_on_own_side = 1.0 if cake_progress < 0.45 else 0.0
        enemy_dist = _dist(own_hero, enemy_hero)
        enemy_pressure = 1.0 if enemy_hero and enemy_dist <= DEFAULT_ATTACK_RANGE * 1.25 else 0.0

        hp_deficit = max(0.0, _hp_ratio(enemy_hero) - own_hp)
        level_deficit = max(0.0, _unit_value(enemy_hero, "level") - _unit_value(own_hero, "level"))
        money_deficit = max(0.0, _unit_value(enemy_hero, "money_cnt") - _unit_value(own_hero, "money_cnt"))
        relative_danger = _clip01(
            1.55 * hp_deficit
            + 0.10 * min(level_deficit, 3.0)
            + 0.18 * _ratio(money_deficit, 2500.0)
            + 0.18 * enemy_pressure
            + (0.08 if hp_trend < -0.004 else 0.0)
        )
        safe_lane_hold = (
            1.0
            if own_hp < 0.42
            and enemy_near_own > 0
            and enemy_pressure == 0.0
            and enemy_hero
            and _hp_ratio(enemy_hero) <= own_hp + 0.12
            else 0.0
        )
        low_hp_execute_risk = (
            own_hp < HARD_RECALL_HP
            and _hp_ratio(enemy_hero) >= ENEMY_HEALTHY_HP
            and (enemy_pressure > 0.0 or relative_danger > 0.18)
        )
        enemy_weaker_window = (
            bool(enemy_hero)
            and _hp_ratio(enemy_hero) > 0.0
            and _hp_ratio(enemy_hero) + LOW_HP_RELATIVE_ADVANTAGE <= own_hp
        )
        core_skill_down_risk = (
            own_hp < 0.42
            and _hp_ratio(enemy_hero) >= max(0.45, own_hp + 0.06)
            and skill_ready_ratio < 0.45
            and enemy_pressure > 0.0
        )
        low_hp_hold_under_tower_window = (
            1.0
            if own_hp < 0.44
            and enemy_near_own > 0
            and enemy_hero
            and _hp_ratio(enemy_hero) <= min(0.56, own_hp + LOW_HP_HOLD_MAX_ENEMY_ADVANTAGE)
            and relative_danger < 0.52
            else 0.0
        )
        low_hp_grass_ambush_window = (
            1.0
            if own_hp < SHORT_RESET_EXIT_HP
            and bool(own_hero and own_hero.get("is_in_grass"))
            and enemy_hero
            and _hp_ratio(enemy_hero) > 0.0
            and _hp_ratio(enemy_hero) <= LOW_HP_GRASS_RESET_ENEMY_HP
            and enemy_dist <= DEFAULT_ATTACK_RANGE + LOW_HP_GRASS_AMBUSH_ENEMY_RANGE_PAD
            and not low_hp_execute_risk
            and not no_wave_tower_dive
            else 0.0
        )
        low_hp_commit_trade_window = (
            1.0
            if own_hp >= TOWER_CRASH_KILL_MIN_OWN_HP
            and own_hp < SHORT_RESET_EXIT_HP
            and enemy_hero
            and _hp_ratio(enemy_hero) > 0.0
            and (_hp_ratio(enemy_hero) <= LOW_HP_TRADE_ENEMY_HP or enemy_weaker_window)
            and enemy_dist <= DEFAULT_ATTACK_RANGE + 2200.0
            and not no_wave_tower_dive
            else 0.0
        )
        recovery_trigger = (
            (low_hp_execute_risk and low_hp_commit_trade_window == 0.0 and low_hp_grass_ambush_window == 0.0)
            or (core_skill_down_risk and low_hp_commit_trade_window == 0.0 and not enemy_weaker_window)
            or (
                own_hp < 0.38
                and relative_danger > 0.22
                and low_hp_commit_trade_window == 0.0
                and not enemy_weaker_window
            )
        )
        if recovery_trigger:
            self.recovery_commit_active = True
        elif own_hp >= SHORT_RESET_EXIT_HP or not own_hero:
            self.recovery_commit_active = False
        recovery_commit = 1.0 if self.recovery_commit_active and own_hp < SHORT_RESET_EXIT_HP else 0.0
        safe_lane_hold = 0.0 if recovery_commit or low_hp_commit_trade_window > 0.0 else safe_lane_hold

        moved_toward_own_tower = (
            self.last_own_tower_dist is not None and dist_own_tower < self.last_own_tower_dist - 100.0
        )
        moved_toward_cake = (
            self.last_cake_dist is not None and nearest_cake_dist < self.last_cake_dist - 100.0
        )
        moving_to_recovery = moved_toward_own_tower or moved_toward_cake
        if recovery_commit and hp_trend <= 0.002 and not moving_to_recovery:
            self.low_hp_stall_steps = min(self.low_hp_stall_steps + 1, 12)
        elif own_hp >= SHORT_RESET_EXIT_HP or hp_trend > 0.015 or moving_to_recovery:
            self.low_hp_stall_steps = max(0, self.low_hp_stall_steps - 2)
        else:
            self.low_hp_stall_steps = max(0, self.low_hp_stall_steps - 1)

        self.last_own_tower_dist = dist_own_tower if dist_own_tower < 99999.0 else self.last_own_tower_dist
        self.last_cake_dist = nearest_cake_dist if nearest_cake_dist < 99999.0 else self.last_cake_dist

        low_hp_cake_window = (
            1.0
            if own_hp < 0.40
            and cake_visible
            and cake_on_own_side
            and nearest_cake_dist <= 18000.0
            and recovery_commit == 0.0
            and safe_lane_hold == 0.0
            and not enemy_weaker_window
            else 0.0
        )
        near_own_tower = _dist(own_hero, own_tower) <= 13000.0
        safe_position_for_recall = near_own_tower or bool(own_hero and own_hero.get("is_in_grass"))
        enemy_not_interrupting_recall = enemy_pressure == 0.0 and enemy_near_own == 0
        safe_recall_window = (
            1.0
            if (
                safe_position_for_recall
                and enemy_not_interrupting_recall
                and not no_wave_tower_dive
                and safe_lane_hold == 0.0
                and low_hp_hold_under_tower_window == 0.0
                and low_hp_grass_ambush_window == 0.0
                and low_hp_commit_trade_window == 0.0
                and not enemy_weaker_window
                and (
                    low_hp_execute_risk
                    or core_skill_down_risk
                    or recovery_commit
                    or (own_hp < 0.40 and self.low_hp_stall_steps >= SHORT_RESET_STALL_LIMIT)
                )
            )
            else 0.0
        )
        return {
            "cake_visible": cake_visible,
            "nearest_cake_dist": _ratio(nearest_cake_dist, 30000.0, default=1.0),
            "cake_on_own_side": cake_on_own_side,
            "low_hp_cake_window": low_hp_cake_window,
            "hp_recover_trend": _clip(hp_trend * 8.0),
            "enemy_pressure_nearby": enemy_pressure,
            "safe_recall_window": safe_recall_window,
            "recovery_commit": recovery_commit,
            "low_hp_execute_risk": 1.0 if low_hp_execute_risk else 0.0,
            "core_skill_down_risk": 1.0 if core_skill_down_risk else 0.0,
            "low_hp_hold_under_tower_window": low_hp_hold_under_tower_window,
            "low_hp_grass_ambush_window": low_hp_grass_ambush_window,
            "low_hp_commit_trade_window": low_hp_commit_trade_window,
            "recovery_need": _clip01((SHORT_RESET_EXIT_HP - own_hp) / SHORT_RESET_EXIT_HP),
        }

    def _tower_state(
        self,
        own_hero,
        enemy_hero,
        enemy_tower,
        own_hp,
        attack_range,
        ally_near_enemy,
        no_wave_tower_dive,
        enemy_tower_hp,
        own_is_luban,
        own_is_direnjie,
    ):
        dist_enemy_tower = _dist(own_hero, enemy_tower)
        enemy_tower_range = _safe_float(enemy_tower.get("attack_range") if enemy_tower else 8800.0, 8800.0)
        pressure_outer_range = max(attack_range + TOWER_ATTACK_RANGE_PAD, enemy_tower_range + TOWER_EDGE_OUTER_PAD)
        tower_in_normal_attack_range = 1.0 if enemy_tower and dist_enemy_tower <= pressure_outer_range else 0.0
        enemy_tower_can_hit_self = 1.0 if enemy_tower and dist_enemy_tower <= enemy_tower_range else 0.0
        ally_wave_tanking_enemy_tower = 1.0 if ally_near_enemy > 0 else 0.0
        enemy_close = bool(enemy_hero and _dist(own_hero, enemy_hero) <= attack_range * 1.10)
        enemy_under_tower = (
            1.0
            if enemy_tower
            and enemy_hero
            and _hp_ratio(enemy_hero) > 0.0
            and _dist(enemy_hero, enemy_tower) <= enemy_tower_range + TOWER_TURTLE_RANGE_PAD
            else 0.0
        )
        tower_edge_band = (
            1.0
            if enemy_tower
            and enemy_tower_range + TOWER_EDGE_SAFE_MARGIN <= dist_enemy_tower <= pressure_outer_range
            else 0.0
        )
        tower_edge_poke_window = (
            1.0
            if enemy_under_tower
            and max(own_is_luban, own_is_direnjie) > 0.0
            and own_hp >= TOWER_PRESSURE_MIN_HP
            and tower_edge_band
            and ally_wave_tanking_enemy_tower
            else 0.0
        )
        tower_wave_lost_reset = (
            1.0
            if enemy_under_tower
            and not ally_wave_tanking_enemy_tower
            and dist_enemy_tower <= pressure_outer_range + 2500.0
            else 0.0
        )
        tower_edge_overstep = (
            1.0
            if enemy_under_tower
            and enemy_tower_can_hit_self
            and not ally_wave_tanking_enemy_tower
            else 0.0
        )
        if tower_edge_poke_window:
            self.tower_edge_hold_steps = min(self.tower_edge_hold_steps + 1, 16)
        else:
            self.tower_edge_hold_steps = max(0, self.tower_edge_hold_steps - 2)
        if tower_edge_overstep:
            self.tower_edge_overstep_steps = min(self.tower_edge_overstep_steps + 1, 16)
        else:
            self.tower_edge_overstep_steps = max(0, self.tower_edge_overstep_steps - 2)
        tower_attack_safe_window = (
            1.0
            if tower_in_normal_attack_range
            and own_hp >= TOWER_PRESSURE_MIN_HP
            and not no_wave_tower_dive
            and (ally_wave_tanking_enemy_tower or tower_edge_poke_window)
            and (not enemy_close or tower_edge_poke_window)
            else 0.0
        )
        tower_finish_window = 1.0 if tower_attack_safe_window and 0.0 < enemy_tower_hp <= 0.25 else 0.0
        return {
            "tower_in_normal_attack_range": tower_in_normal_attack_range,
            "enemy_tower_can_hit_self": enemy_tower_can_hit_self,
            "ally_wave_tanking_enemy_tower": ally_wave_tanking_enemy_tower,
            "enemy_under_tower": enemy_under_tower,
            "tower_edge_band": tower_edge_band,
            "tower_edge_poke_window": tower_edge_poke_window,
            "tower_wave_lost_reset": tower_wave_lost_reset,
            "tower_edge_overstep": tower_edge_overstep,
            "tower_edge_hold": _ratio(self.tower_edge_hold_steps, 16.0),
            "tower_attack_safe_window": tower_attack_safe_window,
            "tower_finish_window": tower_finish_window,
        }
