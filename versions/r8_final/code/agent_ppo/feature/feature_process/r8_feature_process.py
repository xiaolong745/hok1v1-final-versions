#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
R8 lane-centric feature builder.

The contract is intentionally explicit:
dense scalars + one-hot bins + semantic unit slots + lane grid + temporal state.
"""

import math

from agent_ppo.conf.conf import DimConfig
from agent_ppo.feature.feature_process.tactical_process import TacticalProcess


ACTOR_TYPE_HERO = 0
ACTOR_TYPE_MONSTER = 1
ACTOR_TYPE_ORGAN = 2
SUB_TYPE_LANE_SOLDIER = 11
SUB_TYPE_TOWER = 21
SUB_TYPE_CRYSTAL = 24
RIVER_SPIRIT_CONFIG_ID = 6827
LANE_SOLDIER_CONFIG_IDS = {6800, 6801, 6802, 6803, 6804, 6805}
TOWER_CONFIG_IDS = {1111, 1112}
CRYSTAL_CONFIG_IDS = {1113, 1114}
LUBAN_HERO_ID = 112
DIRENJIE_HERO_ID = 133
MAX_DIST = 60000.0
MAX_MONEY = 14000.0
MAX_EXP = 2200.0
MAX_LEVEL = 15.0
MAX_ATTACK = 3000.0
MAX_DEFENSE = 3000.0
MAX_SPEED = 14000.0
MAX_HURT = 30000.0
LANE_LATERAL_SCALE = 14000.0
DEFAULT_ATTACK_RANGE = 8200.0
TOWER_ATTACK_RANGE_PAD = 1800.0
TOWER_EDGE_OUTER_PAD = 1800.0
TOWER_HIT_RANGE_PAD = 650.0
TOWER_STABLE_WAVE_MIN_COUNT = 2
TOWER_PRESSURE_MIN_HP = 0.42
LOW_HP_OBJECTIVE_MIN_HP = 0.18
TOWER_HP_SAFE_EDGE = 0.70
TOWER_HP_WARN_EDGE = 0.45
TOWER_HP_CRITICAL_EDGE = 0.28


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


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clip(value, low=-1.0, high=1.0):
    return max(low, min(high, _safe_float(value)))


def _clip01(value):
    return _clip(value, 0.0, 1.0)


def _ratio(value, max_value, default=0.0):
    max_value = _safe_float(max_value)
    if max_value <= 0.0:
        return default
    return _clip01(_safe_float(value) / max_value)


def _signed_ratio(value, scale):
    scale = _safe_float(scale)
    if scale <= 0.0:
        return 0.0
    return _clip(_safe_float(value) / scale)


def _pos(unit):
    location = unit.get("location", {}) if unit else {}
    return _safe_float(location.get("x")), _safe_float(location.get("z"))


def _dist(a, b):
    if not a or not b:
        return MAX_DIST
    ax, az = _pos(a)
    bx, bz = _pos(b)
    return math.hypot(ax - bx, az - bz)


def _dist_pos(a_pos, b_pos):
    ax, az = a_pos
    bx, bz = b_pos
    return math.hypot(ax - bx, az - bz)


def _hp_ratio(unit):
    if not unit:
        return 0.0
    return _ratio(unit.get("hp"), unit.get("max_hp"))


def _ep_ratio(unit):
    if not unit:
        return 1.0, 0.0
    max_ep = _safe_float(unit.get("max_ep"), 0.0)
    if max_ep <= 0.0:
        return 1.0, 0.0
    return _ratio(unit.get("ep"), max_ep, default=1.0), 1.0


def _skill_slots(hero):
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


def _skill_ready(hero, slot_id):
    for slot in _skill_slots(hero):
        slot_type = _safe_int(slot.get("slot_type", slot.get("slot")), -1)
        if slot_type == slot_id:
            return 1.0 if slot.get("usable") else 0.0
    return 0.0


def _skill_cd_ratio(hero, slot_id):
    for slot in _skill_slots(hero):
        slot_type = _safe_int(slot.get("slot_type", slot.get("slot")), -1)
        if slot_type == slot_id:
            return _ratio(slot.get("cooldown"), slot.get("max_cooldown"), default=0.0)
    return 0.0


def _append_one_hot(out, index, size, scale=1.0):
    index = max(0, min(size - 1, int(index)))
    for i in range(size):
        out.append(scale if i == index else 0.0)


def _bin_index(value, edges):
    value = _safe_float(value)
    for idx, edge in enumerate(edges):
        if value <= edge:
            return idx
    return len(edges)


def _append_bin(out, value, edges):
    _append_one_hot(out, _bin_index(value, edges), len(edges) + 1)


def _tower_hp_stage(hp_ratio):
    hp = _safe_float(hp_ratio)
    if hp <= 0.0:
        return 0
    if hp <= TOWER_HP_CRITICAL_EDGE:
        return 1
    if hp <= TOWER_HP_WARN_EDGE:
        return 2
    if hp <= TOWER_HP_SAFE_EDGE:
        return 3
    return 4


class R8FeatureProcess:
    def __init__(self, camp):
        self.reset(camp)

    def reset(self, camp):
        self.camp = camp
        self.camp_key = "PLAYERCAMP_2" if camp == "PLAYERCAMP_2" or camp == 2 or str(camp) == "2" else "PLAYERCAMP_1"
        self.last = None
        self.enemy_missing_steps = 0
        self.post_kill_steps = 0
        self.post_kill_wait_enemy_seen = False
        self.ep_hold_steps = 0
        self.tactical_process = TacticalProcess(camp)

    def process_feature(self, observation):
        frame_state = observation["frame_state"]
        player_id = observation.get("player_id")
        own, enemy = self._find_heroes(frame_state, player_id)
        ally_minions, enemy_minions = self._find_minions(frame_state)
        own_tower, enemy_tower = self._find_towers(frame_state)
        monsters = self._find_monsters(frame_state)
        cakes = frame_state.get("cakes", []) or []

        lane = self._lane_context(own_tower, enemy_tower)
        tactical = self.tactical_process.process_vec_tactical(observation)
        state = self._semantic_state(
            frame_state,
            own,
            enemy,
            own_tower,
            enemy_tower,
            ally_minions,
            enemy_minions,
            monsters,
            cakes,
            lane,
            tactical,
        )

        dense = self._build_dense(state)
        discrete = self._build_discrete(state)
        units = self._build_units(state)
        lane_grid = self._build_lane_grid(state)
        temporal = self._build_temporal(state)

        feature = dense + discrete + units + lane_grid + temporal
        expected = int(DimConfig.DIM_OF_FEATURE[0])
        if len(feature) != expected:
            raise ValueError(f"R8 feature dim mismatch: {len(feature)} != {expected}")
        self.last = state
        return [float(_clip(value)) for value in feature]

    def _find_heroes(self, frame_state, player_id):
        own = None
        enemy = None
        for hero in frame_state.get("hero_states", []) or []:
            if hero.get("player_id") == player_id or hero.get("runtime_id") == player_id:
                own = hero
            elif hero.get("camp") != self.camp:
                enemy = hero
        if own is None:
            for hero in frame_state.get("hero_states", []) or []:
                if hero.get("camp") == self.camp:
                    own = hero
                    break
        return own, enemy

    def _find_minions(self, frame_state):
        ally, enemy = [], []
        for npc in frame_state.get("npc_states", []) or []:
            config_id = _safe_int(npc.get("config_id"), 0)
            if npc.get("sub_type") != SUB_TYPE_LANE_SOLDIER and config_id not in LANE_SOLDIER_CONFIG_IDS:
                continue
            if npc.get("camp") == self.camp:
                ally.append(npc)
            else:
                enemy.append(npc)
        key = lambda unit: _safe_int(unit.get("runtime_id"), 0)
        return sorted(ally, key=key), sorted(enemy, key=key)

    def _find_towers(self, frame_state):
        own_tower, enemy_tower = None, None
        own_fallback, enemy_fallback = None, None
        for npc in frame_state.get("npc_states", []) or []:
            config_id = _safe_int(npc.get("config_id"), 0)
            is_tower = npc.get("sub_type") == SUB_TYPE_TOWER or config_id in TOWER_CONFIG_IDS
            is_crystal = npc.get("sub_type") == SUB_TYPE_CRYSTAL or config_id in CRYSTAL_CONFIG_IDS
            if not is_tower and not is_crystal:
                continue
            if npc.get("camp") == self.camp:
                if is_tower and own_tower is None:
                    own_tower = npc
                elif own_fallback is None:
                    own_fallback = npc
            else:
                if is_tower and enemy_tower is None:
                    enemy_tower = npc
                elif enemy_fallback is None:
                    enemy_fallback = npc
        return own_tower or own_fallback, enemy_tower or enemy_fallback

    def _find_monsters(self, frame_state):
        monsters = []
        for npc in frame_state.get("npc_states", []) or []:
            config_id = _safe_int(npc.get("config_id"), 0)
            if npc.get("actor_type") == ACTOR_TYPE_MONSTER or config_id == RIVER_SPIRIT_CONFIG_ID:
                monsters.append(npc)
        return monsters

    def _lane_context(self, own_tower, enemy_tower):
        if own_tower and enemy_tower:
            ox, oz = _pos(own_tower)
            ex, ez = _pos(enemy_tower)
        else:
            cakes = CAKE_LOCATIONS_BY_CAMP[self.camp_key]
            ox, oz = cakes["main"]
            ex, ez = cakes["enemy"]
        vx, vz = ex - ox, ez - oz
        norm = math.hypot(vx, vz)
        if norm <= 1.0:
            norm = 1.0
            vx, vz = 1.0, 0.0
        return {
            "origin": (ox, oz),
            "enemy": (ex, ez),
            "axis": (vx / norm, vz / norm),
            "span": norm,
        }

    def _lane_coords(self, unit, lane):
        if not unit:
            return 0.0, 0.0
        ux, uz = _pos(unit)
        ox, oz = lane["origin"]
        ax, az = lane["axis"]
        dx, dz = ux - ox, uz - oz
        progress = (dx * ax + dz * az) / max(lane["span"], 1.0)
        lateral = (dx * (-az) + dz * ax) / LANE_LATERAL_SCALE
        return _clip(progress, -0.25, 1.25), _clip(lateral)

    def _semantic_state(self, frame_state, own, enemy, own_tower, enemy_tower, ally_minions, enemy_minions, monsters, cakes, lane, tactical):
        own_hp = _hp_ratio(own)
        enemy_hp = _hp_ratio(enemy)
        own_ep, own_has_ep = _ep_ratio(own)
        enemy_ep, enemy_has_ep = _ep_ratio(enemy)
        if own_has_ep and own_ep <= 0.34:
            self.ep_hold_steps = min(self.ep_hold_steps + 2, 16)
        elif own_has_ep and own_ep >= 0.50:
            self.ep_hold_steps = max(self.ep_hold_steps - 3, 0)
        else:
            self.ep_hold_steps = max(self.ep_hold_steps - 1, 0)

        enemy_alive = bool(enemy and _safe_float(enemy.get("hp")) > 0.0)
        if not enemy_alive:
            self.post_kill_steps = max(self.post_kill_steps, 56)
            self.post_kill_wait_enemy_seen = True
            self.enemy_missing_steps = min(self.enemy_missing_steps + 1, 64)
        elif self.post_kill_wait_enemy_seen:
            self.post_kill_steps = 0
            self.post_kill_wait_enemy_seen = False
            self.enemy_missing_steps = 0
        else:
            self.enemy_missing_steps = 0
        post_kill_active = 1.0 if self.post_kill_steps > 0 or self.post_kill_wait_enemy_seen else 0.0
        if self.post_kill_steps > 0:
            self.post_kill_steps -= 1

        own_u, own_v = self._lane_coords(own, lane)
        enemy_u, enemy_v = self._lane_coords(enemy, lane)
        ally_us = [self._lane_coords(unit, lane)[0] for unit in ally_minions]
        enemy_us = [self._lane_coords(unit, lane)[0] for unit in enemy_minions]
        ally_front = max(ally_us) if ally_us else own_u
        enemy_front = min(enemy_us) if enemy_us else enemy_u
        wave_gap = ally_front - enemy_front
        ally_near_enemy_tower = sum(1 for unit in ally_minions if self._lane_coords(unit, lane)[0] >= 0.82)
        enemy_near_own_tower = sum(1 for unit in enemy_minions if self._lane_coords(unit, lane)[0] <= 0.22)
        enemy_between = any(min(own_u, enemy_u) <= u <= max(own_u, enemy_u) for u in enemy_us)
        own_tower_hp = _hp_ratio(own_tower)
        enemy_tower_hp = _hp_ratio(enemy_tower)
        own_tower_hp_stage = _tower_hp_stage(own_tower_hp)
        enemy_tower_hp_stage = _tower_hp_stage(enemy_tower_hp)
        own_tower_warn = 1.0 if 0.0 < own_tower_hp <= TOWER_HP_WARN_EDGE else 0.0
        own_tower_critical = 1.0 if 0.0 < own_tower_hp <= TOWER_HP_CRITICAL_EDGE else 0.0
        enemy_tower_finishable = 1.0 if 0.0 < enemy_tower_hp <= TOWER_HP_CRITICAL_EDGE else 0.0
        tower_hp_advantage = _clip(enemy_tower_hp - own_tower_hp)
        dist_enemy = _dist(own, enemy)
        dist_enemy_tower = _dist(own, enemy_tower)
        dist_own_tower = _dist(own, own_tower)
        attack_range = max(_safe_float(own.get("attack_range") if own else 0.0, DEFAULT_ATTACK_RANGE), DEFAULT_ATTACK_RANGE)
        tower_range = _safe_float(enemy_tower.get("attack_range") if enemy_tower else 0.0, 9000.0)
        if tower_range <= 0:
            tower_range = 9000.0
        tower_hit_range = max(attack_range + TOWER_HIT_RANGE_PAD, tower_range * 0.72)
        tower_pressure_range = max(attack_range + TOWER_ATTACK_RANGE_PAD, tower_range + TOWER_EDGE_OUTER_PAD)
        in_enemy_tower = 1.0 if enemy_tower and dist_enemy_tower <= tower_range + 600.0 else 0.0
        enemy_tower_can_hit_self = 1.0 if enemy_tower and dist_enemy_tower <= tower_range else 0.0
        hero_can_hit_tower = 1.0 if enemy_tower and dist_enemy_tower <= tower_hit_range else 0.0
        hero_near_tower_stage = 1.0 if enemy_tower and dist_enemy_tower <= tower_pressure_range + 2600.0 else 0.0
        wave_tanking = 1.0 if ally_near_enemy_tower > 0 else 0.0
        stable_wave_tanking = 1.0 if ally_near_enemy_tower >= TOWER_STABLE_WAVE_MIN_COUNT else 0.0
        no_wave_dive = 1.0 if enemy_tower_can_hit_self > 0.0 and wave_tanking <= 0.0 else 0.0
        tower_defense_crisis = 1.0 if own_tower_critical > 0.0 and enemy_near_own_tower > 0 else 0.0
        tower_trade_deficit = 1.0 if own_tower_hp + 0.18 < enemy_tower_hp else 0.0
        force_lane_out = 1.0 if tower_defense_crisis > 0.0 or tower_trade_deficit > 0.0 else 0.0
        tower_defend_clear = 1.0 if enemy_near_own_tower > 0 or (own_tower_warn > 0.0 and enemy_between) else 0.0
        tower_out_pressure = 1.0 if force_lane_out > 0.0 and tower_defend_clear <= 0.0 else 0.0
        own_id = _safe_int(own.get("config_id") if own else 0, 0)
        enemy_id = _safe_int(enemy.get("config_id") if enemy else 0, 0)
        own_is_luban = 1.0 if own_id == LUBAN_HERO_ID else 0.0
        own_in_grass = 1.0 if own and own.get("is_in_grass") else 0.0
        enemy_in_grass = 1.0 if enemy and enemy.get("is_in_grass") else 0.0
        enemy_missing = 1.0 if not enemy_alive else 0.0
        enemy_missing_ratio = _ratio(self.enemy_missing_steps, 64.0)
        grass_vision_probe = 1.0 if enemy_missing_ratio > 0.08 and own_hp >= 0.38 and no_wave_dive <= 0.0 else 0.0
        grass_ambush_window = 1.0 if own_in_grass > 0.0 and enemy_alive and dist_enemy <= attack_range * 1.25 else 0.0
        objective_hp_ok = 1.0 if own_hp >= TOWER_PRESSURE_MIN_HP or (stable_wave_tanking > 0.0 and own_hp >= LOW_HP_OBJECTIVE_MIN_HP) else 0.0
        push_lane_phase = 1.0 if enemy_tower and ally_front >= 0.52 and wave_tanking <= 0.0 else 0.0
        if tower_out_pressure > 0.0 and enemy_tower:
            push_lane_phase = max(push_lane_phase, 0.70)
        tower_entry_phase = (
            1.0
            if enemy_tower
            and wave_tanking > 0.0
            and stable_wave_tanking <= 0.0
            and hero_near_tower_stage > 0.0
            and objective_hp_ok > 0.0
            and no_wave_dive <= 0.0
            else 0.0
        )
        tower_hit_phase = (
            1.0
            if enemy_tower
            and stable_wave_tanking > 0.0
            and hero_can_hit_tower > 0.0
            and objective_hp_ok > 0.0
            and no_wave_dive <= 0.0
            else 0.0
        )
        tower_safe = tower_hit_phase
        tower_stage = 1.0 if max(push_lane_phase, tower_entry_phase, stable_wave_tanking * (1.0 - tower_hit_phase)) > 0.0 else 0.0
        tower_distance_band = 0
        if enemy_tower:
            if enemy_tower_can_hit_self > 0.0:
                tower_distance_band = 4
            elif hero_can_hit_tower > 0.0:
                tower_distance_band = 3
            elif dist_enemy_tower <= tower_pressure_range:
                tower_distance_band = 2
            elif hero_near_tower_stage > 0.0:
                tower_distance_band = 1
        tower_phase_index = 0
        if no_wave_dive > 0.0:
            tower_phase_index = 4
        elif tower_hit_phase > 0.0:
            tower_phase_index = 3
        elif tower_entry_phase > 0.0:
            tower_phase_index = 2
        elif push_lane_phase > 0.0:
            tower_phase_index = 1
        tactical = tactical or []
        tactical = [float(_clip(v, 0.0, 1.0)) for v in tactical]
        tactical_map = self._tactical_state_map(tactical)

        own_defend = max(tower_defend_clear, tactical_map["own_tower_defend"] * (1.0 - 0.75 * tower_out_pressure))
        kill_window = max(1.0 if enemy_alive and enemy_hp <= 0.30 and own_hp >= 0.20 else 0.0, tactical_map["kill_window"])
        should_clear = max(
            1.0 if own_defend or (enemy_between and wave_tanking <= 0.0) or (post_kill_active and ally_near_enemy_tower <= 0) else 0.0,
            tactical_map["should_clear_wave"],
        )
        can_fight = max(
            1.0 if enemy_alive and dist_enemy <= max(_safe_float(own.get("attack_range") if own else 0.0, 8200.0), 8200.0) * 1.35 else 0.0,
            tactical_map["can_fight"],
        )
        if tower_out_pressure > 0.0:
            should_clear *= 0.45
            tower_stage = max(tower_stage, 0.70)
            can_fight = max(can_fight, 0.86 if own_is_luban > 0.0 and enemy_alive else 0.56)
        should_recover = max(
            1.0 if ((own_hp <= 0.30 or (own_has_ep and own_ep <= 0.22)) and not post_kill_active and not tower_safe) or no_wave_dive > 0.0 else 0.0,
            tactical_map["should_recover"],
        )
        if tower_out_pressure > 0.0 and no_wave_dive <= 0.0:
            should_recover *= 0.45
        tactical_map["tower_out_pressure"] = tower_out_pressure
        tactical_map["tower_defend_clear"] = tower_defend_clear
        tactical_map["own_is_luban"] = own_is_luban

        tower_safe = max(tower_hit_phase, tactical_map["must_hit_tower_now"])
        tower_stage = max(tower_stage, tactical_map["can_pressure_tower"])
        if enemy_tower_finishable > 0.0 and stable_wave_tanking > 0.0 and hero_can_hit_tower > 0.0 and no_wave_dive <= 0.0:
            tower_safe = 1.0
            tower_hit_phase = 1.0
        if own_tower_warn > 0.0 and enemy_near_own_tower > 0 and wave_tanking <= 0.0:
            tower_stage = max(tower_stage, 0.55)
        push_lane_phase = max(push_lane_phase, tactical_map["post_kill_clear_wave"])
        tower_entry_phase = max(tower_entry_phase, tactical_map["post_kill_tower_stage"])
        tower_hit_phase = max(tower_hit_phase, tactical_map["post_kill_tower_plan"])
        main_task = self._main_task(
            post_kill_active,
            tower_safe,
            tower_stage,
            own_defend,
            kill_window,
            can_fight,
            should_clear,
            should_recover,
            monsters,
            no_wave_dive,
            tactical_map,
        )

        last = self.last or {}
        deltas = {
            "own_hp": own_hp - _safe_float(last.get("own_hp"), own_hp),
            "enemy_hp": enemy_hp - _safe_float(last.get("enemy_hp"), enemy_hp),
            "own_ep": own_ep - _safe_float(last.get("own_ep"), own_ep),
            "enemy_tower_hp": _hp_ratio(enemy_tower) - _safe_float(last.get("enemy_tower_hp"), _hp_ratio(enemy_tower)),
            "own_tower_hp": _hp_ratio(own_tower) - _safe_float(last.get("own_tower_hp"), _hp_ratio(own_tower)),
            "money": _safe_float(own.get("money") if own else 0.0) - _safe_float(last.get("money_raw"), _safe_float(own.get("money") if own else 0.0)),
            "exp": _safe_float(own.get("exp") if own else 0.0) - _safe_float(last.get("exp_raw"), _safe_float(own.get("exp") if own else 0.0)),
        }

        return {
            "frame_no": _safe_float(frame_state.get("frame_no"), 0.0),
            "own": own,
            "enemy": enemy,
            "own_tower": own_tower,
            "enemy_tower": enemy_tower,
            "ally_minions": ally_minions,
            "enemy_minions": enemy_minions,
            "monsters": monsters,
            "cakes": cakes,
            "lane": lane,
            "own_hp": own_hp,
            "enemy_hp": enemy_hp,
            "own_ep": own_ep,
            "enemy_ep": enemy_ep,
            "own_has_ep": own_has_ep,
            "enemy_has_ep": enemy_has_ep,
            "enemy_alive": 1.0 if enemy_alive else 0.0,
            "own_u": own_u,
            "own_v": own_v,
            "enemy_u": enemy_u,
            "enemy_v": enemy_v,
            "ally_front": ally_front,
            "enemy_front": enemy_front,
            "wave_gap": wave_gap,
            "ally_near_enemy_tower": ally_near_enemy_tower,
            "enemy_near_own_tower": enemy_near_own_tower,
            "enemy_between": 1.0 if enemy_between else 0.0,
            "dist_enemy": dist_enemy,
            "dist_enemy_tower": dist_enemy_tower,
            "dist_own_tower": dist_own_tower,
            "in_enemy_tower": in_enemy_tower,
            "enemy_tower_can_hit_self": enemy_tower_can_hit_self,
            "hero_can_hit_tower": hero_can_hit_tower,
            "hero_near_tower_stage": hero_near_tower_stage,
            "wave_tanking": wave_tanking,
            "stable_wave_tanking": stable_wave_tanking,
            "no_wave_dive": no_wave_dive,
            "objective_hp_ok": objective_hp_ok,
            "push_lane_phase": push_lane_phase,
            "tower_entry_phase": tower_entry_phase,
            "tower_hit_phase": tower_hit_phase,
            "tower_distance_band": tower_distance_band,
            "tower_phase_index": tower_phase_index,
            "tower_safe": tower_safe,
            "tower_stage": tower_stage,
            "own_defend": own_defend,
            "kill_window": kill_window,
            "should_clear": should_clear,
            "can_fight": can_fight,
            "should_recover": should_recover,
            "post_kill_active": post_kill_active,
            "post_kill_ratio": _ratio(self.post_kill_steps, 56.0),
            "enemy_missing_ratio": _ratio(self.enemy_missing_steps, 64.0),
            "ep_hold": 1.0 if self.ep_hold_steps > 0 else 0.0,
            "ep_critical": 1.0 if own_has_ep and own_ep <= 0.22 else 0.0,
            "ep_resume": 1.0 if own_has_ep and own_ep >= 0.50 and self.ep_hold_steps == 0 else 0.0,
            "main_task": main_task,
            "tactical": tactical_map,
            "deltas": deltas,
            "money_raw": _safe_float(own.get("money") if own else 0.0),
            "exp_raw": _safe_float(own.get("exp") if own else 0.0),
            "enemy_tower_hp": enemy_tower_hp,
            "own_tower_hp": own_tower_hp,
            "own_tower_hp_stage": own_tower_hp_stage,
            "enemy_tower_hp_stage": enemy_tower_hp_stage,
            "own_tower_warn": own_tower_warn,
            "own_tower_critical": own_tower_critical,
            "enemy_tower_finishable": enemy_tower_finishable,
            "tower_hp_advantage": tower_hp_advantage,
            "tower_defense_crisis": tower_defense_crisis,
            "tower_trade_deficit": tower_trade_deficit,
            "force_lane_out": force_lane_out,
            "tower_defend_clear": tower_defend_clear,
            "tower_out_pressure": tower_out_pressure,
            "own_is_luban": own_is_luban,
            "own_in_grass": own_in_grass,
            "enemy_in_grass": enemy_in_grass,
            "enemy_missing": enemy_missing,
            "grass_vision_probe": grass_vision_probe,
            "grass_ambush_window": grass_ambush_window,
        }

    def _main_task(self, post_kill, tower_safe, tower_stage, own_defend, kill_window, can_fight, should_clear, should_recover, monsters, no_wave_dive=0.0, tactical=None):
        tactical = tactical or {}
        hero_contest = _clip01(tactical.get("hero_contest", can_fight))
        commit_kill = _clip01(tactical.get("commit_kill", kill_window))
        short_reset = _clip01(tactical.get("short_reset", should_recover))
        tower_out_pressure = _clip01(tactical.get("tower_out_pressure", 0.0))
        own_is_luban = _clip01(tactical.get("own_is_luban", 0.0))
        if tower_out_pressure > 0.0:
            tower_stage = max(tower_stage, 0.70 * tower_out_pressure)
            hero_contest = max(hero_contest, 0.82 * tower_out_pressure * own_is_luban)
            should_clear *= 1.0 - 0.65 * tower_out_pressure
            should_recover *= 1.0 - 0.45 * tower_out_pressure
        post_kill_clear_wave = _clip01(tactical.get("post_kill_clear_wave", 0.0))
        post_kill_tower_stage = _clip01(tactical.get("post_kill_tower_stage", 0.0))
        post_kill_tower_plan = _clip01(tactical.get("post_kill_tower_plan", 0.0))
        post_kill = post_kill * (1.0 - no_wave_dive)
        post_kill_push = post_kill * (1.0 - max(post_kill_tower_plan, post_kill_tower_stage))
        scores = [
            max(post_kill_push, post_kill_clear_wave) * (1.0 - max(post_kill_tower_plan, post_kill_tower_stage)),
            max(tower_safe, post_kill_tower_plan) * (1.0 - post_kill_clear_wave),
            max(tower_stage, post_kill_tower_stage) * (1.0 - max(tower_safe, post_kill_tower_plan)),
            own_defend,
            max(kill_window, commit_kill),
            hero_contest,
            should_clear,
            max(should_recover, short_reset),
            0.35 if monsters else 0.0,
            0.1,
        ]
        order = [1, 2, 0, 3, 4, 7, 5, 6, 8, 9]
        best = max(order, key=lambda idx: (scores[idx], -order.index(idx)))
        return {"index": best, "scores": [_clip01(score) for score in scores]}

    def _tactical_state_map(self, tactical):
        value = lambda idx: _clip01(tactical[idx]) if idx < len(tactical) else 0.0
        out = {
            "should_clear_wave": value(36),
            "can_pressure_tower": value(39),
            "must_hit_tower_now": value(40),
            "own_tower_defend": max(value(36), value(41)),  # clear_wave or tower_dive_danger
            "can_fight": value(42),
            "should_recover": max(value(47), value(48)),  # should_recall or low_hp_stall
            "kill_window": value(44),
            "hero_contest": max(value(42), value(51), value(66), value(69), value(70), value(188), value(189)),
            "commit_kill": max(value(44), value(64), value(158), value(188), value(189)),
            "short_reset": max(value(46), value(47), value(48), value(178), value(181), value(195), value(197)),
            "post_kill_active": value(172),
            "post_kill_clear_wave": value(172) * value(36),
            "post_kill_tower_stage": value(172) * max(value(39), value(154)),
            "post_kill_tower_plan": value(172) * max(value(40), value(154), value(160)),
            "combo_start": value(188),
            "combo_chain": value(189),
            "combo_abort": value(190),
            "direnjie_skill1_poke": value(61),
            "direnjie_skill2_escape": value(62),
            "direnjie_ult_kill": value(64),
            "direnjie_followup": value(66),
            "enemy_direnjie_threat": value(67),
            "luban_vs_direnjie": value(69),
            "direnjie_vs_luban": value(70),
            "luban_safe_passive": value(73),
            "direnjie_punish_luban": value(74),
            "luban_passive_tower": value(154),
            "luban_passive_hero": value(155),
            "luban_followup": value(156),
            "luban_skill2_save": value(157),
            "luban_skill2_execute": value(158),
            "luban_skill_waste": value(159),
            "luban_tower_ignore_enemy": value(160),
            "enemy_missing": value(185),
            "enemy_missing_ratio": value(186),
            "enemy_close_extended": value(187),
            "direnjie_skill2_combo": value(191),
            "luban_skill2_combo": value(192),
            "ep_ratio": value(193),
            "ep_conserve": value(194),
            "ep_hold_skills": value(195),
            "ep_resume_ready": value(196),
            "ep_critical": value(197),
            "ep_recover": value(198),
            # R8-c6: HP advantage and ratio level for chase/retreat decisions
            "hp_advantage": _clip(tactical[76]) if len(tactical) > 76 else 0.0,
            "hp_ratio_level": _clip(tactical[77]) if len(tactical) > 77 else 0.0,
            # Grass tactical features
            "in_grass": value(32),
            "purposeful_grass": value(53),
        }
        return out

    def _build_dense(self, state):
        own = state["own"]
        enemy = state["enemy"]
        own_tower = state["own_tower"]
        enemy_tower = state["enemy_tower"]
        d = state["deltas"]
        tactical = state["tactical"]
        values = [
            state["own_hp"], state["enemy_hp"], state["own_ep"], state["enemy_ep"],
            state["own_has_ep"], state["enemy_has_ep"], state["enemy_alive"],
            _ratio(own.get("level") if own else 0.0, MAX_LEVEL), _ratio(enemy.get("level") if enemy else 0.0, MAX_LEVEL),
            _ratio(own.get("money") if own else 0.0, MAX_MONEY), _ratio(enemy.get("money") if enemy else 0.0, MAX_MONEY),
            _ratio(own.get("exp") if own else 0.0, MAX_EXP), _ratio(enemy.get("exp") if enemy else 0.0, MAX_EXP),
            _ratio(own.get("attack_range") if own else 0.0, MAX_DIST), _ratio(enemy.get("attack_range") if enemy else 0.0, MAX_DIST),
            _ratio(own.get("phy_atk") if own else 0.0, MAX_ATTACK), _ratio(enemy.get("phy_atk") if enemy else 0.0, MAX_ATTACK),
            _ratio(own.get("phy_def") if own else 0.0, MAX_DEFENSE), _ratio(enemy.get("phy_def") if enemy else 0.0, MAX_DEFENSE),
            _ratio(own.get("mov_spd") if own else 0.0, MAX_SPEED), _ratio(enemy.get("mov_spd") if enemy else 0.0, MAX_SPEED),
            _ratio(own.get("kill_cnt") if own else 0.0, 10.0), _ratio(own.get("dead_cnt") if own else 0.0, 10.0),
            _ratio(enemy.get("kill_cnt") if enemy else 0.0, 10.0), _ratio(enemy.get("dead_cnt") if enemy else 0.0, 10.0),
            _hp_ratio(own_tower), _hp_ratio(enemy_tower),
            _ratio(state["dist_enemy"], MAX_DIST), _ratio(state["dist_enemy_tower"], MAX_DIST), _ratio(state["dist_own_tower"], MAX_DIST),
            state["own_u"], state["own_v"], state["enemy_u"], state["enemy_v"],
            state["ally_front"], state["enemy_front"], _signed_ratio(state["wave_gap"], 1.0),
            _ratio(len(state["ally_minions"]), 6.0), _ratio(len(state["enemy_minions"]), 6.0),
            _ratio(state["ally_near_enemy_tower"], 4.0), _ratio(state["enemy_near_own_tower"], 4.0),
            state["enemy_between"], state["in_enemy_tower"], state["enemy_tower_can_hit_self"], state["hero_can_hit_tower"],
            state["hero_near_tower_stage"], state["wave_tanking"], state["stable_wave_tanking"], state["no_wave_dive"],
            state["objective_hp_ok"], state["push_lane_phase"], state["tower_entry_phase"], state["tower_hit_phase"],
            state["tower_safe"], state["tower_stage"], state["own_defend"], state["kill_window"], state["should_clear"],
            state["can_fight"], state["should_recover"], state["post_kill_active"], state["ep_hold"],
            state["ep_critical"], state["ep_resume"],
            _signed_ratio(d["own_hp"], 0.30), _signed_ratio(d["enemy_hp"], 0.30), _signed_ratio(d["own_ep"], 0.30),
            _signed_ratio(d["enemy_tower_hp"], 0.30), _signed_ratio(d["own_tower_hp"], 0.30),
            _signed_ratio(d["money"], 500.0), _signed_ratio(d["exp"], 500.0),
            state["own_tower_warn"], state["own_tower_critical"], state["enemy_tower_finishable"],
            state["tower_defense_crisis"], state["tower_trade_deficit"], state["force_lane_out"],
            state["tower_defend_clear"], state["tower_out_pressure"], state["own_is_luban"],
            state["own_in_grass"], state["enemy_in_grass"], state["enemy_missing"],
            state["grass_vision_probe"], state["grass_ambush_window"],
            _signed_ratio(state["tower_hp_advantage"], 1.0),
            _ratio(own.get("total_hurt_to_hero") if own else 0.0, MAX_HURT),
            _ratio(own.get("total_be_hurt_by_hero") if own else 0.0, MAX_HURT),
        ]
        for slot_id in range(1, 5):
            values.append(_skill_ready(own, slot_id))
            values.append(_skill_cd_ratio(own, slot_id))
        for slot_id in range(1, 5):
            values.append(_skill_ready(enemy, slot_id))
            values.append(_skill_cd_ratio(enemy, slot_id))
        values.extend([
            tactical["hero_contest"],
            tactical["commit_kill"],
            tactical["short_reset"],
            tactical["post_kill_clear_wave"],
            tactical["post_kill_tower_stage"],
            tactical["post_kill_tower_plan"],
            tactical["combo_start"],
            tactical["combo_chain"],
            tactical["combo_abort"],
            tactical["direnjie_skill1_poke"],
            tactical["direnjie_skill2_escape"],
            tactical["direnjie_ult_kill"],
            tactical["direnjie_followup"],
            tactical["enemy_direnjie_threat"],
            tactical["luban_vs_direnjie"],
            tactical["direnjie_vs_luban"],
            tactical["luban_safe_passive"],
            tactical["direnjie_punish_luban"],
            tactical["luban_passive_tower"],
            tactical["luban_passive_hero"],
            tactical["luban_followup"],
            tactical["luban_skill2_save"],
            tactical["luban_skill2_execute"],
            tactical["luban_skill_waste"],
            tactical["luban_tower_ignore_enemy"],
            tactical["enemy_missing"],
            tactical["enemy_missing_ratio"],
            tactical["enemy_close_extended"],
            tactical["direnjie_skill2_combo"],
            tactical["luban_skill2_combo"],
            tactical["ep_ratio"],
            tactical["ep_conserve"],
            tactical["ep_hold_skills"],
            tactical["ep_resume_ready"],
            tactical["ep_critical"],
            tactical["ep_recover"],
            tactical["hp_advantage"],
            tactical["hp_ratio_level"],
            tactical["in_grass"],
            tactical["purposeful_grass"],
        ])
        return self._pad(values, DimConfig.R8_DENSE_DIM)

    def _build_discrete(self, state):
        out = []
        tactical = state["tactical"]
        hp_edges = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
        ep_edges = [0.22, 0.34, 0.50, 0.70, 0.90]
        dist_edges = [3500, 6500, 9000, 12000, 18000, 26000, 38000]
        prog_edges = [i / 16.0 for i in range(1, 16)]
        for value in [state["own_hp"], state["enemy_hp"], state["own_tower_hp"], state["enemy_tower_hp"]]:
            _append_bin(out, value, hp_edges)
        for value in [state["own_ep"], state["enemy_ep"]]:
            _append_bin(out, value, ep_edges)
        for value in [state["dist_enemy"], state["dist_enemy_tower"], state["dist_own_tower"]]:
            _append_bin(out, value, dist_edges)
        for value in [state["own_u"], state["enemy_u"], state["ally_front"], state["enemy_front"]]:
            _append_bin(out, value, prog_edges)
        tower_zone = 0
        if state["in_enemy_tower"] and state["wave_tanking"]:
            tower_zone = 2
        elif state["in_enemy_tower"]:
            tower_zone = 3
        elif state["tower_safe"]:
            tower_zone = 1
        _append_one_hot(out, tower_zone, 5)
        _append_one_hot(out, state["tower_phase_index"], 5)
        _append_one_hot(out, state["tower_distance_band"], 5)
        _append_one_hot(out, state["own_tower_hp_stage"], 5)
        _append_one_hot(out, state["enemy_tower_hp_stage"], 5)
        _append_one_hot(out, 1 if state["tower_defense_crisis"] > 0.0 else 0, 2)
        _append_one_hot(out, 1 if state["tower_trade_deficit"] > 0.0 else 0, 2)
        _append_one_hot(out, 1 if state["tower_defend_clear"] > 0.0 else 0, 2)
        _append_one_hot(out, 1 if state["tower_out_pressure"] > 0.0 else 0, 2)
        _append_one_hot(out, 1 if state["enemy_tower_finishable"] > 0.0 else 0, 2)
        _append_one_hot(out, 1 if state["own_in_grass"] > 0.0 else 0, 2)
        _append_one_hot(out, 1 if state["enemy_in_grass"] > 0.0 else 0, 2)
        _append_one_hot(out, 1 if state["grass_vision_probe"] > 0.0 else 0, 2)
        _append_one_hot(out, 1 if state["grass_ambush_window"] > 0.0 else 0, 2)
        _append_one_hot(out, min(int(state["ally_near_enemy_tower"]), 4), 5)
        _append_one_hot(out, min(int(state["enemy_near_own_tower"]), 4), 5)
        _append_bin(out, state["wave_gap"], [-0.35, -0.18, -0.04, 0.04, 0.18, 0.35, 0.55])
        own = state["own"]
        enemy = state["enemy"]
        matchup = 0
        own_id = _safe_int(own.get("config_id") if own else 0, 0)
        enemy_id = _safe_int(enemy.get("config_id") if enemy else 0, 0)
        if own_id == LUBAN_HERO_ID and enemy_id == DIRENJIE_HERO_ID:
            matchup = 1
        elif own_id == DIRENJIE_HERO_ID and enemy_id == LUBAN_HERO_ID:
            matchup = 2
        elif own_id == DIRENJIE_HERO_ID and enemy_id == DIRENJIE_HERO_ID:
            matchup = 3
        _append_one_hot(out, matchup, 4)
        _append_one_hot(out, state["main_task"]["index"], 10)
        _append_one_hot(out, 1 if tactical["post_kill_tower_plan"] > 0.55 else 0, 2)
        _append_one_hot(out, 1 if tactical["post_kill_tower_stage"] > 0.35 else 0, 2)
        _append_one_hot(out, 1 if tactical["post_kill_clear_wave"] > 0.35 else 0, 2)
        _append_one_hot(out, 1 if tactical["hero_contest"] > 0.55 else 0, 2)
        _append_one_hot(out, 1 if tactical["commit_kill"] > 0.55 else 0, 2)
        _append_one_hot(out, 1 if tactical["short_reset"] > 0.55 else 0, 2)
        _append_one_hot(out, 1 if tactical["combo_start"] > 0.55 else 0, 2)
        _append_one_hot(out, 1 if tactical["combo_chain"] > 0.55 else 0, 2)
        _append_one_hot(out, 1 if tactical["combo_abort"] > 0.55 else 0, 2)
        _append_one_hot(out, 1 if tactical["luban_passive_tower"] > 0.55 else 0, 2)
        _append_one_hot(out, 1 if tactical["luban_passive_hero"] > 0.55 else 0, 2)
        _append_one_hot(out, 1 if tactical["direnjie_skill1_poke"] > 0.55 else 0, 2)
        _append_one_hot(out, 1 if tactical["direnjie_skill2_escape"] > 0.55 else 0, 2)
        _append_one_hot(out, 1 if tactical["direnjie_ult_kill"] > 0.55 else 0, 2)
        _append_one_hot(out, 1 if tactical["ep_hold_skills"] > 0.55 else 0, 2)
        _append_one_hot(out, 1 if tactical["ep_resume_ready"] > 0.55 else 0, 2)
        _append_one_hot(out, 1 if tactical["ep_critical"] > 0.55 else 0, 2)
        _append_one_hot(out, 1 if tactical["enemy_missing"] > 0.55 else 0, 2)
        for flag in [
            state["enemy_alive"], state["enemy_between"], state["wave_tanking"], state["stable_wave_tanking"],
            state["no_wave_dive"], state["hero_can_hit_tower"], state["push_lane_phase"], state["tower_entry_phase"],
            state["tower_hit_phase"], state["tower_safe"], state["tower_stage"], state["own_defend"], state["kill_window"], state["should_clear"],
            state["can_fight"], state["should_recover"], state["post_kill_active"], state["ep_hold"],
            state["ep_critical"], state["ep_resume"],
        ]:
            _append_one_hot(out, 1 if flag > 0.0 else 0, 2)
        return self._pad(out, DimConfig.R8_DISCRETE_DIM)

    def _build_units(self, state):
        slots = []
        slots.append(("self_hero", state["own"]))
        slots.append(("enemy_hero", state["enemy"]))
        slots.append(("own_tower", state["own_tower"]))
        slots.append(("enemy_tower", state["enemy_tower"]))
        slots.extend(("ally_minion", unit) for unit in state["ally_minions"][:4])
        slots.extend(("enemy_minion", unit) for unit in state["enemy_minions"][:4])
        slots.append(("monster", state["monsters"][0] if state["monsters"] else None))
        slots.append(("own_cake", self._fixed_cake_unit("main")))
        slots.append(("enemy_cake", self._fixed_cake_unit("enemy")))
        slots.append(("reserved", None))
        slots = slots[: DimConfig.R8_UNIT_SLOT_COUNT]
        while len(slots) < DimConfig.R8_UNIT_SLOT_COUNT:
            slots.append(("empty", None))
        out = []
        for slot_type, unit in slots:
            out.extend(self._unit_slot(slot_type, unit, state))
        return self._pad(out, DimConfig.R8_UNIT_FEATURE_DIM)

    def _unit_slot(self, slot_type, unit, state):
        lane = state["lane"]
        own = state["own"]
        enemy_tower = state["enemy_tower"]
        own_tower = state["own_tower"]
        present = 1.0 if unit else 0.0
        u, v = self._lane_coords(unit, lane)
        role_ids = {
            "empty": 0, "self_hero": 1, "enemy_hero": 2, "own_tower": 3,
            "enemy_tower": 4, "ally_minion": 5, "enemy_minion": 6,
            "monster": 7, "own_cake": 8, "enemy_cake": 9, "reserved": 10,
        }
        out = [present]
        _append_one_hot(out, role_ids.get(slot_type, 0), 11)
        out.extend([
            _hp_ratio(unit),
            u,
            v,
            _ratio(_dist(unit, own), MAX_DIST),
            _ratio(_dist(unit, own_tower), MAX_DIST),
            _ratio(_dist(unit, enemy_tower), MAX_DIST),
            1.0 if unit and unit.get("camp") == self.camp else 0.0,
            1.0 if unit and unit.get("camp") != self.camp else 0.0,
            1.0 if unit and _safe_float(unit.get("hp"), 1.0) > 0.0 else 0.0,
            _ratio(unit.get("level") if unit else 0.0, MAX_LEVEL),
            _ratio(unit.get("attack_range") if unit else 0.0, MAX_DIST),
            _ratio(unit.get("phy_atk") if unit else 0.0, MAX_ATTACK),
            _ratio(unit.get("mov_spd") if unit else 0.0, MAX_SPEED),
            1.0 if slot_type == "enemy_minion" and state["enemy_between"] else 0.0,
            1.0 if slot_type == "ally_minion" and u >= 0.82 else 0.0,
            1.0 if slot_type == "enemy_minion" and u <= 0.22 else 0.0,
        ])
        _append_bin(out, _hp_ratio(unit), [0.2, 0.4, 0.6, 0.8])
        _append_bin(out, _dist(unit, own), [3500, 6500, 9000, 14000, 22000])
        return self._pad(out, DimConfig.R8_UNIT_SLOT_DIM)

    def _fixed_cake_unit(self, key):
        x, z = CAKE_LOCATIONS_BY_CAMP[self.camp_key][key]
        return {"location": {"x": x, "z": z}, "hp": 1.0, "max_hp": 1.0, "camp": self.camp if key == "main" else -1}

    def _build_lane_grid(self, state):
        grid = [0.0] * DimConfig.R8_LANE_GRID_DIM
        for unit in state["ally_minions"]:
            self._add_grid(grid, unit, state, 0, max(_hp_ratio(unit), 0.15))
            self._add_grid(grid, unit, state, 1, _hp_ratio(unit))
        for unit in state["enemy_minions"]:
            self._add_grid(grid, unit, state, 2, max(_hp_ratio(unit), 0.15))
            self._add_grid(grid, unit, state, 3, _hp_ratio(unit))
        self._add_grid(grid, state["own"], state, 4, 0.5)
        self._add_grid(grid, state["enemy"], state, 4, -0.5 if state["enemy_alive"] else -0.2)
        self._add_grid(grid, state["own_tower"], state, 5, -0.6)
        self._add_grid(grid, state["enemy_tower"], state, 5, 0.8)
        tower_grid_signal = max(
            state["tower_hit_phase"],
            0.70 * state["tower_entry_phase"],
            0.40 * state["push_lane_phase"],
            state["post_kill_active"],
        )
        if tower_grid_signal > 0.0:
            self._add_grid(grid, state["enemy_tower"], state, 6, tower_grid_signal)
        for monster in state["monsters"][:1]:
            self._add_grid(grid, monster, state, 7, 0.5)
        for i, value in enumerate(grid):
            grid[i] = _clip(value)
        return grid

    def _add_grid(self, grid, unit, state, channel, value):
        if not unit:
            return
        u, v = self._lane_coords(unit, state["lane"])
        progress_idx = int(_clip01((u + 0.05) / 1.10) * (DimConfig.R8_LANE_PROGRESS_BINS - 1))
        lateral_idx = int(_clip01((v + 1.0) / 2.0) * (DimConfig.R8_LANE_LATERAL_BINS - 1))
        channel = max(0, min(DimConfig.R8_LANE_GRID_CHANNELS - 1, channel))
        index = (
            channel * DimConfig.R8_LANE_PROGRESS_BINS * DimConfig.R8_LANE_LATERAL_BINS
            + progress_idx * DimConfig.R8_LANE_LATERAL_BINS
            + lateral_idx
        )
        grid[index] += value

    def _build_temporal(self, state):
        d = state["deltas"]
        tactical = state["tactical"]
        out = [
            _ratio(state["frame_no"], 20000.0),
            state["post_kill_active"], state["post_kill_ratio"], state["enemy_missing_ratio"],
            state["ep_hold"], state["ep_critical"], state["ep_resume"],
            state["push_lane_phase"], state["tower_entry_phase"], state["tower_hit_phase"], state["no_wave_dive"],
            _signed_ratio(d["own_hp"], 0.30), _signed_ratio(d["enemy_hp"], 0.30), _signed_ratio(d["own_ep"], 0.30),
            _signed_ratio(d["enemy_tower_hp"], 0.30), _signed_ratio(d["own_tower_hp"], 0.30),
            _signed_ratio(d["money"], 500.0), _signed_ratio(d["exp"], 500.0),
            state["own_tower_warn"], state["own_tower_critical"], state["enemy_tower_finishable"],
            state["tower_defense_crisis"], state["tower_trade_deficit"], state["force_lane_out"],
            state["tower_defend_clear"], state["tower_out_pressure"], state["own_is_luban"],
            state["own_in_grass"], state["enemy_in_grass"], state["enemy_missing"],
            state["grass_vision_probe"], state["grass_ambush_window"],
            _signed_ratio(state["tower_hp_advantage"], 1.0),
        ]
        out.extend(state["main_task"]["scores"])
        out.extend([
            tactical["hero_contest"],
            tactical["commit_kill"],
            tactical["short_reset"],
            tactical["post_kill_clear_wave"],
            tactical["post_kill_tower_stage"],
            tactical["post_kill_tower_plan"],
            tactical["combo_start"],
            tactical["combo_chain"],
            tactical["combo_abort"],
            tactical["direnjie_skill1_poke"],
            tactical["direnjie_skill2_escape"],
            tactical["direnjie_ult_kill"],
            tactical["luban_passive_tower"],
            tactical["luban_passive_hero"],
            tactical["luban_followup"],
            tactical["luban_skill2_execute"],
            tactical["enemy_missing"],
            tactical["enemy_missing_ratio"],
            tactical["ep_ratio"],
            tactical["ep_conserve"],
            tactical["ep_hold_skills"],
            tactical["ep_resume_ready"],
            tactical["ep_critical"],
            tactical["ep_recover"],
            tactical["hp_advantage"],
            tactical["hp_ratio_level"],
            tactical["in_grass"],
            tactical["purposeful_grass"],
        ])
        _append_one_hot(out, state["main_task"]["index"], 10)
        _append_one_hot(out, state["tower_phase_index"], 5)
        _append_one_hot(out, state["tower_distance_band"], 5)
        _append_one_hot(out, state["own_tower_hp_stage"], 5)
        _append_one_hot(out, state["enemy_tower_hp_stage"], 5)
        return self._pad(out, DimConfig.R8_TEMPORAL_DIM)

    def _pad(self, values, target_dim):
        out = [float(_clip(value)) for value in values[:target_dim]]
        if len(out) < target_dim:
            out.extend([0.0] * (target_dim - len(out)))
        return out
