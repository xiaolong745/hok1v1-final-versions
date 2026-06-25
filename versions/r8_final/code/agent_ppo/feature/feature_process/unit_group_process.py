#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Structured unit-group features for R5.1.

The layout follows the community model note: friendly/enemy heroes, soldiers,
and organs are kept as typed unit groups. The model receives the flat contract
but splits it back into groups for per-type MLP encoders and max-pooling.
"""

import math

from agent_ppo.conf.conf import DimConfig


MINION_SUB_TYPE = 11
TOWER_SUB_TYPE = 21
CRYSTAL_SUB_TYPE = 24
LANE_SOLDIER_CONFIG_IDS = {6800, 6801, 6802, 6803, 6804, 6805}
TOWER_CONFIG_IDS = {1111, 1112}
CRYSTAL_CONFIG_IDS = {1113, 1114}
MAX_POS = 60000.0
MAX_DIST = 60000.0
MAX_MONEY = 12000.0
MAX_EXP = 2200.0
MAX_HURT = 10000.0
MAX_ATTACK = 2000.0
MAX_DEFENSE = 2000.0
MAX_SPEED = 12000.0
DEFAULT_ATTACK_RANGE = 8200.0


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


def _signed_ratio(value, scale):
    scale = _safe_float(scale)
    if scale <= 0:
        return 0.0
    return _clip(_safe_float(value) / scale)


def _hp_ratio(unit):
    if not unit:
        return 0.0
    return _ratio(unit.get("hp"), unit.get("max_hp"))


def _pos(unit):
    location = unit.get("location", {}) if unit else {}
    return (_safe_float(location.get("x")), _safe_float(location.get("z")))


def _dist(a, b):
    if not a or not b:
        return MAX_DIST
    ax, az = _pos(a)
    bx, bz = _pos(b)
    return math.hypot(ax - bx, az - bz)


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


def _skill_ready_by_slot(hero, slot_id):
    for slot in _slot_states(hero):
        slot_type = slot.get("slot_type", slot.get("slot"))
        if int(_safe_float(slot_type, -1)) == slot_id:
            return 1.0 if slot.get("usable") else 0.0
    return 0.0


class UnitGroupProcess:
    def __init__(self, camp):
        self.camp = camp
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2" or camp == 2 or str(camp) == "2"

    def reset(self, camp):
        self.camp = camp
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2" or camp == 2 or str(camp) == "2"

    def process_vec_unit_group(self, observation):
        frame_state = observation["frame_state"]
        player_id = observation.get("player_id")
        own_hero, enemy_hero = self._find_heroes(frame_state, player_id)
        ally_soldiers, enemy_soldiers = self._find_soldiers(frame_state)
        ally_organs, enemy_organs = self._find_organs(frame_state)

        features = []
        features.extend(self._pad_units([own_hero], self._hero_features, DimConfig.HERO_UNIT_COUNT, DimConfig.HERO_UNIT_FEATURE_DIM, own_hero, enemy_hero))
        features.extend(self._pad_units([enemy_hero], self._hero_features, DimConfig.HERO_UNIT_COUNT, DimConfig.HERO_UNIT_FEATURE_DIM, own_hero, enemy_hero))
        features.extend(self._pad_units(ally_soldiers, self._soldier_features, DimConfig.SOLDIER_UNIT_COUNT, DimConfig.SOLDIER_UNIT_FEATURE_DIM, own_hero, enemy_hero))
        features.extend(self._pad_units(enemy_soldiers, self._soldier_features, DimConfig.SOLDIER_UNIT_COUNT, DimConfig.SOLDIER_UNIT_FEATURE_DIM, own_hero, enemy_hero))
        features.extend(self._pad_units(ally_organs, self._organ_features, DimConfig.ORGAN_UNIT_COUNT, DimConfig.ORGAN_UNIT_FEATURE_DIM, own_hero, enemy_hero))
        features.extend(self._pad_units(enemy_organs, self._organ_features, DimConfig.ORGAN_UNIT_COUNT, DimConfig.ORGAN_UNIT_FEATURE_DIM, own_hero, enemy_hero))

        if len(features) != DimConfig.UNIT_GROUP_FEATURE_DIM:
            raise ValueError(f"UNIT_GROUP_FEATURE_DIM mismatch: {len(features)} != {DimConfig.UNIT_GROUP_FEATURE_DIM}")
        return [float(_clip(value)) for value in features]

    def _find_heroes(self, frame_state, player_id):
        own_hero = None
        enemy_hero = None
        for hero in frame_state.get("hero_states", []):
            if hero.get("player_id") == player_id or hero.get("runtime_id") == player_id:
                own_hero = hero
            elif hero.get("camp") != self.camp:
                enemy_hero = hero
        if own_hero is None:
            for hero in frame_state.get("hero_states", []):
                if hero.get("camp") == self.camp:
                    own_hero = hero
                    break
        return own_hero, enemy_hero

    def _find_soldiers(self, frame_state):
        ally = []
        enemy = []
        for npc in frame_state.get("npc_states", []):
            config_id = int(_safe_float(npc.get("config_id"), 0))
            if npc.get("sub_type") != MINION_SUB_TYPE and config_id not in LANE_SOLDIER_CONFIG_IDS:
                continue
            if npc.get("camp") == self.camp:
                ally.append(npc)
            else:
                enemy.append(npc)
        key = lambda unit: (int(_safe_float(unit.get("runtime_id"), 0)), int(_safe_float(unit.get("config_id"), 0)))
        return sorted(ally, key=key)[: DimConfig.SOLDIER_UNIT_COUNT], sorted(enemy, key=key)[: DimConfig.SOLDIER_UNIT_COUNT]

    def _find_organs(self, frame_state):
        ally = []
        enemy = []
        for npc in frame_state.get("npc_states", []):
            config_id = int(_safe_float(npc.get("config_id"), 0))
            is_known_organ = config_id in TOWER_CONFIG_IDS or config_id in CRYSTAL_CONFIG_IDS
            if npc.get("sub_type") not in {TOWER_SUB_TYPE, CRYSTAL_SUB_TYPE} and not is_known_organ:
                continue
            if npc.get("camp") == self.camp:
                ally.append(npc)
            else:
                enemy.append(npc)
        def key(unit):
            config_id = int(_safe_float(unit.get("config_id"), 0))
            if unit.get("sub_type") == TOWER_SUB_TYPE or config_id in TOWER_CONFIG_IDS:
                order = 0
            elif unit.get("sub_type") == CRYSTAL_SUB_TYPE or config_id in CRYSTAL_CONFIG_IDS:
                order = 1
            else:
                order = 9
            return order, int(_safe_float(unit.get("runtime_id"), 0))
        return sorted(ally, key=key)[: DimConfig.ORGAN_UNIT_COUNT], sorted(enemy, key=key)[: DimConfig.ORGAN_UNIT_COUNT]

    def _pad_units(self, units, feature_fn, count, dim, own_hero, enemy_hero):
        features = []
        for unit in units[:count]:
            features.extend(feature_fn(unit, own_hero, enemy_hero))
        missing = count - len(units[:count])
        if missing > 0:
            features.extend([0.0] * missing * dim)
        expected = count * dim
        if len(features) != expected:
            raise ValueError(f"unit feature mismatch: {len(features)} != {expected}")
        return features

    def _relative_xy(self, unit, own_hero):
        ux, uz = _pos(unit)
        ox, oz = _pos(own_hero)
        dx = ux - ox
        dz = uz - oz
        if self.transform_camp2_to_camp1:
            ux = -ux
            uz = -uz
            dx = -dx
            dz = -dz
        return ux, uz, dx, dz

    def _hero_features(self, hero, own_hero, enemy_hero):
        if not hero:
            return [0.0] * DimConfig.HERO_UNIT_FEATURE_DIM
        ux, uz, dx, dz = self._relative_xy(hero, own_hero)
        is_enemy = 1.0 if hero.get("camp") != self.camp else 0.0
        dist_self = math.hypot(dx, dz)
        enemy_dist = _dist(hero, enemy_hero if hero is own_hero else own_hero)
        features = [
            1.0 if _safe_float(hero.get("hp")) > 0 else 0.0,
            is_enemy,
            _hp_ratio(hero),
            _ratio(hero.get("level"), 15.0),
            _ratio(hero.get("money_cnt"), MAX_MONEY),
            _ratio(hero.get("exp"), MAX_EXP),
            _signed_ratio(ux, MAX_POS),
            _signed_ratio(uz, MAX_POS),
            _signed_ratio(dx, MAX_POS),
            _signed_ratio(dz, MAX_POS),
            _ratio(dist_self, MAX_DIST),
            _ratio(enemy_dist, MAX_DIST),
            _ratio(hero.get("attack_range", DEFAULT_ATTACK_RANGE), 12000.0),
            _ratio(hero.get("phy_atk"), MAX_ATTACK),
            _ratio(hero.get("mgc_atk"), MAX_ATTACK),
            _ratio(hero.get("phy_def"), MAX_DEFENSE),
            _ratio(hero.get("mgc_def"), MAX_DEFENSE),
            _ratio(hero.get("mov_spd"), MAX_SPEED),
            _ratio(hero.get("atk_spd"), 20000.0),
            _ratio(hero.get("ep"), max(_safe_float(hero.get("max_ep")), 1.0)),
            _ratio(hero.get("hp_recover"), 1000.0),
            _ratio(hero.get("ep_recover"), 1000.0),
            _ratio(hero.get("kill_cnt"), 8.0),
            _ratio(hero.get("dead_cnt"), 8.0),
            _ratio(hero.get("total_hurt_to_hero"), MAX_HURT),
            _ratio(hero.get("total_be_hurt_by_hero"), MAX_HURT),
            _skill_ready_by_slot(hero, 1),
            _skill_ready_by_slot(hero, 2),
            _skill_ready_by_slot(hero, 3),
            1.0 if hero.get("is_in_grass") else 0.0,
            _ratio(len(hero.get("take_hurt_infos", []) or []), 6.0),
            _ratio(len(hero.get("hit_target_info", []) or []), 6.0),
        ]
        return features

    def _soldier_features(self, soldier, own_hero, enemy_hero):
        if not soldier:
            return [0.0] * DimConfig.SOLDIER_UNIT_FEATURE_DIM
        ux, uz, dx, dz = self._relative_xy(soldier, own_hero)
        is_enemy = 1.0 if soldier.get("camp") != self.camp else 0.0
        features = [
            1.0 if _safe_float(soldier.get("hp")) > 0 else 0.0,
            is_enemy,
            _hp_ratio(soldier),
            _signed_ratio(ux, MAX_POS),
            _signed_ratio(uz, MAX_POS),
            _signed_ratio(dx, MAX_POS),
            _signed_ratio(dz, MAX_POS),
            _ratio(math.hypot(dx, dz), MAX_DIST),
            _ratio(_dist(soldier, enemy_hero), MAX_DIST),
            _ratio(soldier.get("attack_range", 4500.0), 12000.0),
            _ratio(soldier.get("phy_atk"), MAX_ATTACK),
            _ratio(soldier.get("phy_def"), MAX_DEFENSE),
            _ratio(soldier.get("mgc_def"), MAX_DEFENSE),
            _ratio(soldier.get("mov_spd"), MAX_SPEED),
            1.0 if _hp_ratio(soldier) <= 0.30 else 0.0,
            1.0 if _hp_ratio(soldier) <= 0.15 else 0.0,
            _ratio(soldier.get("level"), 15.0),
            _ratio(soldier.get("sub_type"), 30.0),
            _ratio(len(soldier.get("take_hurt_infos", []) or []), 6.0),
            _ratio(len(soldier.get("hit_target_info", []) or []), 6.0),
        ]
        return features

    def _organ_features(self, organ, own_hero, enemy_hero):
        if not organ:
            return [0.0] * DimConfig.ORGAN_UNIT_FEATURE_DIM
        ux, uz, dx, dz = self._relative_xy(organ, own_hero)
        is_enemy = 1.0 if organ.get("camp") != self.camp else 0.0
        config_id = int(_safe_float(organ.get("config_id"), 0))
        is_tower = 1.0 if organ.get("sub_type") == TOWER_SUB_TYPE or config_id in TOWER_CONFIG_IDS else 0.0
        is_crystal = 1.0 if organ.get("sub_type") == CRYSTAL_SUB_TYPE or config_id in CRYSTAL_CONFIG_IDS else 0.0
        features = [
            1.0 if _safe_float(organ.get("hp")) > 0 else 0.0,
            is_enemy,
            is_tower,
            is_crystal,
            _hp_ratio(organ),
            _signed_ratio(ux, MAX_POS),
            _signed_ratio(uz, MAX_POS),
            _signed_ratio(dx, MAX_POS),
            _signed_ratio(dz, MAX_POS),
            _ratio(math.hypot(dx, dz), MAX_DIST),
            _ratio(_dist(organ, enemy_hero), MAX_DIST),
            _ratio(organ.get("attack_range", 8800.0), 12000.0),
            _ratio(organ.get("phy_atk"), 3000.0),
            _ratio(organ.get("phy_def"), 4000.0),
            _ratio(organ.get("mgc_def"), 4000.0),
            1.0 if _hp_ratio(organ) <= 0.30 else 0.0,
            1.0 if _hp_ratio(organ) <= 0.15 else 0.0,
            _ratio(organ.get("level"), 15.0),
            _ratio(len(organ.get("take_hurt_infos", []) or []), 6.0),
            _ratio(len(organ.get("hit_target_info", []) or []), 6.0),
        ]
        return features
