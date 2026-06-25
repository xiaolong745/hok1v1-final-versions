#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Explicit local-map features for the R5 temporal policy.

The vector is a compact multi-channel ego-centric grid. Channels are kept
separate so unit types do not cancel each other inside the same cell.
"""

import math

from agent_ppo.conf.conf import DimConfig

LOCAL_MAP_FEATURE_DIM = DimConfig.LOCAL_MAP_FEATURE_DIM
GRID_SIZE = DimConfig.LOCAL_MAP_GRID_SIZE
CHANNELS = DimConfig.LOCAL_MAP_CHANNELS
CELL_SIZE = 3500.0
MINION_SUB_TYPE = 11
TOWER_SUB_TYPE = 21
CRYSTAL_SUB_TYPE = 24
LANE_SOLDIER_CONFIG_IDS = {6800, 6801, 6802, 6803, 6804, 6805}
TOWER_CONFIG_IDS = {1111, 1112}
CRYSTAL_CONFIG_IDS = {1113, 1114}
ALLY_MINION_CH = 0
ENEMY_MINION_CH = 1
ALLY_ORGAN_CH = 2
ENEMY_ORGAN_CH = 3
ENEMY_HERO_CH = 4
CAKE_CH = 5


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


def _hp_ratio(unit):
    max_hp = _safe_float(unit.get("max_hp") if unit else 0.0)
    if max_hp <= 0:
        return 0.0
    return _clip(_safe_float(unit.get("hp")) / max_hp, 0.0, 1.0)


def _pos(unit):
    location = unit.get("location", {}) if unit else {}
    return (_safe_float(location.get("x")), _safe_float(location.get("z")))


class LocalMapProcess:
    def __init__(self, camp):
        self.camp = camp
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2" or camp == 2 or str(camp) == "2"

    def reset(self, camp):
        self.camp = camp
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2" or camp == 2 or str(camp) == "2"

    def process_vec_local_map(self, observation):
        frame_state = observation["frame_state"]
        player_id = observation.get("player_id")
        own_hero = self._find_own_hero(frame_state, player_id)
        if not own_hero:
            return [0.0] * LOCAL_MAP_FEATURE_DIM

        ox, oz = _pos(own_hero)
        grid = [0.0] * LOCAL_MAP_FEATURE_DIM

        for hero in frame_state.get("hero_states", []):
            if hero is own_hero or hero.get("player_id") == player_id or hero.get("runtime_id") == player_id:
                continue
            self._add_unit(grid, ox, oz, hero, ENEMY_HERO_CH, value=max(_hp_ratio(hero), 0.2))

        for npc in frame_state.get("npc_states", []):
            sub_type = npc.get("sub_type")
            camp = npc.get("camp")
            config_id = int(_safe_float(npc.get("config_id"), 0))
            if sub_type == MINION_SUB_TYPE or config_id in LANE_SOLDIER_CONFIG_IDS:
                value = 0.35 * max(_hp_ratio(npc), 0.15)
                channel = ALLY_MINION_CH if camp == self.camp else ENEMY_MINION_CH
                self._add_unit(grid, ox, oz, npc, channel, value=value)
            elif sub_type in {TOWER_SUB_TYPE, CRYSTAL_SUB_TYPE} or config_id in TOWER_CONFIG_IDS or config_id in CRYSTAL_CONFIG_IDS:
                value = 0.8 * max(_hp_ratio(npc), 0.25)
                channel = ALLY_ORGAN_CH if camp == self.camp else ENEMY_ORGAN_CH
                self._add_unit(grid, ox, oz, npc, channel, value=value)

        for cake in frame_state.get("cakes", []):
            self._add_unit(grid, ox, oz, cake, CAKE_CH, value=0.25)

        return [float(_clip(value)) for value in grid]

    def _find_own_hero(self, frame_state, player_id):
        for hero in frame_state.get("hero_states", []):
            if hero.get("player_id") == player_id or hero.get("runtime_id") == player_id:
                return hero
        for hero in frame_state.get("hero_states", []):
            if hero.get("camp") == self.camp:
                return hero
        return None

    def _add_unit(self, grid, ox, oz, unit, channel, value):
        ux, uz = _pos(unit)
        dx = ux - ox
        dz = uz - oz
        if self.transform_camp2_to_camp1:
            dx = -dx
            dz = -dz

        half = GRID_SIZE // 2
        gx = int(math.floor(dx / CELL_SIZE)) + half
        gz = int(math.floor(dz / CELL_SIZE)) + half
        if gx < 0 or gx >= GRID_SIZE or gz < 0 or gz >= GRID_SIZE:
            return
        if channel < 0 or channel >= CHANNELS:
            return
        index = channel * GRID_SIZE * GRID_SIZE + gz * GRID_SIZE + gx
        grid[index] = _clip(grid[index] + value)
