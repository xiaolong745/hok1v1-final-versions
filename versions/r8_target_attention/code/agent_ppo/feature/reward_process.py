#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
R2.3 PPO reward for HOK1v1.

Keep the reward API and configured keys stable, but only let a small set of
signals drive learning: tower damage, own tower protection, economy progress,
death avoidance, low-HP/tower-dive safety, and light lane pressure.
R2.2 focused on objective conversion: when allied minions enter tower range,
convert the wave into tower damage instead of drifting, brawling, or hiding
without progress.
Grass is only rewarded when it serves a tactical purpose: disengaging under
pressure, vision-poking the enemy, or staging just outside tower range before
a minion wave converts. Aimless grass wandering is a negative signal.
R2.3 adds marksman fundamentals: clear the wave before fighting, take neutral
resources only in safe windows, use health packs/recall when low, and stop
wasting Luban skill 1 on towers.
R2.4 makes recall/recovery relative to the enemy state. A low-HP agent should
hold lane only when the enemy is also weak or the wave is valuable; otherwise it
must commit to cake, tower-side disengage, or Recall instead of orbiting.
R4.2 makes recovery a hard commitment until at least 70% HP, finishes low-HP
neutral monsters before abandoning them, and changes healthy early lane play
from fast clear to last-hit/freeze control.
R4.2.1 narrows recovery exceptions: reset from a safe tower/grass position
prefers BUTTON_RECALL when the runtime exposes it, but moving back toward our
tower/cake remains a valid recovery action. Low-HP fighting is only allowed
when our wave has crashed into the enemy tower and the enemy is already in an
executable state.
R5.2 adds Luban tower-edge poke pressure: when the enemy turtles under tower,
standing just outside tower range and using passive/skills to chip tower or
hero is rewarded, while no-wave oversteps into tower range are punished.
R5.2.1 makes that pressure objective-first: with our wave tanking tower, prefer
two or three tower hits and tolerate small hero poke; when the wave is gone,
reset instead of trading or tanking tower.
R5.2.2 adds explicit hero matchup awareness and gives DiRenjie the same
objective-first tower-edge pressure path without treating it as Luban passive
follow-up.
R5.2.3 adds DiRenJie skill semantics for 1v1: passive auto pressure, skill 1
poke/clear, skill 2 defensive escape, and skill 3 close-range kill confirm.
R5.2.4 adds four explicit matchup strategy gates so Luban mirror, Luban into
DiRenJie, DiRenJie into Luban, and DiRenJie mirror are not learned as one
generic marksman plan.
R5.3.4 treats BUTTON_RECALL as unavailable in this runtime. Recovery is a
short tower/cake/grass reset, not a long walk back to fountain; if short reset
does not create value and the enemy/tower state is favorable, low-HP play may
trade for kill or tower progress instead of being forced to retreat to 70% HP.
R5.4 activates tactical matchup profiles on top of the frozen R5.3 feature
schema. The model and dimensions stay unchanged; reward terms are scaled by
the observed 112/133 matchup so each lane learns its own tactical priorities.
"""

import math

from agent_ppo.conf.conf import GameConfig


MINION_SUB_TYPE = 11
TOWER_SPRING_SUB_TYPE = 20
TOWER_SUB_TYPE = 21
CRYSTAL_SUB_TYPE = 24
MAX_FRAME_NO = 20000.0
DEFAULT_ATTACK_RANGE = 8200.0
BUTTON_MOVE = 2
BUTTON_ATTACK = 3
BUTTON_SKILL_1 = 4
BUTTON_SKILL_2 = 5
BUTTON_SKILL_3 = 6
BUTTON_HEAL = 7
BUTTON_SUMMONER = 8
BUTTON_RECALL = 9
HERO_SKILL_BUTTONS = {BUTTON_SKILL_1, BUTTON_SKILL_2, BUTTON_SKILL_3}
SKILL_BUTTONS = {4, 5, 6, 8, 10, 11}
LUBAN_PRESSURE_SKILLS = {BUTTON_SKILL_1, BUTTON_SKILL_2, BUTTON_SKILL_3}
DIRENJIE_PRESSURE_SKILLS = {BUTTON_SKILL_1, BUTTON_SKILL_2, BUTTON_SKILL_3}
TARGET_ENEMY = 1
TARGET_NONE = 0
TARGET_SELF = 2
TARGET_SOLDIER_MIN = 3
TARGET_SOLDIER_MAX = 6
TARGET_TOWER = 7
TARGET_MONSTER = 8
TOWER_ATTACK_RANGE_PAD = 1800.0
LANE_LATERAL_SCALE = 12000.0
LUBAN_HERO_ID = 112
DIRENJIE_HERO_ID = 133
LOW_HP_OBJECTIVE_MIN_HP = 0.18
POST_KILL_EARLY_END_FRAME = 9000.0
POST_KILL_FINISH_START_FRAME = 12500.0
POST_KILL_FINISH_FULL_FRAME = 15000.0
RECOVERY_HEALTHY_HP = 0.70
SHORT_RESET_EXIT_HP = 0.48
SHORT_RESET_STALL_LIMIT = 5
TOWER_PRESSURE_MIN_HP = 0.42
LOW_HP_HOLD_MAX_ENEMY_ADVANTAGE = 0.15
LOW_HP_GRASS_RESET_ENEMY_HP = 0.62
LOW_HP_GRASS_AMBUSH_ENEMY_RANGE_PAD = 2400.0
LOW_HP_RELATIVE_ADVANTAGE = 0.08
LANE_CONTROL_END_FRAME = 8500.0
HARD_RECALL_HP = 0.36
ENEMY_HEALTHY_HP = 0.55
TOWER_CRASH_KILL_ENEMY_HP = 0.34
TOWER_CRASH_KILL_MIN_OWN_HP = 0.14
LOW_HP_TRADE_ENEMY_HP = 0.42
LOW_HP_TRADE_MIN_OWN_HP = 0.14
TOWER_EDGE_SAFE_MARGIN = 250.0
TOWER_EDGE_OUTER_PAD = 1800.0
TOWER_TURTLE_RANGE_PAD = 1200.0
TOWER_HIT_RANGE_PAD = 650.0
TOWER_STABLE_WAVE_MIN_COUNT = 2
BASELINE_OPENING_PRESSURE_END_FRAME = 3800.0
TOWER_POKE_TOLERATED_HURT = 520.0
TOWER_POKE_HEAVY_HURT = 1100.0
EP_HOLD_SKILL_RATIO = 0.34
EP_RESUME_SKILL_RATIO = 0.50
EP_CRITICAL_RATIO = 0.22
EP_RECOVER_SCALE = 1000.0


class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


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


def _clip_head_reward(head_name, value):
    low, high = GameConfig.VALUE_HEAD_CLIPS.get(head_name, (-20.0, 20.0))
    return _clip(value, low, high)


def _reward_vector_from_heads(total_reward, reward_vector_by_head):
    reward_vector = []
    for head_name in GameConfig.VALUE_HEAD_NAMES:
        if head_name == "total":
            reward_vector.append(_clip(total_reward, -20.0, 20.0))
        else:
            reward_vector.append(_clip_head_reward(head_name, reward_vector_by_head.get(head_name, 0.0)))
    return reward_vector


def _ratio(value, max_value, default=0.0):
    max_value = _safe_float(max_value)
    if max_value <= 0:
        return default
    return max(0.0, min(1.0, _safe_float(value) / max_value))


def _phase(value, start, end):
    if end <= start:
        return 1.0 if value >= end else 0.0
    return _clip((_safe_float(value) - start) / (end - start), 0.0, 1.0)


def _hp_ratio(unit):
    if not unit:
        return 0.0
    return _ratio(_get(unit, "hp"), _get(unit, "max_hp"))


def _ep_ratio(unit):
    if not unit:
        return 1.0, 0.0
    max_ep = _safe_float(_get(unit, "max_ep"), 0.0)
    if max_ep <= 0.0:
        return 1.0, 0.0
    return _ratio(_get(unit, "ep"), max_ep, default=1.0), 1.0


def _pos(unit):
    location = _get(unit, "location", {}) if unit else {}
    return (_safe_float(_get(location, "x")), _safe_float(_get(location, "z")))


def _dist(unit_a, unit_b):
    if not unit_a or not unit_b:
        return 99999.0
    ax, az = _pos(unit_a)
    bx, bz = _pos(unit_b)
    return math.hypot(ax - bx, az - bz)


def _get(item, key, default=None):
    if item is None:
        return default
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _slot_states(hero):
    if not hero:
        return []
    skill_state = _get(hero, "skill_state", {})
    if isinstance(skill_state, dict):
        states = skill_state.get("slot_states", [])
    elif isinstance(skill_state, list):
        states = skill_state
    else:
        states = []
    return [state for state in states if isinstance(state, dict)]


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.last_main = None
        self.last_enemy = None
        self.last_action = None
        self.pending_action = None
        self.tower_window_stall_steps = 0
        self.low_hp_stall_steps = 0
        self.post_kill_steps = 0
        self.post_kill_window_steps = 1
        self.post_kill_wait_enemy_seen = False
        self.recovery_commit_active = False
        self.post_clear_steps = 0
        self.ult_follow_steps = 0
        self.enemy_missing_steps = 0
        self.ep_hold_steps = 0
        self.episode_stats = {
            "tower_window_frames": 0,
            "attack_tower_actions": 0,
            "tower_damage_events": 0,
            "missed_tower_windows": 0,
            "late_urgency_frames": 0,
            "no_wave_dive_frames": 0,
            "skill_then_attack": 0,
            "skill_without_followup": 0,
            "purposeful_grass_frames": 0,
            "aimless_grass_frames": 0,
            "grass_tower_staging_frames": 0,
            "lane_clear_actions": 0,
            "last_hit_windows": 0,
            "missed_last_hit_windows": 0,
            "over_wave_fight_frames": 0,
            "neutral_window_frames": 0,
            "neutral_attack_actions": 0,
            "neutral_damage_events": 0,
            "neutral_finish_windows": 0,
            "missed_neutral_finish_windows": 0,
            "low_hp_recovery_windows": 0,
            "health_pack_recover_events": 0,
            "safe_recall_windows": 0,
            "recall_actions": 0,
            "missed_recall_windows": 0,
            "early_lane_control_windows": 0,
            "low_hp_stall_frames": 0,
            "skill_tower_waste_events": 0,
            "passive_followup_actions": 0,
            "direnjie_pressure_windows": 0,
            "direnjie_pressure_actions": 0,
            "direnjie_auto_pressure_actions": 0,
            "direnjie_skill1_poke_actions": 0,
            "direnjie_skill2_escape_actions": 0,
            "direnjie_skill2_waste_actions": 0,
            "direnjie_ult_pressure_actions": 0,
            "direnjie_ult_waste_actions": 0,
            "matchup_luban_mirror_frames": 0,
            "matchup_luban_direnjie_frames": 0,
            "matchup_direnjie_luban_frames": 0,
            "matchup_direnjie_mirror_frames": 0,
            "matchup_strategy_actions": 0,
            "post_kill_window_frames": 0,
            "post_kill_lane_clear_actions": 0,
            "post_kill_tower_actions": 0,
            "post_kill_tower_damage_events": 0,
            "post_kill_recovery_actions": 0,
            "post_kill_idle_frames": 0,
            "tower_crash_kill_windows": 0,
            "tower_crash_kill_attack_actions": 0,
            "tower_crash_kill_kills": 0,
            "tower_crash_kill_deaths": 0,
            "tower_edge_poke_windows": 0,
            "tower_edge_poke_actions": 0,
            "tower_edge_poke_damage_events": 0,
            "tower_edge_oversteps": 0,
            "tower_wave_lost_resets": 0,
            "ep_low_frames": 0,
            "ep_critical_frames": 0,
            "ep_low_skill_actions": 0,
            "ep_conserve_actions": 0,
            "ep_resume_events": 0,
            "safe_recall_button_actions": 0,
            "missed_safe_recall_windows": 0,
            "post_clear_idle_frames": 0,
            "ult_follow_windows": 0,
            "baseline_luban_pressure_windows": 0,
            "baseline_luban_pressure_actions": 0,
            "baseline_direnjie_safe_farm_windows": 0,
            "vision_probe_windows": 0,
            "tower_damage": 0.0,
        }

    def record_action(self, action):
        if action is None:
            self.pending_action = None
            return
        try:
            self.pending_action = [int(v) for v in action]
        except (TypeError, ValueError):
            self.pending_action = None

    def result(self, frame_data):
        frame_data = frame_data or {}
        frame_no = _get(frame_data, "frame_no", 0)
        main = self._snapshot(frame_data, self._find_main_camp(frame_data))
        enemy = self._snapshot(frame_data, main["enemy_camp"]) if main["enemy_camp"] != -1 else self.last_enemy
        if enemy is None:
            enemy = self._empty_snapshot()

        if self.last_main is None or self.last_enemy is None or not self.last_main["hero_present"] or not main["hero_present"]:
            reward_dict = {key: 0.0 for key in GameConfig.REWARD_WEIGHT_DICT}
            reward_dict["reward_sum"] = 0.0
            reward_dict["reward_vector"] = [0.0 for _ in GameConfig.VALUE_HEAD_NAMES]
            self.low_hp_stall_steps = 0
            self.post_kill_steps = 0
            self.post_kill_window_steps = 1
            self.post_kill_wait_enemy_seen = False
            self.recovery_commit_active = False
            self.post_clear_steps = 0
            self.ult_follow_steps = 0
            self.enemy_missing_steps = 0
            self.ep_hold_steps = 0
            self.last_main = main
            self.last_enemy = enemy
            self.m_reward_value = reward_dict
            return reward_dict

        components = self._calc_components(frame_no, main, enemy)
        reward_sum = 0.0
        reward_dict = {}
        reward_vector_by_head = {head_name: 0.0 for head_name in GameConfig.VALUE_HEAD_NAMES if head_name != "total"}
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            value = components.get(reward_name, 0.0)
            reward_struct.value = value
            weighted_value = value * reward_struct.weight
            reward_sum += weighted_value
            head_name = GameConfig.REWARD_HEAD_BY_ITEM.get(reward_name)
            if head_name in reward_vector_by_head:
                reward_vector_by_head[head_name] += weighted_value
            reward_dict[reward_name] = value

        if self.time_scale_arg > 0:
            reward_sum *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)
            decay = math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)
            for head_name in list(reward_vector_by_head):
                reward_vector_by_head[head_name] *= decay

        reward_dict["reward_sum"] = _clip(reward_sum, -20.0, 20.0)
        reward_dict["reward_vector"] = _reward_vector_from_heads(reward_dict["reward_sum"], reward_vector_by_head)
        self.last_main = main
        self.last_enemy = enemy
        self.last_action = self.pending_action
        self.pending_action = None
        self.m_reward_value = reward_dict
        return reward_dict

    def _find_main_camp(self, frame_data):
        for hero in _get(frame_data, "hero_states", []) or []:
            if _get(hero, "runtime_id") == self.main_hero_player_id or _get(hero, "player_id") == self.main_hero_player_id:
                self.main_hero_camp = _get(hero, "camp", self.main_hero_camp)
                return self.main_hero_camp
        if self.main_hero_camp != -1:
            return self.main_hero_camp
        for hero in _get(frame_data, "hero_states", []) or []:
            self.main_hero_camp = _get(hero, "camp", -1)
            return self.main_hero_camp
        return -1

    def _snapshot(self, frame_data, camp):
        hero, enemy_hero = None, None
        for item in _get(frame_data, "hero_states", []) or []:
            if _get(item, "camp") == camp:
                hero = item
            else:
                enemy_hero = item

        own_tower, enemy_tower = None, None
        ally_minions, enemy_minions, monsters = [], [], []
        for npc in _get(frame_data, "npc_states", []) or []:
            sub_type = _get(npc, "sub_type")
            npc_camp = _get(npc, "camp")
            if sub_type == TOWER_SUB_TYPE:
                if npc_camp == camp:
                    own_tower = npc
                else:
                    enemy_tower = npc
            elif sub_type == MINION_SUB_TYPE:
                if npc_camp == camp:
                    ally_minions.append(npc)
                else:
                    enemy_minions.append(npc)
            elif self._is_monster(npc):
                monsters.append(npc)

        hero_ep, hero_has_ep = _ep_ratio(hero)
        return {
            "camp": camp,
            "hero_present": hero is not None,
            "enemy_hero_present": enemy_hero is not None,
            "hero_config_id": int(_safe_float(_get(hero, "config_id", -1), -1)),
            "enemy_camp": _get(enemy_hero, "camp", -1) if enemy_hero else -1,
            "hero": hero,
            "enemy_hero": enemy_hero,
            "own_tower": own_tower,
            "enemy_tower": enemy_tower,
            "own_tower_present": own_tower is not None,
            "enemy_tower_present": enemy_tower is not None,
            "ally_minions": ally_minions,
            "enemy_minions": enemy_minions,
            "monsters": monsters,
            "cakes": [cake for cake in _get(frame_data, "cakes", []) or [] if isinstance(cake, dict)],
            "hero_hp": _hp_ratio(hero),
            "hero_ep": hero_ep,
            "hero_has_ep": hero_has_ep,
            "hero_ep_recover": _ratio(_get(hero, "ep_recover", 0.0), EP_RECOVER_SCALE),
            "enemy_hp": _hp_ratio(enemy_hero),
            "own_tower_hp": _hp_ratio(own_tower),
            "enemy_tower_hp": _hp_ratio(enemy_tower),
            "money_cnt": _safe_float(_get(hero, "money_cnt", 0)),
            "exp": _safe_float(_get(hero, "exp", 0)),
            "level": _safe_float(_get(hero, "level", 0)),
            "kill_cnt": _safe_float(_get(hero, "kill_cnt", 0)),
            "dead_cnt": _safe_float(_get(hero, "dead_cnt", 0)),
            "hurt_to_hero": _safe_float(_get(hero, "total_hurt_to_hero", 0)),
            "be_hurt_by_hero": _safe_float(_get(hero, "total_be_hurt_by_hero", 0)),
            "attack_range": _safe_float(_get(hero, "attack_range", DEFAULT_ATTACK_RANGE), DEFAULT_ATTACK_RANGE),
            "is_in_grass": bool(_get(hero, "is_in_grass", False)),
            "frame_no": _get(frame_data, "frame_no", 0),
            "skill_ready_ratio": self._skill_ready_ratio(hero),
            "skill1_ready": self._skill_ready_by_slot(hero, 1),
            "skill2_ready": self._skill_ready_by_slot(hero, 2),
            "skill3_ready": self._skill_ready_by_slot(hero, 3),
        }

    def _empty_snapshot(self):
        return {
            "camp": -1,
            "hero_present": False,
            "enemy_hero_present": False,
            "hero_config_id": -1,
            "enemy_camp": -1,
            "hero": None,
            "enemy_hero": None,
            "own_tower": None,
            "enemy_tower": None,
            "own_tower_present": False,
            "enemy_tower_present": False,
            "ally_minions": [],
            "enemy_minions": [],
            "monsters": [],
            "cakes": [],
            "hero_hp": 0.0,
            "hero_ep": 1.0,
            "hero_has_ep": 0.0,
            "hero_ep_recover": 0.0,
            "enemy_hp": 0.0,
            "own_tower_hp": 0.0,
            "enemy_tower_hp": 0.0,
            "money_cnt": 0.0,
            "exp": 0.0,
            "level": 0.0,
            "kill_cnt": 0.0,
            "dead_cnt": 0.0,
            "hurt_to_hero": 0.0,
            "be_hurt_by_hero": 0.0,
            "attack_range": DEFAULT_ATTACK_RANGE,
            "is_in_grass": False,
            "frame_no": 0,
            "skill_ready_ratio": 0.0,
            "skill1_ready": 0.0,
            "skill2_ready": 0.0,
            "skill3_ready": 0.0,
        }

    def _calc_components(self, frame_no, main, enemy):
        enemy_tower_delta = 0.0
        if main["enemy_tower_present"] and self.last_main["enemy_tower_present"]:
            enemy_tower_delta = self.last_main["enemy_tower_hp"] - main["enemy_tower_hp"]
        elif self.last_main["enemy_tower_present"] and not main["enemy_tower_present"]:
            enemy_tower_delta = self.last_main["enemy_tower_hp"]
        own_tower_delta = 0.0
        if main["own_tower_present"] and self.last_main["own_tower_present"]:
            own_tower_delta = main["own_tower_hp"] - self.last_main["own_tower_hp"]
        elif self.last_main["own_tower_present"] and not main["own_tower_present"]:
            own_tower_delta = -self.last_main["own_tower_hp"]

        money_delta = max(0.0, main["money_cnt"] - self.last_main["money_cnt"])
        exp_delta = max(0.0, main["exp"] - self.last_main["exp"])
        level_delta = max(0.0, main["level"] - self.last_main["level"])
        economy_progress = money_delta / 120.0 + exp_delta / 180.0 + level_delta * 0.25
        death_delta = main["dead_cnt"] - self.last_main["dead_cnt"]
        kill_delta = main["kill_cnt"] - self.last_main["kill_cnt"]
        enemy_dead_delta = enemy["dead_cnt"] - self.last_enemy["dead_cnt"]
        post_kill_delta = max(kill_delta, enemy_dead_delta)
        if death_delta > 0:
            self.post_kill_steps = 0
            self.post_kill_window_steps = 1
            self.post_kill_wait_enemy_seen = False
        enemy_minion_clear_delta = len(self.last_main["enemy_minions"]) - len(main["enemy_minions"])
        if enemy_minion_clear_delta > 0 and len(main["enemy_minions"]) == 0:
            self.post_clear_steps = 6
        elif self.post_clear_steps > 0:
            self.post_clear_steps -= 1
        hurt_to_delta = max(0.0, main["hurt_to_hero"] - self.last_main["hurt_to_hero"])
        hurt_by_delta = max(0.0, main["be_hurt_by_hero"] - self.last_main["be_hurt_by_hero"])
        if self.last_action is not None and self._action_head(self.last_action) == BUTTON_SKILL_3 and hurt_to_delta > 0:
            self.ult_follow_steps = 5
        elif self.ult_follow_steps > 0:
            self.ult_follow_steps -= 1
        hp_trade = (hurt_to_delta - hurt_by_delta) / 1600.0

        ally_near_enemy_tower = self._count_near(main["ally_minions"], main["enemy_tower"], main["enemy_tower"])
        stable_ally_near_enemy_tower = ally_near_enemy_tower >= TOWER_STABLE_WAVE_MIN_COUNT
        enemy_near_own_tower = self._count_near(main["enemy_minions"], main["own_tower"], main["own_tower"])
        dist_enemy_tower_now = _dist(main["hero"], main["enemy_tower"])
        dist_enemy_tower_last = _dist(self.last_main["hero"], self.last_main["enemy_tower"])
        dist_own_tower_now = _dist(main["hero"], main["own_tower"])
        dist_own_tower_last = _dist(self.last_main["hero"], self.last_main["own_tower"])
        dist_enemy_hero_now = _dist(main["hero"], main["enemy_hero"])
        dist_enemy_hero_last = _dist(self.last_main["hero"], self.last_main["enemy_hero"])
        move_delta = _dist(main["hero"], self.last_main["hero"])

        enemy_tower_range = _safe_float(_get(main["enemy_tower"], "attack_range", 8800.0), 8800.0)
        in_enemy_tower_range = main["enemy_tower"] and dist_enemy_tower_now <= enemy_tower_range
        no_wave_dive = bool(in_enemy_tower_range and ally_near_enemy_tower == 0)
        low_hp = main["hero_present"] and main["hero_hp"] < 0.4
        action_head = self._transition_action_head()
        previous_action_head = self._action_head(self.last_action)
        enemy_visible = main["enemy_hero"] is not None
        if enemy_visible and main["enemy_hp"] > 0.0:
            self.enemy_missing_steps = 0
        else:
            self.enemy_missing_steps = min(self.enemy_missing_steps + 1, 32)
        attack_range = max(main["attack_range"], DEFAULT_ATTACK_RANGE)
        enemy_in_attack_range = enemy_visible and dist_enemy_hero_now <= attack_range
        enemy_close_range = enemy_visible and dist_enemy_hero_now <= attack_range * 1.18
        took_hero_hurt = hurt_by_delta > 0
        dealt_hero_hurt = hurt_to_delta > 0
        action_target = self._transition_action_target()
        close_tower_range = max(attack_range + TOWER_HIT_RANGE_PAD, enemy_tower_range * 0.72)
        pressure_outer_range = max(attack_range + TOWER_ATTACK_RANGE_PAD, enemy_tower_range + TOWER_EDGE_OUTER_PAD)
        hero_can_hit_tower = main["enemy_tower"] and dist_enemy_tower_now <= close_tower_range
        target_soldier_action = action_head == BUTTON_ATTACK and self._is_soldier_target(action_target)
        target_monster_action = action_head == BUTTON_ATTACK and action_target == TARGET_MONSTER
        target_enemy_action = action_target == TARGET_ENEMY
        attack_tower_action = action_head == BUTTON_ATTACK and action_target == TARGET_TOWER
        ep_context = self._calc_ep_context(
            main=main,
            action_head=action_head,
            dealt_hero_hurt=dealt_hero_hurt,
            took_hero_hurt=took_hero_hurt,
            enemy_tower_delta=enemy_tower_delta,
            target_enemy_action=target_enemy_action,
            target_soldier_action=target_soldier_action,
            attack_tower_action=attack_tower_action,
            enemy_minion_clear_delta=enemy_minion_clear_delta,
            money_delta=money_delta,
            exp_delta=exp_delta,
        )
        own_is_luban = main["hero_config_id"] == LUBAN_HERO_ID
        own_is_direnjie = main["hero_config_id"] == DIRENJIE_HERO_ID
        enemy_is_luban = enemy["hero_config_id"] == LUBAN_HERO_ID
        enemy_is_direnjie = enemy["hero_config_id"] == DIRENJIE_HERO_ID
        matchup_context = {
            "luban_vs_luban": bool(own_is_luban and enemy_is_luban),
            "luban_vs_direnjie": bool(own_is_luban and enemy_is_direnjie),
            "direnjie_vs_luban": bool(own_is_direnjie and enemy_is_luban),
            "direnjie_vs_direnjie": bool(own_is_direnjie and enemy_is_direnjie),
        }
        enemy_recent_missing = self.enemy_missing_steps > 0
        enemy_under_tower = bool(
            enemy_visible
            and main["enemy_tower_present"]
            and main["enemy_hp"] > 0.0
            and _dist(main["enemy_hero"], main["enemy_tower"]) <= enemy_tower_range + TOWER_TURTLE_RANGE_PAD
        )
        tower_edge_band = bool(
            main["enemy_tower_present"]
            and enemy_tower_range + TOWER_EDGE_SAFE_MARGIN <= dist_enemy_tower_now <= pressure_outer_range
        )
        tower_wave_lost_reset = bool(
            enemy_under_tower
            and ally_near_enemy_tower == 0
            and dist_enemy_tower_now <= pressure_outer_range + 2500.0
        )
        tower_edge_overstep = bool(enemy_under_tower and in_enemy_tower_range and ally_near_enemy_tower == 0)
        raw_tower_edge_poke_window = bool(
            (own_is_luban or own_is_direnjie)
            and enemy_under_tower
            and tower_edge_band
            and stable_ally_near_enemy_tower
            and (
                main["hero_hp"] > TOWER_PRESSURE_MIN_HP
                or main["hero_hp"] >= LOW_HP_OBJECTIVE_MIN_HP
            )
        )
        tower_stage_window = bool(
            main["enemy_tower_present"]
            and ally_near_enemy_tower > 0
            and (
                main["hero_hp"] >= TOWER_PRESSURE_MIN_HP
                or (stable_ally_near_enemy_tower and main["hero_hp"] >= LOW_HP_OBJECTIVE_MIN_HP)
            )
            and not no_wave_dive
            and dist_enemy_tower_now <= pressure_outer_range + 2600.0
        )
        wave_crashed_enemy_tower = bool(
            main["enemy_tower_present"]
            and ally_near_enemy_tower > 0
            and not no_wave_dive
        )
        core_skills_ready = main["skill_ready_ratio"] >= 0.45
        enemy_executable = bool(
            enemy_visible
            and 0.0 < main["enemy_hp"] <= TOWER_CRASH_KILL_ENEMY_HP
            and main["hero_hp"] >= TOWER_CRASH_KILL_MIN_OWN_HP
            and dist_enemy_hero_now <= attack_range + 1800.0
        )
        low_hp_value_trade_window = bool(
            enemy_visible
            and main["hero_hp"] >= LOW_HP_TRADE_MIN_OWN_HP
            and main["hero_hp"] < SHORT_RESET_EXIT_HP
            and 0.0 < main["enemy_hp"] <= LOW_HP_TRADE_ENEMY_HP
            and dist_enemy_hero_now <= attack_range + 2200.0
            and (
                wave_crashed_enemy_tower
                or raw_tower_edge_poke_window
                or main["enemy_tower_hp"] <= 0.35
                or main["enemy_hp"] <= 0.26
                or dealt_hero_hurt
            )
            and not no_wave_dive
        )
        if hurt_to_delta > 0 and hurt_by_delta <= TOWER_POKE_TOLERATED_HURT:
            hp_trade += min(0.08, hurt_to_delta / 4200.0)
        if target_enemy_action and hurt_to_delta <= 0 and hurt_by_delta > 0 and not low_hp_value_trade_window:
            hp_trade -= min(0.06, hurt_by_delta / 3600.0)
        if (
            previous_action_head in LUBAN_PRESSURE_SKILLS
            and action_head == BUTTON_ATTACK
            and (target_enemy_action or attack_tower_action or self._is_soldier_target(action_target))
            and hurt_by_delta <= TOWER_POKE_TOLERATED_HURT
        ):
            hp_trade += 0.025
        tower_crash_kill_window = bool(
            wave_crashed_enemy_tower
            and enemy_executable
            and (core_skills_ready or main["enemy_hp"] <= 0.24 or dealt_hero_hurt)
        )
        tower_hit_window = bool(
            main["enemy_tower_present"]
            and stable_ally_near_enemy_tower
            and hero_can_hit_tower
            and (
                main["hero_hp"] >= TOWER_PRESSURE_MIN_HP
                or main["hero_hp"] >= LOW_HP_OBJECTIVE_MIN_HP
            )
            and not no_wave_dive
        )
        tower_window = tower_hit_window
        late_phase = _clip((frame_no - 9000.0) / 9000.0, 0.0, 1.0)
        lane_context = self._lane_context(
            main=main,
            enemy=enemy,
            frame_no=frame_no,
            attack_range=attack_range,
            enemy_near_own_tower=enemy_near_own_tower,
            ally_near_enemy_tower=ally_near_enemy_tower,
        )
        neutral_context = self._neutral_context(
            main=main,
            attack_range=attack_range,
            lane_clear_priority=lane_context["lane_clear_priority"],
        )
        recovery_context = self._recovery_context(
            main=main,
            enemy=enemy,
            no_wave_dive=no_wave_dive,
            enemy_near_own_tower=enemy_near_own_tower,
            dist_own_tower_now=dist_own_tower_now,
        )
        if tower_crash_kill_window:
            recovery_context["tower_crash_kill_window"] = True
            recovery_context["recovery_commit"] = False
            recovery_context["safe_recall_window"] = False
            recovery_context["low_hp_cake_window"] = False
        else:
            recovery_context["tower_crash_kill_window"] = False
        recovery_context["low_hp_value_trade_window"] = low_hp_value_trade_window
        tower_edge_poke_window = bool(
            raw_tower_edge_poke_window
            and (
                not recovery_context["recovery_commit"]
                or lane_context["enemy_front_count"] == 0
                or ally_near_enemy_tower > 0
                or main["enemy_tower_hp"] <= 0.45
            )
        )
        if tower_window or tower_edge_poke_window or lane_context["lane_clear_priority"]:
            recovery_context["safe_recall_window"] = False
            if tower_window or tower_edge_poke_window or lane_context["lane_clear_priority"]:
                recovery_context["recovery_commit"] = False
        post_kill_context = self._post_kill_context(
            frame_no=frame_no,
            kill_delta=post_kill_delta,
            main=main,
            lane_context=lane_context,
            neutral_context=neutral_context,
            recovery_context=recovery_context,
            tower_stage_window=tower_stage_window,
            tower_window=tower_window,
            tower_edge_poke_window=tower_edge_poke_window,
            hero_can_hit_tower=hero_can_hit_tower,
            no_wave_dive=no_wave_dive,
            dist_own_tower_now=dist_own_tower_now,
            dist_own_tower_last=dist_own_tower_last,
            dist_enemy_tower_now=dist_enemy_tower_now,
            dist_enemy_tower_last=dist_enemy_tower_last,
        )
        luban_baseline_pressure = bool(
            own_is_luban
            and enemy_visible
            and main["enemy_hp"] > 0.0
            and main["hero_hp"] >= 0.42
            and not recovery_context["recovery_commit"]
            and not lane_context["own_tower_defend_clear"]
            and not no_wave_dive
            and (
                enemy_in_attack_range
                or (frame_no <= BASELINE_OPENING_PRESSURE_END_FRAME and lane_context["enemy_wave_near_self"])
                or matchup_context["luban_vs_direnjie"]
            )
        )
        direnjie_early_develop_window = bool(
            own_is_direnjie
            and frame_no <= 6500.0
            and (
                lane_context["enemy_wave_near_self"]
                or lane_context["own_tower_defend_clear"]
                or enemy_near_own_tower > 0
            )
            and not tower_window
            and not tower_edge_poke_window
            and not tower_crash_kill_window
            and not low_hp_value_trade_window
        )
        direnjie_safe_farm_window = bool(
            direnjie_early_develop_window
            and (
                matchup_context["direnjie_vs_luban"]
                or main["hero_hp"] < 0.66
                or lane_context["own_tower_defend_clear"]
                or enemy_near_own_tower > 0
            )
            and (
                lane_context["own_tower_defend_clear"]
                or recovery_context.get("low_hp_hold_under_tower_window")
                or not enemy_visible
                or main["enemy_hp"] >= main["hero_hp"] - 0.08
                or (took_hero_hurt and main["enemy_hp"] > main["hero_hp"] + 0.08)
            )
        )
        vision_probe_window = bool(
            enemy_recent_missing
            and main["hero_hp"] >= 0.48
            and not lane_context["enemy_wave_near_self"]
            and not recovery_context["recovery_commit"]
            and not no_wave_dive
        )
        direnjie_enemy_poke_window = bool(
            own_is_direnjie
            and enemy_visible
            and dist_enemy_hero_now <= attack_range * 1.20
            and main["hero_hp"] > 0.30
            and not recovery_context["recovery_commit"]
        )
        direnjie_skill1_poke_window = bool(
            own_is_direnjie
            and (
                direnjie_enemy_poke_window
                or tower_edge_poke_window
                or lane_context["lane_clear_priority"]
                or lane_context["last_hit_window"]
            )
        )
        direnjie_skill2_escape_window = bool(
            own_is_direnjie
            and (
                (took_hero_hurt and (main["hero_hp"] < 0.48 or hurt_by_delta >= 360.0))
                or hurt_by_delta >= 720.0
                or (main["hero_hp"] < 0.36 and dist_enemy_hero_now <= attack_range * 1.35)
                or recovery_context["recovery_commit"]
                or no_wave_dive
                or tower_edge_overstep
            )
        )
        combo_start_window = bool(
            enemy_visible
            and main["enemy_hp"] > 0.0
            and main["hero_hp"] >= 0.38
            and dist_enemy_hero_now <= attack_range * 1.25
            and not direnjie_skill2_escape_window
            and not recovery_context["recovery_commit"]
            and not no_wave_dive
            and (
                main["enemy_hp"] <= 0.62
                or main["hero_hp"] >= main["enemy_hp"] - 0.06
                or enemy_in_attack_range
                or tower_edge_poke_window
                or tower_window
            )
        )
        combo_chain_window = bool(
            previous_action_head in HERO_SKILL_BUTTONS
            and enemy_visible
            and main["enemy_hp"] > 0.0
            and dist_enemy_hero_now <= attack_range * 1.35
            and not direnjie_skill2_escape_window
        )
        direnjie_ult_control_window = bool(
            own_is_direnjie
            and enemy_visible
            and dist_enemy_hero_now <= attack_range * 1.35
            and main["hero_hp"] >= 0.34
            and main["skill3_ready"] > 0.0
            and not recovery_context["recovery_commit"]
            and not no_wave_dive
            and not direnjie_skill2_escape_window
            and (
                0.0 < main["enemy_hp"] <= 0.72
                or main["level"] >= 4.0
                or main["money_cnt"] >= 1200.0
                or tower_edge_poke_window
                or combo_start_window
            )
        )
        direnjie_ult_kill_window = bool(
            own_is_direnjie
            and enemy_visible
            and dist_enemy_hero_now <= attack_range * 1.22
            and main["hero_hp"] > 0.25
            and (
                direnjie_ult_control_window
                or 0.0 < main["enemy_hp"] <= 0.50
                or tower_crash_kill_window
                or main["hero_hp"] > main["enemy_hp"] + 0.18
                or (matchup_context["direnjie_vs_luban"] and main["enemy_hp"] <= 0.58 and main["hero_hp"] >= 0.36)
            )
        )

        safety = 0.0
        if low_hp:
            safety -= 0.04
            if dist_own_tower_now < dist_own_tower_last:
                safety += 0.05
            if dist_enemy_tower_now > dist_enemy_tower_last:
                safety += 0.04
        if no_wave_dive:
            safety -= 0.08
            if low_hp:
                safety -= 0.08
        if tower_edge_overstep:
            safety -= 0.07
            if took_hero_hurt:
                safety -= 0.06

        low_hp_retreat = 0.0
        if low_hp:
            if recovery_context.get("low_hp_commit_trade_window"):
                if target_enemy_action or dealt_hero_hurt:
                    low_hp_retreat += 0.08
                elif target_soldier_action or target_monster_action:
                    low_hp_retreat -= 0.06
            elif recovery_context.get("low_hp_hold_under_tower_window"):
                if target_soldier_action:
                    low_hp_retreat += 0.06
                elif target_enemy_action:
                    low_hp_retreat += 0.03
                elif dist_own_tower_now < dist_own_tower_last:
                    low_hp_retreat += 0.02
                else:
                    low_hp_retreat -= 0.02
            elif recovery_context.get("low_hp_grass_ambush_window"):
                if target_enemy_action or dealt_hero_hurt:
                    low_hp_retreat += 0.06
                elif main["is_in_grass"]:
                    low_hp_retreat += 0.03
                else:
                    low_hp_retreat -= 0.03
            elif dist_own_tower_now < dist_own_tower_last:
                low_hp_retreat += 0.10
            if recovery_context.get("enemy_weaker_window"):
                low_hp_retreat -= 0.10
            if dist_enemy_tower_now > dist_enemy_tower_last:
                low_hp_retreat += 0.08
            if dist_enemy_tower_now < dist_enemy_tower_last and no_wave_dive:
                low_hp_retreat -= 0.12

        wave_balance = (ally_near_enemy_tower - enemy_near_own_tower) / 4.0
        safe_push = 0.0
        if ally_near_enemy_tower > 0 and main["hero_hp"] >= LOW_HP_OBJECTIVE_MIN_HP:
            tower_span = max(_dist(main["own_tower"], main["enemy_tower"]), 1.0)
            safe_push = (dist_enemy_tower_last - dist_enemy_tower_now) / tower_span
        lane_pressure = 0.6 * wave_balance + 0.4 * safe_push
        own_tower_low = main["own_tower_present"] and 0.0 < main["own_tower_hp"] <= 0.45
        own_tower_critical = main["own_tower_present"] and 0.0 < main["own_tower_hp"] <= 0.28
        tower_trade_deficit = (
            main["own_tower_present"]
            and main["enemy_tower_present"]
            and main["own_tower_hp"] + 0.18 < main["enemy_tower_hp"]
        )
        tower_defense_pressure = 0.0
        if own_tower_delta < 0:
            tower_defense_pressure += own_tower_delta * (1.0 + late_phase + (0.65 if own_tower_low else 0.0))
        if own_tower_critical and enemy_near_own_tower > 0:
            if target_soldier_action or lane_context["lane_clear_priority"]:
                tower_defense_pressure += 0.050
            elif target_enemy_action or dealt_hero_hurt:
                tower_defense_pressure += 0.030
            else:
                tower_defense_pressure -= 0.055
        if tower_trade_deficit and enemy_near_own_tower == 0 and ally_near_enemy_tower == 0:
            if dist_enemy_tower_now < dist_enemy_tower_last:
                tower_defense_pressure += 0.025
            else:
                tower_defense_pressure -= 0.020
        safe_tower_pressure = 0.0
        if enemy_tower_delta > 0 and ally_near_enemy_tower > 0 and main["hero_hp"] >= LOW_HP_OBJECTIVE_MIN_HP:
            safe_tower_pressure = enemy_tower_delta * (1.0 + late_phase)
        elif no_wave_dive and enemy_tower_delta <= 0:
            safe_tower_pressure = -0.04

        tower_window_conversion = self._calc_tower_window_conversion(
            enemy_tower_delta=enemy_tower_delta,
            tower_window=tower_window,
            hero_can_hit_tower=hero_can_hit_tower,
            attack_tower_action=attack_tower_action,
            late_phase=late_phase,
        )
        missed_tower_window = self._calc_missed_tower_window(
            enemy_tower_delta=enemy_tower_delta,
            tower_window=tower_window,
            hero_can_hit_tower=hero_can_hit_tower,
            attack_tower_action=attack_tower_action,
            enemy_in_attack_range=enemy_in_attack_range,
            dealt_hero_hurt=dealt_hero_hurt,
            took_hero_hurt=took_hero_hurt,
            low_hp=low_hp,
            enemy_weaker_window=recovery_context["enemy_weaker_window"],
        )
        late_tower_urgency = self._calc_late_tower_urgency(
            frame_no=frame_no,
            enemy_tower_delta=enemy_tower_delta,
            enemy_tower_hp=main["enemy_tower_hp"],
            tower_window=tower_window,
            hero_can_hit_tower=hero_can_hit_tower,
            hero_hp=main["hero_hp"],
            late_phase=late_phase,
        )
        tower_edge_poke = self._calc_tower_edge_poke(
            tower_edge_poke_window=tower_edge_poke_window,
            tower_edge_overstep=tower_edge_overstep,
            enemy_under_tower=enemy_under_tower,
            action_head=action_head,
            action_target=action_target,
            enemy_tower_delta=enemy_tower_delta,
            dealt_hero_hurt=dealt_hero_hurt,
            hurt_by_delta=hurt_by_delta,
            took_hero_hurt=took_hero_hurt,
            dist_enemy_tower_now=dist_enemy_tower_now,
            dist_enemy_tower_last=dist_enemy_tower_last,
            pressure_outer_range=pressure_outer_range,
            low_hp=low_hp,
            no_wave_dive=no_wave_dive,
            tower_wave_lost_reset=tower_wave_lost_reset,
            attack_tower_action=attack_tower_action,
        )
        post_clear_idle = 0.0
        if (
            self.post_clear_steps > 0
            and not low_hp
            and not lane_context["lane_clear_priority"]
            and not tower_window
            and not attack_tower_action
            and not target_enemy_action
            and action_head not in SKILL_BUTTONS
            and move_delta < 180.0
        ):
            post_clear_idle = -0.055
        elif (
            self.post_clear_steps > 0
            and enemy_visible
            and not low_hp
            and not lane_context["lane_clear_priority"]
            and dist_enemy_hero_now <= attack_range * 1.35
            and not target_enemy_action
            and action_head not in SKILL_BUTTONS
            and move_delta < 260.0
        ):
            post_clear_idle = -0.035
        elif (
            self.post_clear_steps > 0
            and (attack_tower_action or target_enemy_action or dealt_hero_hurt or tower_window or main["is_in_grass"])
        ):
            post_clear_idle = 0.02

        attack_move_rhythm = self._calc_attack_move_rhythm(
            action_head=action_head,
            move_delta=move_delta,
            enemy_in_attack_range=enemy_in_attack_range,
            dealt_hero_hurt=dealt_hero_hurt,
            took_hero_hurt=took_hero_hurt,
        )
        spacing_control = self._calc_spacing_control(
            action_head=action_head,
            dist_enemy_hero_now=dist_enemy_hero_now,
            dist_enemy_hero_last=dist_enemy_hero_last,
            dist_own_tower_now=dist_own_tower_now,
            dist_own_tower_last=dist_own_tower_last,
            attack_range=attack_range,
            enemy_visible=enemy_visible,
            low_hp=low_hp,
            took_hero_hurt=took_hero_hurt,
        )
        post_skill_attack = self._calc_post_skill_attack(
            current_action_head=action_head,
            previous_action_head=self._action_head(self.last_action),
            dealt_hero_hurt=dealt_hero_hurt,
            enemy_tower_delta=enemy_tower_delta,
            enemy_in_attack_range=enemy_in_attack_range,
            tower_window=bool(tower_window or tower_edge_poke_window),
            move_delta=move_delta,
            low_hp=low_hp,
            combo_chain_window=combo_chain_window,
        )
        skill_discipline = self._calc_skill_discipline(
            action_head=action_head,
            action_target=action_target,
            hero_config_id=main["hero_config_id"],
            enemy_visible=enemy_visible,
            enemy_in_attack_range=enemy_in_attack_range,
            target_enemy_action=target_enemy_action,
            dealt_hero_hurt=dealt_hero_hurt,
            low_enemy_minions=sum(1 for minion in main["enemy_minions"] if _hp_ratio(minion) < 0.35),
            enemy_minion_clear_delta=enemy_minion_clear_delta,
            money_delta=money_delta,
            exp_delta=exp_delta,
            enemy_tower_delta=enemy_tower_delta,
            no_wave_dive=no_wave_dive,
            lane_clear_window=lane_context["lane_clear_priority"],
            last_hit_window=lane_context["last_hit_window"],
            defensive_skill_window=direnjie_skill2_escape_window,
            pressure_window=bool(tower_window or tower_edge_poke_window),
            combo_start_window=combo_start_window,
            combo_chain_window=combo_chain_window,
        )
        lane_clear = self._calc_lane_clear(
            lane_context=lane_context,
            target_soldier_action=target_soldier_action,
            target_enemy_action=target_enemy_action,
            money_delta=money_delta,
            exp_delta=exp_delta,
            action_head=action_head,
            recovery_context=recovery_context,
            took_hero_hurt=took_hero_hurt,
            enemy_in_attack_range=enemy_in_attack_range,
            low_hp=low_hp,
            enemy_near_own_tower=enemy_near_own_tower,
            no_wave_dive=no_wave_dive,
            post_clear_active=self.post_clear_steps > 0,
        )
        over_wave_fight = self._calc_over_wave_fight(
            lane_context=lane_context,
            action_head=action_head,
            target_enemy_action=target_enemy_action,
            target_soldier_action=target_soldier_action,
            tower_window=tower_window,
            low_hp=low_hp,
            enemy_hp=main["enemy_hp"],
            took_hero_hurt=took_hero_hurt,
            tower_crash_kill_window=tower_crash_kill_window,
            low_hp_value_trade_window=low_hp_value_trade_window,
            post_clear_active=self.post_clear_steps > 0,
        )
        neutral_resource = self._calc_neutral_resource(
            neutral_context=neutral_context,
            target_monster_action=target_monster_action,
            money_delta=money_delta,
            recovery_context=recovery_context,
        )
        recovery_supply = self._calc_recovery_supply(
            recovery_context=recovery_context,
            action_head=action_head,
            target_enemy_action=target_enemy_action,
            target_soldier_action=target_soldier_action,
            target_monster_action=target_monster_action,
            own_hp_delta=main["hero_hp"] - self.last_main["hero_hp"],
            low_hp=low_hp,
            tower_crash_kill_window=tower_crash_kill_window,
            enemy_weaker_window=recovery_context["enemy_weaker_window"],
        )
        recall_discipline = self._calc_recall_discipline(
            recovery_context=recovery_context,
            action_head=action_head,
            low_hp=low_hp,
            target_enemy_action=target_enemy_action,
            target_soldier_action=target_soldier_action,
            target_monster_action=target_monster_action,
            tower_crash_kill_window=tower_crash_kill_window,
        )
        skill_tower_waste = self._calc_skill_tower_waste(
            action_head=action_head,
            action_target=action_target,
            hero_config_id=main["hero_config_id"],
            tower_edge_poke_window=tower_edge_poke_window,
            enemy_tower_delta=enemy_tower_delta,
            dealt_hero_hurt=dealt_hero_hurt,
        )
        passive_followup = self._calc_passive_followup(
            current_action_head=action_head,
            previous_action_head=self._action_head(self.last_action),
            action_target=action_target,
            hero_config_id=main["hero_config_id"],
            dealt_hero_hurt=dealt_hero_hurt,
            enemy_tower_delta=enemy_tower_delta,
            low_hp=low_hp,
        )
        if luban_baseline_pressure:
            if target_enemy_action or dealt_hero_hurt or action_head in LUBAN_PRESSURE_SKILLS:
                passive_followup += 0.030
            elif target_soldier_action and not dealt_hero_hurt:
                passive_followup -= 0.025
        hero_skill_pressure = self._calc_hero_skill_pressure(
            hero_config_id=main["hero_config_id"],
            enemy_hero_config_id=enemy["hero_config_id"],
            current_action_head=action_head,
            previous_action_head=previous_action_head,
            action_target=action_target,
            tower_edge_poke_window=tower_edge_poke_window,
            tower_window=tower_window,
            target_enemy_action=target_enemy_action,
            attack_tower_action=attack_tower_action,
            enemy_tower_delta=enemy_tower_delta,
            dealt_hero_hurt=dealt_hero_hurt,
            hurt_by_delta=hurt_by_delta,
            took_hero_hurt=took_hero_hurt,
            low_hp=low_hp,
            enemy_in_attack_range=enemy_in_attack_range,
            no_wave_dive=no_wave_dive,
            tower_edge_overstep=tower_edge_overstep,
            recovery_commit=recovery_context["recovery_commit"],
            lane_clear_window=lane_context["lane_clear_priority"],
            last_hit_window=lane_context["last_hit_window"],
            direnjie_skill1_poke_window=direnjie_skill1_poke_window,
            direnjie_skill2_escape_window=direnjie_skill2_escape_window,
            direnjie_ult_kill_window=direnjie_ult_kill_window,
            ult_follow_window=self.ult_follow_steps > 0,
            combo_start_window=combo_start_window,
            combo_chain_window=combo_chain_window,
        )
        if luban_baseline_pressure:
            if target_enemy_action or dealt_hero_hurt or action_head in LUBAN_PRESSURE_SKILLS:
                hero_skill_pressure += 0.055
                if dealt_hero_hurt:
                    hp_trade += min(0.055, hurt_to_delta / 5200.0)
                if target_soldier_action and not dealt_hero_hurt:
                    lane_clear -= 0.045
                    hero_skill_pressure -= 0.025
            elif target_soldier_action:
                lane_clear -= 0.040
                hero_skill_pressure -= 0.030
        if direnjie_safe_farm_window:
            if target_soldier_action:
                lane_clear += 0.060
                recovery_supply += 0.018
                if money_delta > 0 or exp_delta > 0:
                    lane_clear += 0.035
            elif action_head == BUTTON_SKILL_1 and lane_context["lane_clear_priority"]:
                lane_clear += 0.045
                recovery_supply += 0.012
                if money_delta > 0 or exp_delta > 0:
                    lane_clear += 0.020
            elif action_head == BUTTON_SKILL_3 and lane_context["lane_clear_priority"]:
                lane_clear += 0.015
                hero_skill_pressure += 0.012
            elif target_enemy_action and not dealt_hero_hurt:
                hero_skill_pressure -= 0.055
        matchup_strategy = self._calc_matchup_strategy(
            matchup_context=matchup_context,
            action_head=action_head,
            previous_action_head=self._action_head(self.last_action),
            action_target=action_target,
            target_enemy_action=target_enemy_action,
            attack_tower_action=attack_tower_action,
            target_soldier_action=target_soldier_action,
            enemy_tower_delta=enemy_tower_delta,
            dealt_hero_hurt=dealt_hero_hurt,
            hurt_by_delta=hurt_by_delta,
            took_hero_hurt=took_hero_hurt,
            enemy_in_attack_range=enemy_in_attack_range,
            enemy_close_range=enemy_close_range,
            tower_window=tower_window,
            tower_edge_poke_window=tower_edge_poke_window,
            tower_edge_overstep=tower_edge_overstep,
            no_wave_dive=no_wave_dive,
            low_hp=low_hp,
            own_hp=main["hero_hp"],
            enemy_hp=main["enemy_hp"],
            enemy_skill1_ready=enemy["skill1_ready"],
            enemy_skill2_ready=enemy["skill2_ready"],
            enemy_skill3_ready=enemy["skill3_ready"],
            direnjie_skill1_poke_window=direnjie_skill1_poke_window,
            direnjie_skill2_escape_window=direnjie_skill2_escape_window,
            direnjie_ult_kill_window=direnjie_ult_kill_window,
            combo_start_window=combo_start_window,
            combo_chain_window=combo_chain_window,
        )
        summoner_discipline = self._calc_summoner_discipline(
            action_head=action_head,
            previous_action_head=previous_action_head,
            target_enemy_action=target_enemy_action,
            dealt_hero_hurt=dealt_hero_hurt,
            took_hero_hurt=took_hero_hurt,
            low_hp=low_hp,
            own_hp_delta=main["hero_hp"] - self.last_main["hero_hp"],
            own_hp=main["hero_hp"],
            enemy_hp=main["enemy_hp"],
            enemy_visible=enemy_visible,
            enemy_in_attack_range=enemy_in_attack_range,
            tower_window=tower_window,
            tower_crash_kill_window=tower_crash_kill_window,
            low_hp_value_trade_window=low_hp_value_trade_window,
            recovery_context=recovery_context,
        )
        hero_skill_pressure += summoner_discipline["hero_skill_pressure"]
        recovery_supply += summoner_discipline["recovery_supply"]
        recall_discipline += summoner_discipline["recall_discipline"]
        post_kill_conversion = self._calc_post_kill_conversion(
            post_kill_context=post_kill_context,
            lane_context=lane_context,
            neutral_context=neutral_context,
            recovery_context=recovery_context,
            action_head=action_head,
            target_soldier_action=target_soldier_action,
            target_monster_action=target_monster_action,
            attack_tower_action=attack_tower_action,
            enemy_tower_delta=enemy_tower_delta,
            money_delta=money_delta,
            exp_delta=exp_delta,
            low_hp=low_hp,
            no_wave_dive=no_wave_dive,
            tower_crash_kill_window=tower_crash_kill_window,
            target_enemy_action=target_enemy_action,
            dealt_hero_hurt=dealt_hero_hurt,
            kill_delta=kill_delta,
            death_delta=death_delta,
        )
        grass_reposition, grass_tag = self._calc_purposeful_grass(
            in_grass=main["is_in_grass"],
            was_in_grass=self.last_main["is_in_grass"],
            enemy_visible=enemy_visible,
            enemy_in_attack_range=enemy_in_attack_range,
            dealt_hero_hurt=dealt_hero_hurt,
            took_hero_hurt=took_hero_hurt,
            low_hp=low_hp,
            tower_window=tower_window,
            hero_can_hit_tower=hero_can_hit_tower,
            no_wave_dive=no_wave_dive,
            dist_enemy_hero_now=dist_enemy_hero_now,
            dist_enemy_hero_last=dist_enemy_hero_last,
            dist_own_tower_now=dist_own_tower_now,
            dist_own_tower_last=dist_own_tower_last,
            dist_enemy_tower_now=dist_enemy_tower_now,
            dist_enemy_tower_last=dist_enemy_tower_last,
            close_tower_range=close_tower_range,
            enemy_tower_delta=enemy_tower_delta,
            action_head=action_head,
            tower_crash_kill_window=tower_crash_kill_window,
            target_enemy_action=target_enemy_action,
            attack_tower_action=attack_tower_action,
            tower_edge_poke_window=tower_edge_poke_window,
            kill_delta=kill_delta,
            death_delta=death_delta,
            own_is_luban=own_is_luban,
            own_is_direnjie=own_is_direnjie,
            enemy_is_luban=enemy_is_luban,
            enemy_is_direnjie=enemy_is_direnjie,
            own_hp=main["hero_hp"],
            enemy_hp=main["enemy_hp"],
        )
        if vision_probe_window:
            if main["is_in_grass"] or action_head in {BUTTON_MOVE, BUTTON_SKILL_1, BUTTON_SKILL_3}:
                grass_reposition += 0.030
            elif action_head not in {BUTTON_MOVE, BUTTON_ATTACK}:
                grass_reposition -= 0.020
        forward_pressure = self._calc_forward_pressure(
            enemy_visible=enemy_visible,
            enemy_in_attack_range=enemy_in_attack_range,
            dist_enemy_hero_now=dist_enemy_hero_now,
            dist_enemy_hero_last=dist_enemy_hero_last,
            dist_enemy_tower_now=dist_enemy_tower_now,
            dist_enemy_tower_last=dist_enemy_tower_last,
            attack_range=attack_range,
            move_delta=move_delta,
            action_head=action_head,
            target_enemy_action=target_enemy_action,
            dealt_hero_hurt=dealt_hero_hurt,
            low_hp=low_hp,
            no_wave_dive=no_wave_dive,
            lane_clear_priority=lane_context["lane_clear_priority"],
            own_tower_defend_clear=lane_context["own_tower_defend_clear"],
            tower_window=tower_window,
            recovery_commit=recovery_context["recovery_commit"],
            enemy_hp=main["enemy_hp"],
            own_hp=main["hero_hp"],
        )
        if luban_baseline_pressure and (target_enemy_action or dealt_hero_hurt or action_head in LUBAN_PRESSURE_SKILLS):
            forward_pressure += 0.025
        if direnjie_safe_farm_window and (target_soldier_action or dist_own_tower_now <= dist_own_tower_last + 120.0):
            forward_pressure -= 0.006
        if vision_probe_window and (main["is_in_grass"] or dist_enemy_tower_now < dist_enemy_tower_last - 120.0):
            forward_pressure += 0.015
        if post_kill_context["active"] and own_is_luban and action_head == BUTTON_SKILL_2:
            hero_skill_pressure += 0.055
            forward_pressure += 0.020
            grass_reposition += 0.018
        main_task = self._main_task_priority(
            lane_context=lane_context,
            recovery_context=recovery_context,
            post_kill_context=post_kill_context,
            tower_window=tower_window,
            tower_edge_poke_window=tower_edge_poke_window,
            tower_wave_lost_reset=tower_wave_lost_reset,
            tower_crash_kill_window=tower_crash_kill_window,
            low_hp_value_trade_window=low_hp_value_trade_window,
            post_clear_active=self.post_clear_steps > 0,
            enemy_visible=enemy_visible,
            enemy_in_attack_range=enemy_in_attack_range,
            took_hero_hurt=took_hero_hurt,
            dealt_hero_hurt=dealt_hero_hurt,
            own_is_luban=own_is_luban,
            own_is_direnjie=own_is_direnjie,
            enemy_is_luban=enemy_is_luban,
            enemy_is_direnjie=enemy_is_direnjie,
            action_head=action_head,
            target_enemy_action=target_enemy_action,
            dist_enemy_hero_now=dist_enemy_hero_now,
            attack_range=attack_range,
            own_hp=main["hero_hp"],
            enemy_hp=main["enemy_hp"],
            frame_no=frame_no,
        )
        self._record_episode_stats(
            enemy_tower_delta=enemy_tower_delta,
            tower_window=tower_window,
            attack_tower_action=attack_tower_action,
            missed_tower_window=missed_tower_window,
            late_tower_urgency=late_tower_urgency,
            no_wave_dive=no_wave_dive,
            previous_action_head=self._action_head(self.last_action),
            current_action_head=action_head,
            post_skill_attack=post_skill_attack,
            grass_tag=grass_tag,
            lane_context=lane_context,
            neutral_context=neutral_context,
            recovery_context=recovery_context,
            lane_clear=lane_clear,
            over_wave_fight=over_wave_fight,
            neutral_resource=neutral_resource,
            recovery_supply=recovery_supply,
            recall_discipline=recall_discipline,
            skill_tower_waste=skill_tower_waste,
            passive_followup=passive_followup,
            hero_skill_pressure=hero_skill_pressure,
            hero_config_id=main["hero_config_id"],
            matchup_context=matchup_context,
            matchup_strategy=matchup_strategy,
            post_kill_context=post_kill_context,
            post_kill_conversion=post_kill_conversion,
            target_soldier_action=target_soldier_action,
            target_monster_action=target_monster_action,
            action_head=action_head,
            tower_crash_kill_window=tower_crash_kill_window,
            target_enemy_action=target_enemy_action,
            kill_delta=kill_delta,
            death_delta=death_delta,
            tower_edge_poke_window=tower_edge_poke_window,
            tower_edge_poke=tower_edge_poke,
            tower_edge_overstep=tower_edge_overstep,
            tower_wave_lost_reset=tower_wave_lost_reset,
            dealt_hero_hurt=dealt_hero_hurt,
            ep_context=ep_context,
        )

        reward_components = {
            "tower_hp_point": _clip(enemy_tower_delta),
            "own_tower_hp_point": _clip(own_tower_delta),
            "hp_trade": _clip(hp_trade),
            "economy": _clip(economy_progress),
            "exp": 0.0,
            "kill": _clip(kill_delta),
            "death": _clip(
                -death_delta
                * (
                    0.55
                    if tower_crash_kill_window and kill_delta > 0
                    else 0.65
                    if low_hp_value_trade_window and (kill_delta > 0 or enemy_tower_delta > 0)
                    else 1.0
                )
            ),
            "safe_tower_pressure": _clip(safe_tower_pressure + post_clear_idle),
            "tower_edge_poke": _clip(tower_edge_poke),
            "tower_window_conversion": _clip(tower_window_conversion),
            "missed_tower_window": _clip(missed_tower_window),
            "late_tower_urgency": _clip(late_tower_urgency),
            "tower_defense_pressure": _clip(tower_defense_pressure),
            "unsafe_tower_dive": _clip(safety),
            "low_hp_retreat": _clip(low_hp_retreat),
            "lane_pressure": _clip(lane_pressure),
            "lane_clear": _clip(lane_clear),
            "over_wave_fight": _clip(over_wave_fight),
            "neutral_resource": _clip(neutral_resource),
            "recovery_supply": _clip(recovery_supply),
            "recall_discipline": _clip(recall_discipline),
            "attack_move_rhythm": _clip(attack_move_rhythm),
            "spacing_control": _clip(spacing_control),
            "post_skill_attack": _clip(post_skill_attack),
            "skill_discipline": _clip(skill_discipline),
            "ep_discipline": _clip(ep_context["ep_discipline"]),
            "skill_tower_waste": _clip(skill_tower_waste),
            "passive_followup": _clip(passive_followup),
            "hero_skill_pressure": _clip(hero_skill_pressure),
            "matchup_strategy": _clip(matchup_strategy),
            "post_kill_conversion": _clip(post_kill_conversion),
            "grass_reposition": _clip(grass_reposition),
            "forward": _clip(forward_pressure),
        }
        reward_components = self._apply_main_task_priority(
            reward_components=reward_components,
            main_task=main_task,
            action_head=action_head,
            action_target=action_target,
            target_soldier_action=target_soldier_action,
            target_enemy_action=target_enemy_action,
            attack_tower_action=attack_tower_action,
            tower_edge_poke_window=tower_edge_poke_window,
            post_kill_context=post_kill_context,
            lane_context=lane_context,
            recovery_context=recovery_context,
        )
        return self._apply_matchup_profile(reward_components, matchup_context)

    def get_episode_stats(self):
        stats = dict(self.episode_stats)
        stats["tower_damage"] = round(stats["tower_damage"], 5)
        return stats

    def _matchup_key(self, matchup_context):
        if matchup_context.get("luban_vs_luban"):
            return "112v112"
        if matchup_context.get("luban_vs_direnjie"):
            return "112v133"
        if matchup_context.get("direnjie_vs_luban"):
            return "133v112"
        if matchup_context.get("direnjie_vs_direnjie"):
            return "133v133"
        return "default"

    def _apply_matchup_profile(self, reward_components, matchup_context):
        phase = str(getattr(GameConfig, "TRAINING_PHASE", "base")).strip().lower()
        if phase not in {"r8", "tactic", "profile", "strategy", "r5_4"}:
            return reward_components
        scale_fn = getattr(GameConfig, "matchup_profile_scales", None)
        if not callable(scale_fn):
            return reward_components

        scales = scale_fn(self._matchup_key(matchup_context))
        if not scales:
            return reward_components

        scaled = dict(reward_components)
        for reward_name, scale in scales.items():
            if reward_name in scaled:
                scaled[reward_name] = _clip(scaled[reward_name] * scale)
        return scaled

    def _main_task_priority(
        self,
        lane_context,
        recovery_context,
        post_kill_context,
        tower_window,
        tower_edge_poke_window,
        tower_wave_lost_reset,
        tower_crash_kill_window,
        low_hp_value_trade_window,
        post_clear_active,
        enemy_visible,
        enemy_in_attack_range,
        took_hero_hurt,
        dealt_hero_hurt,
        own_is_luban,
        own_is_direnjie,
        enemy_is_luban,
        enemy_is_direnjie,
        action_head,
        target_enemy_action,
        dist_enemy_hero_now,
        attack_range,
        own_hp,
        enemy_hp,
        frame_no,
    ):
        relative_hero_threat = bool(
            enemy_visible
            and dist_enemy_hero_now <= attack_range * 1.25
            and (
                took_hero_hurt
                or enemy_hp >= own_hp + 0.10
                or (own_hp < 0.48 and enemy_hp > 0.28)
            )
        )
        hero_contesting = bool(
            enemy_visible
            and (enemy_in_attack_range or relative_hero_threat)
            and (took_hero_hurt or dealt_hero_hurt or target_enemy_action)
        )
        defend_clear_first = bool(lane_context.get("own_tower_defend_clear") or lane_context.get("own_tower_low_hp"))
        luban_pressure_first = bool(
            own_is_luban
            and enemy_visible
            and enemy_hp > 0.0
            and own_hp >= 0.42
            and not defend_clear_first
            and not recovery_context.get("recovery_commit")
            and (
                enemy_in_attack_range
                or dealt_hero_hurt
                or target_enemy_action
                or (frame_no <= BASELINE_OPENING_PRESSURE_END_FRAME and lane_context.get("enemy_wave_near_self"))
                or enemy_is_direnjie
            )
        )
        direnjie_safe_farm_first = bool(
            own_is_direnjie
            and frame_no <= 6500.0
            and (lane_context.get("enemy_wave_near_self") or defend_clear_first)
            and (
                enemy_is_luban
                or own_hp < 0.66
                or defend_clear_first
            )
            and (
                defend_clear_first
                or recovery_context.get("low_hp_hold_under_tower_window")
                or not enemy_visible
                or enemy_hp >= own_hp - 0.08
                or (took_hero_hurt and enemy_hp > own_hp + 0.08)
            )
        )
        hero_interrupts_clear = bool(
            lane_context["lane_clear_priority"]
            and (hero_contesting or relative_hero_threat or luban_pressure_first)
            and not defend_clear_first
            and not recovery_context.get("low_hp_hold_under_tower_window")
        )
        direnjie_ult_follow = bool(
            own_is_direnjie
            and self.ult_follow_steps > 0
            and enemy_visible
            and dist_enemy_hero_now <= attack_range * 1.20
        )
        direnjie_luban_pressure_first = bool(
            own_is_direnjie
            and enemy_is_luban
            and enemy_visible
            and not defend_clear_first
            and not recovery_context.get("recovery_commit")
            and not direnjie_safe_farm_first
            and own_hp >= 0.34
            and dist_enemy_hero_now <= attack_range * 1.25
            and (
                enemy_hp <= 0.58
                or own_hp >= enemy_hp - 0.04
                or dealt_hero_hurt
                or target_enemy_action
                or action_head in DIRENJIE_PRESSURE_SKILLS
            )
        )
        if post_kill_context.get("active"):
            if tower_window:
                return "tower_hit"
            if tower_edge_poke_window:
                return "tower_stage"
            return "clear_wave"
        if tower_crash_kill_window or low_hp_value_trade_window or direnjie_ult_follow:
            return "commit_kill"
        if tower_wave_lost_reset:
            return "short_reset"
        if direnjie_safe_farm_first:
            return "clear_wave"
        if lane_context["lane_clear_priority"] and defend_clear_first:
            return "clear_wave"
        if tower_window:
            return "tower_hit"
        if tower_edge_poke_window:
            return "tower_stage"
        if recovery_context["safe_recall_window"] or recovery_context["recovery_commit"]:
            return "short_reset"
        if direnjie_luban_pressure_first:
            return "hero_contest"
        if luban_pressure_first:
            return "hero_contest"
        if post_clear_active and enemy_visible and (enemy_in_attack_range or relative_hero_threat):
            return "hero_contest"
        if hero_interrupts_clear:
            return "hero_contest"
        if lane_context["lane_clear_priority"]:
            return "clear_wave"
        if enemy_visible and (enemy_in_attack_range or relative_hero_threat) and not recovery_context["recovery_commit"]:
            return "hero_contest"
        return "clear_wave"

    def _apply_main_task_priority(
        self,
        reward_components,
        main_task,
        action_head,
        action_target,
        target_soldier_action,
        target_enemy_action,
        attack_tower_action,
        tower_edge_poke_window,
        post_kill_context,
        lane_context,
        recovery_context,
    ):
        adjusted = dict(reward_components)
        clearing_wave = target_soldier_action or (
            lane_context["lane_clear_priority"] and action_head in {BUTTON_SKILL_1, BUTTON_SKILL_3}
        )
        pressure_skill = action_head in (LUBAN_PRESSURE_SKILLS | DIRENJIE_PRESSURE_SKILLS)
        hero_pressure_action = target_enemy_action or pressure_skill
        reset_move = action_head == BUTTON_MOVE and (
            recovery_context.get("moved_toward_own_tower") or recovery_context.get("moved_toward_cake")
        )
        luban_pressure_recover = bool(
            reward_components.get("hero_skill_pressure", 0.0) > 0.045
            and reward_components.get("matchup_strategy", 0.0) >= 0.0
            and not lane_context.get("own_tower_defend_clear")
            and not recovery_context.get("low_hp_hold_under_tower_window")
            and not recovery_context.get("recovery_commit")
        )

        if main_task == "clear_wave":
            if clearing_wave:
                adjusted["lane_clear"] = _clip(adjusted.get("lane_clear", 0.0) + (0.090 if post_kill_context.get("active") else 0.045))
            elif hero_pressure_action:
                if luban_pressure_recover:
                    adjusted["hero_skill_pressure"] = _clip(adjusted.get("hero_skill_pressure", 0.0) + 0.045)
                    adjusted["forward"] = _clip(adjusted.get("forward", 0.0) + 0.025)
                    adjusted["lane_clear"] = min(adjusted.get("lane_clear", 0.0), 0.035)
                else:
                    adjusted["over_wave_fight"] = _clip(adjusted.get("over_wave_fight", 0.0) - 0.055)
                    adjusted["hero_skill_pressure"] = min(adjusted.get("hero_skill_pressure", 0.0), 0.035)
            if post_kill_context.get("active"):
                if not clearing_wave:
                    adjusted["post_kill_conversion"] = _clip(adjusted.get("post_kill_conversion", 0.0) - 0.075)
                adjusted["neutral_resource"] = min(adjusted.get("neutral_resource", 0.0), -0.035 if action_target == TARGET_MONSTER else 0.0)
                adjusted["recovery_supply"] = min(adjusted.get("recovery_supply", 0.0), 0.0)
                adjusted["recall_discipline"] = min(adjusted.get("recall_discipline", 0.0), 0.0)
            if luban_pressure_recover:
                adjusted["tower_edge_poke"] = _clip(max(adjusted.get("tower_edge_poke", 0.0), 0.045))
                adjusted["matchup_strategy"] = _clip(max(adjusted.get("matchup_strategy", 0.0), 0.055))
            else:
                adjusted["tower_edge_poke"] = min(adjusted.get("tower_edge_poke", 0.0), 0.02)
                adjusted["matchup_strategy"] = min(adjusted.get("matchup_strategy", 0.0), 0.04)
        elif main_task == "hero_contest":
            if hero_pressure_action:
                adjusted["hero_skill_pressure"] = _clip(adjusted.get("hero_skill_pressure", 0.0) + 0.080)
                adjusted["forward"] = _clip(adjusted.get("forward", 0.0) + 0.030)
                adjusted["lane_clear"] = min(adjusted.get("lane_clear", 0.0), 0.02)
            elif clearing_wave:
                adjusted["lane_clear"] = _clip(adjusted.get("lane_clear", 0.0) - 0.07)
                adjusted["over_wave_fight"] = _clip(adjusted.get("over_wave_fight", 0.0) - 0.035)
                adjusted["forward"] = _clip(adjusted.get("forward", 0.0) - 0.025)
        elif main_task == "tower_hit":
            if attack_tower_action:
                adjusted["tower_window_conversion"] = _clip(
                    adjusted.get("tower_window_conversion", 0.0) + (0.10 if post_kill_context.get("active") else 0.04)
                )
            elif hero_pressure_action and tower_edge_poke_window:
                adjusted["tower_edge_poke"] = _clip(adjusted.get("tower_edge_poke", 0.0) + 0.025)
            elif clearing_wave:
                adjusted["lane_clear"] = min(adjusted.get("lane_clear", 0.0), 0.025)
            if post_kill_context.get("active"):
                if not attack_tower_action:
                    adjusted["post_kill_conversion"] = _clip(adjusted.get("post_kill_conversion", 0.0) - 0.090)
                    adjusted["missed_tower_window"] = _clip(adjusted.get("missed_tower_window", 0.0) - 0.050)
                adjusted["hero_skill_pressure"] = min(adjusted.get("hero_skill_pressure", 0.0), 0.055)
                adjusted["neutral_resource"] = min(adjusted.get("neutral_resource", 0.0), 0.0)
                adjusted["recovery_supply"] = min(adjusted.get("recovery_supply", 0.0), 0.0)
                adjusted["recall_discipline"] = min(adjusted.get("recall_discipline", 0.0), 0.0)
            if hero_pressure_action or attack_tower_action:
                adjusted["recovery_supply"] = min(adjusted.get("recovery_supply", 0.0), 0.02)
                adjusted["recall_discipline"] = min(adjusted.get("recall_discipline", 0.0), 0.02)
        elif main_task == "tower_stage":
            if attack_tower_action:
                adjusted["tower_window_conversion"] = _clip(adjusted.get("tower_window_conversion", 0.0) - 0.030)
                adjusted["unsafe_tower_dive"] = _clip(adjusted.get("unsafe_tower_dive", 0.0) - 0.025)
            elif hero_pressure_action and tower_edge_poke_window:
                adjusted["tower_edge_poke"] = _clip(adjusted.get("tower_edge_poke", 0.0) + 0.030)
            elif clearing_wave:
                adjusted["lane_clear"] = _clip(adjusted.get("lane_clear", 0.0) + 0.020)
            if post_kill_context.get("active"):
                adjusted["recovery_supply"] = min(adjusted.get("recovery_supply", 0.0), 0.0)
                adjusted["recall_discipline"] = min(adjusted.get("recall_discipline", 0.0), 0.0)
        elif main_task == "commit_kill":
            if hero_pressure_action:
                adjusted["hero_skill_pressure"] = _clip(adjusted.get("hero_skill_pressure", 0.0) + 0.055)
                adjusted["low_hp_retreat"] = min(adjusted.get("low_hp_retreat", 0.0), 0.02)
            elif clearing_wave or action_target == TARGET_MONSTER:
                adjusted["lane_clear"] = _clip(adjusted.get("lane_clear", 0.0) - 0.045)
                adjusted["neutral_resource"] = min(adjusted.get("neutral_resource", 0.0), 0.0)
        elif main_task == "short_reset":
            if reset_move:
                adjusted["recovery_supply"] = _clip(adjusted.get("recovery_supply", 0.0) + 0.04)
                adjusted["recall_discipline"] = _clip(adjusted.get("recall_discipline", 0.0) + 0.035)
            elif hero_pressure_action or attack_tower_action:
                adjusted["hero_skill_pressure"] = min(adjusted.get("hero_skill_pressure", 0.0), 0.0)
                adjusted["tower_edge_poke"] = min(adjusted.get("tower_edge_poke", 0.0), 0.0)
                adjusted["recall_discipline"] = _clip(adjusted.get("recall_discipline", 0.0) - 0.045)
        return adjusted

    def _count_near(self, units, anchor, tower):
        if not anchor:
            return 0
        tower_range = _safe_float(_get(tower, "attack_range", 8800.0), 8800.0)
        return sum(1 for unit in units if _dist(unit, anchor) <= tower_range)

    def _transition_action_head(self):
        action = self.pending_action if self.pending_action is not None else self.last_action
        return self._action_head(action)

    def _transition_action_target(self):
        action = self.pending_action if self.pending_action is not None else self.last_action
        if not action:
            return None
        try:
            return int(action[5])
        except (TypeError, ValueError, IndexError):
            return None

    def _action_head(self, action):
        if not action:
            return None
        try:
            return int(action[0])
        except (TypeError, ValueError, IndexError):
            return None

    def _is_soldier_target(self, action_target):
        return action_target is not None and TARGET_SOLDIER_MIN <= action_target <= TARGET_SOLDIER_MAX

    def _skill_ready_ratio(self, hero):
        slots = _slot_states(hero)
        if not slots:
            return 0.0
        return _ratio(sum(1 for slot in slots if _get(slot, "usable", False)), len(slots))

    def _skill_ready_by_slot(self, hero, slot_type):
        for slot in _slot_states(hero):
            if int(_safe_float(_get(slot, "slot_type", -1), -1)) != slot_type:
                continue
            if _get(slot, "usable", False):
                return 1.0
            cooldown = _safe_float(_get(slot, "cooldown", 0.0), 0.0)
            cooldown_max = _safe_float(_get(slot, "cooldown_max", 0.0), 0.0)
            if cooldown_max <= 0.0:
                return 0.0
            return _clip(1.0 - cooldown / cooldown_max, 0.0, 1.0)
        return 0.0

    def _is_monster(self, npc):
        actor_type = _get(npc, "actor_type")
        if isinstance(actor_type, str) and "MONSTER" in actor_type.upper():
            return True
        sub_type = _get(npc, "sub_type")
        if sub_type in {MINION_SUB_TYPE, TOWER_SPRING_SUB_TYPE, TOWER_SUB_TYPE, CRYSTAL_SUB_TYPE}:
            return False
        max_hp = _safe_float(_get(npc, "max_hp"), 0.0)
        if max_hp <= 0.0:
            return False
        camp = _get(npc, "camp")
        return camp in {0, "0", None}

    def _nearest(self, units, anchor):
        if not units or not anchor:
            return None, 99999.0
        best = min(units, key=lambda unit: _dist(unit, anchor))
        return best, _dist(best, anchor)

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
        return max(0.0, min(1.0, ((ux - ox) * vx + (uz - oz) * vz) / denom))

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

    def _lane_context(self, main, enemy, frame_no, attack_range, enemy_near_own_tower, ally_near_enemy_tower):
        hero = main["hero"]
        enemy_hero = main["enemy_hero"]
        own_tower = main["own_tower"]
        enemy_tower = main["enemy_tower"]
        hero_progress = self._lane_progress(hero, own_tower, enemy_tower)
        enemy_progress = self._lane_progress(enemy_hero, own_tower, enemy_tower)
        ally_progresses = [self._lane_progress(unit, own_tower, enemy_tower) for unit in main["ally_minions"]]
        enemy_progresses = [self._lane_progress(unit, own_tower, enemy_tower) for unit in main["enemy_minions"]]
        max_ally_progress = max(ally_progresses) if ally_progresses else hero_progress

        enemy_minion_between = False
        if hero and enemy_hero:
            low, high = sorted((hero_progress, enemy_progress))
            for minion in main["enemy_minions"]:
                p = self._lane_progress(minion, own_tower, enemy_tower)
                if low <= p <= high and self._lane_lateral_ratio(minion, own_tower, enemy_tower) < 0.55:
                    enemy_minion_between = True
                    break

        self_ahead_of_ally_wave = bool(main["ally_minions"] and hero_progress > max_ally_progress + 0.08)
        nearest_enemy_minion, nearest_enemy_minion_dist = self._nearest(main["enemy_minions"], hero)
        nearest_enemy_minion_hp = _hp_ratio(nearest_enemy_minion)
        last_hit_window = bool(
            nearest_enemy_minion
            and 0.0 < nearest_enemy_minion_hp < 0.35
            and nearest_enemy_minion_dist <= attack_range + 1800.0
        )
        early_lane_phase = frame_no < LANE_CONTROL_END_FRAME
        enemy_front_count = sum(1 for p in enemy_progresses if p >= hero_progress - 0.03)
        ally_front_count = sum(1 for p in ally_progresses if p >= hero_progress - 0.03)
        enemy_wave_near_self = nearest_enemy_minion_dist <= attack_range + 3500.0
        own_tower_low_hp = main["own_tower_present"] and 0.0 < main["own_tower_hp"] <= 0.35
        own_tower_defend_clear = bool(enemy_near_own_tower > 0 or (own_tower_low_hp and enemy_wave_near_self))
        lane_control_window = bool(
            early_lane_phase
            and main["hero_hp"] >= RECOVERY_HEALTHY_HP
            and nearest_enemy_minion
            and nearest_enemy_minion_dist <= attack_range + 2200.0
            and ally_near_enemy_tower == 0
            and enemy_near_own_tower == 0
            and not self_ahead_of_ally_wave
            and not enemy_minion_between
        )
        lane_clear_priority = bool(
            enemy_wave_near_self
            and (own_tower_defend_clear or self_ahead_of_ally_wave or enemy_minion_between)
            and ally_near_enemy_tower == 0
            and not lane_control_window
        )
        return {
            "enemy_minion_between": enemy_minion_between,
            "self_ahead_of_ally_wave": self_ahead_of_ally_wave,
            "nearest_enemy_minion_dist": nearest_enemy_minion_dist,
            "nearest_enemy_minion_hp": nearest_enemy_minion_hp,
            "last_hit_window": last_hit_window,
            "lane_control_window": lane_control_window,
            "lane_clear_priority": lane_clear_priority,
            "own_tower_defend_clear": own_tower_defend_clear,
            "own_tower_low_hp": own_tower_low_hp,
            "enemy_wave_near_self": enemy_wave_near_self,
            "enemy_front_count": enemy_front_count,
            "ally_front_count": ally_front_count,
        }

    def _monster_hp_total(self, units):
        return sum(max(0.0, _safe_float(_get(unit, "hp"), 0.0)) for unit in units)

    def _neutral_context(self, main, attack_range, lane_clear_priority):
        nearest_monster, nearest_monster_dist = self._nearest(main["monsters"], main["hero"])
        nearest_monster_hp = _hp_ratio(nearest_monster)
        last_monster_hp = self._monster_hp_total(self.last_main["monsters"])
        cur_monster_hp = self._monster_hp_total(main["monsters"])
        monster_hp_delta = max(0.0, last_monster_hp - cur_monster_hp)
        enemy_close = main["enemy_hero"] and _dist(main["hero"], main["enemy_hero"]) <= attack_range * 1.25
        monster_finish_window = bool(
            nearest_monster
            and 0.0 < nearest_monster_hp <= 0.28
            and nearest_monster_dist <= attack_range + 1800.0
            and main["hero_hp"] > 0.45
            and not enemy_close
        )
        neutral_safe_window = bool(
            nearest_monster
            and main["hero_hp"] > 0.48
            and (not lane_clear_priority or monster_finish_window)
            and not enemy_close
            and nearest_monster_dist <= 18000.0
        )
        return {
            "neutral_safe_window": neutral_safe_window,
            "monster_finish_window": monster_finish_window,
            "nearest_monster_hp": nearest_monster_hp,
            "monster_hp_delta": monster_hp_delta,
            "nearest_monster_dist": nearest_monster_dist,
        }

    def _cake_pos(self, cake):
        collider = _get(cake, "collider", {}) if cake else {}
        location = _get(collider, "location", {}) if isinstance(collider, dict) else {}
        return (_safe_float(_get(location, "x")), _safe_float(_get(location, "z")))

    def _cake_unit(self, cake):
        x, z = self._cake_pos(cake)
        return {"location": {"x": x, "z": z}}

    def _cake_dist(self, cake, unit):
        if not cake or not unit:
            return 99999.0
        cx, cz = self._cake_pos(cake)
        ux, uz = _pos(unit)
        return math.hypot(cx - ux, cz - uz)

    def _nearest_cake(self, cakes, hero):
        if not cakes or not hero:
            return None, 99999.0
        best = min(cakes, key=lambda cake: self._cake_dist(cake, hero))
        return best, self._cake_dist(best, hero)

    def _recovery_context(self, main, enemy, no_wave_dive, enemy_near_own_tower, dist_own_tower_now):
        nearest_cake, nearest_cake_dist = self._nearest_cake(main["cakes"], main["hero"])
        last_cake, last_cake_dist = self._nearest_cake(self.last_main["cakes"], self.last_main["hero"])
        cake_on_own_side = False
        if nearest_cake and main["own_tower"] and main["enemy_tower"]:
            cake_on_own_side = self._lane_progress(self._cake_unit(nearest_cake), main["own_tower"], main["enemy_tower"]) < 0.45
        enemy_dist = _dist(main["hero"], main["enemy_hero"])
        enemy_pressure = bool(main["enemy_hero"] and enemy_dist <= DEFAULT_ATTACK_RANGE * 1.25)
        hp_deficit = max(0.0, main["enemy_hp"] - main["hero_hp"])
        level_deficit = max(0.0, enemy["level"] - main["level"])
        money_deficit = max(0.0, enemy["money_cnt"] - main["money_cnt"])
        hp_delta = main["hero_hp"] - self.last_main["hero_hp"]
        relative_danger = _clip(
            1.55 * hp_deficit
            + 0.10 * min(level_deficit, 3.0)
            + 0.18 * _ratio(money_deficit, 2500.0)
            + (0.18 if enemy_pressure else 0.0)
            + (0.08 if hp_delta < -0.004 else 0.0),
            0.0,
            1.0,
        )
        safe_lane_hold = bool(
            main["hero_hp"] < 0.42
            and enemy_near_own_tower > 0
            and not enemy_pressure
            and main["enemy_hero"] is not None
            and main["enemy_hp"] <= main["hero_hp"] + 0.12
        )
        low_hp_execute_risk = bool(
            main["hero_hp"] < HARD_RECALL_HP
            and main["enemy_hp"] >= ENEMY_HEALTHY_HP
            and (enemy_pressure or relative_danger > 0.18)
        )
        enemy_weaker_window = bool(
            enemy["hero_present"]
            and 0.0 < main["enemy_hp"] + LOW_HP_RELATIVE_ADVANTAGE <= main["hero_hp"]
        )
        core_skill_down_risk = bool(
            main["hero_hp"] < 0.42
            and main["enemy_hp"] >= max(0.45, main["hero_hp"] + 0.06)
            and main["skill_ready_ratio"] < 0.45
            and enemy_pressure
        )
        low_hp_hold_under_tower_window = bool(
            main["hero_hp"] < 0.44
            and enemy_near_own_tower > 0
            and main["own_tower_present"]
            and main["enemy_hero"] is not None
            and main["enemy_hp"] <= min(0.56, main["hero_hp"] + LOW_HP_HOLD_MAX_ENEMY_ADVANTAGE)
            and relative_danger < 0.52
        )
        low_hp_grass_ambush_window = bool(
            main["hero_hp"] < SHORT_RESET_EXIT_HP
            and main["is_in_grass"]
            and enemy["hero_present"]
            and 0.0 < main["enemy_hp"] <= LOW_HP_GRASS_RESET_ENEMY_HP
            and enemy_dist <= DEFAULT_ATTACK_RANGE + LOW_HP_GRASS_AMBUSH_ENEMY_RANGE_PAD
            and not no_wave_dive
            and not low_hp_execute_risk
        )
        low_hp_commit_trade_window = bool(
            enemy["hero_present"]
            and main["hero_hp"] >= LOW_HP_TRADE_MIN_OWN_HP
            and main["hero_hp"] < SHORT_RESET_EXIT_HP
            and (
                0.0 < main["enemy_hp"] <= LOW_HP_TRADE_ENEMY_HP
                or enemy_weaker_window
            )
            and enemy_dist <= DEFAULT_ATTACK_RANGE + 2200.0
            and not no_wave_dive
        )
        recovery_trigger = (
            (low_hp_execute_risk and not low_hp_commit_trade_window and not low_hp_grass_ambush_window)
            or (core_skill_down_risk and not low_hp_commit_trade_window and not enemy_weaker_window)
            or (
                main["hero_hp"] < 0.38
                and relative_danger > 0.22
                and not low_hp_commit_trade_window
                and not enemy_weaker_window
            )
        )
        if recovery_trigger:
            self.recovery_commit_active = True
        elif main["hero_hp"] >= SHORT_RESET_EXIT_HP or not main["hero_present"]:
            self.recovery_commit_active = False
        recovery_commit = bool(self.recovery_commit_active and main["hero_hp"] < SHORT_RESET_EXIT_HP)
        if recovery_commit or low_hp_commit_trade_window:
            safe_lane_hold = False
        low_hp_cake_window = bool(
            main["hero_hp"] < 0.40
            and nearest_cake
            and cake_on_own_side
            and nearest_cake_dist <= 18000.0
            and not recovery_commit
            and not safe_lane_hold
            and not enemy_weaker_window
        )
        near_own_tower = _dist(main["hero"], main["own_tower"]) <= 13000.0
        safe_position_for_recall = near_own_tower or main["is_in_grass"]
        enemy_not_interrupting_recall = not enemy_pressure and enemy_near_own_tower == 0
        moved_toward_cake = nearest_cake and last_cake and nearest_cake_dist < last_cake_dist - 100.0
        moved_toward_own_tower = dist_own_tower_now < _dist(self.last_main["hero"], self.last_main["own_tower"]) - 100.0
        moving_to_recovery = bool(moved_toward_cake or moved_toward_own_tower)
        if recovery_commit and hp_delta <= 0.002 and not moving_to_recovery:
            self.low_hp_stall_steps = min(self.low_hp_stall_steps + 1, 16)
        elif main["hero_hp"] >= SHORT_RESET_EXIT_HP or hp_delta > 0.015 or moving_to_recovery:
            self.low_hp_stall_steps = max(0, self.low_hp_stall_steps - 2)
        else:
            self.low_hp_stall_steps = max(0, self.low_hp_stall_steps - 1)
        safe_recall_window = bool(
            safe_position_for_recall
            and enemy_not_interrupting_recall
            and not no_wave_dive
            and not safe_lane_hold
            and not low_hp_hold_under_tower_window
            and not low_hp_grass_ambush_window
            and not low_hp_commit_trade_window
            and not enemy_weaker_window
            and (
                low_hp_execute_risk
                or core_skill_down_risk
                or recovery_commit
                or (main["hero_hp"] < 0.40 and self.low_hp_stall_steps >= SHORT_RESET_STALL_LIMIT)
            )
        )
        return {
            "low_hp_cake_window": low_hp_cake_window,
            "safe_recall_window": safe_recall_window,
            "moved_toward_cake": moved_toward_cake,
            "moved_toward_own_tower": moved_toward_own_tower,
            "nearest_cake_dist": nearest_cake_dist,
            "enemy_pressure": enemy_pressure,
            "relative_danger": relative_danger,
            "safe_lane_hold": safe_lane_hold,
            "recovery_commit": recovery_commit,
            "low_hp_execute_risk": low_hp_execute_risk,
            "core_skill_down_risk": core_skill_down_risk,
            "low_hp_hold_under_tower_window": low_hp_hold_under_tower_window,
            "low_hp_grass_ambush_window": low_hp_grass_ambush_window,
            "low_hp_commit_trade_window": low_hp_commit_trade_window,
            "enemy_weaker_window": enemy_weaker_window,
            "recovery_need": _clip((SHORT_RESET_EXIT_HP - main["hero_hp"]) / SHORT_RESET_EXIT_HP, 0.0, 1.0),
            "low_hp_stall_steps": self.low_hp_stall_steps,
        }

    def _post_kill_context(
        self,
        frame_no,
        kill_delta,
        main,
        lane_context,
        neutral_context,
        recovery_context,
        tower_stage_window,
        tower_window,
        tower_edge_poke_window,
        hero_can_hit_tower,
        no_wave_dive,
        dist_own_tower_now,
        dist_own_tower_last,
        dist_enemy_tower_now,
        dist_enemy_tower_last,
    ):
        finish_phase = _phase(frame_no, POST_KILL_FINISH_START_FRAME, POST_KILL_FINISH_FULL_FRAME)
        early_reset_phase = 1.0 - _phase(frame_no, POST_KILL_EARLY_END_FRAME, POST_KILL_FINISH_FULL_FRAME)

        if kill_delta > 0:
            window_steps = int(26 + 54 * finish_phase)
            self.post_kill_wait_enemy_seen = True
            self.post_kill_steps = max(self.post_kill_steps, window_steps)
            self.post_kill_window_steps = max(1, window_steps)
        elif self.post_kill_wait_enemy_seen and main["enemy_hero_present"] and main["enemy_hp"] > 0.0:
            self.post_kill_wait_enemy_seen = False
            self.post_kill_steps = 0

        active = self.post_kill_steps > 0 or self.post_kill_wait_enemy_seen
        window_ratio = max(
            _ratio(self.post_kill_steps, max(self.post_kill_window_steps, 1), default=0.0),
            0.35 if self.post_kill_wait_enemy_seen else 0.0,
        )
        enemy_wave_present = lane_context["enemy_front_count"] > 0 or lane_context["enemy_wave_near_self"]
        tower_ready = bool(tower_window)
        tower_stage_ready = bool(tower_stage_window or tower_ready)

        lane_clear_plan = bool(active and not tower_ready and (enemy_wave_present or not tower_stage_ready))
        tower_stage_plan = bool(
            active
            and main["enemy_tower_present"]
            and tower_stage_ready
            and not tower_ready
            and not no_wave_dive
        )
        tower_plan = bool(
            active
            and main["enemy_tower_present"]
            and tower_ready
            and not no_wave_dive
        )
        reset_plan = False
        resource_plan = False

        moved_toward_own_tower = dist_own_tower_now < dist_own_tower_last - 100.0
        moved_toward_enemy_tower = dist_enemy_tower_now < dist_enemy_tower_last - 100.0

        if self.post_kill_steps > 0:
            self.post_kill_steps -= 1

        return {
            "active": active,
            "window_ratio": window_ratio,
            "finish_phase": finish_phase,
            "early_reset_phase": early_reset_phase,
            "waiting_enemy_return": self.post_kill_wait_enemy_seen,
            "lane_clear_plan": lane_clear_plan,
            "tower_stage_plan": tower_stage_plan,
            "tower_plan": tower_plan,
            "tower_ready": tower_ready,
            "tower_stage_ready": tower_stage_ready,
            "reset_plan": reset_plan,
            "resource_plan": resource_plan,
            "moved_toward_own_tower": moved_toward_own_tower,
            "moved_toward_enemy_tower": moved_toward_enemy_tower,
        }

    def _calc_lane_clear(
        self,
        lane_context,
        target_soldier_action,
        target_enemy_action,
        money_delta,
        exp_delta,
        action_head,
        recovery_context,
        took_hero_hurt,
        enemy_in_attack_range,
        low_hp,
        enemy_near_own_tower,
        no_wave_dive,
        post_clear_active,
    ):
        if recovery_context["recovery_commit"]:
            if target_soldier_action or action_head in {BUTTON_SKILL_1, BUTTON_SKILL_3}:
                return -0.045
            return 0.0
        value = 0.0
        skill_clear_action = lane_context["lane_clear_priority"] and action_head in {BUTTON_SKILL_1, BUTTON_SKILL_3}
        clearing_wave = target_soldier_action or skill_clear_action
        if lane_context["lane_control_window"] and not lane_context["lane_clear_priority"]:
            if lane_context["last_hit_window"]:
                if target_soldier_action:
                    value += 0.11
                else:
                    value -= 0.055
            else:
                if target_soldier_action and lane_context["nearest_enemy_minion_hp"] > 0.55:
                    value -= 0.055
                if action_head in {BUTTON_SKILL_1, BUTTON_SKILL_3}:
                    value -= 0.04
        if lane_context["last_hit_window"]:
            if target_soldier_action:
                value += 0.08
            elif skill_clear_action:
                value += 0.025
            else:
                value -= 0.05
        if lane_context["lane_clear_priority"]:
            if enemy_near_own_tower > 0:
                value += 0.07 if clearing_wave else -0.06
            if clearing_wave:
                value += 0.07
            else:
                value -= 0.055
        if (
            lane_context["lane_clear_priority"]
            and took_hero_hurt
            and enemy_in_attack_range
            and enemy_near_own_tower == 0
            and not no_wave_dive
        ):
            if target_enemy_action:
                value += 0.060
            elif clearing_wave:
                value -= 0.075 if low_hp else 0.060
        if (
            clearing_wave
            and enemy_in_attack_range
            and low_hp
            and not recovery_context.get("low_hp_hold_under_tower_window")
            and not recovery_context.get("low_hp_commit_trade_window")
            and enemy_near_own_tower == 0
        ):
            value -= 0.050
        if clearing_wave and (money_delta > 0 or exp_delta > 0):
            value += 0.03
        if post_clear_active and not clearing_wave and not target_enemy_action:
            value -= 0.035
        return value

    def _calc_over_wave_fight(
        self,
        lane_context,
        action_head,
        target_enemy_action,
        target_soldier_action,
        tower_window,
        low_hp,
        enemy_hp,
        took_hero_hurt,
        tower_crash_kill_window,
        low_hp_value_trade_window,
        post_clear_active,
    ):
        clearing_wave = target_soldier_action or (
            lane_context["lane_clear_priority"] and action_head in {BUTTON_SKILL_1, BUTTON_SKILL_3}
        )
        fighting_enemy = target_enemy_action or (action_head in SKILL_BUTTONS and not clearing_wave)
        if post_clear_active and fighting_enemy and not low_hp and enemy_hp > 0.30:
            return 0.02
        if tower_crash_kill_window and fighting_enemy:
            return 0.065
        if low_hp_value_trade_window and fighting_enemy:
            return 0.045
        if not fighting_enemy or tower_window:
            return 0.0
        if enemy_hp < 0.22:
            return 0.0
        risky = lane_context["enemy_minion_between"] or lane_context["self_ahead_of_ally_wave"]
        if not risky:
            return 0.0
        value = -0.07
        if low_hp:
            value -= 0.05
        if took_hero_hurt:
            value -= 0.04
        return value

    def _calc_neutral_resource(self, neutral_context, target_monster_action, money_delta, recovery_context):
        if recovery_context["recovery_commit"]:
            return -0.035 if target_monster_action else 0.0
        value = 0.0
        if neutral_context["monster_finish_window"]:
            if target_monster_action:
                value += 0.12
            else:
                value -= 0.10
        if neutral_context["neutral_safe_window"]:
            value += 0.05 if target_monster_action else -0.015
        if target_monster_action:
            if neutral_context["monster_hp_delta"] > 0:
                value += min(0.12, neutral_context["monster_hp_delta"] / 4000.0)
            elif neutral_context["neutral_safe_window"]:
                value += 0.02
        if target_monster_action and money_delta > 0:
            value += 0.10 if neutral_context["monster_finish_window"] else 0.04
        return value

    def _calc_recovery_supply(
        self,
        recovery_context,
        action_head,
        target_enemy_action,
        target_soldier_action,
        target_monster_action,
        own_hp_delta,
        low_hp,
        tower_crash_kill_window,
        enemy_weaker_window,
    ):
        if tower_crash_kill_window:
            value = 0.0
            if target_enemy_action or action_head in SKILL_BUTTONS:
                value += 0.02
            if own_hp_delta < -0.02:
                value -= 0.03
            return value
        if recovery_context.get("low_hp_commit_trade_window"):
            value = 0.0
            if target_enemy_action or action_head in SKILL_BUTTONS:
                value += 0.045
            if target_soldier_action or target_monster_action:
                value -= 0.025
            if own_hp_delta < -0.035:
                value -= 0.025
            return value
        if low_hp and enemy_weaker_window:
            if target_enemy_action or action_head in SKILL_BUTTONS:
                return 0.04
            if action_head == BUTTON_MOVE:
                return -0.01
            return -0.02
        if recovery_context.get("low_hp_hold_under_tower_window"):
            value = 0.0
            if target_soldier_action:
                value += 0.055
            elif recovery_context["moved_toward_own_tower"] or recovery_context["moved_toward_cake"]:
                value += 0.02
            elif target_enemy_action or action_head in SKILL_BUTTONS:
                value += 0.015
            else:
                value -= 0.02
            if own_hp_delta > 0.012:
                value += min(0.08, own_hp_delta * 2.0)
            return value
        if recovery_context.get("low_hp_grass_ambush_window"):
            value = 0.0
            if target_enemy_action or action_head in SKILL_BUTTONS:
                value += 0.05
            elif action_head == BUTTON_MOVE and not recovery_context["moved_toward_own_tower"]:
                value += 0.012
            else:
                value -= 0.02
            if own_hp_delta < -0.035 and not target_enemy_action:
                value -= 0.025
            return value
        if not low_hp and not recovery_context["low_hp_cake_window"]:
            return 0.0
        value = 0.0
        if own_hp_delta > 0.015:
            value += min(0.12, own_hp_delta * 2.5)
        if recovery_context["low_hp_cake_window"] or recovery_context["recovery_commit"]:
            if recovery_context["moved_toward_cake"] or recovery_context["moved_toward_own_tower"]:
                value += 0.035 if recovery_context["recovery_commit"] else 0.06
            elif action_head == BUTTON_MOVE:
                value -= 0.01 if recovery_context["recovery_commit"] else -0.005
        if recovery_context["recovery_commit"] and recovery_context["low_hp_stall_steps"] >= 4:
            value -= min(0.10, 0.012 * recovery_context["low_hp_stall_steps"])
        if target_enemy_action or action_head in SKILL_BUTTONS:
            value -= 0.06
        if recovery_context["recovery_commit"] and (target_soldier_action or target_monster_action):
            value -= 0.06
        return value

    def _calc_recall_discipline(
        self,
        recovery_context,
        action_head,
        low_hp,
        target_enemy_action,
        target_soldier_action,
        target_monster_action,
        tower_crash_kill_window,
    ):
        recovery_move = action_head == BUTTON_MOVE and (
            recovery_context["moved_toward_own_tower"] or recovery_context["moved_toward_cake"]
        )
        if tower_crash_kill_window:
            if target_enemy_action or action_head in SKILL_BUTTONS:
                return 0.04
            if target_soldier_action or target_monster_action:
                return -0.035
            if action_head == BUTTON_RECALL:
                return -0.025
            return 0.0
        if recovery_context.get("low_hp_commit_trade_window"):
            if target_enemy_action or action_head in SKILL_BUTTONS:
                return 0.055
            if recovery_move:
                return 0.012
            if target_soldier_action or target_monster_action:
                return -0.035
            return -0.015
        if recovery_context.get("low_hp_hold_under_tower_window"):
            if target_soldier_action:
                return 0.05
            if target_enemy_action or action_head in SKILL_BUTTONS:
                return 0.025
            if recovery_move:
                return 0.01
            return -0.02
        if recovery_context.get("low_hp_grass_ambush_window"):
            if target_enemy_action or action_head in SKILL_BUTTONS:
                return 0.06
            if recovery_move and not recovery_context["moved_toward_own_tower"]:
                return 0.01
            if target_soldier_action or target_monster_action:
                return -0.03
            return -0.025
        if recovery_context["safe_recall_window"]:
            if action_head == BUTTON_RECALL:
                return -0.10
            if recovery_move:
                return 0.055 if recovery_context["recovery_commit"] else 0.035
            if target_soldier_action or target_monster_action:
                return -0.08 if recovery_context["recovery_commit"] else -0.025
            if target_enemy_action or action_head in SKILL_BUTTONS:
                return -0.10
            return -0.055 if recovery_context["recovery_commit"] else -0.03
        if recovery_context["recovery_commit"]:
            if target_soldier_action or target_monster_action:
                return -0.07
            if recovery_move:
                return 0.035
            if target_enemy_action or action_head in SKILL_BUTTONS:
                return -0.07
            if recovery_context["low_hp_stall_steps"] >= 4 and action_head != BUTTON_RECALL:
                return -0.05
        if low_hp and target_enemy_action and recovery_context.get("enemy_weaker_window"):
            return 0.025
        if low_hp and target_enemy_action and recovery_context["enemy_pressure"]:
            return -0.035
        return 0.0

    def _calc_skill_tower_waste(
        self,
        action_head,
        action_target,
        hero_config_id,
        tower_edge_poke_window,
        enemy_tower_delta,
        dealt_hero_hurt,
    ):
        if action_head not in SKILL_BUTTONS or action_target != TARGET_TOWER:
            return 0.0
        if enemy_tower_delta > 0 or dealt_hero_hurt:
            return 0.0
        if hero_config_id == LUBAN_HERO_ID and action_head == BUTTON_SKILL_1:
            return -0.12
        if hero_config_id == LUBAN_HERO_ID and action_head == BUTTON_SKILL_3 and tower_edge_poke_window:
            return 0.0
        if hero_config_id == DIRENJIE_HERO_ID and tower_edge_poke_window:
            return -0.04
        return -0.06

    def _calc_passive_followup(
        self,
        current_action_head,
        previous_action_head,
        action_target,
        hero_config_id,
        dealt_hero_hurt,
        enemy_tower_delta,
        low_hp,
    ):
        if hero_config_id != LUBAN_HERO_ID or previous_action_head not in SKILL_BUTTONS:
            return 0.0
        if current_action_head == BUTTON_ATTACK:
            if dealt_hero_hurt or enemy_tower_delta > 0 or action_target in {TARGET_ENEMY, TARGET_TOWER} or self._is_soldier_target(action_target):
                return 0.11 if low_hp else 0.08
            if action_target in {None, TARGET_NONE} and not low_hp:
                return 0.018
            return -0.02
        if not low_hp:
            return -0.05
        return 0.0

    def _calc_hero_skill_pressure(
        self,
        hero_config_id,
        enemy_hero_config_id,
        current_action_head,
        previous_action_head,
        action_target,
        tower_edge_poke_window,
        tower_window,
        target_enemy_action,
        attack_tower_action,
        enemy_tower_delta,
        dealt_hero_hurt,
        hurt_by_delta,
        took_hero_hurt,
        low_hp,
        enemy_in_attack_range,
        no_wave_dive,
        tower_edge_overstep,
        recovery_commit,
        lane_clear_window,
        last_hit_window,
        direnjie_skill1_poke_window,
        direnjie_skill2_escape_window,
        direnjie_ult_kill_window,
        ult_follow_window,
        combo_start_window,
        combo_chain_window,
    ):
        if hero_config_id not in {LUBAN_HERO_ID, DIRENJIE_HERO_ID}:
            return 0.0

        value = 0.0
        pressure_window = bool(tower_edge_poke_window or tower_window)
        combo_window = bool(combo_start_window or combo_chain_window)
        skill_action = current_action_head in HERO_SKILL_BUTTONS
        attack_pressure = current_action_head == BUTTON_ATTACK and (
            target_enemy_action or attack_tower_action or action_target in {TARGET_ENEMY, TARGET_TOWER}
        )

        if hero_config_id == DIRENJIE_HERO_ID:
            if combo_window and skill_action:
                value += 0.060
                if target_enemy_action or enemy_in_attack_range:
                    value += 0.030
                if current_action_head == BUTTON_SKILL_2 and not direnjie_skill2_escape_window:
                    value += 0.020
            if pressure_window and attack_pressure:
                value += 0.05
            if current_action_head == BUTTON_ATTACK and (
                dealt_hero_hurt or enemy_tower_delta > 0 or enemy_in_attack_range or pressure_window
            ):
                value += 0.035
                if hurt_by_delta <= TOWER_POKE_TOLERATED_HURT:
                    value += 0.020
            if dealt_hero_hurt and hurt_by_delta <= TOWER_POKE_TOLERATED_HURT:
                value += 0.04
            if enemy_tower_delta > 0 and pressure_window:
                value += 0.04

            if current_action_head == BUTTON_SKILL_1:
                if direnjie_skill1_poke_window:
                    value += 0.045
                    if dealt_hero_hurt:
                        value += 0.045
                    if lane_clear_window or last_hit_window:
                        value += 0.015
                    if hurt_by_delta <= TOWER_POKE_TOLERATED_HURT:
                        value += 0.015
                elif not low_hp:
                    value -= 0.045

            if current_action_head == BUTTON_SKILL_2:
                if direnjie_skill2_escape_window:
                    value += 0.060
                    if took_hero_hurt or hurt_by_delta > 0:
                        value += 0.025
                    if tower_edge_overstep or no_wave_dive or recovery_commit:
                        value += 0.025
                elif combo_window:
                    value += 0.055
                    if target_enemy_action or enemy_in_attack_range:
                        value += 0.025
                    if dealt_hero_hurt:
                        value += 0.040
                else:
                    value -= 0.035

            if current_action_head == BUTTON_SKILL_3:
                ult_context_action = bool(
                    target_enemy_action
                    or enemy_in_attack_range
                    or lane_clear_window
                    or last_hit_window
                    or self._is_soldier_target(action_target)
                    or pressure_window
                )
                if target_enemy_action and dealt_hero_hurt:
                    value += 0.165
                    if direnjie_ult_kill_window:
                        value += 0.050
                elif target_enemy_action and direnjie_ult_kill_window:
                    value += 0.115
                elif ult_context_action:
                    value += 0.035
                    if enemy_in_attack_range:
                        value += 0.020
                    if lane_clear_window or last_hit_window or self._is_soldier_target(action_target):
                        value += 0.012
                elif not low_hp and not recovery_commit:
                    value -= 0.015

            useful_skill = (
                current_action_head == BUTTON_SKILL_1
                and (direnjie_skill1_poke_window or dealt_hero_hurt)
            ) or (
                current_action_head == BUTTON_SKILL_3
                and (
                    target_enemy_action
                    or enemy_in_attack_range
                    or lane_clear_window
                    or last_hit_window
                    or self._is_soldier_target(action_target)
                    or pressure_window
                    or dealt_hero_hurt
                )
            ) or (
                current_action_head == BUTTON_SKILL_2
                and (direnjie_skill2_escape_window or combo_window)
            )
            if skill_action and not useful_skill and not low_hp:
                value -= 0.020
            if current_action_head == BUTTON_ATTACK and previous_action_head in DIRENJIE_PRESSURE_SKILLS:
                if dealt_hero_hurt or enemy_tower_delta > 0 or pressure_window:
                    value += 0.070 if combo_chain_window else 0.045
                elif not low_hp:
                    value -= 0.02
            if ult_follow_window:
                if target_enemy_action and current_action_head == BUTTON_ATTACK:
                    value += 0.07
                elif current_action_head in {BUTTON_SKILL_1, BUTTON_SKILL_3} and target_enemy_action:
                    value += 0.03
                elif lane_clear_window or action_target == TARGET_SOLDIER_MIN or action_target == TARGET_SOLDIER_MAX:
                    value -= 0.06

        elif hero_config_id == LUBAN_HERO_ID:
            if current_action_head == BUTTON_SKILL_3:
                luban_ult_area = bool(
                    target_enemy_action
                    or enemy_in_attack_range
                    or lane_clear_window
                    or last_hit_window
                    or self._is_soldier_target(action_target)
                    or pressure_window
                    or attack_tower_action
                )
                if luban_ult_area:
                    value += 0.040
                    if dealt_hero_hurt:
                        value += 0.050
                    if lane_clear_window or last_hit_window or self._is_soldier_target(action_target):
                        value += 0.015
                    if pressure_window and (target_enemy_action or attack_tower_action or enemy_tower_delta > 0):
                        value += 0.025
                elif not low_hp:
                    value -= 0.010
            if combo_window and skill_action:
                value += 0.040
            if pressure_window and current_action_head in LUBAN_PRESSURE_SKILLS:
                if dealt_hero_hurt or enemy_tower_delta > 0 or enemy_in_attack_range:
                    value += 0.04
                elif not low_hp:
                    value -= 0.02

        if enemy_hero_config_id == LUBAN_HERO_ID and pressure_window and dealt_hero_hurt:
            value += 0.01
        return _clip(value, -0.14, 0.24)

    def _calc_matchup_strategy(
        self,
        matchup_context,
        action_head,
        previous_action_head,
        action_target,
        target_enemy_action,
        attack_tower_action,
        target_soldier_action,
        enemy_tower_delta,
        dealt_hero_hurt,
        hurt_by_delta,
        took_hero_hurt,
        enemy_in_attack_range,
        enemy_close_range,
        tower_window,
        tower_edge_poke_window,
        tower_edge_overstep,
        no_wave_dive,
        low_hp,
        own_hp,
        enemy_hp,
        enemy_skill1_ready,
        enemy_skill2_ready,
        enemy_skill3_ready,
        direnjie_skill1_poke_window,
        direnjie_skill2_escape_window,
        direnjie_ult_kill_window,
        combo_start_window,
        combo_chain_window,
    ):
        value = 0.0
        pressure_window = bool(tower_window or tower_edge_poke_window)
        combo_window = bool(combo_start_window or combo_chain_window)
        objective_action = attack_tower_action or enemy_tower_delta > 0
        safe_damage = hurt_by_delta <= TOWER_POKE_TOLERATED_HURT
        skill_pressure_action = action_head in {BUTTON_SKILL_1, BUTTON_SKILL_3}
        enemy_direnjie_control_ready = enemy_skill3_ready > 0.45 or enemy_skill1_ready > 0.65
        enemy_direnjie_escape_ready = enemy_skill2_ready > 0.45
        true_safe_farm = False

        if matchup_context["luban_vs_luban"]:
            if action_head == BUTTON_SKILL_3 and (pressure_window or target_enemy_action or enemy_in_attack_range):
                value += 0.035
                if dealt_hero_hurt:
                    value += 0.035
            if pressure_window and (objective_action or action_head == BUTTON_ATTACK):
                value += 0.055
            if previous_action_head in LUBAN_PRESSURE_SKILLS and action_head == BUTTON_ATTACK:
                value += 0.035 if (dealt_hero_hurt or enemy_tower_delta > 0 or pressure_window) else -0.015
            if target_enemy_action and not pressure_window and not enemy_hp < 0.32:
                value -= 0.030
            if no_wave_dive or tower_edge_overstep:
                value -= 0.060

        elif matchup_context["luban_vs_direnjie"]:
            if action_head == BUTTON_SKILL_3 and (pressure_window or target_enemy_action or enemy_in_attack_range):
                value += 0.035
                if dealt_hero_hurt:
                    value += 0.040
            if (
                enemy_in_attack_range
                and not low_hp
                and not enemy_direnjie_control_ready
                and (target_enemy_action or action_head in LUBAN_PRESSURE_SKILLS or dealt_hero_hurt)
            ):
                value += 0.070
            if target_soldier_action and enemy_in_attack_range and not low_hp:
                value -= 0.055
            if pressure_window and safe_damage and (objective_action or action_head == BUTTON_ATTACK):
                value += 0.055
            if (
                not enemy_direnjie_control_ready
                and enemy_in_attack_range
                and not low_hp
                and (target_enemy_action or action_head in LUBAN_PRESSURE_SKILLS or dealt_hero_hurt)
            ):
                value += 0.045
            if enemy_direnjie_control_ready and enemy_close_range and target_enemy_action and not enemy_hp < 0.30:
                value -= 0.050
            if previous_action_head in LUBAN_PRESSURE_SKILLS and action_head == BUTTON_ATTACK and not took_hero_hurt:
                value += 0.025
            if enemy_direnjie_control_ready and took_hero_hurt and action_head == BUTTON_MOVE:
                value += 0.025
            if target_soldier_action and enemy_in_attack_range and not low_hp and not pressure_window:
                value -= 0.025
            if no_wave_dive or tower_edge_overstep:
                value -= 0.070

        elif matchup_context["direnjie_vs_luban"]:
            true_safe_farm = bool(low_hp and (tower_window or tower_edge_poke_window or not enemy_in_attack_range))
            pressure_trade_window = bool(
                enemy_in_attack_range
                and own_hp >= 0.34
                and (
                    enemy_hp <= 0.58
                    or own_hp >= enemy_hp - 0.04
                    or direnjie_ult_kill_window
                    or safe_damage
                )
            )
            if low_hp and target_soldier_action and true_safe_farm:
                value += 0.030
            if low_hp and target_enemy_action and not direnjie_ult_kill_window and not dealt_hero_hurt:
                value -= 0.045
            if action_head == BUTTON_SKILL_1 and direnjie_skill1_poke_window:
                value += 0.035
                if target_enemy_action:
                    value += 0.025
                if safe_damage and target_enemy_action:
                    value += 0.015
                if dealt_hero_hurt:
                    value += 0.060
            if action_head == BUTTON_SKILL_2 and combo_window and not direnjie_skill2_escape_window:
                value += 0.045
                if target_enemy_action or enemy_in_attack_range:
                    value += 0.025
                if dealt_hero_hurt:
                    value += 0.040
            if action_head == BUTTON_SKILL_3:
                if target_enemy_action and dealt_hero_hurt:
                    value += 0.140
                    if combo_window:
                        value += 0.035
                elif target_enemy_action and direnjie_ult_kill_window:
                    value += 0.100
                elif target_enemy_action or enemy_in_attack_range:
                    value += 0.025
            if action_head == BUTTON_ATTACK and (dealt_hero_hurt or objective_action or pressure_window or combo_chain_window):
                value += 0.035
                if combo_chain_window and target_enemy_action:
                    value += 0.035
            if pressure_trade_window and (target_enemy_action or dealt_hero_hurt):
                value += 0.055
                if dealt_hero_hurt and safe_damage:
                    value += 0.025
            if target_soldier_action and pressure_trade_window and not dealt_hero_hurt and not true_safe_farm:
                value -= 0.070
            if enemy_close_range and took_hero_hurt and action_head == BUTTON_SKILL_2:
                value += 0.045
            if (
                not direnjie_ult_kill_window
                and enemy_in_attack_range
                and action_head == BUTTON_ATTACK
                and target_enemy_action
                and enemy_hp > max(0.42, own_hp + 0.06)
            ):
                value -= 0.045
            if target_enemy_action and enemy_hp > own_hp + 0.12 and own_hp < 0.42:
                value -= 0.050

        elif matchup_context["direnjie_vs_direnjie"]:
            if action_head == BUTTON_SKILL_1 and direnjie_skill1_poke_window and safe_damage:
                value += 0.040
            if action_head == BUTTON_SKILL_2 and (
                direnjie_skill2_escape_window or (enemy_direnjie_control_ready and (took_hero_hurt or enemy_close_range))
            ):
                value += 0.050
            if action_head == BUTTON_SKILL_3:
                if target_enemy_action and dealt_hero_hurt:
                    value += 0.115
                elif target_enemy_action and direnjie_ult_kill_window and not enemy_direnjie_escape_ready:
                    value += 0.075
                elif target_enemy_action and direnjie_ult_kill_window:
                    value += 0.035
                elif enemy_direnjie_escape_ready and enemy_hp > 0.30:
                    value -= 0.015
            if action_head == BUTTON_ATTACK and (pressure_window or dealt_hero_hurt) and safe_damage:
                value += 0.025
            if target_enemy_action and enemy_direnjie_control_ready and own_hp < 0.40 and enemy_hp > own_hp + 0.10:
                value -= 0.045

        if target_soldier_action and low_hp and not pressure_window and not true_safe_farm:
            value -= 0.020
        if objective_action and pressure_window:
            value += 0.020
        return _clip(value, -0.14, 0.18)

    def _calc_summoner_discipline(
        self,
        action_head,
        previous_action_head,
        target_enemy_action,
        dealt_hero_hurt,
        took_hero_hurt,
        low_hp,
        own_hp_delta,
        own_hp,
        enemy_hp,
        enemy_visible,
        enemy_in_attack_range,
        tower_window,
        tower_crash_kill_window,
        low_hp_value_trade_window,
        recovery_context,
    ):
        result = {"hero_skill_pressure": 0.0, "recovery_supply": 0.0, "recall_discipline": 0.0}
        if action_head not in {BUTTON_HEAL, BUTTON_SUMMONER}:
            return result

        kill_or_trade = bool(
            tower_crash_kill_window
            or low_hp_value_trade_window
            or (
                enemy_visible
                and enemy_in_attack_range
                and 0.0 < enemy_hp <= 0.45
                and own_hp > 0.22
            )
            or (target_enemy_action and dealt_hero_hurt)
        )
        recovery_need = bool(
            low_hp
            or recovery_context.get("recovery_commit")
            or recovery_context.get("low_hp_cake_window")
            or recovery_context.get("safe_recall_window")
            or took_hero_hurt
        )
        paired_spell = previous_action_head in {BUTTON_HEAL, BUTTON_SUMMONER} and previous_action_head != action_head

        if action_head == BUTTON_HEAL:
            if recovery_need:
                result["recovery_supply"] += 0.055
                result["recall_discipline"] += 0.035
                if own_hp_delta > 0.012:
                    result["recovery_supply"] += min(0.06, own_hp_delta * 2.0)
            elif kill_or_trade:
                result["hero_skill_pressure"] -= 0.035
            else:
                result["recall_discipline"] -= 0.045
            if paired_spell and not recovery_need:
                result["recall_discipline"] -= 0.045
        else:
            if kill_or_trade and not recovery_context.get("recovery_commit"):
                result["hero_skill_pressure"] += 0.070
                if dealt_hero_hurt:
                    result["hero_skill_pressure"] += 0.035
                if tower_window:
                    result["hero_skill_pressure"] += 0.015
            elif recovery_need and not low_hp_value_trade_window:
                result["recall_discipline"] -= 0.065
                result["recovery_supply"] -= 0.035
            else:
                result["hero_skill_pressure"] -= 0.035
            if paired_spell and not kill_or_trade:
                result["hero_skill_pressure"] -= 0.045

        return {key: _clip(value, -0.12, 0.14) for key, value in result.items()}

    def _calc_tower_edge_poke(
        self,
        tower_edge_poke_window,
        tower_edge_overstep,
        enemy_under_tower,
        action_head,
        action_target,
        enemy_tower_delta,
        dealt_hero_hurt,
        hurt_by_delta,
        took_hero_hurt,
        dist_enemy_tower_now,
        dist_enemy_tower_last,
        pressure_outer_range,
        low_hp,
        no_wave_dive,
        tower_wave_lost_reset,
        attack_tower_action,
    ):
        value = 0.0
        pressure_action = (
            (action_head == BUTTON_ATTACK and action_target in {TARGET_ENEMY, TARGET_TOWER})
            or action_head in LUBAN_PRESSURE_SKILLS
            or action_head in DIRENJIE_PRESSURE_SKILLS
        )
        if tower_edge_poke_window:
            if attack_tower_action:
                value += 0.065
            elif pressure_action:
                value += 0.02
            if enemy_tower_delta > 0:
                value += min(0.22, enemy_tower_delta * 8.5)
            if dealt_hero_hurt:
                value += 0.025
            if pressure_action and hurt_by_delta <= TOWER_POKE_TOLERATED_HURT:
                value += 0.025
            if action_head == BUTTON_MOVE and hurt_by_delta <= TOWER_POKE_TOLERATED_HURT:
                value += 0.006
            if hurt_by_delta > TOWER_POKE_HEAVY_HURT:
                value -= 0.075
        if (
            not tower_edge_poke_window
            and not low_hp
            and action_head == BUTTON_ATTACK
            and action_target not in {TARGET_TOWER, TARGET_ENEMY}
            and dist_enemy_tower_now <= pressure_outer_range
            and enemy_tower_delta <= 0
        ):
            value -= 0.045
        elif enemy_under_tower and dist_enemy_tower_now > pressure_outer_range + 3000.0 and not low_hp:
            value -= 0.018

        if tower_wave_lost_reset:
            if dist_enemy_tower_now > dist_enemy_tower_last + 100.0:
                value += 0.035
            elif pressure_action:
                value -= 0.055

        if no_wave_dive and dealt_hero_hurt and action_head == BUTTON_MOVE and dist_enemy_tower_now > dist_enemy_tower_last + 120.0:
            value += 0.05

        if tower_edge_overstep:
            value -= 0.09
            if took_hero_hurt:
                value -= 0.07
            if pressure_action and not dealt_hero_hurt and enemy_tower_delta <= 0:
                value -= 0.035

        return _clip(value, -0.22, 0.28)

    def _calc_post_kill_conversion(
        self,
        post_kill_context,
        lane_context,
        neutral_context,
        recovery_context,
        action_head,
        target_soldier_action,
        target_monster_action,
        attack_tower_action,
        enemy_tower_delta,
        money_delta,
        exp_delta,
        low_hp,
        no_wave_dive,
        tower_crash_kill_window,
        target_enemy_action,
        dealt_hero_hurt,
        kill_delta,
        death_delta,
    ):
        if tower_crash_kill_window:
            value = 0.0
            if target_enemy_action or dealt_hero_hurt:
                value += 0.09
            else:
                value -= 0.04
            if kill_delta > 0:
                value += 0.28
            if death_delta > 0:
                value -= 0.12 if kill_delta > 0 else 0.28
            return _clip(value, -0.18, 0.42)
        if recovery_context.get("low_hp_commit_trade_window"):
            value = 0.0
            if target_enemy_action or dealt_hero_hurt:
                value += 0.08
            if kill_delta > 0:
                value += 0.24
            if enemy_tower_delta > 0:
                value += min(0.18, enemy_tower_delta * 7.0)
            if death_delta > 0:
                value -= 0.10 if (kill_delta > 0 or enemy_tower_delta > 0) else 0.26
            if target_soldier_action or target_monster_action:
                value -= 0.035
            return _clip(value, -0.16, 0.34)
        if not post_kill_context["active"]:
            return 0.0

        value = 0.0
        finish_phase = post_kill_context["finish_phase"]
        clearing_wave = target_soldier_action or (
            post_kill_context["lane_clear_plan"] and action_head in {BUTTON_SKILL_1, BUTTON_SKILL_3}
        )
        tower_progress = enemy_tower_delta > 0
        objective_setup_move = post_kill_context["moved_toward_enemy_tower"] and not no_wave_dive
        recovery_setup_move = recovery_context["moved_toward_cake"] or recovery_context["moved_toward_own_tower"]
        resource_progress = target_monster_action or neutral_context["monster_hp_delta"] > 0

        if post_kill_context["lane_clear_plan"]:
            if clearing_wave:
                value += 0.09
                if action_head == BUTTON_SKILL_3:
                    value += 0.025
            elif objective_setup_move and not post_kill_context["tower_ready"]:
                value += 0.030
            elif action_head in {BUTTON_ATTACK, BUTTON_SKILL_1, BUTTON_SKILL_3}:
                value -= 0.025
            else:
                value -= 0.04
            if money_delta > 0 or exp_delta > 0:
                value += 0.05
        elif post_kill_context["tower_plan"] and clearing_wave:
            value -= 0.085 + 0.055 * finish_phase

        if post_kill_context["tower_stage_plan"]:
            if objective_setup_move:
                value += 0.050
            elif clearing_wave and lane_context["enemy_front_count"] > 0:
                value += 0.045
            elif attack_tower_action and not post_kill_context["tower_ready"]:
                value -= 0.020
            elif action_head not in {BUTTON_MOVE, BUTTON_ATTACK, BUTTON_SKILL_1, BUTTON_SKILL_3}:
                value -= 0.025

        if post_kill_context["tower_plan"]:
            if tower_progress:
                value += min(0.55, enemy_tower_delta * (9.0 + 10.0 * finish_phase))
            elif attack_tower_action:
                value += 0.110 + 0.035 * finish_phase
            else:
                value -= 0.085 + 0.075 * finish_phase
            if post_kill_context["moved_toward_enemy_tower"] and not no_wave_dive:
                value += 0.035

        if post_kill_context["reset_plan"]:
            if recovery_setup_move:
                value += 0.035
            elif clearing_wave and lane_context["enemy_wave_near_self"]:
                value += 0.015
            else:
                value -= 0.04
            if low_hp and not recovery_setup_move:
                value -= 0.025

        if post_kill_context["resource_plan"]:
            if resource_progress:
                value += 0.055
            elif action_head not in {BUTTON_MOVE, BUTTON_ATTACK}:
                value -= 0.02

        if post_kill_context["tower_plan"]:
            progressed = tower_progress or attack_tower_action or objective_setup_move
        else:
            progressed = clearing_wave or tower_progress or objective_setup_move or money_delta > 0 or exp_delta > 0
        if not progressed:
            value -= 0.025

        return _clip(value, -0.25, 0.45)

    def _calc_tower_window_conversion(
        self,
        enemy_tower_delta,
        tower_window,
        hero_can_hit_tower,
        attack_tower_action,
        late_phase,
    ):
        if enemy_tower_delta > 0:
            value = enemy_tower_delta * (7.0 + 5.0 * late_phase)
            if tower_window:
                value += 0.04
            if attack_tower_action:
                value += 0.03
            return value
        if tower_window and attack_tower_action:
            return 0.05 + 0.03 * late_phase
        return 0.0

    def _calc_missed_tower_window(
        self,
        enemy_tower_delta,
        tower_window,
        hero_can_hit_tower,
        attack_tower_action,
        enemy_in_attack_range,
        dealt_hero_hurt,
        took_hero_hurt,
        low_hp,
        enemy_weaker_window,
    ):
        if enemy_tower_delta > 0 or attack_tower_action or took_hero_hurt:
            self.tower_window_stall_steps = max(0, self.tower_window_stall_steps - 2)
            return 0.0

        if not tower_window:
            self.tower_window_stall_steps = max(0, self.tower_window_stall_steps - 1)
            return 0.0

        if enemy_in_attack_range and dealt_hero_hurt and not (low_hp and enemy_weaker_window):
            self.tower_window_stall_steps = max(0, self.tower_window_stall_steps - 1)
            return 0.0

        self.tower_window_stall_steps = min(self.tower_window_stall_steps + 1, 12)
        if self.tower_window_stall_steps < 3:
            return 0.0
        return -min(0.14, 0.018 * self.tower_window_stall_steps)

    def _calc_late_tower_urgency(
        self,
        frame_no,
        enemy_tower_delta,
        enemy_tower_hp,
        tower_window,
        hero_can_hit_tower,
        hero_hp,
        late_phase,
    ):
        if late_phase <= 0.0 or enemy_tower_hp <= 0.0:
            return 0.0
        if enemy_tower_delta > 0:
            return enemy_tower_delta * (5.0 + 8.0 * late_phase)
        if frame_no >= 15000 and tower_window and hero_hp > 0.45:
            return -0.03 * late_phase
        return 0.0

    def _calc_purposeful_grass(
        self,
        in_grass,
        was_in_grass,
        enemy_visible,
        enemy_in_attack_range,
        dealt_hero_hurt,
        took_hero_hurt,
        low_hp,
        tower_window,
        hero_can_hit_tower,
        no_wave_dive,
        dist_enemy_hero_now,
        dist_enemy_hero_last,
        dist_own_tower_now,
        dist_own_tower_last,
        dist_enemy_tower_now,
        dist_enemy_tower_last,
        close_tower_range,
        enemy_tower_delta,
        action_head,
        tower_crash_kill_window,
        target_enemy_action,
        attack_tower_action,
        tower_edge_poke_window,
        kill_delta,
        death_delta,
        own_is_luban,
        own_is_direnjie,
        enemy_is_luban,
        enemy_is_direnjie,
        own_hp,
        enemy_hp,
    ):
        if not in_grass:
            return 0.0, None

        if tower_crash_kill_window:
            if target_enemy_action or kill_delta > 0:
                return 0.025, "tower_crash_kill"
            if death_delta > 0:
                return -0.06, "aimless"

        entered_grass = in_grass and not was_in_grass
        moved_away_from_enemy = enemy_visible and dist_enemy_hero_now > dist_enemy_hero_last + 100.0
        moved_toward_own_tower = dist_own_tower_now < dist_own_tower_last - 100.0
        moved_out_from_enemy_tower = dist_enemy_tower_now > dist_enemy_tower_last + 100.0
        under_pressure = low_hp or took_hero_hurt or no_wave_dive
        stable_tower_pressure = tower_window and hero_can_hit_tower and not no_wave_dive and not low_hp
        direnjie_safe_develop = bool(
            own_is_direnjie
            and enemy_is_luban
            and not tower_crash_kill_window
            and (
                under_pressure
                or (enemy_visible and enemy_hp >= own_hp + 0.06)
                or not enemy_in_attack_range
            )
        )
        luban_deny_grass = bool(
            own_is_luban
            and enemy_is_direnjie
            and enemy_visible
            and not low_hp
            and (target_enemy_action or dealt_hero_hurt)
        )

        if under_pressure and (entered_grass or moved_away_from_enemy or moved_toward_own_tower or moved_out_from_enemy_tower):
            return 0.08, "defensive"

        if direnjie_safe_develop:
            if dealt_hero_hurt and not took_hero_hurt:
                return 0.070, "vision_poke"
            if entered_grass or moved_away_from_enemy or moved_toward_own_tower:
                return 0.060, "defensive"

        if luban_deny_grass and enemy_in_attack_range:
            return 0.055, "vision_poke"

        if (
            entered_grass
            and enemy_visible
            and dist_enemy_hero_last > 0
            and dist_enemy_hero_now > dist_enemy_hero_last + 180.0
            and not low_hp
            and not no_wave_dive
            and not stable_tower_pressure
        ):
            return 0.065, "break_target"

        if low_hp and enemy_visible and enemy_in_attack_range and entered_grass:
            return 0.075, "ambush"

        if enemy_visible and enemy_in_attack_range and (dealt_hero_hurt or action_head == BUTTON_ATTACK):
            return 0.06, "vision_poke"

        near_tower_stage = dist_enemy_tower_now <= close_tower_range + 4500.0
        if near_tower_stage and not tower_window and not no_wave_dive and dist_enemy_tower_now < dist_enemy_tower_last - 100.0:
            return (0.04 if entered_grass else 0.025), "tower_stage"

        if stable_tower_pressure and entered_grass and not dealt_hero_hurt and not attack_tower_action:
            return -0.055, "aimless"

        if tower_window and enemy_tower_delta <= 0 and action_head != BUTTON_ATTACK:
            return -0.08, "aimless"

        if in_grass and not under_pressure and not tower_window and not enemy_visible and enemy_tower_delta <= 0:
            return -0.035, "aimless"

        if not under_pressure and not tower_window and not dealt_hero_hurt:
            if not enemy_visible or not enemy_in_attack_range:
                return -0.035, "aimless"

        return 0.0, None

    def _calc_forward_pressure(
        self,
        enemy_visible,
        enemy_in_attack_range,
        dist_enemy_hero_now,
        dist_enemy_hero_last,
        dist_enemy_tower_now,
        dist_enemy_tower_last,
        attack_range,
        move_delta,
        action_head,
        target_enemy_action,
        dealt_hero_hurt,
        low_hp,
        no_wave_dive,
        lane_clear_priority,
        own_tower_defend_clear,
        tower_window,
        recovery_commit,
        enemy_hp,
        own_hp,
    ):
        if recovery_commit or no_wave_dive:
            return 0.0
        value = 0.0
        enemy_contact_band = enemy_visible and dist_enemy_hero_now <= attack_range * 1.35
        favorable_contact = enemy_contact_band and enemy_hp > 0.0 and (own_hp >= enemy_hp - 0.05 or enemy_hp <= 0.45)
        if dealt_hero_hurt:
            value += 0.055
        if target_enemy_action and enemy_in_attack_range:
            value += 0.030
        if (
            favorable_contact
            and not low_hp
            and not own_tower_defend_clear
            and dist_enemy_hero_now < dist_enemy_hero_last - 120.0
        ):
            value += 0.025
        if tower_window and dist_enemy_tower_now < dist_enemy_tower_last - 120.0:
            value += 0.020
        if (
            enemy_contact_band
            and not low_hp
            and not lane_clear_priority
            and action_head not in {BUTTON_ATTACK, BUTTON_SKILL_1, BUTTON_SKILL_3}
            and move_delta < 180.0
            and not dealt_hero_hurt
        ):
            value -= 0.035
        return _clip(value, -0.08, 0.12)

    def _record_episode_stats(
        self,
        enemy_tower_delta,
        tower_window,
        attack_tower_action,
        missed_tower_window,
        late_tower_urgency,
        no_wave_dive,
        previous_action_head,
        current_action_head,
        post_skill_attack,
        grass_tag,
        lane_context,
        neutral_context,
        recovery_context,
        lane_clear,
        over_wave_fight,
        neutral_resource,
        recovery_supply,
        recall_discipline,
        skill_tower_waste,
        passive_followup,
        hero_skill_pressure,
        hero_config_id,
        matchup_context,
        matchup_strategy,
        post_kill_context,
        post_kill_conversion,
        target_soldier_action,
        target_monster_action,
        action_head,
        tower_crash_kill_window,
        target_enemy_action,
        kill_delta,
        death_delta,
        tower_edge_poke_window,
        tower_edge_poke,
        tower_edge_overstep,
        tower_wave_lost_reset,
        dealt_hero_hurt,
        ep_context,
    ):
        if tower_window:
            self.episode_stats["tower_window_frames"] += 1
        if attack_tower_action:
            self.episode_stats["attack_tower_actions"] += 1
        if enemy_tower_delta > 0:
            self.episode_stats["tower_damage_events"] += 1
            self.episode_stats["tower_damage"] += enemy_tower_delta
        if missed_tower_window < 0:
            self.episode_stats["missed_tower_windows"] += 1
        if late_tower_urgency != 0:
            self.episode_stats["late_urgency_frames"] += 1
        if no_wave_dive:
            self.episode_stats["no_wave_dive_frames"] += 1
        if previous_action_head in SKILL_BUTTONS:
            if current_action_head == BUTTON_ATTACK and post_skill_attack > 0:
                self.episode_stats["skill_then_attack"] += 1
            elif post_skill_attack < 0:
                self.episode_stats["skill_without_followup"] += 1
        if grass_tag in {"defensive", "vision_poke", "tower_stage", "ambush"}:
            self.episode_stats["purposeful_grass_frames"] += 1
        elif grass_tag == "aimless":
            self.episode_stats["aimless_grass_frames"] += 1
        if grass_tag == "tower_stage":
            self.episode_stats["grass_tower_staging_frames"] += 1
        if target_soldier_action:
            self.episode_stats["lane_clear_actions"] += 1
        if lane_context["last_hit_window"]:
            self.episode_stats["last_hit_windows"] += 1
            if lane_clear < 0:
                self.episode_stats["missed_last_hit_windows"] += 1
        if lane_context["lane_control_window"]:
            self.episode_stats["early_lane_control_windows"] += 1
        if over_wave_fight < 0:
            self.episode_stats["over_wave_fight_frames"] += 1
        if neutral_context["neutral_safe_window"]:
            self.episode_stats["neutral_window_frames"] += 1
        if neutral_context["monster_finish_window"]:
            self.episode_stats["neutral_finish_windows"] += 1
            if not target_monster_action:
                self.episode_stats["missed_neutral_finish_windows"] += 1
        if target_monster_action:
            self.episode_stats["neutral_attack_actions"] += 1
        if neutral_context["monster_hp_delta"] > 0:
            self.episode_stats["neutral_damage_events"] += 1
        if recovery_context["low_hp_cake_window"]:
            self.episode_stats["low_hp_recovery_windows"] += 1
        if recovery_supply > 0 and recovery_context["low_hp_cake_window"]:
            self.episode_stats["health_pack_recover_events"] += 1
        if recovery_context["safe_recall_window"]:
            self.episode_stats["safe_recall_windows"] += 1
            if action_head == BUTTON_RECALL:
                self.episode_stats["safe_recall_button_actions"] += 1
            elif action_head == BUTTON_MOVE and (
                recovery_context["moved_toward_own_tower"] or recovery_context["moved_toward_cake"]
            ):
                pass
            else:
                self.episode_stats["missed_safe_recall_windows"] += 1
                self.episode_stats["missed_recall_windows"] += 1
        if action_head == BUTTON_RECALL:
            self.episode_stats["recall_actions"] += 1
        if recovery_context["low_hp_stall_steps"] >= 4:
            self.episode_stats["low_hp_stall_frames"] += 1
        if skill_tower_waste < 0:
            self.episode_stats["skill_tower_waste_events"] += 1
        if passive_followup > 0:
            self.episode_stats["passive_followup_actions"] += 1
        if hero_config_id == DIRENJIE_HERO_ID and hero_skill_pressure != 0:
            self.episode_stats["direnjie_pressure_windows"] += 1
            if hero_skill_pressure > 0:
                self.episode_stats["direnjie_pressure_actions"] += 1
            if current_action_head == BUTTON_ATTACK and hero_skill_pressure > 0:
                self.episode_stats["direnjie_auto_pressure_actions"] += 1
            elif current_action_head == BUTTON_SKILL_1 and hero_skill_pressure > 0:
                self.episode_stats["direnjie_skill1_poke_actions"] += 1
            elif current_action_head == BUTTON_SKILL_2:
                if hero_skill_pressure > 0:
                    self.episode_stats["direnjie_skill2_escape_actions"] += 1
                else:
                    self.episode_stats["direnjie_skill2_waste_actions"] += 1
            elif current_action_head == BUTTON_SKILL_3:
                if hero_skill_pressure > 0:
                    self.episode_stats["direnjie_ult_pressure_actions"] += 1
                else:
                    self.episode_stats["direnjie_ult_waste_actions"] += 1
        if matchup_context["luban_vs_luban"]:
            self.episode_stats["matchup_luban_mirror_frames"] += 1
        elif matchup_context["luban_vs_direnjie"]:
            self.episode_stats["matchup_luban_direnjie_frames"] += 1
        elif matchup_context["direnjie_vs_luban"]:
            self.episode_stats["matchup_direnjie_luban_frames"] += 1
        elif matchup_context["direnjie_vs_direnjie"]:
            self.episode_stats["matchup_direnjie_mirror_frames"] += 1
        if matchup_strategy > 0:
            self.episode_stats["matchup_strategy_actions"] += 1
        if post_kill_context["active"]:
            self.episode_stats["post_kill_window_frames"] += 1
            if target_soldier_action:
                self.episode_stats["post_kill_lane_clear_actions"] += 1
            if attack_tower_action:
                self.episode_stats["post_kill_tower_actions"] += 1
            if enemy_tower_delta > 0:
                self.episode_stats["post_kill_tower_damage_events"] += 1
            if action_head == BUTTON_RECALL:
                self.episode_stats["post_kill_recovery_actions"] += 1
            if post_kill_conversion < -0.02:
                self.episode_stats["post_kill_idle_frames"] += 1
        if hero_skill_pressure > 0 and matchup_context.get("luban_vs_direnjie"):
            self.episode_stats["baseline_luban_pressure_windows"] += 1
            if target_enemy_action or dealt_hero_hurt or action_head in LUBAN_PRESSURE_SKILLS:
                self.episode_stats["baseline_luban_pressure_actions"] += 1
        if matchup_context.get("direnjie_vs_luban") and recovery_context.get("low_hp_hold_under_tower_window"):
            self.episode_stats["baseline_direnjie_safe_farm_windows"] += 1
        if grass_tag in {"vision_poke", "ambush"}:
            self.episode_stats["vision_probe_windows"] += 1
        if tower_crash_kill_window:
            self.episode_stats["tower_crash_kill_windows"] += 1
            if target_enemy_action:
                self.episode_stats["tower_crash_kill_attack_actions"] += 1
            if kill_delta > 0:
                self.episode_stats["tower_crash_kill_kills"] += 1
            if death_delta > 0:
                self.episode_stats["tower_crash_kill_deaths"] += 1
        if tower_edge_poke_window:
            self.episode_stats["tower_edge_poke_windows"] += 1
            tower_edge_skill_action = (
                (hero_config_id == LUBAN_HERO_ID and action_head in LUBAN_PRESSURE_SKILLS)
                or (hero_config_id == DIRENJIE_HERO_ID and action_head in DIRENJIE_PRESSURE_SKILLS)
            )
            if action_head == BUTTON_ATTACK or tower_edge_skill_action:
                self.episode_stats["tower_edge_poke_actions"] += 1
            if tower_edge_poke > 0 and (enemy_tower_delta > 0 or dealt_hero_hurt):
                self.episode_stats["tower_edge_poke_damage_events"] += 1
        if tower_edge_overstep:
            self.episode_stats["tower_edge_oversteps"] += 1
        if tower_wave_lost_reset:
            self.episode_stats["tower_wave_lost_resets"] += 1
        if ep_context.get("ep_low"):
            self.episode_stats["ep_low_frames"] += 1
        if ep_context.get("ep_critical"):
            self.episode_stats["ep_critical_frames"] += 1
        if ep_context.get("ep_low_skill_action"):
            self.episode_stats["ep_low_skill_actions"] += 1
        if ep_context.get("ep_conserve_action"):
            self.episode_stats["ep_conserve_actions"] += 1
        if ep_context.get("ep_resume_event"):
            self.episode_stats["ep_resume_events"] += 1

    def _calc_attack_move_rhythm(self, action_head, move_delta, enemy_in_attack_range, dealt_hero_hurt, took_hero_hurt):
        if action_head == BUTTON_MOVE:
            if enemy_in_attack_range:
                return 0.04 if move_delta > 250.0 else -0.03
            return 0.01 if move_delta > 250.0 else -0.01
        if action_head == BUTTON_ATTACK:
            value = 0.05 if dealt_hero_hurt or enemy_in_attack_range else -0.04
            if took_hero_hurt and move_delta < 150.0:
                value -= 0.04
            return value
        if took_hero_hurt and move_delta < 150.0:
            return -0.04
        return 0.0

    def _calc_spacing_control(
        self,
        action_head,
        dist_enemy_hero_now,
        dist_enemy_hero_last,
        dist_own_tower_now,
        dist_own_tower_last,
        attack_range,
        enemy_visible,
        low_hp,
        took_hero_hurt,
    ):
        if not enemy_visible:
            return 0.0
        if dist_enemy_hero_now <= 0 or dist_enemy_hero_now >= 99999.0:
            return 0.0
        value = 0.0
        if attack_range * 0.55 <= dist_enemy_hero_now <= attack_range:
            value += 0.04
        elif dist_enemy_hero_now < attack_range * 0.35:
            value -= 0.05
        elif dist_enemy_hero_now > attack_range * 1.2:
            value -= 0.01
        if low_hp or took_hero_hurt:
            if dist_enemy_hero_now > dist_enemy_hero_last:
                value += 0.05
            elif dist_enemy_hero_now < dist_enemy_hero_last:
                value -= 0.04
            if low_hp and action_head == BUTTON_MOVE and dist_enemy_hero_now > dist_enemy_hero_last and dist_own_tower_now > dist_own_tower_last:
                value -= 0.03
        elif action_head == BUTTON_ATTACK and dist_enemy_hero_now <= attack_range:
            value += 0.02
        return value

    def _calc_post_skill_attack(
        self,
        current_action_head,
        previous_action_head,
        dealt_hero_hurt,
        enemy_tower_delta,
        enemy_in_attack_range,
        tower_window,
        move_delta,
        low_hp,
        combo_chain_window,
    ):
        if previous_action_head not in SKILL_BUTTONS:
            return 0.0
        if current_action_head == BUTTON_ATTACK:
            if dealt_hero_hurt or enemy_tower_delta > 0 or enemy_in_attack_range or tower_window:
                return 0.105 if combo_chain_window else 0.07
            return -0.02
        if combo_chain_window and current_action_head in HERO_SKILL_BUTTONS:
            return 0.055
        if current_action_head == BUTTON_MOVE and move_delta > 250.0:
            return 0.005 if combo_chain_window else 0.025
        if not low_hp:
            return -0.04
        return 0.0

    def _calc_skill_discipline(
        self,
        action_head,
        action_target,
        hero_config_id,
        enemy_visible,
        enemy_in_attack_range,
        target_enemy_action,
        dealt_hero_hurt,
        low_enemy_minions,
        enemy_minion_clear_delta,
        money_delta,
        exp_delta,
        enemy_tower_delta,
        no_wave_dive,
        lane_clear_window,
        last_hit_window,
        defensive_skill_window,
        pressure_window,
        combo_start_window,
        combo_chain_window,
    ):
        if action_head not in HERO_SKILL_BUTTONS:
            return 0.0

        cleared_wave = enemy_minion_clear_delta > 0 or money_delta > 0 or exp_delta > 0
        hit_hero = dealt_hero_hurt
        hit_objective = enemy_tower_delta > 0
        combo_window = bool(combo_start_window or combo_chain_window)
        clear_intent = bool(
            lane_clear_window
            or last_hit_window
            or low_enemy_minions > 0
            or (hero_config_id == DIRENJIE_HERO_ID and action_head in {BUTTON_SKILL_1, BUTTON_SKILL_3})
            or (hero_config_id == LUBAN_HERO_ID and pressure_window)
        )
        defensive_escape = bool(hero_config_id == DIRENJIE_HERO_ID and action_head == BUTTON_SKILL_2 and defensive_skill_window)

        value = 0.0
        if hit_hero:
            value += 0.075
        if cleared_wave:
            value += 0.045 if clear_intent else 0.025
        if hit_hero and cleared_wave:
            value += 0.060
        if hit_objective:
            value += 0.045
        if defensive_escape:
            value += 0.055
        if combo_window and hero_config_id in {LUBAN_HERO_ID, DIRENJIE_HERO_ID}:
            value += 0.040
            if target_enemy_action or enemy_in_attack_range:
                value += 0.030
            if action_head == BUTTON_SKILL_2 and not defensive_skill_window:
                value += 0.015
        if value > 0.0:
            return _clip(value, -0.16, 0.20)

        if clear_intent and action_head in {BUTTON_SKILL_1, BUTTON_SKILL_3}:
            return 0.015
        if combo_window and enemy_visible:
            return 0.020
        if pressure_window and enemy_visible and target_enemy_action:
            return 0.005
        value = -0.05
        if target_enemy_action and not enemy_in_attack_range:
            value -= 0.035
        if not enemy_visible and not clear_intent:
            value -= 0.035
        if no_wave_dive:
            value -= 0.03
        return _clip(value, -0.16, 0.20)

    def _calc_ep_context(
        self,
        main,
        action_head,
        dealt_hero_hurt,
        took_hero_hurt,
        enemy_tower_delta,
        target_enemy_action,
        target_soldier_action,
        attack_tower_action,
        enemy_minion_clear_delta,
        money_delta,
        exp_delta,
    ):
        has_ep = bool(main.get("hero_has_ep", 0.0) > 0.0)
        ep_ratio = _clip(main.get("hero_ep", 1.0), 0.0, 1.0)
        last_ep = _clip(self.last_main.get("hero_ep", ep_ratio) if self.last_main else ep_ratio, 0.0, 1.0)
        ep_delta = ep_ratio - last_ep
        if not has_ep:
            self.ep_hold_steps = 0
            return {
                "ep_discipline": 0.0,
                "ep_low": False,
                "ep_critical": False,
                "ep_resume_event": False,
                "ep_conserve_action": False,
                "ep_low_skill_action": False,
            }

        if ep_ratio <= EP_HOLD_SKILL_RATIO:
            self.ep_hold_steps = min(self.ep_hold_steps + 2, 16)
        elif ep_ratio >= EP_RESUME_SKILL_RATIO:
            self.ep_hold_steps = max(0, self.ep_hold_steps - 3)
        else:
            self.ep_hold_steps = max(0, self.ep_hold_steps - 1)

        skill_action = action_head in HERO_SKILL_BUTTONS
        ep_low = ep_ratio <= EP_HOLD_SKILL_RATIO or self.ep_hold_steps > 0
        ep_critical = ep_ratio <= EP_CRITICAL_RATIO
        ep_resume_event = last_ep < EP_RESUME_SKILL_RATIO <= ep_ratio
        productive_skill = bool(
            dealt_hero_hurt
            or enemy_tower_delta > 0
            or enemy_minion_clear_delta > 0
            or money_delta > 0
            or exp_delta > 0
        )
        high_value_intent = bool(
            target_enemy_action
            and main.get("enemy_hp", 1.0) > 0.0
            and (
                main.get("enemy_hp", 1.0) <= 0.24
                or (
                    main.get("hero_config_id") == DIRENJIE_HERO_ID
                    and action_head == BUTTON_SKILL_3
                    and main.get("enemy_hp", 1.0) <= 0.62
                    and main.get("hero_hp", 0.0) >= 0.34
                    and ep_ratio > EP_CRITICAL_RATIO
                )
                or (
                    main.get("hero_config_id") == LUBAN_HERO_ID
                    and action_head == BUTTON_SKILL_3
                    and (target_enemy_action or target_soldier_action or attack_tower_action)
                    and ep_ratio > EP_CRITICAL_RATIO
                )
            )
        )
        useful_skill = productive_skill or high_value_intent
        conserve_action = bool(
            ep_low
            and not skill_action
            and (action_head in {BUTTON_ATTACK, BUTTON_MOVE} or target_soldier_action or attack_tower_action)
            and not took_hero_hurt
        )

        value = 0.0
        if skill_action and ep_critical and not useful_skill:
            value -= 0.13
        elif skill_action and ep_low and not useful_skill:
            value -= 0.09
        elif skill_action and ep_low and useful_skill:
            value -= 0.025

        if conserve_action:
            value += 0.045 if ep_critical else 0.032
        if ep_low and not skill_action and ep_delta >= -0.005:
            value += 0.025
        if ep_resume_event:
            value += 0.055
        if ep_critical and skill_action and took_hero_hurt and not dealt_hero_hurt:
            value -= 0.05

        return {
            "ep_discipline": _clip(value, -0.18, 0.16),
            "ep_low": ep_low,
            "ep_critical": ep_critical,
            "ep_resume_event": ep_resume_event,
            "ep_conserve_action": conserve_action,
            "ep_low_skill_action": bool(skill_action and ep_low),
        }
