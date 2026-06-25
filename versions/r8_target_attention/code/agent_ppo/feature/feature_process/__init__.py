#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.feature.feature_process.hero_process import HeroProcess
from agent_ppo.feature.feature_process.organ_process import OrganProcess
from agent_ppo.feature.feature_process.tactical_process import TacticalProcess
from agent_ppo.feature.feature_process.local_map_process import LocalMapProcess
from agent_ppo.feature.feature_process.unit_group_process import UnitGroupProcess
from agent_ppo.feature.feature_process.r8_feature_process import R8FeatureProcess


class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.tactical_process = TacticalProcess(camp)
        self.local_map_process = LocalMapProcess(camp)
        self.unit_group_process = UnitGroupProcess(camp)
        self.r8_feature_process = R8FeatureProcess(camp)

    def reset(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.tactical_process = TacticalProcess(camp)
        self.local_map_process = LocalMapProcess(camp)
        self.unit_group_process = UnitGroupProcess(camp)
        self.r8_feature_process = R8FeatureProcess(camp)

    def process_organ_feature(self, frame_state):
        return self.organ_process.process_vec_organ(frame_state)

    def process_hero_feature(self, frame_state):
        return self.hero_process.process_vec_hero(frame_state)

    def process_tactical_feature(self, observation):
        return self.tactical_process.process_vec_tactical(observation)

    def process_local_map_feature(self, observation):
        return self.local_map_process.process_vec_local_map(observation)

    def process_unit_group_feature(self, observation):
        return self.unit_group_process.process_vec_unit_group(observation)

    def process_feature(self, observation):
        return self.r8_feature_process.process_feature(observation)

    def process_legacy_feature(self, observation):
        frame_state = observation["frame_state"]

        main_camp_hero_vector_feature = self.process_hero_feature(frame_state)
        organ_feature = self.process_organ_feature(frame_state)
        tactical_feature = self.process_tactical_feature(observation)
        unit_group_feature = self.process_unit_group_feature(observation)
        local_map_feature = self.process_local_map_feature(observation)

        base_feature = main_camp_hero_vector_feature + organ_feature + tactical_feature
        feature = base_feature + unit_group_feature + local_map_feature

        return feature
