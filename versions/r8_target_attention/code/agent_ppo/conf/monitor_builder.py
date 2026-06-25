#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    This function is used to create monitoring panel configurations for custom indicators.
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("智能决策1v1")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="累积回报",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="round(avg(reward{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="round(avg(total_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="round(avg(value_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="R8价值头",
            name_en="r8_value_heads",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss_total",
            expr="round(avg(value_loss_total{}), 0.01)",
        )
        .add_metric(
            metrics_name="value_loss_objective",
            expr="round(avg(value_loss_objective{}), 0.01)",
        )
        .add_metric(
            metrics_name="value_loss_lane",
            expr="round(avg(value_loss_lane{}), 0.01)",
        )
        .add_metric(
            metrics_name="value_loss_combat",
            expr="round(avg(value_loss_combat{}), 0.01)",
        )
        .add_metric(
            metrics_name="value_loss_survival",
            expr="round(avg(value_loss_survival{}), 0.01)",
        )
        .add_metric(
            metrics_name="value_loss_resource",
            expr="round(avg(value_loss_resource{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="round(avg(policy_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="round(avg(entropy_loss{}), 0.01)",
        )
        .end_panel()
        .end_group()
        .add_group(
            group_name="eval matchup",
            group_name_en="eval_matchup",
        )
        .add_panel(
            name="eval win",
            name_en="eval_win",
            type="line",
        )
        .add_metric(metrics_name="eval_112v112_win", expr="round(avg(eval_112v112_win{}), 0.01)")
        .add_metric(metrics_name="eval_112v133_win", expr="round(avg(eval_112v133_win{}), 0.01)")
        .add_metric(metrics_name="eval_133v112_win", expr="round(avg(eval_133v112_win{}), 0.01)")
        .add_metric(metrics_name="eval_133v133_win", expr="round(avg(eval_133v133_win{}), 0.01)")
        .end_panel()
        .add_panel(
            name="eval tower",
            name_en="eval_tower",
            type="line",
        )
        .add_metric(metrics_name="eval_112v112_self_tower_hp", expr="round(avg(eval_112v112_self_tower_hp{}), 0.01)")
        .add_metric(metrics_name="eval_112v112_enemy_tower_hp", expr="round(avg(eval_112v112_enemy_tower_hp{}), 0.01)")
        .add_metric(metrics_name="eval_112v133_self_tower_hp", expr="round(avg(eval_112v133_self_tower_hp{}), 0.01)")
        .add_metric(metrics_name="eval_112v133_enemy_tower_hp", expr="round(avg(eval_112v133_enemy_tower_hp{}), 0.01)")
        .add_metric(metrics_name="eval_133v112_self_tower_hp", expr="round(avg(eval_133v112_self_tower_hp{}), 0.01)")
        .add_metric(metrics_name="eval_133v112_enemy_tower_hp", expr="round(avg(eval_133v112_enemy_tower_hp{}), 0.01)")
        .add_metric(metrics_name="eval_133v133_self_tower_hp", expr="round(avg(eval_133v133_self_tower_hp{}), 0.01)")
        .add_metric(metrics_name="eval_133v133_enemy_tower_hp", expr="round(avg(eval_133v133_enemy_tower_hp{}), 0.01)")
        .end_panel()
        .add_panel(
            name="eval frame",
            name_en="eval_frame",
            type="line",
        )
        .add_metric(metrics_name="eval_112v112_frame", expr="round(avg(eval_112v112_frame{}), 0.01)")
        .add_metric(metrics_name="eval_112v133_frame", expr="round(avg(eval_112v133_frame{}), 0.01)")
        .add_metric(metrics_name="eval_133v112_frame", expr="round(avg(eval_133v112_frame{}), 0.01)")
        .add_metric(metrics_name="eval_133v133_frame", expr="round(avg(eval_133v133_frame{}), 0.01)")
        .end_panel()
        .add_panel(
            name="eval timeout",
            name_en="eval_timeout",
            type="line",
        )
        .add_metric(metrics_name="eval_112v112_timeout", expr="round(avg(eval_112v112_timeout{}), 0.01)")
        .add_metric(metrics_name="eval_112v133_timeout", expr="round(avg(eval_112v133_timeout{}), 0.01)")
        .add_metric(metrics_name="eval_133v112_timeout", expr="round(avg(eval_133v112_timeout{}), 0.01)")
        .add_metric(metrics_name="eval_133v133_timeout", expr="round(avg(eval_133v133_timeout{}), 0.01)")
        .end_panel()
        .add_panel(
            name="eval kd",
            name_en="eval_kd",
            type="line",
        )
        .add_metric(metrics_name="eval_112v112_kill", expr="round(avg(eval_112v112_kill{}), 0.01)")
        .add_metric(metrics_name="eval_112v112_death", expr="round(avg(eval_112v112_death{}), 0.01)")
        .add_metric(metrics_name="eval_112v133_kill", expr="round(avg(eval_112v133_kill{}), 0.01)")
        .add_metric(metrics_name="eval_112v133_death", expr="round(avg(eval_112v133_death{}), 0.01)")
        .add_metric(metrics_name="eval_133v112_kill", expr="round(avg(eval_133v112_kill{}), 0.01)")
        .add_metric(metrics_name="eval_133v112_death", expr="round(avg(eval_133v112_death{}), 0.01)")
        .add_metric(metrics_name="eval_133v133_kill", expr="round(avg(eval_133v133_kill{}), 0.01)")
        .add_metric(metrics_name="eval_133v133_death", expr="round(avg(eval_133v133_death{}), 0.01)")
        .end_panel()
        .add_panel(
            name="eval damage",
            name_en="eval_damage",
            type="line",
        )
        .add_metric(metrics_name="eval_112v112_hurt_to_hero", expr="round(avg(eval_112v112_hurt_to_hero{}), 0.01)")
        .add_metric(metrics_name="eval_112v112_hurt_by_hero", expr="round(avg(eval_112v112_hurt_by_hero{}), 0.01)")
        .add_metric(metrics_name="eval_112v133_hurt_to_hero", expr="round(avg(eval_112v133_hurt_to_hero{}), 0.01)")
        .add_metric(metrics_name="eval_112v133_hurt_by_hero", expr="round(avg(eval_112v133_hurt_by_hero{}), 0.01)")
        .add_metric(metrics_name="eval_133v112_hurt_to_hero", expr="round(avg(eval_133v112_hurt_to_hero{}), 0.01)")
        .add_metric(metrics_name="eval_133v112_hurt_by_hero", expr="round(avg(eval_133v112_hurt_by_hero{}), 0.01)")
        .add_metric(metrics_name="eval_133v133_hurt_to_hero", expr="round(avg(eval_133v133_hurt_to_hero{}), 0.01)")
        .add_metric(metrics_name="eval_133v133_hurt_by_hero", expr="round(avg(eval_133v133_hurt_by_hero{}), 0.01)")
        .end_panel()
        .add_panel(
            name="eval money",
            name_en="eval_money",
            type="line",
        )
        .add_metric(metrics_name="eval_112v112_money_per_frame", expr="round(avg(eval_112v112_money_per_frame{}), 0.01)")
        .add_metric(metrics_name="eval_112v133_money_per_frame", expr="round(avg(eval_112v133_money_per_frame{}), 0.01)")
        .add_metric(metrics_name="eval_133v112_money_per_frame", expr="round(avg(eval_133v112_money_per_frame{}), 0.01)")
        .add_metric(metrics_name="eval_133v133_money_per_frame", expr="round(avg(eval_133v133_money_per_frame{}), 0.01)")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
