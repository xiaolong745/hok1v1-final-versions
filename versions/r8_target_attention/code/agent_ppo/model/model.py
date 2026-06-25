#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import torch.nn as nn
from torch.nn import ModuleDict

import numpy as np
from typing import List

from agent_ppo.conf.conf import DimConfig, Config
from agent_ppo.feature.action_priority import build_torch_action_priority_bias


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # feature configure parameter
        # 特征配置参数
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.m_learning_rate = Config.INIT_LEARNING_RATE_START
        self.m_var_beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.restore_list = []
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.value_loss_coef = Config.VALUE_LOSS_COEF
        self.value_head_dim = Config.VALUE_HEAD_DIM
        self.value_head_loss_weights = torch.tensor(Config.VALUE_HEAD_LOSS_WEIGHTS, dtype=torch.float32)
        self.advantage_norm_eps = Config.ADVANTAGE_NORM_EPS
        self.policy_ratio_max = Config.POLICY_RATIO_MAX
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        self.legal_action_dim = np.sum(Config.LEGAL_ACTION_SIZE_LIST)
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE

        # NETWORK DIM
        # 网络维度
        self.hero_data_len = sum(Config.data_shapes[0])
        self.feature_dim = int(DimConfig.DIM_OF_FEATURE[0])
        self.dense_feature_dim = DimConfig.R8_DENSE_DIM
        self.discrete_feature_dim = DimConfig.R8_DISCRETE_DIM
        self.unit_slot_dim = DimConfig.R8_UNIT_SLOT_DIM
        self.unit_slot_count = DimConfig.R8_UNIT_SLOT_COUNT
        self.unit_feature_dim = DimConfig.R8_UNIT_FEATURE_DIM
        self.local_map_dim = DimConfig.R8_LANE_GRID_DIM
        self.temporal_dim = DimConfig.R8_TEMPORAL_DIM
        self.unit_embed_dim = 64
        self.base_embed_dim = 128
        self.discrete_embed_dim = 160
        self.local_map_embed_dim = 128
        self.temporal_embed_dim = 64
        self.target_entity_count = self.label_size_list[-1]
        # Target order: 0 none, 1 enemy hero, 2 self, 3-6 enemy minions, 7 enemy tower, 8 monster.
        # Values are indices in the R8 unit-slot layout from r8_feature_process.py; -1 is a zero slot.
        self.target_slot_indices = (-1, 1, 0, 8, 9, 10, 11, 3, 12)
        if len(self.target_slot_indices) != self.target_entity_count:
            raise ValueError("target slot mapping does not match target head size")

        self.base_mlp = MLP([self.dense_feature_dim, 256, self.base_embed_dim], "base_mlp", non_linearity_last=True)
        self.discrete_mlp = MLP([self.discrete_feature_dim, 384, self.discrete_embed_dim], "discrete_mlp", non_linearity_last=True)
        self.unit_mlp = MLP([self.unit_slot_dim, 96, self.unit_embed_dim], "unit_mlp", non_linearity_last=True)
        self.target_entity_mlp = MLP(
            [self.unit_slot_dim, 64, self.target_embed_dim],
            "target_entity_mlp",
            non_linearity_last=True,
        )
        self.local_map_mlp = MLP(
            [self.local_map_dim, 256, self.local_map_embed_dim],
            "local_map_mlp",
            non_linearity_last=True,
        )
        self.temporal_mlp = MLP([self.temporal_dim, 128, self.temporal_embed_dim], "temporal_mlp", non_linearity_last=True)
        concat_dim = (
            self.base_embed_dim
            + self.discrete_embed_dim
            + self.unit_embed_dim * 2
            + self.local_map_embed_dim
            + self.temporal_embed_dim
        )
        self.concat_mlp = MLP([concat_dim, 512, self.lstm_hidden_dim], "concat_mlp", non_linearity_last=True)

        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        self.label_mlp = ModuleDict(
            {
                "hero_label{0}_mlp".format(label_index): MLP(
                    [self.lstm_hidden_dim, 256, self.label_size_list[label_index]],
                    "hero_label{0}_mlp".format(label_index),
                )
                for label_index in range(len(self.label_size_list) - 1)
            }
        )
        self.target_query_mlp = MLP([self.lstm_hidden_dim, 128, self.target_embed_dim], "target_query_mlp")

        self.value_mlp = MLP([self.lstm_hidden_dim, 256, self.value_head_dim], "hero_value_mlp")

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list

        sequence_len = 1 if inference else self.lstm_time_steps
        feature_embed, target_units = self._encode_feature(feature_vec)
        lstm_input = feature_embed.reshape(-1, sequence_len, self.lstm_unit_size)
        lstm_hidden_init = lstm_hidden_init.reshape(-1, self.lstm_unit_size).unsqueeze(0)
        lstm_cell_init = lstm_cell_init.reshape(-1, self.lstm_unit_size).unsqueeze(0)
        lstm_output, (self.lstm_hidden_output, self.lstm_cell_output) = self.lstm(
            lstm_input,
            (lstm_hidden_init, lstm_cell_init),
        )

        result_list = []

        # public concat
        # 公共连接层
        fc_public_result = lstm_output.reshape(-1, self.lstm_unit_size)

        # output label
        # 输出标签
        for label_index, label_dim in enumerate(self.label_size_list[:-1]):
            label_mlp_out = self.label_mlp["hero_label{0}_mlp".format(label_index)](fc_public_result)
            result_list.append(label_mlp_out)
        result_list.append(self._target_attention_logits(fc_public_result, target_units))

        # output value
        # 输出价值
        value_result = self.value_mlp(fc_public_result)
        result_list.append(value_result)

        # prepare for infer graph
        # 准备推理图
        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]

        if inference:
            return [logits, value, self.lstm_cell_output, self.lstm_hidden_output]
        else:
            return result_list

    def _encode_feature(self, feature_vec):
        idx = 0
        dense = feature_vec[:, idx : idx + self.dense_feature_dim]
        idx += self.dense_feature_dim
        discrete = feature_vec[:, idx : idx + self.discrete_feature_dim]
        idx += self.discrete_feature_dim
        unit_group = feature_vec[:, idx : idx + self.unit_feature_dim]
        idx += self.unit_feature_dim
        local_map = feature_vec[:, idx : idx + self.local_map_dim]
        idx += self.local_map_dim
        temporal = feature_vec[:, idx : idx + self.temporal_dim]
        idx += self.temporal_dim
        if idx != self.feature_dim:
            raise ValueError(f"feature split mismatch: {idx} != {self.feature_dim}")

        pooled_unit_max, pooled_unit_mean = self._pool_unit_group(unit_group)
        target_units = self._select_target_unit_group(unit_group)
        encoded = [
            self.base_mlp(dense),
            self.discrete_mlp(discrete),
            pooled_unit_max,
            pooled_unit_mean,
            self.local_map_mlp(local_map),
            self.temporal_mlp(temporal),
        ]
        return self.concat_mlp(torch.cat(encoded, dim=1)), target_units

    def _select_target_unit_group(self, group_vec):
        batch_size = group_vec.shape[0]
        units = group_vec.reshape(batch_size, self.unit_slot_count, self.unit_slot_dim)
        zero_unit = torch.zeros(batch_size, self.unit_slot_dim, dtype=group_vec.dtype, device=group_vec.device)
        target_units = []
        for slot_index in self.target_slot_indices:
            if 0 <= slot_index < self.unit_slot_count:
                target_units.append(units[:, slot_index, :])
            else:
                target_units.append(zero_unit)
        return torch.stack(target_units, dim=1)

    def _target_attention_logits(self, public_result, target_units):
        target_embed = self.target_entity_mlp(target_units.reshape(-1, self.unit_slot_dim)).reshape(
            -1,
            self.target_entity_count,
            self.target_embed_dim,
        )
        target_query = self.target_query_mlp(public_result).reshape(-1, self.target_embed_dim, 1)
        return torch.matmul(target_embed, target_query).reshape(-1, self.target_entity_count)

    def _pool_unit_group(self, group_vec):
        batch_size = group_vec.shape[0]
        units = group_vec.reshape(batch_size, self.unit_slot_count, self.unit_slot_dim)
        flat_units = units.reshape(batch_size * self.unit_slot_count, self.unit_slot_dim)
        unit_embed = self.unit_mlp(flat_units).reshape(batch_size, self.unit_slot_count, self.unit_embed_dim)
        present = (torch.sum(torch.abs(units), dim=2, keepdim=True) > 0).float()
        masked_embed = unit_embed.masked_fill(present <= 0, -10000.0)
        pooled_max = torch.max(masked_embed, dim=1).values
        has_any = torch.max(present, dim=1).values
        sum_embed = torch.sum(unit_embed * present, dim=1)
        mean_embed = sum_embed / torch.clamp(torch.sum(present, dim=1), min=1.0)
        return pooled_max * has_any, mean_embed * has_any

    def compute_loss(self, data_list, rst_list):
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])
        usq_advantage = data_list[2].reshape(-1, self.data_split_shape[2])
        usq_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3])

        usq_label_list = data_list[3 : 3 + len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = (
                usq_label_list[shape_index].reshape(-1, self.data_split_shape[3 + shape_index]).long()
            )

        old_label_probability_list = data_list[3 + len(self.label_size_list) : 3 + 2 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(
                -1, self.data_split_shape[3 + len(self.label_size_list) + shape_index]
            )

        usq_weight_list = data_list[3 + 2 * len(self.label_size_list) : 3 + 3 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(
                -1,
                self.data_split_shape[3 + 2 * len(self.label_size_list) + shape_index],
            )

        # squeeze tensor
        # 压缩张量
        reward = usq_reward
        advantage = usq_advantage.squeeze(dim=1)
        label_list = []
        for ele in usq_label_list:
            label_list.append(ele.squeeze(dim=1))
        weight_list = []
        for weight in usq_weight_list:
            weight_list.append(weight.squeeze(dim=1))
        frame_is_train = usq_is_train.squeeze(dim=1)

        label_result = rst_list[:-1]

        value_result = rst_list[-1]

        feature_vec, split_feature_legal_action = torch.split(
            seri_vec,
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(feature_legal_action_shape)

        legal_action_flag_list = torch.split(feature_legal_action, self.label_size_list, dim=1)
        action_priority_bias = None
        if getattr(Config, "ACTION_PRIORITY_ENABLED", False):
            action_priority_bias = build_torch_action_priority_bias(
                feature_vec,
                self.label_size_list,
                getattr(Config, "ACTION_PRIORITY_TRAIN_SCALE", 0.0),
            )

        # loss of value net
        # 值网络的损失
        train_weight = frame_is_train.float()
        train_weight_sum = torch.clamp(train_weight.sum(), min=1.0)
        value_error = reward - value_result
        value_cost_by_head = 0.5 * torch.sum(
            torch.square(value_error) * train_weight.unsqueeze(1),
            dim=0,
        ) / train_weight_sum
        self.value_cost_by_head = value_cost_by_head
        value_head_loss_weights = self.value_head_loss_weights.to(value_error.device)
        self.value_cost = torch.sum(value_cost_by_head * value_head_loss_weights) / torch.clamp(
            value_head_loss_weights.sum(),
            min=1e-6,
        )

        advantage_mean = torch.sum(advantage * train_weight) / train_weight_sum
        advantage_var = torch.sum(torch.square(advantage - advantage_mean) * train_weight) / train_weight_sum
        normalized_advantage = (advantage - advantage_mean) * torch.rsqrt(
            advantage_var + self.advantage_norm_eps
        )
        advantage = torch.where(train_weight_sum > 1.0, normalized_advantage, advantage) * train_weight

        # for entropy loss calculate
        # 用于熵损失计算
        label_probability_list = []

        epsilon = 1e-5

        # policy loss: ppo clip loss
        # 策略损失：PPO剪辑损失
        joint_log_p = torch.zeros_like(advantage)
        old_joint_log_p = torch.zeros_like(advantage)
        joint_weight_sum = torch.zeros_like(advantage)
        joint_weight_count = torch.zeros_like(advantage)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                boundary = torch.tensor(1e20, device=label_result[task_index].device)
                one_hot_actions = nn.functional.one_hot(label_list[task_index].long(), self.label_size_list[task_index])

                legal_action_flag_list_max_mask = (1 - legal_action_flag_list[task_index]) * boundary
                current_label_result = label_result[task_index]
                if action_priority_bias is not None:
                    current_label_result = current_label_result + action_priority_bias[task_index]

                label_logits_subtract_max = torch.clamp(
                    current_label_result
                    - torch.max(
                        current_label_result - legal_action_flag_list_max_mask,
                        dim=1,
                        keepdim=True,
                    ).values,
                    -boundary,
                    1,
                )

                label_exp_logits = legal_action_flag_list[task_index] * (
                    torch.exp(label_logits_subtract_max) + self.min_policy
                )

                label_sum_exp_logits = torch.clamp(label_exp_logits.sum(1, keepdim=True), min=epsilon)

                label_probability = 1.0 * label_exp_logits / label_sum_exp_logits
                label_probability_list.append(label_probability)

                policy_p = (one_hot_actions * label_probability).sum(1)
                policy_log_p = torch.log(policy_p + epsilon)
                old_policy_p = (one_hot_actions * old_label_probability_list[task_index] + epsilon).sum(1)
                old_policy_log_p = torch.log(old_policy_p)
                head_weight = weight_list[task_index].float() * train_weight
                head_active = (head_weight > 0).float()
                joint_log_p = joint_log_p + policy_log_p * head_active
                old_joint_log_p = old_joint_log_p + old_policy_log_p * head_active
                joint_weight_sum = joint_weight_sum + head_weight
                joint_weight_count = joint_weight_count + head_active

        joint_weight = torch.where(
            joint_weight_count > 0,
            joint_weight_sum / torch.clamp(joint_weight_count, min=1.0),
            torch.zeros_like(joint_weight_sum),
        )
        log_ratio = joint_log_p - old_joint_log_p
        ratio = torch.exp(log_ratio).clamp(0.0, self.policy_ratio_max)
        surr1 = ratio * advantage
        surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
        self.policy_cost = -torch.sum(torch.minimum(surr1, surr2) * joint_weight) / torch.clamp(
            joint_weight.sum(),
            min=1.0,
        )

        # cross entropy loss
        # 交叉熵损失
        current_entropy_loss_index = 0
        entropy_loss_list = []
        entropy_weighted_sum = torch.zeros_like(advantage)
        entropy_weight_sum = torch.zeros_like(advantage)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                temp_entropy_loss = -torch.sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * torch.log(label_probability_list[current_entropy_loss_index] + epsilon),
                    dim=1,
                )

                head_weight = weight_list[task_index].float() * train_weight
                entropy_weighted_sum = entropy_weighted_sum + temp_entropy_loss * head_weight
                entropy_weight_sum = entropy_weight_sum + head_weight
                temp_entropy_loss = -torch.sum(temp_entropy_loss * head_weight) / torch.clamp(
                    head_weight.sum(),
                    min=1.0,
                )

                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index = current_entropy_loss_index + 1
            else:
                temp_entropy_loss = torch.zeros_like(advantage).sum()
                entropy_loss_list.append(temp_entropy_loss)

        self.entropy_cost = -torch.sum(entropy_weighted_sum) / torch.clamp(entropy_weight_sum.sum(), min=1.0)

        self.entropy_cost_list = entropy_loss_list

        self.loss = self.value_loss_coef * self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost

        return self.loss, [
            self.loss,
            [self.value_cost, self.policy_cost, self.entropy_cost],
            self.value_cost_by_head,
        ]

    def set_train_mode(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        self.eval()


def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)

    nn.init.orthogonal(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)

    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
