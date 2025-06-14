import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import time
import os
import glob
import copy
import math
from config import Config


class GraphConvLayer(nn.Module):
    """图卷积层，用于处理节点间的消息传递"""

    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 节点特征变换
        self.node_transform = nn.Linear(in_features, out_features)

        # 边特征变换
        self.edge_transform = nn.Linear(in_features, out_features)

        # 更新门，控制信息流
        self.update_gate = nn.Linear(in_features + out_features, out_features)

        # 重置门
        self.reset_gate = nn.Linear(in_features + out_features, out_features)

        # 候选隐藏状态
        self.candidate = nn.Linear(in_features + out_features, out_features)

    def forward(self, x, adj):
        """
        前向传播
        x: 节点特征 [batch_size, num_nodes, in_features]
        adj: 邻接矩阵 [batch_size, num_nodes, num_nodes]
        """
        batch_size, num_nodes, _ = x.size()

        # 特征变换
        transformed = self.node_transform(x)  # [batch_size, num_nodes, out_features]

        # 消息传递
        # 对每个节点，收集来自邻居的信息
        messages = torch.bmm(adj, transformed)  # [batch_size, num_nodes, out_features]

        # 计算更新门
        update_input = torch.cat([x, messages], dim=2)
        update = torch.sigmoid(self.update_gate(update_input))

        # 计算重置门
        reset = torch.sigmoid(self.reset_gate(update_input))

        # 计算候选隐藏状态
        candidate_input = torch.cat([x, reset * messages], dim=2)
        candidate_hidden = torch.tanh(self.candidate(candidate_input))

        # 更新节点表示
        output = (1 - update) * x[:, :, :self.out_features] + update * candidate_hidden

        return output


class GraphAttentionLayer(nn.Module):
    """图注意力层，考虑节点间的注意力权重"""

    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # 特征变换
        self.W = nn.Linear(in_features, out_features, bias=False)

        # 注意力机制
        self.a = nn.Linear(2 * out_features, 1, bias=False)

        # 激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        """
        前向传播
        x: 节点特征 [batch_size, num_nodes, in_features]
        adj: 邻接矩阵 [batch_size, num_nodes, num_nodes]
        """
        batch_size, num_nodes, _ = x.size()

        # 特征变换
        h = self.W(x)  # [batch_size, num_nodes, out_features]

        # 计算注意力系数
        a_input = torch.cat([h.repeat(1, 1, num_nodes).view(batch_size, num_nodes * num_nodes, -1),
                             h.repeat(1, num_nodes, 1)], dim=2)
        a_input = a_input.view(batch_size, num_nodes, num_nodes, 2 * self.out_features)

        e = self.leakyrelu(self.a(a_input).squeeze(3))

        # 应用掩码（邻接矩阵中为0的位置应该被屏蔽）
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # 对每个节点的邻居应用softmax得到注意力权重
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 应用注意力权重
        h_prime = torch.bmm(attention, h)

        return h_prime


class JobShopGraphEncoder(nn.Module):
    """用于作业车间调度问题的图编码器"""

    def __init__(self, node_features, hidden_dim, num_layers=2, use_attention=True):
        super(JobShopGraphEncoder, self).__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention

        # 初始节点特征变换
        self.node_embedding = nn.Linear(node_features, hidden_dim)

        # 图卷积/注意力层
        self.graph_layers = nn.ModuleList()
        for i in range(num_layers):
            if use_attention:
                self.graph_layers.append(GraphAttentionLayer(hidden_dim, hidden_dim))
            else:
                self.graph_layers.append(GraphConvLayer(hidden_dim, hidden_dim))

        # 全局池化层
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, node_features, adj_matrix):
        """
        前向传播
        node_features: 节点特征 [batch_size, num_nodes, node_features]
        adj_matrix: 邻接矩阵 [batch_size, num_nodes, num_nodes]
        """
        # 初始节点嵌入
        x = self.node_embedding(node_features)

        # 图卷积/注意力层
        for layer in self.graph_layers:
            x = F.relu(layer(x, adj_matrix))

        # 全局池化：获取图级别表示
        # 使用节点特征的平均值和最大值
        global_avg_pool = torch.mean(x, dim=1)
        global_max_pool, _ = torch.max(x, dim=1)
        global_rep = global_avg_pool + global_max_pool

        # 进一步处理全局表示
        global_rep = self.global_pool(global_rep)

        return x, global_rep


class GNN_DQN(nn.Module):
    """
    基于图神经网络的普通DQN (非Dueling架构)
    结合了图结构信息和全局状态信息
    """

    def __init__(self, node_features, edge_features, global_features, action_size, hidden_dim=128, graph_hidden_dim=64):
        super(GNN_DQN, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.global_features = global_features
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.graph_hidden_dim = graph_hidden_dim

        # 图编码器
        self.graph_encoder = JobShopGraphEncoder(
            node_features=node_features,
            hidden_dim=graph_hidden_dim,
            num_layers=Config.GNN_LAYERS,
            use_attention=Config.USE_ATTENTION
        )

        # 全局状态编码器
        self.global_encoder = nn.Sequential(
            nn.Linear(global_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 合并图表示和全局状态
        merged_dim = graph_hidden_dim + hidden_dim

        # 普通DQN架构：直接输出每个动作的Q值
        self.q_network = nn.Sequential(
            nn.Linear(merged_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(hidden_dim // 2, action_size)
        )

    def forward(self, node_features, adj_matrix, global_state):
        """
        前向传播
        node_features: 节点特征 [batch_size, num_nodes, node_features]
        adj_matrix: 邻接矩阵 [batch_size, num_nodes, num_nodes]
        global_state: 全局状态 [batch_size, global_features]
        """
        # 图编码
        _, graph_rep = self.graph_encoder(node_features, adj_matrix)

        # 全局状态编码
        global_rep = self.global_encoder(global_state)

        # 合并表示
        combined = torch.cat([graph_rep, global_rep], dim=1)

        # 计算Q值
        q_values = self.q_network(combined)

        return q_values


class GraphBatchBuffer:
    """用于存储和取样图结构数据的经验回放缓冲区"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, node_features, adj_matrix, global_state, action, reward,
             next_node_features, next_adj_matrix, next_global_state, done):
        """存储经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (
            node_features, adj_matrix, global_state, action, reward,
            next_node_features, next_adj_matrix, next_global_state, done
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))

        # 解包批次
        node_features, adj_matrix, global_state, action, reward, \
        next_node_features, next_adj_matrix, next_global_state, done = zip(*batch)

        # 转换为Tensor，处理可能存在的不同形状
        # 对于节点特征和邻接矩阵，我们需要单独处理每个样本
        node_features_tensor = [torch.FloatTensor(nf) for nf in node_features]
        adj_matrix_tensor = [torch.FloatTensor(adj) for adj in adj_matrix]
        next_node_features_tensor = [torch.FloatTensor(nnf) for nnf in next_node_features]
        next_adj_matrix_tensor = [torch.FloatTensor(nadj) for nadj in next_adj_matrix]

        # 全局状态、动作、奖励和完成标志可以直接转换
        global_state_tensor = torch.FloatTensor(np.array(global_state))
        action_tensor = torch.LongTensor(np.array(action))
        reward_tensor = torch.FloatTensor(np.array(reward))
        next_global_state_tensor = torch.FloatTensor(np.array(next_global_state))
        done_tensor = torch.FloatTensor(np.array(done))

        return (
            node_features_tensor,
            adj_matrix_tensor,
            global_state_tensor,
            action_tensor,
            reward_tensor,
            next_node_features_tensor,
            next_adj_matrix_tensor,
            next_global_state_tensor,
            done_tensor
        )

    def __len__(self):
        """返回缓冲区大小"""
        return len(self.buffer)

class GNN_DQNAgent:
    """使用图神经网络的DQN智能体"""

    def __init__(self, node_features, edge_features, global_features, action_size, device='cpu'):
        self.node_features = node_features
        self.edge_features = edge_features
        self.global_features = global_features
        self.action_size = action_size
        self.device = device

        # 超参数设置
        self.gamma = Config.GAMMA  # 折扣因子
        self.epsilon = Config.EPSILON  # 探索率
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        self.learning_rate = Config.LEARNING_RATE

        # 网络初始化 - 使用配置中的HIDDEN_SIZE参数
        hidden_dim = Config.HIDDEN_SIZE
        graph_hidden_dim = Config.GNN_HIDDEN_SIZE

        self.policy_net = GNN_DQN(
            node_features, edge_features, global_features, action_size,
            hidden_dim, graph_hidden_dim
        ).to(device)

        self.target_net = GNN_DQN(
            node_features, edge_features, global_features, action_size,
            hidden_dim, graph_hidden_dim
        ).to(device)

        # 复制策略网络参数到目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器 - 使用Adam并添加L2正则化
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate,
            weight_decay=Config.WEIGHT_DECAY  # L2正则化
        )

        # 损失函数 - 使用平滑L1损失（Huber损失）提高稳定性
        self.criterion = nn.SmoothL1Loss()

        # 经验回放缓冲区
        self.memory = GraphBatchBuffer(Config.MEMORY_SIZE)

        # 全局步数
        self.global_steps = 0

    def remember(self, state_dict, action, reward, next_state_dict, done):
        """
        存储经验到回放缓冲区
        state_dict: 包含图状态信息的字典
        """
        self.memory.push(
            state_dict['node_features'], state_dict['adj_matrix'], state_dict['global_state'],
            action, reward,
            next_state_dict['node_features'], next_state_dict['adj_matrix'], next_state_dict['global_state'],
            done
        )

    def act(self, state_dict):
        """
        基于当前状态选择动作
        state_dict: 包含图状态信息的字典
        """
        # 更新全局步数
        self.global_steps += 1

        # epsilon-贪婪策略
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            # 转换状态为Tensor
            node_features = torch.FloatTensor(state_dict['node_features']).unsqueeze(0).to(self.device)
            adj_matrix = torch.FloatTensor(state_dict['adj_matrix']).unsqueeze(0).to(self.device)
            global_state = torch.FloatTensor(state_dict['global_state']).unsqueeze(0).to(self.device)

            # 获取Q值
            q_values = self.policy_net(node_features, adj_matrix, global_state)
            return q_values.cpu().data.numpy().argmax()

    def replay(self, batch_size):
        """从经验回放缓冲区中学习"""
        if len(self.memory) < batch_size:
            return 0

        # 采样批次
        node_features, adj_matrix, global_state, action, reward, \
        next_node_features, next_adj_matrix, next_global_state, done = self.memory.sample(batch_size)

        # 转移到设备
        global_state = global_state.to(self.device)
        action = action.unsqueeze(1).to(self.device)
        reward = reward.unsqueeze(1).to(self.device)
        next_global_state = next_global_state.to(self.device)
        done = done.unsqueeze(1).to(self.device)

        # 初始化批处理张量
        batch_size = len(node_features)

        # 分别处理每个样本，收集Q值
        q_values_list = []
        next_q_values_list = []

        for i in range(batch_size):
            # 将单个样本的张量移动到设备
            nf = node_features[i].unsqueeze(0).to(self.device)
            adj = adj_matrix[i].unsqueeze(0).to(self.device)
            gs = global_state[i].unsqueeze(0)

            # 当前状态的Q值
            current_q = self.policy_net(nf, adj, gs)
            q_values_list.append(current_q)

            # 下一个状态的Q值
            nnf = next_node_features[i].unsqueeze(0).to(self.device)
            nadj = next_adj_matrix[i].unsqueeze(0).to(self.device)
            ngs = next_global_state[i].unsqueeze(0)

            with torch.no_grad():
                # Double DQN: 使用policy_net选择动作
                next_action = self.policy_net(nnf, nadj, ngs).max(1)[1].unsqueeze(1)
                # 使用target_net评估
                next_q = self.target_net(nnf, nadj, ngs).gather(1, next_action)
                next_q_values_list.append(next_q)

        # 合并所有样本的Q值
        q_values = torch.cat(q_values_list, dim=0).gather(1, action)
        next_q_values = torch.cat(next_q_values_list, dim=0)

        # 计算目标Q值
        target_q_values = reward + (self.gamma * next_q_values * (1 - done))

        # 计算损失
        loss = self.criterion(q_values, target_q_values)

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=Config.GRADIENT_CLIP)

        self.optimizer.step()

        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'global_steps': self.global_steps
        }, filename)
        print(f"模型已保存到 {filename}")

    def load(self, filename):
        """加载模型"""
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            if 'global_steps' in checkpoint:
                self.global_steps = checkpoint['global_steps']
            print(f"已加载模型: {filename}")
            return True
        return False