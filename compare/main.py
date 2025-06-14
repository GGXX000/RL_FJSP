import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
import os
import glob
import argparse
import sys

# 添加当前目录到Python路径，帮助解决导入问题
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs import *
from GNN_DQN_model import *  # 导入新的GNN-DQN模型
from utils import *  # 导入修改后的工具函数

from config import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='作业车间调度强化学习 (GNN版本)')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test',
                        help='运行模式: train-训练, test-测试')
    parser.add_argument('--data_dir', type=str, default="datasets/job_shop_datasets",
                        help='训练数据集目录')
    parser.add_argument('--test_dir', type=str, default="datasets/test_datasets",
                        help='测试数据集目录')
    parser.add_argument('--num_machines', type=int, default=3,
                        help='机器数量')
    parser.add_argument('--model_dir', type=str, default="models",
                        help='模型保存目录')
    parser.add_argument('--episodes', type=int, default=20,
                        help='每个数据集训练的回合数')
    parser.add_argument('--tests', type=int, default=5,
                        help='每个数据集测试的次数')
    parser.add_argument('--task_id', type=int, default=0,
                        help='要分析的特定任务ID')
    parser.add_argument('--dataset', type=str, default=None,
                        help='指定要使用的数据集文件')
    parser.add_argument('--no_charts', action='store_true',
                        help='不生成甘特图')
    # 在参数解析器中添加评估相关参数
    parser.add_argument('--eval_dir', type=str, default="datasets/evaluate_datasets",
                        help='评估数据集目录')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='评估间隔（每训练多少个回合进行一次评估），设为0关闭评估')

    args = parser.parse_args()

    # 检查是否有可用的CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 在训练部分修改调用参数，添加评估参数
    if args.mode == 'train':
        # 训练智能体
        print("\n开始训练GNN-DQN智能体...")
        agent = train_multi_dataset(
            data_dir=args.data_dir,
            episodes_per_dataset=args.episodes,
            num_machines=args.num_machines,
            device=device,
            model_save_dir=args.model_dir,
            log_interval=5,
            eval_dir=args.eval_dir,
            eval_interval=args.eval_interval
        )

    elif args.mode == 'test':
        # 初始化环境以获取图节点信息
        sample_data_files = sorted(glob.glob(os.path.join(args.test_dir, "*.txt")))
        if not sample_data_files:
            print(f"错误：在 {args.test_dir} 目录下未找到数据文件")
            exit(1)

        sample_env = JobShopEnv(sample_data_files[0], args.num_machines)
        sample_state = sample_env.reset()
        graph_state = state_to_graph(sample_state, sample_env)

        # 获取维度信息
        node_features_dim = graph_state['node_features'].shape[1]
        edge_features_dim = 1  # 简单二元邻接矩阵
        global_state_dim = len(graph_state['global_state'])

        # 计算动作空间大小
        action_size = Config.get_action_size(args.num_machines)

        # 加载GNN-DQN模型
        agent = GNN_DQNAgent(
            node_features=node_features_dim,
            edge_features=edge_features_dim,
            global_features=global_state_dim,
            action_size=action_size,
            device=device
        )

        model_path = os.path.join(args.model_dir, "job_shop_gnn_dqn_final.pth")
        if not agent.load(model_path):
            model_path = os.path.join(args.model_dir, "job_shop_gnn_dqn_latest.pth")
            if not agent.load(model_path):
                print(f"错误: 在 {args.model_dir} 目录下未找到GNN模型文件")

                # 尝试加载旧版本DQN模型并转换
                old_model_path = os.path.join(args.model_dir, "job_shop_dqn_final.pth")
                if os.path.exists(old_model_path):
                    print(f"找到旧版本DQN模型: {old_model_path}")
                    print("警告: 无法直接加载旧模型到GNN架构，将使用随机初始化的GNN模型")
                else:
                    exit(1)

        # 测试智能体
        print("\n开始测试GNN-DQN智能体...")

        # 使用指定的测试目录
        test_dir = args.test_dir
        if args.dataset:
            if os.path.isfile(args.dataset):
                test_dir = os.path.dirname(args.dataset)
            else:
                print(f"错误: 指定的数据集文件 {args.dataset} 不存在")
                exit(1)

        if not os.path.isdir(test_dir):
            print(f"错误: 测试目录 {test_dir} 不存在")
            exit(1)

        test_agent(
            agent=agent,
            data_dir=test_dir,  # 使用测试目录
            num_machines=args.num_machines,
            num_tests=args.tests,
            generate_charts=not args.no_charts
        )