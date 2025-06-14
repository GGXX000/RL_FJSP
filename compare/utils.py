import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
import os
import glob
from envs import *
from GNN_DQN_model import *  # 导入新的GNN模型
from config import Config
import matplotlib.pyplot as plt
from envs import JobShopEnv
from validation_logger import ValidationLogger

# 导入甘特图绘制模块
try:
    from gantt_chart_utils import generate_all_charts
    has_gantt_utils = True
except ImportError:
    has_gantt_utils = False
    print("提示: 甘特图工具未找到，将不会生成可视化图表")

import numpy as np
from config import Config


def print_gpu_stats():
    if torch.cuda.is_available():
        print(f"\n===== GPU内存使用情况 =====")
        print(f"分配内存: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        print(f"缓存内存: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")
        print(f"最大分配内存: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")

        # 如果有多个GPU，可以遍历所有设备
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  已用内存: {torch.cuda.memory_allocated(i) / (1024 ** 3):.2f} GB")

def state_to_graph(state, env):
    """
    将状态转换为图结构表示

    返回:
    - node_features: 节点特征矩阵
    - adj_matrix: 邻接矩阵
    - global_state: 全局状态向量
    """
    # 构建图结构
    # 节点类型: 0=机器, 1=任务, 2=工序类型
    num_machines = env.num_machines
    num_jobs = env.num_jobs
    num_op_types = env.num_operation_types

    # 总节点数 = 机器数 + 任务数 + 工序类型数
    num_nodes = num_machines + num_jobs + num_op_types

    # 节点特征维度: 节点类型(3) + 机器属性(3) + 任务属性(3) + 工序类型属性(3) = 12
    node_features = np.zeros((num_nodes, 12), dtype=np.float32)

    # 初始化邻接矩阵
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # 预计算各工序类型的统计信息
    op_type_counts = np.zeros(num_op_types + 1)  # +1 因为工序类型通常从1开始索引
    op_type_total_time = np.zeros(num_op_types + 1)
    total_operations = 0

    # 计算各工序类型的统计信息
    for job_id, job in enumerate(env.jobs):
        for op_type, proc_time in job:
            op_type_counts[op_type] += 1
            op_type_total_time[op_type] += proc_time
            total_operations += 1

    # 计算平均处理时间和频率
    op_type_avg_time = np.zeros(num_op_types + 1)
    op_type_frequency = np.zeros(num_op_types + 1)

    for op_type in range(1, num_op_types + 1):
        if op_type_counts[op_type] > 0:
            op_type_avg_time[op_type] = op_type_total_time[op_type] / op_type_counts[op_type]
            op_type_frequency[op_type] = op_type_counts[op_type] / total_operations

    # 找出最大平均处理时间用于归一化
    max_avg_time = np.max(op_type_avg_time) if np.max(op_type_avg_time) > 0 else 1.0

    # 构建节点特征
    # 1. 机器节点 (索引: 0 to num_machines-1)
    for m_idx in range(num_machines):
        # 节点类型: 机器
        node_features[m_idx, 0] = 1.0

        # 获取机器信息
        machine_info = state['machines'][m_idx]
        remaining_time = machine_info['remaining_time'] / 100.0  # 归一化
        current_op_type = machine_info['current_op_type']

        # 机器特性: 剩余时间
        node_features[m_idx, 3] = remaining_time

        # 机器特性: 当前工序类型
        if current_op_type is not None:
            node_features[m_idx, 4] = current_op_type / num_op_types

            # 连接机器和工序类型
            op_type_node_idx = num_machines + num_jobs + current_op_type - 1
            adj_matrix[m_idx, op_type_node_idx] = 1.0
            adj_matrix[op_type_node_idx, m_idx] = 1.0

        # 机器特性: 已占用槽位数
        num_occupied_slots = sum(1 for slot in machine_info['slots'] if slot[0] != -1)
        node_features[m_idx, 5] = num_occupied_slots / Config.MAX_SLOTS

    # 2. 任务节点 (索引: num_machines to num_machines+num_jobs-1)
    for j_idx in range(num_jobs):
        node_idx = num_machines + j_idx

        # 节点类型: 任务
        node_features[node_idx, 1] = 1.0

        # 任务特性: 完成进度
        if j_idx < len(env.job_operation_index) and j_idx < len(env.jobs):
            progress = env.job_operation_index[j_idx] / len(env.jobs[j_idx]) if len(env.jobs[j_idx]) > 0 else 1.0
            node_features[node_idx, 6] = progress

            # 任务特性: 是否待处理
            is_pending = j_idx in env.pending_jobs
            node_features[node_idx, 7] = 1.0 if is_pending else 0.0

            # 任务特性: 下一个工序类型
            if is_pending and env.job_operation_index[j_idx] < len(env.jobs[j_idx]):
                next_op_type, _ = env.jobs[j_idx][env.job_operation_index[j_idx]]
                node_features[node_idx, 8] = next_op_type / num_op_types

                # 连接任务和工序类型
                op_type_node_idx = num_machines + num_jobs + next_op_type - 1
                adj_matrix[node_idx, op_type_node_idx] = 1.0
                adj_matrix[op_type_node_idx, node_idx] = 1.0

    # 3. 工序类型节点 (索引: num_machines+num_jobs to num_nodes-1)
    for op_idx in range(num_op_types):
        node_idx = num_machines + num_jobs + op_idx
        actual_op_type = op_idx + 1  # 工序类型通常从1开始

        # 节点类型: 工序类型
        node_features[node_idx, 2] = 1.0

        # 工序类型ID (归一化)
        node_features[node_idx, 9] = actual_op_type / num_op_types

        # 工序类型平均处理时间 (归一化)
        node_features[node_idx, 10] = op_type_avg_time[actual_op_type] / max_avg_time if max_avg_time > 0 else 0

        # 工序类型出现频率
        node_features[node_idx, 11] = op_type_frequency[actual_op_type]

    # 根据当前分配创建边
    for m_idx in range(num_machines):
        machine_info = state['machines'][m_idx]

        # 连接机器和分配给它的任务
        for slot in machine_info['slots']:
            job_id, _ = slot
            if job_id != -1:  # 有效任务
                job_node_idx = num_machines + job_id

                # 添加机器->任务的边
                adj_matrix[m_idx, job_node_idx] = 1.0
                adj_matrix[job_node_idx, m_idx] = 1.0

    # 构建全局状态向量
    global_features = [
        state['tool_changes'] / 100.0,  # 归一化工具更换次数
        state['total_time'] / 1000.0,  # 归一化总时间
        state['completed_jobs'] / env.num_jobs,  # 完成任务比例
        len(env.pending_jobs) / env.num_jobs,  # 待处理任务比例
        sum(len(slots) for slots in env.machine_slots) / (env.num_machines * Config.MAX_SLOTS)  # 槽位占用率
    ]

    # 添加自环，确保消息可以传递到自身
    np.fill_diagonal(adj_matrix, 1.0)

    return {
        'node_features': node_features,
        'adj_matrix': adj_matrix,
        'global_state': np.array(global_features, dtype=np.float32)
    }

def get_valid_actions(state, env):
    """获取当前状态下的有效动作"""
    valid_actions = []

    for machine_id in range(env.num_machines):
        if len(env.machine_slots[machine_id]) < Config.MAX_SLOTS:  # 使用配置中的最大槽位数
            for job_id in env.pending_jobs:
                valid_actions.append((machine_id, job_id))

    return valid_actions


def convert_action_index_to_action(action_idx, valid_actions):
    """将动作索引转换为实际动作"""
    if not valid_actions:
        return None
    return valid_actions[action_idx % len(valid_actions)]


def train_multi_dataset(data_dir, episodes_per_dataset=50, num_machines=3, device='cpu',
                        model_save_dir='models', log_interval=5,
                        eval_dir='datasets/evaluate_datasets', eval_interval=10):
    """
    使用多个数据集训练图神经网络智能体

    参数:
    data_dir: 包含数据集文件的目录
    episodes_per_dataset: 每个数据集训练的回合数
    num_machines: 机器数量
    device: 训练设备 ('cpu' 或 'cuda')
    model_save_dir: 模型保存目录
    log_interval: 日志输出间隔
    eval_dir: 评估数据集目录
    eval_interval: 评估间隔（每训练多少个回合进行一次评估）
    """
    # 创建模型保存目录
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # 初始化评估历史记录字典
    eval_history = {}

    # 初始化损失记录
    loss_history = []

    # 创建验证结果记录器
    validation_logger = ValidationLogger('validation_results.xlsx')

    # 获取所有数据文件
    data_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not data_files:
        print(f"错误：在 {data_dir} 目录下未找到数据文件")
        return

    print(f"找到 {len(data_files)} 个数据文件")

    # 初始化环境获取图节点信息
    sample_env = JobShopEnv(data_files[0], num_machines)
    sample_state = sample_env.reset()
    graph_state = state_to_graph(sample_state, sample_env)

    # 获取维度信息
    node_features_dim = graph_state['node_features'].shape[1]
    edge_features_dim = 1  # 简单二元邻接矩阵
    global_state_dim = len(graph_state['global_state'])

    # 计算动作空间大小
    action_size = Config.get_action_size(num_machines)  # 机器数 * 最大任务数

    # 初始化GNN-DQN智能体
    agent = GNN_DQNAgent(
        node_features=node_features_dim,
        edge_features=edge_features_dim,
        global_features=global_state_dim,
        action_size=action_size,
        device=device
    )

    # 定义批大小和目标网络更新频率
    batch_size = Config.BATCH_SIZE
    target_update = Config.TARGET_UPDATE

    # 加载已有模型（如果存在）
    model_path = os.path.join(model_save_dir, "job_shop_gnn_dqn_latest.pth")
    agent.load(model_path)

    total_episodes = 0
    training_start_time = time.time()

    # 循环训练每个数据集
    for dataset_idx, data_file in enumerate(data_files):
        print(f"\n{'=' * 50}")
        print(f"训练数据集 {dataset_idx + 1}/{len(data_files)}: {os.path.basename(data_file)}")
        print(f"{'=' * 50}")

        env = JobShopEnv(data_file, num_machines)
        print(f"数据集有 {env.num_jobs} 个任务, {env.num_operation_types} 种工序类型")

        dataset_start_time = time.time()

        # 记录当前数据集的损失
        dataset_losses = []

        # 对当前数据集进行训练
        for e in range(episodes_per_dataset):
            # 在每个回合开始时
            # print_gpu_stats()

            total_episodes += 1
            state = env.reset()
            graph_state = state_to_graph(state, env)
            total_reward = 0
            steps_taken = 0
            episode_losses = []  # 记录当前回合的损失

            for time_step in range(Config.MAX_STEPS):  # 使用配置中的最大步数
                valid_actions = get_valid_actions(state, env)

                if not valid_actions:
                    break

                # 选择动作
                action_idx = agent.act(graph_state)
                action = convert_action_index_to_action(action_idx, valid_actions)

                # 执行动作
                next_state, reward, done, _ = env.step(action)
                next_graph_state = state_to_graph(next_state, env)
                total_reward += reward
                steps_taken += 1

                # 记忆
                agent.remember(graph_state, action_idx, reward, next_graph_state, done)

                state = next_state
                graph_state = next_graph_state

                # 训练 - 使用GNN模型的replay方法
                if len(agent.memory) > batch_size:
                    loss = agent.replay(batch_size)
                    episode_losses.append(loss)  # 记录损失

                if done:
                    break

            # 计算并保存当前回合的平均损失
            if episode_losses:
                avg_loss = sum(episode_losses) / len(episode_losses)
                dataset_losses.append(avg_loss)
                loss_history.append((total_episodes, avg_loss))

                # 每隔固定回合数绘制损失曲线
                if total_episodes % log_interval == 0:
                    plot_loss_curve(loss_history)

            # 更新目标网络
            if total_episodes % target_update == 0:
                agent.update_target_network()

            # 评估部分：在指定间隔评估模型性能
            if eval_interval > 0 and total_episodes % eval_interval == 0:
                print(f"\n评估回合 {total_episodes} 的模型性能...")
                eval_results = evaluate_on_datasets(
                    agent,
                    eval_dir,
                    num_machines,
                    state_to_graph,
                    get_valid_actions,
                    convert_action_index_to_action,
                    num_eval_runs=3,
                    validation_logger=validation_logger
                )

                # 更新评估历史
                eval_history = update_eval_history(eval_history, total_episodes, eval_results)

                # 绘制评估指标图
                plot_eval_metrics(eval_history)
                print("评估完成\n")

            # 输出训练日志
            if (e + 1) % log_interval == 0:
                loss_info = ""
                if dataset_losses:
                    avg_dataset_loss = sum(dataset_losses[-log_interval:]) / max(1, len(dataset_losses[-log_interval:]))
                    loss_info = f", 平均损失: {avg_dataset_loss:.6f}"

                print(f"数据集 {dataset_idx + 1}, 回合 {e + 1}/{episodes_per_dataset}, "
                      f"工具更换: {env.tool_changes}, 总时间: {env.total_time}, "
                      f"任务完成: {env.completed_jobs}/{env.num_jobs}, "
                      f"Epsilon: {agent.epsilon:.4f}{loss_info}")

            # 在回合结束时
            # print_gpu_stats()
            # 添加内存高峰重置
            # torch.cuda.reset_peak_memory_stats()

        # 保存模型
        agent.save(model_path)
        print(f"模型已保存到 {model_path}")

        # 每10个数据集保存一个检查点
        if (dataset_idx + 1) % 10 == 0:
            checkpoint_path = os.path.join(model_save_dir, f"job_shop_gnn_dqn_checkpoint_{dataset_idx + 1}.pth")
            agent.save(checkpoint_path)
            print(f"检查点已保存到 {checkpoint_path}")

        dataset_time = time.time() - dataset_start_time
        print(f"数据集 {dataset_idx + 1} 训练完成, 用时: {dataset_time:.2f}秒")

    # 保存最终模型
    final_model_path = os.path.join(model_save_dir, "job_shop_gnn_dqn_final.pth")
    agent.save(final_model_path)

    total_training_time = time.time() - training_start_time
    print(f"\n训练完成, 总用时: {total_training_time:.2f}秒")
    print(f"最终模型已保存到 {final_model_path}")

    # 训练结束后进行最终评估
    print(f"\n对最终模型进行评估...")
    final_eval_results = evaluate_on_datasets(
        agent,
        eval_dir,
        num_machines,
        state_to_graph,
        get_valid_actions,
        convert_action_index_to_action,
        num_eval_runs=5,  # 最终评估运行更多次
        validation_logger=validation_logger
    )

    # 更新并绘制最终评估结果
    eval_history = update_eval_history(eval_history, total_episodes, final_eval_results)
    plot_eval_metrics(eval_history)

    # 绘制最终的损失曲线
    plot_loss_curve(loss_history, final=True)

    return agent


def test_agent(agent, data_dir, num_machines=3, num_tests=5, generate_charts=True):
    """在多个数据集上测试图神经网络智能体"""
    # 获取所有数据文件
    data_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not data_files:
        print(f"错误：在 {data_dir} 目录下未找到数据文件")
        return

    # 选择一些数据集进行测试
    test_files = random.sample(data_files, min(10, len(data_files)))

    print(f"\n{'=' * 50}")
    print(f"测试智能体在 {len(test_files)} 个数据集上的表现")
    print(f"{'=' * 50}")

    total_tool_changes = 0
    total_processing_time = 0
    total_tests = 0
    successful_tests = 0

    # 创建结果目录
    results_dir = "results/scheduling_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 对每个测试数据集进行测试
    for file_idx, data_file in enumerate(test_files):
        print(f"\n测试数据集 {file_idx + 1}/{len(test_files)}: {os.path.basename(data_file)}")

        env = JobShopEnv(data_file, num_machines)
        print(f"数据集有 {env.num_jobs} 个任务, {env.num_operation_types} 种工序类型")

        dataset_tool_changes = 0
        dataset_processing_time = 0
        dataset_successful = 0

        # 对当前数据集进行多次测试
        for test_idx in range(num_tests):
            print(f"  测试 {test_idx + 1}/{num_tests}")

            state = env.reset()
            graph_state = state_to_graph(state, env)
            steps_taken = 0

            for time_step in range(Config.MAX_STEPS):  # 使用配置中的最大步数
                valid_actions = get_valid_actions(state, env)

                if not valid_actions:
                    break

                # 选择动作 - 使用图状态
                action_idx = agent.act(graph_state)
                action = convert_action_index_to_action(action_idx, valid_actions)

                # 执行动作
                next_state, reward, done, _ = env.step(action)
                next_graph_state = state_to_graph(next_state, env)

                state = next_state
                graph_state = next_graph_state
                steps_taken += 1

                if done:
                    break

            total_tests += 1

            # 判断测试是否成功
            if env.completed_jobs == env.num_jobs:
                successful_tests += 1
                dataset_successful += 1
                dataset_tool_changes += env.tool_changes
                dataset_processing_time += env.total_time
                print(f"    成功! 工具更换次数: {env.tool_changes}, 处理时间: {env.total_time}")

                # 打印每台机器处理的任务序列
                env.print_job_sequences()

                # 保存详细调度结果到txt文件
                file_basename = os.path.basename(data_file).replace('.txt', '')
                result_file = os.path.join(results_dir, f"{file_basename}_test{test_idx + 1}_results.txt")
                env.save_detailed_schedule(result_file)
                print(f"    详细调度结果已保存到: {result_file}")

                # 生成甘特图
                if generate_charts and has_gantt_utils:
                    try:
                        chart_dir = os.path.join(results_dir, "charts", f"{file_basename}_test{test_idx + 1}")
                        os.makedirs(chart_dir, exist_ok=True)
                        generate_all_charts(env, chart_dir)
                    except Exception as e:
                        print(f"    生成甘特图时出错: {e}")
            else:
                print(f"    失败! 完成任务: {env.completed_jobs}/{env.num_jobs}")

        # 计算当前数据集的平均结果
        if dataset_successful > 0:
            avg_tool_changes = dataset_tool_changes / dataset_successful
            avg_processing_time = dataset_processing_time / dataset_successful
            print(f"  数据集平均结果 (成功率: {dataset_successful}/{num_tests}):")
            print(f"    平均工具更换次数: {avg_tool_changes:.2f}")
            print(f"    平均处理时间: {avg_processing_time:.2f}")

            total_tool_changes += dataset_tool_changes
            total_processing_time += dataset_processing_time
        else:
            print(f"  该数据集所有测试均失败")

    # 计算总体平均结果
    if successful_tests > 0:
        overall_avg_tool_changes = total_tool_changes / successful_tests
        overall_avg_processing_time = total_processing_time / successful_tests
        print(f"\n总体测试结果 (成功率: {successful_tests}/{total_tests}):")
        print(f"  平均工具更换次数: {overall_avg_tool_changes:.2f}")
        print(f"  平均处理时间: {overall_avg_processing_time:.2f}")
    else:
        print("\n所有测试均失败")


def evaluate_on_datasets(agent, eval_dir, num_machines, state_transform_fn, get_valid_actions_fn,
                         convert_action_fn, num_eval_runs=3, validation_logger=None):
    """
    在评估数据集上评估当前图神经网络模型的性能

    参数:
    agent: GNN_DQNAgent实例
    eval_dir: 评估数据集目录
    num_machines: 机器数量
    state_transform_fn: 状态转换函数 (转换为图结构)
    get_valid_actions_fn: 获取有效动作函数
    convert_action_fn: 动作索引转换函数
    num_eval_runs: 每个数据集评估的次数
    validation_logger: 可选，ValidationLogger实例，用于记录验证结果

    返回:
    results: 包含每个数据集评估结果的字典
    """
    # 获取评估数据集
    eval_files = sorted(glob.glob(os.path.join(eval_dir, "*.txt")))
    if not eval_files:
        print(f"警告: 在 {eval_dir} 目录下未找到评估数据文件")
        return {}

    print(f"\n正在评估模型性能 (在 {len(eval_files)} 个数据集上)")

    results = {}

    for data_file in eval_files:
        file_name = os.path.basename(data_file)
        print(f"  评估数据集: {file_name}")

        env = JobShopEnv(data_file, num_machines)

        # 每个数据集的统计数据
        successful_runs = 0
        total_completed_jobs = 0
        total_tool_changes = 0
        total_time = 0

        for run in range(num_eval_runs):
            state = env.reset()
            graph_state = state_transform_fn(state, env)

            for time_step in range(Config.MAX_STEPS):
                valid_actions = get_valid_actions_fn(state, env)

                if not valid_actions:
                    break

                # 在评估模式下，使用贪婪策略选择动作
                epsilon_backup = agent.epsilon
                agent.epsilon = 0  # 关闭探索
                action_idx = agent.act(graph_state)
                agent.epsilon = epsilon_backup  # 恢复探索率

                action = convert_action_fn(action_idx, valid_actions)

                # 执行动作
                next_state, reward, done, _ = env.step(action)
                next_graph_state = state_transform_fn(next_state, env)

                state = next_state
                graph_state = next_graph_state

                if done:
                    break

            # 收集该次运行的结果
            total_completed_jobs += env.completed_jobs
            if env.completed_jobs == env.num_jobs:
                successful_runs += 1
                total_tool_changes += env.tool_changes
                total_time += env.total_time

        # 计算该数据集的平均指标
        completion_rate = total_completed_jobs / (env.num_jobs * num_eval_runs)
        success_rate = successful_runs / num_eval_runs

        avg_tool_changes = 0
        avg_time = 0
        if successful_runs > 0:
            avg_tool_changes = total_tool_changes / successful_runs
            avg_time = total_time / successful_runs

            # 记录验证结果到Excel
            if validation_logger is not None:
                validation_logger.log_validation_result(
                    dataset_name=file_name,
                    completion_time=avg_time,
                    tool_change_count=avg_tool_changes
                )

        # 存储结果
        results[file_name] = {
            'completion_rate': completion_rate,
            'success_rate': success_rate,
            'avg_tool_changes': avg_tool_changes if successful_runs > 0 else None,
            'avg_time': avg_time if successful_runs > 0 else None,
            'successful_runs': successful_runs,
            'num_eval_runs': num_eval_runs,
            'num_jobs': env.num_jobs
        }

        # 打印该数据集的评估结果
        print(f"    任务完成率: {completion_rate:.2f}")
        print(f"    成功率: {success_rate:.2f} ({successful_runs}/{num_eval_runs})")
        if successful_runs > 0:
            print(f"    平均工具更换: {avg_tool_changes:.2f}")
            print(f"    平均时间: {avg_time:.2f}")

    # 如果有验证记录器，完成一轮验证
    if validation_logger is not None:
        validation_logger.complete_validation_round()

    return results

def update_eval_history(history, epoch, results):
    """将当前评估结果添加到历史记录中"""
    history[epoch] = results
    return history


def plot_eval_metrics(eval_history, save_dir='results/eval_results'):
    """绘制评估指标随训练进展的变化图"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 获取所有训练轮次
    epochs = sorted(eval_history.keys())

    # 初始化数据收集器
    makespan_history = []
    tool_changes_history = []

    # 对每个轮次，计算所有数据集的平均值
    for epoch in epochs:
        # 计算该轮次所有数据集的平均值
        total_makespan = 0
        total_tool_changes = 0
        valid_datasets = 0

        for dataset, metrics in eval_history[epoch].items():
            if metrics['avg_time'] is not None:  # 只统计成功的运行
                total_makespan += metrics['avg_time']
                total_tool_changes += metrics['avg_tool_changes']
                valid_datasets += 1

        # 添加平均值到历史记录
        if valid_datasets > 0:
            makespan_history.append((epoch, total_makespan / valid_datasets))
            tool_changes_history.append((epoch, total_tool_changes / valid_datasets))

    # 绘制完工时间曲线（类似损失函数曲线）
    plt.figure(figsize=(10, 6))

    # 提取数据
    makespan_epochs = [item[0] for item in makespan_history]
    makespan_values = [item[1] for item in makespan_history]

    plt.plot(makespan_epochs, makespan_values, 'b-', alpha=0.7)

    # 添加移动平均线
    window_size = min(50, len(makespan_values))
    if window_size > 1:
        moving_avg = []
        for i in range(len(makespan_values)):
            start_idx = max(0, i - window_size + 1)
            moving_avg.append(sum(makespan_values[start_idx:i + 1]) / (i - start_idx + 1))
        plt.plot(makespan_epochs, moving_avg, 'r-', linewidth=2, label=f'移动平均 (窗口={window_size})')

    plt.title('训练完工时间曲线')
    plt.xlabel('训练回合')
    plt.ylabel('完工时间')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eval_makespan.png'), dpi=300)
    plt.close()

    # 绘制模具更换次数曲线
    plt.figure(figsize=(10, 6))

    # 提取数据
    tool_epochs = [item[0] for item in tool_changes_history]
    tool_values = [item[1] for item in tool_changes_history]

    plt.plot(tool_epochs, tool_values, 'b-', alpha=0.7)

    # 添加移动平均线
    window_size = min(50, len(tool_values))
    if window_size > 1:
        moving_avg = []
        for i in range(len(tool_values)):
            start_idx = max(0, i - window_size + 1)
            moving_avg.append(sum(tool_values[start_idx:i + 1]) / (i - start_idx + 1))
        plt.plot(tool_epochs, moving_avg, 'r-', linewidth=2, label=f'移动平均 (窗口={window_size})')

    plt.title('训练模具更换次数曲线')
    plt.xlabel('训练回合')
    plt.ylabel('模具更换次数')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eval_tool_changes.png'), dpi=300)
    plt.close()

    # 为了保持原有功能，也绘制成功率图表（可以保留原有代码）
    # 这里保留原有代码的功能，绘制其他指标图表...
    # [原有的其他图表绘制代码]

    print(f"评估图表已保存到 {save_dir} 目录")


def plot_loss_curve(loss_history, save_dir='results/eval_results', final=False):
    """绘制训练损失曲线

    参数:
    loss_history: 包含(episode, loss)元组的列表
    save_dir: 保存目录
    final: 是否为最终的损失曲线
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 提取数据
    episodes = [item[0] for item in loss_history]
    losses = [item[1] for item in loss_history]

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, losses, 'b-', alpha=0.7)

    # 添加移动平均线以便更清晰地看到趋势
    window_size = min(50, len(losses))
    if window_size > 1:
        moving_avg = []
        for i in range(len(losses)):
            start_idx = max(0, i - window_size + 1)
            moving_avg.append(sum(losses[start_idx:i + 1]) / (i - start_idx + 1))
        plt.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'移动平均 (窗口={window_size})')

    plt.title('训练损失曲线')
    plt.xlabel('训练回合')
    plt.ylabel('损失值')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 设置y轴范围，确保显示所有数据，包括最大值
    if len(losses) > 0:
        max_loss = max(losses)
        # 给顶部增加5-10%的空间以确保显示完整
        plt.ylim(0, max_loss * 1.1)

    # 不要使用自动缩放或百分位数截断，以确保显示所有数据点
    # 下面的代码被替换：
    # if len(losses) > 10:
    #     sorted_losses = sorted(losses)
    #     lower_bound = sorted_losses[int(len(sorted_losses) * 0.05)]  # 下5%分位数
    #     upper_bound = sorted_losses[int(len(sorted_losses) * 0.95)]  # 上95%分位数
    #     plt.ylim(max(0, lower_bound * 0.9), upper_bound * 1.1)

    plt.tight_layout()

    if final:
        filename = os.path.join(save_dir, 'final_training_loss.png')
    else:
        filename = os.path.join(save_dir, 'training_loss.png')

    plt.savefig(filename, dpi=300)
    plt.close()

    if final:
        print(f"最终训练损失曲线已保存到 {filename}")

    # 保存损失数据到CSV文件，方便后续分析
    if final:
        csv_file = os.path.join(save_dir, 'training_loss_data.csv')
        with open(csv_file, 'w') as f:
            f.write('episode,loss\n')
            for episode, loss in loss_history:
                f.write(f'{episode},{loss}\n')
        print(f"训练损失数据已保存到 {csv_file}")


def plot_reward_curve(reward_history, save_dir='results/eval_results', final=False):
    """绘制训练奖励曲线

    参数:
    reward_history: 包含(episode, reward)元组的列表
    save_dir: 保存目录
    final: 是否为最终的奖励曲线
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 提取数据
    episodes = [item[0] for item in reward_history]
    rewards = [item[1] for item in reward_history]

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, 'g-', alpha=0.7)

    # 添加移动平均线以便更清晰地看到趋势
    window_size = min(50, len(rewards))
    if window_size > 1:
        moving_avg = []
        for i in range(len(rewards)):
            start_idx = max(0, i - window_size + 1)
            moving_avg.append(sum(rewards[start_idx:i + 1]) / (i - start_idx + 1))
        plt.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'移动平均 (窗口={window_size})')

    plt.title('训练奖励曲线')
    plt.xlabel('训练回合')
    plt.ylabel('奖励值')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    if final:
        filename = os.path.join(save_dir, 'final_training_reward.png')
    else:
        filename = os.path.join(save_dir, 'training_reward.png')

    plt.savefig(filename, dpi=300)
    plt.close()

    if final:
        print(f"最终训练奖励曲线已保存到 {filename}")

    # 保存奖励数据到CSV文件，方便后续分析
    if final:
        csv_file = os.path.join(save_dir, 'training_reward_data.csv')
        with open(csv_file, 'w') as f:
            f.write('episode,reward\n')
            for episode, reward in reward_history:
                f.write(f'{episode},{reward}\n')
        print(f"训练奖励数据已保存到 {csv_file}")


# 修改训练函数以收集奖励数据
# 以下代码片段应该集成到 train_multi_dataset 函数中

"""
# 在函数开始处添加
# 初始化奖励记录
reward_history = []

# 在每个回合结束后添加
reward_history.append((total_episodes, total_reward))

# 每隔固定回合数绘制奖励曲线
if total_episodes % log_interval == 0:
    plot_reward_curve(reward_history)

# 在训练结束后添加
# 绘制最终的奖励曲线
plot_reward_curve(reward_history, final=True)
"""