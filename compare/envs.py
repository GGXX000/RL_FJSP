import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
import os
import glob
from datetime import datetime
from config import Config


class JobShopEnv:
    def __init__(self, data_file, num_machines):
        self.num_machines = num_machines
        self.load_data(data_file)

        # 添加变量存储上一次模拟结果
        self.last_simulated_makespan = None
        self.last_simulated_tool_changes = None

        self.reset()

    def load_data(self, data_file):
        with open(data_file, 'r') as f:
            first_line = f.readline().strip().split()
            self.num_jobs = int(first_line[0])
            self.num_operation_types = int(first_line[1])

            self.jobs = []
            for i in range(self.num_jobs):
                line = f.readline().strip()
                if not line:  # 跳过空行
                    line = f.readline().strip()
                job_data = list(map(int, line.split()))
                operations = []
                j = 0
                while j < len(job_data):
                    operations.append((job_data[j], job_data[j + 1]))  # (operation_type, processing_time)
                    j += 2
                self.jobs.append(operations)

    def reset(self):
        # 每台机器上有最大槽位数
        self.machine_slots = [[] for _ in range(self.num_machines)]

        # 当前每台机器上的工序类型（模具）
        self.current_operation_type = [None for _ in range(self.num_machines)]

        # 每台机器的剩余处理时间
        self.remaining_time = [0 for _ in range(self.num_machines)]

        # 模具更换次数
        self.tool_changes = 0

        # 总处理时间
        self.total_time = 0

        # 未分配的任务
        self.pending_jobs = list(range(self.num_jobs))

        # 任务的当前操作索引
        self.job_operation_index = [0 for _ in range(self.num_jobs)]

        # 已完成的任务数量
        self.completed_jobs = 0

        # 初始化当前槽位索引，用于"一圈一圈"处理
        self.current_slot_index = [0 for _ in range(self.num_machines)]

        # 重置机器任务序列记录
        self.machine_job_sequences = [[] for _ in range(self.num_machines)]

        # 记录任务的分配顺序
        self.machine_allocation_order = [[] for _ in range(self.num_machines)]

        # 记录每一轮分配的任务
        self.machine_rounds = [[] for _ in range(self.num_machines)]

        # 每台机器当前的轮数
        self.current_round = [0 for _ in range(self.num_machines)]

        # 每台机器的最后处理时间点，用于确保时间连续性
        self.last_machine_times = [0 for _ in range(self.num_machines)]

        # 重置模拟结果
        self.last_simulated_makespan = None
        self.last_simulated_tool_changes = None

        # 状态表示
        self.state = self._get_state()

        return self.state

    def _get_state(self):
        # 状态表示包括：
        # 1. 每台机器上的槽位状态
        # 2. 每台机器的当前工序类型
        # 3. 每台机器的剩余处理时间
        # 4. 未分配任务的下一个工序

        machine_state = []
        for machine_id in range(self.num_machines):
            slots = self.machine_slots[machine_id]
            # 对于每个槽位，我们记录任务ID和当前工序
            slot_state = []
            for slot in range(Config.MAX_SLOTS):  # 使用配置中的最大槽位数
                if slot < len(slots):
                    job_id, op_idx = slots[slot]
                    if op_idx < len(self.jobs[job_id]):
                        op_type, _ = self.jobs[job_id][op_idx]
                    else:
                        op_type = 0  # 表示任务完成
                    slot_state.append((job_id, op_type))
                else:
                    slot_state.append((-1, 0))  # 空槽位

            machine_state.append({
                'slots': slot_state,
                'current_op_type': self.current_operation_type[machine_id],
                'remaining_time': self.remaining_time[machine_id]
            })

        pending_jobs_state = []
        for job_id in self.pending_jobs:
            if self.job_operation_index[job_id] < len(self.jobs[job_id]):
                next_op = self.jobs[job_id][self.job_operation_index[job_id]]
                pending_jobs_state.append((job_id, next_op[0]))
            else:
                pending_jobs_state.append((job_id, -1))  # 任务完成

        return {
            'machines': machine_state,
            'pending_jobs': pending_jobs_state,
            'tool_changes': self.tool_changes,
            'total_time': self.total_time,
            'completed_jobs': self.completed_jobs
        }

    def simulate_complete_schedule(self, action):
        """
        模拟从当前状态执行给定动作后，完成所有任务的整个调度过程

        参数:
        action: (machine_id, job_id) 或 None（表示不执行额外动作，直接模拟当前状态）

        返回:
        simulated_makespan: 模拟的总完工时间
        simulated_tool_changes: 模拟的总工具更换次数
        """
        # 创建环境状态的深拷贝
        machine_slots_copy = [list(slots) for slots in self.machine_slots]
        current_op_type_copy = list(self.current_operation_type)
        pending_jobs_copy = list(self.pending_jobs)
        job_operation_index_copy = list(self.job_operation_index)
        completed_jobs_copy = self.completed_jobs
        current_slot_index_copy = list(self.current_slot_index)
        last_machine_times_copy = list(self.last_machine_times)

        # 模拟执行当前动作（如果提供）
        if action is not None:
            machine_id, job_id = action
            if job_id in pending_jobs_copy and len(machine_slots_copy[machine_id]) < Config.MAX_SLOTS:
                machine_slots_copy[machine_id].append((job_id, job_operation_index_copy[job_id]))
                pending_jobs_copy.remove(job_id)
            else:
                # 无效动作，返回极高的惩罚值
                return float('inf'), float('inf')

        # 模拟工具更换计数
        simulated_tool_changes = self.tool_changes

        # 循环直到所有任务完成
        max_iterations = 100  # 防止无限循环
        iterations = 0

        while completed_jobs_copy < self.num_jobs and iterations < max_iterations:
            iterations += 1

            # 如果所有槽位都已满或没有待分配任务，处理当前的操作
            if all(len(slots) >= Config.MAX_SLOTS for slots in machine_slots_copy) or not pending_jobs_copy:
                # 模拟处理当前批次的操作
                tool_changes_delta, machine_times, processed_jobs, emptied_machines = self._simulate_batch_processing(
                    machine_slots_copy,
                    current_op_type_copy,
                    job_operation_index_copy,
                    current_slot_index_copy,
                    last_machine_times_copy,
                    completed_jobs_copy
                )

                # 更新模拟状态
                simulated_tool_changes += tool_changes_delta
                completed_jobs_copy += processed_jobs

                # 移除处理完的任务槽位
                for m_id in emptied_machines:
                    machine_slots_copy[m_id] = []
                    current_slot_index_copy[m_id] = 0

            # 如果还有待分配的任务，使用贪婪策略分配下一个任务
            if pending_jobs_copy and not all(len(slots) >= Config.MAX_SLOTS for slots in machine_slots_copy):
                next_action = self._greedy_action_selection(
                    machine_slots_copy,
                    current_op_type_copy,
                    pending_jobs_copy,
                    job_operation_index_copy,
                    last_machine_times_copy
                )

                if next_action:
                    next_machine_id, next_job_id = next_action
                    machine_slots_copy[next_machine_id].append((next_job_id, job_operation_index_copy[next_job_id]))
                    pending_jobs_copy.remove(next_job_id)
                else:
                    # 如果找不到有效动作，但仍有未分配任务和可用槽位，可能是算法问题
                    break

            # 检查是否所有任务都完成了
            if completed_jobs_copy >= self.num_jobs:
                break

        # 获取模拟的总完工时间
        simulated_makespan = max(last_machine_times_copy)

        return simulated_makespan, simulated_tool_changes

    def _simulate_batch_processing(self, machine_slots, current_op_types, job_op_indices,
                                   current_slot_indices, machine_times, completed_jobs):
        """
        模拟处理一批操作的过程

        返回:
        tool_changes: 此批次的工具更换次数
        updated_machine_times: 更新后的机器时间
        completed_jobs_delta: 完成的任务数增量
        emptied_machines: 已清空的机器列表
        """
        tool_changes = 0
        completed_jobs_delta = 0
        emptied_machines = []

        # 记录各机器是否处理完所有任务
        machines_completed = [False for _ in range(len(machine_slots))]

        # 单次处理的最大迭代次数，防止无限循环
        max_iterations = 1000
        iterations = 0

        while not all(machines_completed) and iterations < max_iterations:
            iterations += 1
            any_processed = False

            # 处理每台机器的下一个工序
            for machine_id in range(len(machine_slots)):
                # 如果机器已完成所有工序，跳过
                if machines_completed[machine_id]:
                    continue

                slots = machine_slots[machine_id]
                if not slots:
                    machines_completed[machine_id] = True
                    emptied_machines.append(machine_id)
                    continue

                # 检查是否所有槽位的工序都已完成
                all_slots_completed = True
                for job_id, op_idx in slots:
                    if op_idx < len(self.jobs[job_id]):
                        all_slots_completed = False
                        break

                if all_slots_completed:
                    machines_completed[machine_id] = True
                    emptied_machines.append(machine_id)
                    continue

                # 尝试找到下一个有效槽位
                found_valid_slot = False
                slots_checked = 0

                while slots_checked < len(slots) and not found_valid_slot:
                    current_idx = current_slot_indices[machine_id]
                    current_idx = current_idx % len(slots)

                    job_id, op_idx = slots[current_idx]

                    if op_idx < len(self.jobs[job_id]):
                        # 找到有效工序
                        found_valid_slot = True
                        any_processed = True

                        # 获取工序信息
                        op_type, proc_time = self.jobs[job_id][op_idx]

                        # 工具更换时间
                        tool_change_time = 0

                        # 检查是否需要更换模具
                        if current_op_types[machine_id] != op_type:
                            current_op_types[machine_id] = op_type
                            tool_changes += 1
                            tool_change_time = Config.TOOL_CHANGE_TIME

                        # 计算时间
                        start_time = machine_times[machine_id]
                        if tool_change_time > 0:
                            start_time += tool_change_time

                        end_time = start_time + proc_time
                        machine_times[machine_id] = end_time

                        # 更新操作索引
                        job_op_indices[job_id] += 1
                        slots[current_idx] = (job_id, job_op_indices[job_id])

                        # 检查任务是否完成
                        if job_op_indices[job_id] >= len(self.jobs[job_id]):
                            completed_jobs_delta += 1

                    # 移动到下一个槽位
                    current_slot_indices[machine_id] = (current_idx + 1) % len(slots)
                    slots_checked += 1

                # 如果没有找到有效槽位但还没标记为完成，继续检查
                if not found_valid_slot and not machines_completed[machine_id]:
                    # 检查是否所有槽位的工序确实都已完成
                    all_completed = True
                    for job_id, op_idx in slots:
                        if op_idx < len(self.jobs[job_id]):
                            all_completed = False
                            break

                    if all_completed:
                        machines_completed[machine_id] = True
                        emptied_machines.append(machine_id)

            # 如果没有任何机器处理了工序，并且不是所有机器都完成了，说明需要重新分配
            if not any_processed and not all(machines_completed):
                break

        return tool_changes, machine_times, completed_jobs_delta, emptied_machines

    def _greedy_action_selection(self, machine_slots, current_op_types, pending_jobs, job_op_indices, machine_times):
        """
        使用贪婪策略为模拟选择下一个动作

        返回:
        (machine_id, job_id) 或 None 如果没有有效动作
        """
        best_action = None
        min_score = float('inf')

        for machine_id in range(len(machine_slots)):
            if len(machine_slots[machine_id]) >= Config.MAX_SLOTS:
                continue

            # 获取当前机器的负载情况
            machine_load = machine_times[machine_id]
            current_types = set()
            for job_id, op_idx in machine_slots[machine_id]:
                if op_idx < len(self.jobs[job_id]):
                    current_types.add(self.jobs[job_id][op_idx][0])

            for job_id in pending_jobs:
                if job_op_indices[job_id] < len(self.jobs[job_id]):
                    op_idx = job_op_indices[job_id]
                    op_type, proc_time = self.jobs[job_id][op_idx]

                    # 计算当前动作的影响
                    action_score = 0

                    # 考虑机器负载平衡
                    action_score += machine_load * 0.5

                    # 考虑处理时间
                    action_score += proc_time

                    # 考虑工具更换
                    if op_type not in current_types and current_op_types[machine_id] != op_type:
                        action_score += Config.TOOL_CHANGE_TIME * 2

                    # 选择分数最低的动作
                    if action_score < min_score:
                        min_score = action_score
                        best_action = (machine_id, job_id)

        return best_action

    def step(self, action):
        """
        执行调度动作
        动作定义为：(machine_id, job_id)，即将job_id分配到machine_id的下一个可用槽位
        """
        machine_id, job_id = action

        # 检查动作是否有效
        if job_id not in self.pending_jobs or len(self.machine_slots[machine_id]) >= Config.MAX_SLOTS:
            return self.state, -100, False, {}

        # 记录动作执行前的模拟结果
        if self.last_simulated_makespan is None:
            # 首次执行，需要进行一次初始模拟
            pre_action_makespan, pre_action_tool_changes = self.simulate_complete_schedule(None)
        else:
            # 使用上一次保存的模拟结果
            pre_action_makespan = self.last_simulated_makespan
            pre_action_tool_changes = self.last_simulated_tool_changes

        # 记录动作执行前的状态
        prev_tool_changes = self.tool_changes

        # 执行动作后的模拟结果
        post_action_makespan, post_action_tool_changes = self.simulate_complete_schedule(action)

        # 保存本次模拟结果，供下一次决策使用
        self.last_simulated_makespan = post_action_makespan
        self.last_simulated_tool_changes = post_action_tool_changes

        # 将任务分配到机器
        self.machine_slots[machine_id].append((job_id, self.job_operation_index[job_id]))
        self.pending_jobs.remove(job_id)

        # 记录分配顺序
        self.machine_allocation_order[machine_id].append(job_id)

        # 记录轮次信息
        current_round = self.current_round[machine_id]
        if len(self.machine_rounds[machine_id]) <= current_round:
            self.machine_rounds[machine_id].append([])
        self.machine_rounds[machine_id][current_round].append(job_id)

        # 如果所有机器的槽位都满了，或者没有更多待分配的任务，就开始处理
        if all(len(slots) == Config.MAX_SLOTS for slots in self.machine_slots) or not self.pending_jobs:
            self._process_operations(self.last_machine_times)
            # 处理完一批操作后，重置模拟结果缓存
            self.last_simulated_makespan = None
            self.last_simulated_tool_changes = None

        # 获取新状态
        self.state = self._get_state()

        # 检查是否所有任务都完成了
        done = self.completed_jobs == self.num_jobs

        # ===== 归一化的基于差值的奖励函数 =====
        # 计算完工时间的改进
        makespan_improvement = pre_action_makespan - post_action_makespan

        # 计算工具更换次数的改进
        tool_changes_improvement = pre_action_tool_changes - post_action_tool_changes

        # 归一化改进值
        # 1. 对完工时间改进进行归一化
        if pre_action_makespan > 0:
            norm_makespan_improvement = makespan_improvement / pre_action_makespan
        else:
            norm_makespan_improvement = 0

        # 2. 对工具更换改进进行归一化
        # 计算理想情况下的最小工具更换次数（每种工序类型至少一次）
        unique_op_types = set()
        for job in self.jobs:
            for op_type, _ in job:
                unique_op_types.add(op_type)
        min_possible_tool_changes = len(unique_op_types)

        # 最大可能的工具更换次数（粗略估计）
        total_operations = sum(len(job) for job in self.jobs)
        max_possible_changes = min(total_operations, self.num_jobs * len(self.jobs[0]) if self.jobs else 0)
        tool_changes_range = max(1, max_possible_changes - min_possible_tool_changes)

        # 如果有工具更换改进，进行归一化
        if tool_changes_improvement != 0 and tool_changes_range > 0:
            norm_tool_improvement = tool_changes_improvement / tool_changes_range
        else:
            norm_tool_improvement = 0

        # 3. 使用归一化后的值计算奖励
        makespan_weight = 0.7  # 完工时间的权重
        tool_weight = 0.3  # 工具更换的权重

        # 归一化后的综合奖励，缩放到合适的范围
        base_reward = (norm_makespan_improvement * makespan_weight +
                       norm_tool_improvement * tool_weight) * 100

        # 添加小的即时奖励，鼓励智能体尽快完成任务
        immediate_reward = 0
        if self.tool_changes > prev_tool_changes:
            immediate_reward -= 1  # 小惩罚，避免不必要的工具更换

        # 最终奖励
        reward = base_reward + immediate_reward

        # 任务完成额外奖励
        if done:
            completion_bonus = 50  # 完成所有任务的奖励
            reward += completion_bonus
        elif self.pending_jobs == []:
            # 所有任务都已分配但尚未完成处理，给予中等奖励
            allocation_bonus = 20
            reward += allocation_bonus

        return self.state, reward, done, {
            'tool_changes': self.tool_changes - prev_tool_changes,
            'total_time': self.total_time,
            'makespan_improvement': makespan_improvement,
            'norm_makespan_improvement': norm_makespan_improvement,
            'tool_changes_improvement': tool_changes_improvement,
            'norm_tool_improvement': norm_tool_improvement,
            'reward': reward,
            'pre_action_makespan': pre_action_makespan,
            'post_action_makespan': post_action_makespan
        }

    def _process_operations(self, start_times=None):
        """
        模拟真正并行的方式处理所有机器上的操作直到需要重新分配任务

        参数:
        start_times: 可选，每台机器的开始处理时间，确保时间连续性
        """
        # 记录每台机器的当前时间
        if start_times is None:
            machine_times = [0 for _ in range(self.num_machines)]
        else:
            machine_times = list(start_times)  # 复制一份避免修改原始数据

        # 记录每台机器是否处理完所有任务
        machines_completed = [False for _ in range(self.num_machines)]

        # 单次处理的最大迭代次数，防止无限循环
        max_iterations = 1000
        iterations = 0

        # 调试信息
        debug_info = {}
        for machine_id in range(self.num_machines):
            debug_info[machine_id] = {
                "start_time": machine_times[machine_id],
                "tasks": []
            }

        while not all(machines_completed) and iterations < max_iterations:
            iterations += 1
            any_processed = False

            # 处理每台机器的下一个工序
            for machine_id in range(self.num_machines):
                # 如果机器已完成所有工序，跳过
                if machines_completed[machine_id]:
                    continue

                slots = self.machine_slots[machine_id]
                if not slots:
                    machines_completed[machine_id] = True
                    continue

                # 检查是否所有槽位的工序都已完成
                all_slots_completed = True
                for job_id, op_idx in slots:
                    if op_idx < len(self.jobs[job_id]):
                        all_slots_completed = False
                        break

                if all_slots_completed:
                    self.machine_slots[machine_id] = []
                    self.current_slot_index[machine_id] = 0
                    machines_completed[machine_id] = True

                    # 增加当前轮数，表示这一批处理完毕
                    self.current_round[machine_id] += 1

                    # 注意：这里不重置机器时间，确保时间连续性
                    debug_info[machine_id]["batch_completed"] = True
                    debug_info[machine_id]["final_time"] = machine_times[machine_id]
                    continue

                # 尝试找到下一个有效槽位
                found_valid_slot = False
                slots_checked = 0

                while slots_checked < len(slots) and not found_valid_slot:
                    current_idx = self.current_slot_index[machine_id]
                    current_idx = current_idx % len(slots)

                    job_id, op_idx = slots[current_idx]

                    if op_idx < len(self.jobs[job_id]):
                        # 找到有效工序
                        found_valid_slot = True
                        any_processed = True

                        # 获取工序信息
                        op_type, proc_time = self.jobs[job_id][op_idx]

                        # 工具更换时间
                        tool_change_time = 0

                        # 检查是否需要更换模具
                        need_tool_change = False
                        if self.current_operation_type[machine_id] != op_type:
                            self.current_operation_type[machine_id] = op_type
                            self.tool_changes += 1
                            tool_change_time = Config.TOOL_CHANGE_TIME  # 使用配置中的工具更换时间
                            need_tool_change = True

                        # 计算时间
                        start_time = machine_times[machine_id]
                        if tool_change_time > 0:
                            machine_times[machine_id] += tool_change_time

                        end_time = machine_times[machine_id] + proc_time
                        machine_times[machine_id] = end_time

                        # 更新总时间（取所有机器时间的最大值）
                        self.total_time = max(self.total_time, end_time)

                        # 更新最后处理时间点，用于下一批任务
                        self.last_machine_times[machine_id] = end_time

                        # 添加处理记录到机器任务序列
                        job_info = {
                            "job_id": job_id,
                            "operation": op_idx + 1,
                            "type": op_type,
                            "start_time": start_time,
                            "end_time": end_time,
                            "proc_time": proc_time,
                            "tool_change": need_tool_change,
                            "tool_change_time": tool_change_time,
                            "slot_index": current_idx,
                            "round": self.current_round[machine_id]
                        }
                        self.machine_job_sequences[machine_id].append(job_info)

                        # 添加调试信息
                        debug_info[machine_id]["tasks"].append({
                            "job_id": job_id,
                            "op_idx": op_idx,
                            "start": start_time,
                            "end": end_time,
                            "round": self.current_round[machine_id]
                        })

                        # 更新操作索引
                        self.job_operation_index[job_id] += 1
                        slots[current_idx] = (job_id, self.job_operation_index[job_id])

                        # 检查任务是否完成
                        if self.job_operation_index[job_id] >= len(self.jobs[job_id]):
                            self.completed_jobs += 1

                    # 移动到下一个槽位
                    self.current_slot_index[machine_id] = (current_idx + 1) % len(slots)
                    slots_checked += 1

                # 如果没有找到有效槽位但还没标记为完成，说明可能需要再转一圈
                if not found_valid_slot and not machines_completed[machine_id]:
                    # 检查是否所有槽位的工序确实都已完成
                    all_completed = True
                    for job_id, op_idx in slots:
                        if op_idx < len(self.jobs[job_id]):
                            all_completed = False
                            break

                    if all_completed:
                        self.machine_slots[machine_id] = []
                        self.current_slot_index[machine_id] = 0
                        machines_completed[machine_id] = True

                        # 增加当前轮数，表示这一批处理完毕
                        self.current_round[machine_id] += 1

            # 如果没有任何机器处理了工序，并且不是所有机器都完成了，说明需要重新分配
            if not any_processed and not all(machines_completed):
                break

        # 处理完一批操作后，重置模拟结果缓存
        self.last_simulated_makespan = None
        self.last_simulated_tool_changes = None

    def print_job_sequences(self):
        """打印每台机器处理的任务序列摘要"""
        print("\n=== 数据集处理摘要 ===")
        print(f"总处理时间: {self.total_time}, 总工具更换次数: {self.tool_changes}")
        print(f"完成任务数: {self.completed_jobs}/{self.num_jobs}")

        print("\n各机器分配的任务（按分配顺序）:")
        for machine_id in range(self.num_machines):
            if self.machine_allocation_order[machine_id]:
                jobs_str = " ".join([str(job_id) for job_id in self.machine_allocation_order[machine_id]])
                print(f"机器 {machine_id}: {jobs_str}")
            else:
                print(f"机器 {machine_id}: 未分配任何任务")

        print("\n各机器轮询处理顺序（按轮次）:")
        for machine_id in range(self.num_machines):
            print(f"机器 {machine_id} 轮询顺序:")
            for round_idx, round_jobs in enumerate(self.machine_rounds[machine_id]):
                if round_jobs:
                    jobs_str = " ".join([str(job_id) for job_id in round_jobs])
                    print(f"  第{round_idx + 1}轮: {jobs_str}")

            # 打印机器完成时间
            job_sequence = self.machine_job_sequences[machine_id]
            if job_sequence:
                # 检查时间连续性
                times_by_round = {}
                for job in job_sequence:
                    round_num = job["round"]
                    if round_num not in times_by_round:
                        times_by_round[round_num] = []
                    times_by_round[round_num].append((job["start_time"], job["end_time"]))

                # 输出每轮的时间范围
                for round_num in sorted(times_by_round.keys()):
                    times = times_by_round[round_num]
                    min_start = min([t[0] for t in times])
                    max_end = max([t[1] for t in times])
                    print(f"  第{round_num + 1}轮时间范围: {min_start} - {max_end}")

                last_end_time = max(job["end_time"] for job in job_sequence)
                print(f"  完成时间: {last_end_time}")
            else:
                print(f"  未处理任何任务")

    def save_detailed_schedule(self, filename):
        """保存详细的调度信息到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            # 写入标题和摘要信息
            f.write(f"===== 作业车间调度详细结果 =====\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总任务数: {self.num_jobs}\n")
            f.write(f"操作类型数: {self.num_operation_types}\n")
            f.write(f"机器数量: {self.num_machines}\n\n")

            f.write(f"总处理时间: {self.total_time}\n")
            f.write(f"总工具更换次数: {self.tool_changes}\n")
            f.write(f"完成任务数: {self.completed_jobs}/{self.num_jobs}\n\n")

            # 写入机器的任务分配顺序（智能体决策顺序）
            f.write("===== 机器任务分配顺序 =====\n")
            f.write("这显示智能体将任务分配到机器的顺序\n\n")
            for machine_id in range(self.num_machines):
                if self.machine_allocation_order[machine_id]:
                    jobs_str = " ".join([str(job_id) for job_id in self.machine_allocation_order[machine_id]])
                    f.write(f"机器 {machine_id} 分配顺序: {jobs_str}\n")
                else:
                    f.write(f"机器 {machine_id}: 未分配任何任务\n")

            # 写入机器的轮询处理顺序
            f.write("\n===== 机器轮询处理顺序 =====\n")
            f.write("这显示机器轮询处理任务的顺序（按照轮次）\n\n")
            for machine_id in range(self.num_machines):
                f.write(f"机器 {machine_id} 轮询顺序:\n")
                for round_idx, round_jobs in enumerate(self.machine_rounds[machine_id]):
                    if round_jobs:
                        jobs_str = " ".join([str(job_id) for job_id in round_jobs])
                        f.write(f"  第{round_idx + 1}轮: {jobs_str}\n")

                # 检查时间连续性并记录
                job_sequence = self.machine_job_sequences[machine_id]
                if job_sequence:
                    times_by_round = {}
                    for job in job_sequence:
                        round_num = job["round"]
                        if round_num not in times_by_round:
                            times_by_round[round_num] = []
                        times_by_round[round_num].append((job["start_time"], job["end_time"]))

                    # 输出每轮的时间范围
                    for round_num in sorted(times_by_round.keys()):
                        times = times_by_round[round_num]
                        min_start = min([t[0] for t in times])
                        max_end = max([t[1] for t in times])
                        f.write(f"  第{round_num + 1}轮时间范围: {min_start} - {max_end}\n")

                    # 添加机器完成时间
                    last_end_time = max(job["end_time"] for job in job_sequence)
                    f.write(f"  完成时间: {last_end_time}\n")