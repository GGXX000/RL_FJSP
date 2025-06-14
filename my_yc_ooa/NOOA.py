import numpy as np
import random
import time
import copy
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong', 'WenQuanYi Micro Hei']  # 优先使用的中文字体列表
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
from envs import JobShopEnv
from config import Config


class OspreyOptimizationAlgorithm:
    """鱼鹰优化算法用于作业车间调度问题"""

    def __init__(self, env, population_size=30, max_iterations=100,
                 makespan_weight=0.7, tool_change_weight=0.3, mutation_rate=0.2):
        """
        初始化鱼鹰优化算法

        参数:
            env: JobShopEnv实例，用于模拟环境
            population_size: 种群大小（鱼鹰数量）
            max_iterations: 最大迭代次数
            makespan_weight: 完工时间目标权重
            tool_change_weight: 工具更换目标权重
            mutation_rate: 变异概率
        """
        self.env = env
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.mutation_rate = mutation_rate

        # 目标权重
        self.makespan_weight = makespan_weight
        self.tool_change_weight = tool_change_weight

        # 鱼鹰种群
        self.ospreys = []

        # 最优解记录
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.best_makespan = float('inf')
        self.best_tool_changes = float('inf')

        # 设置任务和机器数量
        self.num_jobs = env.num_jobs
        self.num_machines = env.num_machines

        # 历史记录，用于跟踪算法进展
        self.history = {
            'best_fitness': [],  # 最优适应度历史
            'avg_fitness': [],  # 平均适应度历史
            'best_makespan': [],  # 最优完工时间历史
            'best_tool_changes': []  # 最优工具更换次数历史
        }

    def initialize_population(self):
        """初始化鱼鹰种群，随机生成初始解，考虑7个槽位和轮询处理方式"""
        self.ospreys = []

        for _ in range(self.population_size):
            # 每个鱼鹰代表一个调度方案，即(machine_id, job_id)对的序列
            osprey = []

            # 为每个任务随机分配机器，考虑每台机器最多7个槽位
            machine_slots = [0 for _ in range(self.num_machines)]  # 记录每台机器已分配的槽位数
            remaining_jobs = list(range(self.num_jobs))
            
            # 首先确保每台机器的槽位尽量平均分配
            while remaining_jobs:
                # 找出槽位数最少的机器
                min_slots = min(machine_slots)
                eligible_machines = [i for i, slots in enumerate(machine_slots) if slots == min_slots and slots < Config.MAX_SLOTS]
                
                if not eligible_machines:  # 所有机器槽位已满
                    break
                    
                machine_id = random.choice(eligible_machines)
                job_id = random.choice(remaining_jobs)
                
                osprey.append((machine_id, job_id))
                machine_slots[machine_id] += 1
                remaining_jobs.remove(job_id)
            
            # 处理剩余任务（如果有）
            for job_id in remaining_jobs:
                # 随机选择一个未满的机器
                available_machines = [i for i, slots in enumerate(machine_slots) if slots < Config.MAX_SLOTS]
                if available_machines:
                    machine_id = random.choice(available_machines)
                    osprey.append((machine_id, job_id))
                    machine_slots[machine_id] += 1
                else:  # 所有机器都已满，随机分配
                    machine_id = random.randint(0, self.num_machines - 1)
                    osprey.append((machine_id, job_id))

            # 按照轮询处理的逻辑对鱼鹰解进行排序
            # 首先按机器ID分组
            machine_groups = [[] for _ in range(self.num_machines)]
            for gene in osprey:
                machine_id, job_id = gene
                machine_groups[machine_id].append(job_id)
                
            # 然后重新组织解，使得同一机器的任务按照添加顺序排列（模拟轮询处理）
            new_osprey = []
            for machine_id, jobs in enumerate(machine_groups):
                for job_id in jobs:
                    new_osprey.append((machine_id, job_id))
                    
            # 随机打乱不同机器之间的任务顺序，但保持同一机器内的顺序
            # 这样可以增加种群多样性，同时尊重轮询处理的约束
            machine_sections = []
            current_machine = -1
            current_section = []
            
            for gene in new_osprey:
                machine_id, _ = gene
                if machine_id != current_machine:
                    if current_section:
                        machine_sections.append(current_section)
                    current_section = [gene]
                    current_machine = machine_id
                else:
                    current_section.append(gene)
            
            if current_section:  # 添加最后一个部分
                machine_sections.append(current_section)
                
            # 随机打乱机器部分的顺序
            random.shuffle(machine_sections)
            
            # 重建鱼鹰解
            final_osprey = []
            for section in machine_sections:
                final_osprey.extend(section)
                
            self.ospreys.append(final_osprey)

    def evaluate_osprey(self, osprey):
        """评估单个鱼鹰的适应度（解的质量），考虑7个槽位和轮询处理方式"""
        # 重置环境
        state = self.env.reset()
        
        # 按机器ID对鱼鹰解进行分组，保持每组内的顺序
        machine_groups = [[] for _ in range(self.num_machines)]
        for action in osprey:
            machine_id, job_id = action
            machine_groups[machine_id].append(job_id)
        
        # 检查每台机器的槽位数是否超过最大限制
        for machine_id, jobs in enumerate(machine_groups):
            if len(jobs) > Config.MAX_SLOTS:
                # 如果超过限制，只保留前MAX_SLOTS个任务
                machine_groups[machine_id] = jobs[:Config.MAX_SLOTS]
        
        # 按照轮询方式执行任务分配
        # 首先分配每台机器的第一个槽位，然后是第二个槽位，以此类推
        max_slots_used = max([len(jobs) for jobs in machine_groups])
        
        for slot_idx in range(max_slots_used):
            for machine_id, jobs in enumerate(machine_groups):
                if slot_idx < len(jobs):
                    job_id = jobs[slot_idx]
                    
                    # 检查任务是否还在待分配列表中
                    if job_id in self.env.pending_jobs:
                        # 检查机器槽位是否有空间
                        if len(self.env.machine_slots[machine_id]) < Config.MAX_SLOTS:
                            # 执行动作
                            _, _, done, _ = self.env.step((machine_id, job_id))
                            
                            # 如果所有任务都完成了，跳出循环
                            if done:
                                break
        
        # 处理剩余未分配任务，使用贪婪策略
        self._process_remaining_tasks()

        # 获取调度结果
        makespan = self.env.total_time
        tool_changes = self.env.tool_changes

        # 计算适应度（负值，因为我们在最小化目标）
        fitness = -(self.makespan_weight * makespan +
                    self.tool_change_weight * tool_changes * 10)

        return fitness, makespan, tool_changes

    def _process_remaining_tasks(self):
        """使用贪婪策略处理剩余未分配的任务，考虑轮询处理方式"""
        # 首先按照轮询方式分配任务
        # 记录每台机器当前的槽位数
        machine_slots_count = [len(slots) for slots in self.env.machine_slots]
        max_slots = max(machine_slots_count)
        
        # 如果有机器的槽位数小于最大值，先尝试平衡分配
        if min(machine_slots_count) < max_slots and self.env.pending_jobs:
            for slot_idx in range(min(machine_slots_count), max_slots):
                for machine_id in range(self.num_machines):
                    if machine_slots_count[machine_id] <= slot_idx and self.env.pending_jobs:
                        # 选择最适合当前机器的任务（减少工具更换）
                        best_job = self._select_best_job_for_machine(machine_id)
                        
                        if best_job is not None:
                            # 执行动作
                            _, _, done, _ = self.env.step((machine_id, best_job))
                            machine_slots_count[machine_id] += 1
                            
                            # 如果所有任务完成，跳出循环
                            if done:
                                return
        
        # 处理剩余任务，使用贪婪策略
        while self.env.pending_jobs and any(len(slots) < Config.MAX_SLOTS
                                            for slots in self.env.machine_slots):
            # 查找有可用槽位的机器
            available_machines = [i for i, slots in enumerate(self.env.machine_slots)
                                  if len(slots) < Config.MAX_SLOTS]

            if available_machines and self.env.pending_jobs:
                # 对于剩余任务，使用贪婪方法分配
                # 查找负载最少的机器
                machine_loads = [sum(self.env.jobs[job_id][op_idx][1] if op_idx < len(self.env.jobs[job_id]) else 0 
                                   for job_id, op_idx in slots) 
                                 for slots in self.env.machine_slots]
                machine_id = available_machines[np.argmin([machine_loads[m] for m in available_machines])]

                # 选择最适合当前机器的任务
                best_job = self._select_best_job_for_machine(machine_id)
                
                # 如果没有找到合适的任务，选择第一个待分配任务
                if best_job is None and self.env.pending_jobs:
                    best_job = self.env.pending_jobs[0]

                # 执行动作
                _, _, done, _ = self.env.step((machine_id, best_job))

                # 如果所有任务完成，跳出循环
                if done:
                    break
            else:
                break
                
    def _select_best_job_for_machine(self, machine_id):
        """为指定机器选择最适合的任务，尽量减少工具更换"""
        # 获取当前机器上已加载的工具类型
        current_types = set()
        if self.env.current_operation_type[machine_id] is not None:
            current_types.add(self.env.current_operation_type[machine_id])
            
        for job_id, op_idx in self.env.machine_slots[machine_id]:
            if op_idx < len(self.env.jobs[job_id]):
                op_type, _ = self.env.jobs[job_id][op_idx]
                current_types.add(op_type)
        
        # 评估每个待分配任务
        best_job = None
        min_tool_impact = float('inf')
        
        for job_id in self.env.pending_jobs:
            if self.env.job_operation_index[job_id] < len(self.env.jobs[job_id]):
                op_type, proc_time = self.env.jobs[job_id][self.env.job_operation_index[job_id]]
                
                # 计算工具影响和处理时间
                if op_type in current_types:
                    tool_impact = 0  # 无需工具更换
                else:
                    tool_impact = 1  # 需要工具更换
                
                # 综合考虑工具更换和处理时间
                impact = tool_impact * Config.TOOL_CHANGE_TIME + proc_time * 0.1
                
                if impact < min_tool_impact:
                    min_tool_impact = impact
                    best_job = job_id
        
        return best_job

    def find_best_fish(self):
        """评估所有鱼鹰并找到最佳解（鱼）"""
        evaluated_ospreys = []
        for osprey in self.ospreys:
            fitness, makespan, tool_changes = self.evaluate_osprey(osprey)
            evaluated_ospreys.append((osprey, fitness, makespan, tool_changes))

        # 按适应度排序（降序，因为适应度为负值）
        evaluated_ospreys.sort(key=lambda x: x[1], reverse=True)

        # 如果有更好的解，更新最优解
        if evaluated_ospreys[0][1] > self.best_fitness:
            self.best_solution = evaluated_ospreys[0][0]
            self.best_fitness = evaluated_ospreys[0][1]
            self.best_makespan = evaluated_ospreys[0][2]
            self.best_tool_changes = evaluated_ospreys[0][3]

        # 记录历史
        self.history['best_fitness'].append(self.best_fitness)
        self.history['avg_fitness'].append(sum(o[1] for o in evaluated_ospreys) / len(evaluated_ospreys))
        self.history['best_makespan'].append(self.best_makespan)
        self.history['best_tool_changes'].append(self.best_tool_changes)

        return evaluated_ospreys

    def mutate_osprey(self, osprey):
        """对鱼鹰解进行变异，考虑7个槽位和轮询处理方式"""
        if random.random() > self.mutation_rate:
            return osprey

        mutated = copy.deepcopy(osprey)
        
        # 按机器ID对鱼鹰解进行分组
        machine_groups = [[] for _ in range(self.num_machines)]
        for gene in mutated:
            machine_id, job_id = gene
            machine_groups[machine_id].append(job_id)
        
        # 选择变异类型
        mutation_type = random.choice(['swap', 'machine_change', 'insert', 'slot_balance'])
        
        if mutation_type == 'swap':
            # 在同一机器内交换两个任务的位置（保持轮询顺序）
            # 或者交换两个不同机器上的任务（改变分配）
            if random.random() < 0.5 and any(len(jobs) >= 2 for jobs in machine_groups):
                # 同一机器内交换
                eligible_machines = [i for i, jobs in enumerate(machine_groups) if len(jobs) >= 2]
                if eligible_machines:
                    machine_id = random.choice(eligible_machines)
                    jobs = machine_groups[machine_id]
                    idx1, idx2 = random.sample(range(len(jobs)), 2)
                    jobs[idx1], jobs[idx2] = jobs[idx2], jobs[idx1]
            else:
                # 不同机器间交换任务
                non_empty_machines = [i for i, jobs in enumerate(machine_groups) if jobs]
                if len(non_empty_machines) >= 2:
                    m1, m2 = random.sample(non_empty_machines, 2)
                    if machine_groups[m1] and machine_groups[m2]:
                        j1 = random.choice(machine_groups[m1])
                        j2 = random.choice(machine_groups[m2])
                        
                        # 从原机器中移除
                        machine_groups[m1].remove(j1)
                        machine_groups[m2].remove(j2)
                        
                        # 添加到新机器
                        machine_groups[m1].append(j2)
                        machine_groups[m2].append(j1)
        
        elif mutation_type == 'machine_change':
            # 改变一个任务的分配机器
            non_empty_machines = [i for i, jobs in enumerate(machine_groups) if jobs]
            if non_empty_machines:
                source_machine = random.choice(non_empty_machines)
                if machine_groups[source_machine]:
                    # 选择一个任务
                    job_id = random.choice(machine_groups[source_machine])
                    
                    # 从源机器移除
                    machine_groups[source_machine].remove(job_id)
                    
                    # 选择目标机器，优先选择槽位未满的机器
                    target_machines = [i for i in range(self.num_machines) 
                                      if len(machine_groups[i]) < Config.MAX_SLOTS and i != source_machine]
                    
                    if not target_machines:  # 如果所有机器都已满，随机选择
                        target_machines = [i for i in range(self.num_machines) if i != source_machine]
                    
                    if target_machines:  # 确保有可用的目标机器
                        target_machine = random.choice(target_machines)
                        machine_groups[target_machine].append(job_id)
                    else:  # 如果没有其他机器，放回原机器
                        machine_groups[source_machine].append(job_id)
        
        elif mutation_type == 'insert':
            # 改变任务在同一机器内的处理顺序
            eligible_machines = [i for i, jobs in enumerate(machine_groups) if len(jobs) >= 2]
            if eligible_machines:
                machine_id = random.choice(eligible_machines)
                jobs = machine_groups[machine_id]
                
                # 选择一个任务并改变其位置
                idx1 = random.randint(0, len(jobs) - 1)
                idx2 = random.randint(0, len(jobs) - 1)
                while idx1 == idx2 and len(jobs) > 1:
                    idx2 = random.randint(0, len(jobs) - 1)
                
                job = jobs.pop(idx1)
                jobs.insert(idx2, job)
        
        elif mutation_type == 'slot_balance':
            # 平衡各机器的槽位分配
            # 从槽位较多的机器移动任务到槽位较少的机器
            max_slots = max(len(jobs) for jobs in machine_groups)
            min_slots = min(len(jobs) for jobs in machine_groups)
            
            if max_slots > min_slots + 1:  # 如果不平衡
                # 找出槽位最多和最少的机器
                max_machine = [i for i, jobs in enumerate(machine_groups) if len(jobs) == max_slots]
                min_machine = [i for i, jobs in enumerate(machine_groups) if len(jobs) == min_slots]
                
                if max_machine and min_machine:
                    source = random.choice(max_machine)
                    target = random.choice(min_machine)
                    
                    if machine_groups[source]:
                        # 移动一个任务
                        job_id = random.choice(machine_groups[source])
                        machine_groups[source].remove(job_id)
                        machine_groups[target].append(job_id)
        
        # 重建鱼鹰解
        new_osprey = []
        for machine_id, jobs in enumerate(machine_groups):
            for job_id in jobs:
                new_osprey.append((machine_id, job_id))
        
        # 确保所有任务都被分配
        assigned_jobs = set(job_id for _, job_id in new_osprey)
        all_jobs = set(range(self.num_jobs))
        
        # 处理未分配的任务
        for job_id in all_jobs - assigned_jobs:
            # 找一个未满的机器
            machine_counts = [len(machine_groups[mid]) for mid in range(self.num_machines)]
            available_machines = [mid for mid, count in enumerate(machine_counts) if count < Config.MAX_SLOTS]
            
            if available_machines:
                machine_id = random.choice(available_machines)
            else:
                machine_id = random.randint(0, self.num_machines - 1)
                
            new_osprey.append((machine_id, job_id))
            machine_groups[machine_id].append(job_id)
        
        return new_osprey
        
    def phase1_hunting_fish(self, osprey, iteration, fish_positions):
        """
        第一阶段：识别鱼的位置并猎捕（探索阶段）
        直接使用OOA论文中的公式: x_new = x_old + r * (fish_pos - I * x_old)
        """
        new_osprey = copy.deepcopy(osprey)
        
        # 如果没有更好的鱼位置，使用随机探索
        if not fish_positions:
            # 随机探索 - 交换操作
            if len(new_osprey) > 1:
                idx1, idx2 = random.sample(range(len(new_osprey)), 2)
                new_osprey[idx1], new_osprey[idx2] = new_osprey[idx2], new_osprey[idx1]
            return new_osprey
        
        # 随机选择一条鱼（更好的解）去猎捕
        selected_fish = random.choice(fish_positions)
        
        # 使用OOA公式: x_new = x_old + r * (fish_pos - I * x_old)
        I = random.randint(1, 2)  # 随机强度因子，1或2
        r = random.random()       # 随机系数
        
        # 对于离散问题，我们需要将公式应用到解的表示上
        # 我们可以通过概率来决定是否采用鱼的位置元素
        for i in range(min(len(new_osprey), len(selected_fish))):
            # 计算概率 p = r * (1 - I * 相似度)
            # 相似度可以简单定义为0或1（相同为1，不同为0）
            similarity = 1 if new_osprey[i] == selected_fish[i] else 0
            p = r * (1 - I * similarity)
            
            # 根据概率决定是否采用鱼的位置
            if random.random() < p:
                new_osprey[i] = selected_fish[i]
        
        return new_osprey

    def phase2_carrying_fish(self, osprey, iteration):
        """
        第二阶段：携带鱼回巢（利用阶段）
        在这个阶段，鱼鹰会携带捕获的鱼回到巢穴，这个过程中会进行局部搜索
        """
        # 复制当前解
        new_osprey = copy.deepcopy(osprey)
        
        # 计算当前迭代的搜索强度
        # 随着迭代次数增加，搜索范围逐渐缩小（从探索到利用）
        intensity = 1 - (iteration / self.max_iterations)
        
        # 对解进行变异操作，增加局部搜索能力
        mutated_osprey = self.mutate_osprey(new_osprey)
        
        # 根据当前迭代阶段，在原解和变异解之间进行插值
        # 如果是早期迭代，更倾向于变异解（探索）
        # 如果是后期迭代，更倾向于原解（利用）
        final_osprey = []
        
        # 按机器ID对原始解和变异解进行分组
        orig_machine_groups = [[] for _ in range(self.num_machines)]
        for gene in osprey:
            machine_id, job_id = gene
            orig_machine_groups[machine_id].append(job_id)
            
        mut_machine_groups = [[] for _ in range(self.num_machines)]
        for gene in mutated_osprey:
            machine_id, job_id = gene
            mut_machine_groups[machine_id].append(job_id)
        
        # 对每台机器，根据强度选择原始解或变异解的任务
        for machine_id in range(self.num_machines):
            # 如果随机值小于强度，选择变异解的任务分配
            if random.random() < intensity:
                for job_id in mut_machine_groups[machine_id]:
                    final_osprey.append((machine_id, job_id))
            else:
                # 否则选择原始解的任务分配
                for job_id in orig_machine_groups[machine_id]:
                    final_osprey.append((machine_id, job_id))
        
        # 确保所有任务都被分配
        assigned_jobs = set(job_id for _, job_id in final_osprey)
        all_jobs = set(range(self.num_jobs))
        
        # 处理未分配的任务
        for job_id in all_jobs - assigned_jobs:
            # 找一个未满的机器
            machine_slots = [sum(1 for _, j in final_osprey if m == _) for m in range(self.num_machines)]
            available_machines = [m for m, slots in enumerate(machine_slots) if slots < Config.MAX_SLOTS]
            
            if available_machines:
                machine_id = random.choice(available_machines)
            else:
                machine_id = random.randint(0, self.num_machines - 1)
                
            final_osprey.append((machine_id, job_id))
        
        return final_osprey
        
    def phase3_hunting_fish(self, osprey, iteration, best_solution):
        """
        第三阶段：与最优解交互（全局最优引导）
        在这个阶段，鱼鹰会受到全局最优解的引导，向最优解方向移动
        """
        # 如果没有最优解，返回原始解
        if best_solution is None:
            return osprey
            
        # 复制当前解
        new_osprey = copy.deepcopy(osprey)
        
        # 计算当前迭代的引导强度
        # 随着迭代次数增加，引导强度逐渐增强（更多地向最优解靠拢）
        guidance_strength = iteration / self.max_iterations
        
        # 按机器ID对当前解和最优解进行分组
        current_machine_groups = [[] for _ in range(self.num_machines)]
        for gene in osprey:
            machine_id, job_id = gene
            current_machine_groups[machine_id].append(job_id)
            
        best_machine_groups = [[] for _ in range(self.num_machines)]
        for gene in best_solution:
            machine_id, job_id = gene
            best_machine_groups[machine_id].append(job_id)
        
        # 创建新解
        final_osprey = []
        
        # 对每台机器，根据引导强度决定是否采用最优解的任务分配
        for machine_id in range(self.num_machines):
            # 如果随机值小于引导强度，采用最优解的任务分配
            if random.random() < guidance_strength and best_machine_groups[machine_id]:
                # 从最优解中选择一部分任务（比例由引导强度决定）
                num_tasks_to_take = max(1, int(len(best_machine_groups[machine_id]) * guidance_strength))
                tasks_to_take = random.sample(best_machine_groups[machine_id], 
                                             min(num_tasks_to_take, len(best_machine_groups[machine_id])))
                
                # 将这些任务添加到新解中
                for job_id in tasks_to_take:
                    final_osprey.append((machine_id, job_id))
                    
                # 从当前解中选择剩余任务
                remaining_tasks = [job for job in current_machine_groups[machine_id] 
                                 if job not in tasks_to_take]
                
                # 确保不超过最大槽位数
                remaining_slots = Config.MAX_SLOTS - len(tasks_to_take)
                if remaining_slots > 0 and remaining_tasks:
                    for job_id in remaining_tasks[:remaining_slots]:
                        final_osprey.append((machine_id, job_id))
            else:
                # 否则保持当前解的任务分配
                for job_id in current_machine_groups[machine_id]:
                    final_osprey.append((machine_id, job_id))
        
        # 确保所有任务都被分配
        assigned_jobs = set(job_id for _, job_id in final_osprey)
        all_jobs = set(range(self.num_jobs))
        
        # 处理未分配的任务
        for job_id in all_jobs - assigned_jobs:
            # 找一个未满的机器
            machine_slots = [sum(1 for _, j in final_osprey if m == _) for m in range(self.num_machines)]
            available_machines = [m for m, slots in enumerate(machine_slots) if slots < Config.MAX_SLOTS]
            
            if available_machines:
                machine_id = random.choice(available_machines)
            else:
                machine_id = random.randint(0, self.num_machines - 1)
                
            final_osprey.append((machine_id, job_id))
        
        return final_osprey
        """
        第二阶段：将鱼携带到合适位置（利用阶段）
        直接使用OOA论文中的公式: x_new = x_old + (lb + r * (ub - lb))/t
        """
        new_osprey = copy.deepcopy(osprey)
        
        # 使用OOA公式: x_new = x_old + (lb + r * (ub - lb))/t
        # 对于离散问题，我们需要适当转换
        
        # 计算随迭代减少的调整因子
        adjustment_factor = 1 / iteration
        
        # 根据调整因子确定要进行的修改数量
        num_modifications = max(1, int(len(new_osprey) * adjustment_factor * 0.3))
        
        # 随机选择位置进行修改
        positions_to_modify = random.sample(range(len(new_osprey)), min(num_modifications, len(new_osprey)))
        
        for pos in positions_to_modify:
            # 对于作业车间调度，我们可以通过以下方式模拟公式效果:
            # 1. 随机选择一种修改类型
            mod_type = random.choice(['swap', 'change_machine', 'insert'])
            
            if mod_type == 'swap' and len(new_osprey) > 1:
                # 交换两个操作
                idx2 = random.randint(0, len(new_osprey) - 1)
                while idx2 == pos and len(new_osprey) > 1:
                    idx2 = random.randint(0, len(new_osprey) - 1)
                new_osprey[pos], new_osprey[idx2] = new_osprey[idx2], new_osprey[pos]
                
            elif mod_type == 'change_machine':
                # 更改任务的机器分配
                machine_id, job_id = new_osprey[pos]
                new_machine_id = random.randint(0, self.num_machines - 1)
                new_osprey[pos] = (new_machine_id, job_id)
                
            elif mod_type == 'insert' and len(new_osprey) > 1:
                # 将操作插入到不同位置
                idx2 = random.randint(0, len(new_osprey) - 1)
                if pos != idx2:
                    job = new_osprey.pop(pos)
                    new_osprey.insert(idx2, job)
        
        return new_osprey

    def run(self, verbose=True):
        """运行鱼鹰优化算法"""
        start_time = time.time()

        # 初始化种群
        self.initialize_population()

        # 迭代
        for iteration in range(1, self.max_iterations + 1):
            # 评估所有鱼鹰并找到最优解
            evaluated_ospreys = self.find_best_fish()

            # 创建新种群
            new_ospreys = []

            # 对种群中的每个鱼鹰
            for osprey, fitness, _, _ in evaluated_ospreys:
                # 找出比当前鱼鹰更好的解（鱼的位置）
                fish_positions = [better_osprey for better_osprey, better_fitness, _, _ in evaluated_ospreys
                                  if better_fitness > fitness]

                # 始终包含最优解
                if self.best_solution not in fish_positions:
                    fish_positions.append(self.best_solution)

                # 第一阶段：猎捕鱼（探索）
                new_osprey1 = self.phase1_hunting_fish(osprey, iteration, fish_positions)

                # 评估新位置
                fitness1, _, _ = self.evaluate_osprey(new_osprey1)

                # 如果改进了，更新鱼鹰位置
                if fitness1 > fitness:
                    osprey = new_osprey1
                    fitness = fitness1

                # 第二阶段：携带鱼（利用）
                new_osprey2 = self.phase2_carrying_fish(osprey, iteration)

                # 评估新位置
                fitness2, _, _ = self.evaluate_osprey(new_osprey2)

                # 如果改进了，更新鱼鹰位置
                if fitness2 > fitness:
                    osprey = new_osprey2
                    fitness = fitness2
                    
                # 第三阶段：与最优解交互（全局最优引导）
                new_osprey3 = self.phase3_hunting_fish(osprey, iteration, self.best_solution)
                
                # 评估新位置
                fitness3, _, _ = self.evaluate_osprey(new_osprey3)
                
                # 如果改进了，更新鱼鹰位置
                if fitness3 > fitness:
                    osprey = new_osprey3

                # 添加到新种群
                new_ospreys.append(osprey)

            # 更新下一代的种群
            self.ospreys = new_ospreys

            # 打印进度
            if verbose and (iteration % 10 == 0 or iteration == 1):
                elapsed_time = time.time() - start_time
                print(f"迭代 {iteration}/{self.max_iterations} | "
                      f"最优适应度: {self.best_fitness:.2f} | "
                      f"最优完工时间: {self.best_makespan:.2f} | "
                      f"最优工具更换: {self.best_tool_changes} | "
                      f"运行时间: {elapsed_time:.2f}秒")

        if verbose:
            print("\n优化完成!")
            print(f"总运行时间: {time.time() - start_time:.2f} 秒")
            print(f"最优完工时间: {self.best_makespan}")
            print(f"最优工具更换次数: {self.best_tool_changes}")
            print(f"最优适应度: {self.best_fitness}")

        return self.best_solution, self.best_makespan, self.best_tool_changes

    def plot_progress(self, save_dir='osprey_progress_charts'):
        """绘制并保存优化进度图表"""
        # 如果保存目录不存在，则创建
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        iterations = range(1, len(self.history['best_fitness']) + 1)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 1. 适应度曲线
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, self.history['best_fitness'], 'b-', label='最优适应度')
        plt.plot(iterations, self.history['avg_fitness'], 'r--', label='平均适应度')
        plt.xlabel('迭代次数')
        plt.ylabel('适应度')
        plt.title('迭代过程中的适应度变化')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'fitness_curve_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 完工时间曲线
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, self.history['best_makespan'], 'g-')
        plt.xlabel('迭代次数')
        plt.ylabel('完工时间')
        plt.title('迭代过程中的最优完工时间')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'makespan_curve_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 工具更换次数曲线
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, self.history['best_tool_changes'], 'm-')
        plt.xlabel('迭代次数')
        plt.ylabel('工具更换次数')
        plt.title('迭代过程中的最优工具更换次数')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'tool_changes_curve_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. 多目标权衡图
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.history['best_makespan'], self.history['best_tool_changes'],
                              c=iterations, cmap='viridis', s=30)
        plt.xlabel('完工时间')
        plt.ylabel('工具更换次数')
        plt.title('完工时间与工具更换次数的权衡关系')
        plt.grid(True)
        plt.colorbar(scatter, label='迭代次数')
        plt.savefig(os.path.join(save_dir, f'tradeoff_curve_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. 组合图表
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # 适应度曲线
        axs[0, 0].plot(iterations, self.history['best_fitness'], 'b-', label='最优适应度')
        axs[0, 0].plot(iterations, self.history['avg_fitness'], 'r--', label='平均适应度')
        axs[0, 0].set_xlabel('迭代次数')
        axs[0, 0].set_ylabel('适应度')
        axs[0, 0].set_title('迭代过程中的适应度变化')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # 完工时间曲线
        axs[0, 1].plot(iterations, self.history['best_makespan'], 'g-')
        axs[0, 1].set_xlabel('迭代次数')
        axs[0, 1].set_ylabel('完工时间')
        axs[0, 1].set_title('迭代过程中的最优完工时间')
        axs[0, 1].grid(True)

        # 工具更换次数曲线
        axs[1, 0].plot(iterations, self.history['best_tool_changes'], 'm-')
        axs[1, 0].set_xlabel('迭代次数')
        axs[1, 0].set_ylabel('工具更换次数')
        axs[1, 0].set_title('迭代过程中的最优工具更换次数')
        axs[1, 0].grid(True)

        # 多目标权衡图
        scatter = axs[1, 1].scatter(self.history['best_makespan'], self.history['best_tool_changes'],
                                    c=iterations, cmap='viridis', s=30)
        axs[1, 1].set_xlabel('完工时间')
        axs[1, 1].set_ylabel('工具更换次数')
        axs[1, 1].set_title('完工时间与工具更换次数的权衡关系')
        axs[1, 1].grid(True)
        fig.colorbar(scatter, ax=axs[1, 1], label='迭代次数')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'all_charts_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"图表已保存到 {save_dir} 目录")

    def apply_best_solution(self, env=None):
        """应用找到的最优解到环境中"""
        if env is None:
            env = self.env

        # 重置环境
        state = env.reset()

        # 执行最优解
        for action in self.best_solution:
            machine_id, job_id = action

            # 检查任务是否仍在待分配列表中且机器有空间
            if job_id in env.pending_jobs:
                if len(env.machine_slots[machine_id]) < Config.MAX_SLOTS:
                    # 执行动作
                    _, _, done, _ = env.step(action)

                    # 如果所有任务完成，则跳出循环
                    if done:
                        break

        # 处理剩余任务
        while env.pending_jobs and any(len(slots) < Config.MAX_SLOTS
                                       for slots in env.machine_slots):
            # 查找有可用槽位的机器
            available_machines = [i for i, slots in enumerate(env.machine_slots)
                                  if len(slots) < Config.MAX_SLOTS]

            if available_machines and env.pending_jobs:
                # 对剩余任务使用简单的贪婪分配
                machine_id = random.choice(available_machines)
                job_id = random.choice(env.pending_jobs)

                # 执行动作
                _, _, done, _ = env.step((machine_id, job_id))

                # 如果所有任务完成，则跳出循环
                if done:
                    break
            else:
                break

        return env.total_time, env.tool_changes, env


def run_osprey_optimization(data_file, num_machines=3, population_size=30, max_iterations=100,
                            makespan_weight=0.7, tool_change_weight=0.3):
    """运行鱼鹰优化算法求解作业车间调度问题"""
    # 创建环境
    env = JobShopEnv(data_file, num_machines)
    print(f"数据集: {data_file}")
    print(f"任务数: {env.num_jobs}, 工序类型数: {env.num_operation_types}, 机器数: {num_machines}")

    # 创建OOA实例
    ooa = OspreyOptimizationAlgorithm(
        env=env,
        population_size=population_size,
        max_iterations=max_iterations,
        makespan_weight=makespan_weight,
        tool_change_weight=tool_change_weight
    )

    print(f"\n开始运行鱼鹰优化算法...")
    # 运行算法
    best_solution, best_makespan, best_tool_changes = ooa.run(verbose=True)

    # 绘制进度图
    ooa.plot_progress()

    # 应用最优解
    makespan, tool_changes, env = ooa.apply_best_solution()

    print("\n最优调度结果:")
    print(f"完工时间: {makespan}")
    print(f"工具更换次数: {tool_changes}")

    # 打印详细的任务序列
    env.print_job_sequences()

    # 如果可用，生成甘特图
    try:
        from gantt_chart_utils import generate_all_charts
        print("\n生成甘特图...")
        generate_all_charts(env, output_dir='ooa_charts')
    except ImportError:
        print("\n提示: 甘特图工具未找到，未生成可视化图表。")

    return best_solution, best_makespan, best_tool_changes, env


# 多目标权重分析函数
def analyze_weight_combinations(data_file, num_machines=3, population_size=30, max_iterations=50, step_size=0.1, num_trials=10):
    """分析不同权重组合下的优化结果，可自定义步长测试权重参数
    
    参数:
        data_file: 数据文件路径
        num_machines: 机器数量
        population_size: 种群大小
        max_iterations: 最大迭代次数
        step_size: 权重分析的步长，默认为0.1
        num_trials: 每种权重组合的测试次数，默认为10次
    """
    # 计算权重组合数量
    num_steps = int(1.0 / step_size) + 1
    
    # 权重组合，使用自定义步长
    weight_combinations = []
    for i in range(num_steps):
        makespan_w = round(1.0 - i * step_size, 2)  # 保留两位小数
        tool_w = round(i * step_size, 2)  # 保留两位小数
        weight_combinations.append((makespan_w, tool_w))
    
    # 权重组合说明
    weight_descriptions = {
        (1.0, 0.0): "仅考虑完工时间",
        (0.0, 1.0): "仅考虑工具更换"
    }
    # 其他权重组合的描述
    for i in range(1, num_steps-1):
        makespan_w = round(1.0 - i * step_size, 2)
        tool_w = round(i * step_size, 2)
        weight_descriptions[(makespan_w, tool_w)] = f"完工时间权重={makespan_w}, 工具更换权重={tool_w}"

    results = []

    for makespan_w, tool_w in weight_combinations:
        print(f"\n测试权重组合: 完工时间={makespan_w}, 工具更换={tool_w}")
        
        # 存储多次测试的结果
        trial_makespans = []
        trial_tool_changes = []
        
        for trial in range(num_trials):
            print(f"  正在进行第 {trial+1}/{num_trials} 次测试...")
            
            # 创建环境
            env = JobShopEnv(data_file, num_machines)

            # 创建OOA实例
            ooa = OspreyOptimizationAlgorithm(
                env=env,
                population_size=population_size,
                max_iterations=max_iterations,
                makespan_weight=makespan_w,
                tool_change_weight=tool_w
            )

            # 以减少输出的方式运行算法
            _, makespan, tool_changes = ooa.run(verbose=False)
            
            # 收集本次测试结果
            trial_makespans.append(makespan)
            trial_tool_changes.append(tool_changes)
        
        # 计算平均值
        avg_makespan = sum(trial_makespans) / len(trial_makespans)
        avg_tool_changes = sum(trial_tool_changes) / len(trial_tool_changes)
        
        # 记录平均结果
        results.append({
            'makespan_weight': makespan_w,
            'tool_weight': tool_w,
            'best_makespan': avg_makespan,
            'best_tool_changes': avg_tool_changes
        })

        print(f"平均完工时间: {avg_makespan:.2f}, 平均工具更换次数: {avg_tool_changes:.2f}")
        print(f"完工时间标准差: {np.std(trial_makespans):.2f}, 工具更换次数标准差: {np.std(trial_tool_changes):.2f}")

    # 绘制帕累托前沿
    plt.figure(figsize=(12, 8))

    makespan_values = [r['best_makespan'] for r in results]
    tool_values = [r['best_tool_changes'] for r in results]
    weights = [f"({r['makespan_weight']}, {r['tool_weight']})" for r in results]

    # 按照权重排序，确保曲线连接正确
    sorted_indices = sorted(range(len(results)), key=lambda i: results[i]['makespan_weight'], reverse=True)
    sorted_makespan = [makespan_values[i] for i in sorted_indices]
    sorted_tool = [tool_values[i] for i in sorted_indices]
    sorted_weights = [weights[i] for i in sorted_indices]

    # 绘制散点图
    plt.scatter(sorted_makespan, sorted_tool, s=100, c='blue', zorder=5)

    # 添加权重标签
    for i, txt in enumerate(sorted_weights):
        plt.annotate(txt, (sorted_makespan[i], sorted_tool[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.xlabel('完工时间', fontsize=12)
    plt.ylabel('工具更换次数', fontsize=12)
    plt.title('鱼鹰优化算法 - 不同权重组合的帕累托前沿', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加算法名称标识
    plt.text(0.02, 0.98, 'OOA算法', transform=plt.gca().transAxes, 
             fontsize=12, fontweight='bold', verticalalignment='top')
    
    # 添加权重说明
    plt.figtext(0.5, 0.01, f'权重从(1.0,0.0)到(0.0,1.0)，步长{step_size}，每种权重测试{num_trials}次取平均值', 
                ha='center', fontsize=10, bbox={'facecolor':'lightgray', 'alpha':0.5, 'pad':5})
    
    plt.tight_layout()
    plt.savefig('ooa_pareto_front.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results


# 主程序
if __name__ == "__main__":
    # 指定数据文件路径
    data_file = "datasets/data_1.txt"  # 替换为您的数据文件路径

    # 运行单次优化
    '''
      best_solution, best_makespan, best_tool_changes, env = run_osprey_optimization(
        data_file=data_file,
        num_machines=3,
        population_size=300,
        max_iterations=100,
        makespan_weight=0.7,
        tool_change_weight=0.3
    )
    
    '''
    # 可选：分析不同权重组合
    weight_results = analyze_weight_combinations(data_file)