import numpy as np
import random
import time
import copy
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib
# 导入您现有的环境类，假设它在同一目录下的envs.py文件中
# 如果需要从其他地方导入，请修改导入路径
from envs import JobShopEnv
from config import Config

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong', 'WenQuanYi Micro Hei']  # 优先使用的中文字体列表
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
class GeneticAlgorithm:
    """遗传算法求解作业车间调度问题"""

    def __init__(self, env, population_size=100, generations=100,
                 crossover_rate=0.8, mutation_rate=0.2, elitism_size=50,
                 makespan_weight=0.7, tool_change_weight=0.3):
        """
        初始化遗传算法

        参数:
            env: JobShopEnv实例，作业车间环境
            population_size: 种群大小
            generations: 最大迭代代数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            elitism_size: 精英个体数量
            makespan_weight: 完工时间权重
            tool_change_weight: 模具更换次数权重
        """
        self.env = env  # 环境引用
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_size = elitism_size

        # 多目标优化权重
        self.makespan_weight = makespan_weight
        self.tool_change_weight = tool_change_weight

        # 种群
        self.population = []

        # 最优解
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.best_makespan = float('inf')
        self.best_tool_changes = float('inf')

        # 设置任务数和机器数
        self.num_jobs = env.num_jobs
        self.num_machines = env.num_machines

        # 历史记录
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_makespan': [],
            'best_tool_changes': []
        }

    def initialize_population(self):
        """初始化种群，每个个体是一个任务分配序列，考虑7个槽位和轮询处理方式"""
        self.population = []

        for _ in range(self.population_size):
            # 创建一个编码为(机器ID, 任务ID)对的序列
            # 每个任务需要分配到一个机器
            chromosome = []

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
                
                chromosome.append((machine_id, job_id))
                machine_slots[machine_id] += 1
                remaining_jobs.remove(job_id)
            
            # 处理剩余任务（如果有）
            for job_id in remaining_jobs:
                # 随机选择一个未满的机器
                available_machines = [i for i, slots in enumerate(machine_slots) if slots < Config.MAX_SLOTS]
                if available_machines:
                    machine_id = random.choice(available_machines)
                    chromosome.append((machine_id, job_id))
                    machine_slots[machine_id] += 1
                else:  # 所有机器都已满，随机分配
                    machine_id = random.randint(0, self.num_machines - 1)
                    chromosome.append((machine_id, job_id))

            # 按照轮询处理的逻辑对染色体进行排序
            # 首先按机器ID分组
            machine_groups = [[] for _ in range(self.num_machines)]
            for gene in chromosome:
                machine_id, job_id = gene
                machine_groups[machine_id].append(job_id)
                
            # 然后重新组织染色体，使得同一机器的任务按照添加顺序排列（模拟轮询处理）
            new_chromosome = []
            for machine_id, jobs in enumerate(machine_groups):
                for job_id in jobs:
                    new_chromosome.append((machine_id, job_id))
                    
            # 随机打乱不同机器之间的任务顺序，但保持同一机器内的顺序
            # 这样可以增加种群多样性，同时尊重轮询处理的约束
            machine_sections = []
            current_machine = -1
            current_section = []
            
            for gene in new_chromosome:
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
            
            # 重建染色体
            final_chromosome = []
            for section in machine_sections:
                final_chromosome.extend(section)
                
            self.population.append(final_chromosome)

    def evaluate_chromosome(self, chromosome):
        """评估单个染色体的适应度，考虑7个槽位和轮询处理方式"""
        # 重置环境
        state = self.env.reset()
        
        # 按机器ID对染色体进行分组，保持每组内的顺序
        machine_groups = [[] for _ in range(self.num_machines)]
        for action in chromosome:
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
        
        # 如果还有未分配任务，且机器有空槽位，使用贪婪策略分配
        # 这里使用贪婪策略而不是随机分配，以减少工具更换
        while self.env.pending_jobs and any(len(slots) < Config.MAX_SLOTS
                                            for slots in self.env.machine_slots):
            # 找出有空槽的机器
            available_machines = [i for i, slots in enumerate(self.env.machine_slots)
                                  if len(slots) < Config.MAX_SLOTS]
            
            if available_machines and self.env.pending_jobs:
                # 使用贪婪策略选择机器和任务
                best_action = None
                min_tool_impact = float('inf')
                
                for machine_id in available_machines:
                    # 获取当前机器上已加载的工具类型
                    current_types = set()
                    current_op_type = self.env.current_operation_type[machine_id]
                    if current_op_type is not None:
                        current_types.add(current_op_type)
                    
                    for slot in self.env.machine_slots[machine_id]:
                        job_id, op_idx = slot
                        if op_idx < len(self.env.jobs[job_id]):
                            op_type, _ = self.env.jobs[job_id][op_idx]
                            current_types.add(op_type)
                    
                    # 评估每个待分配任务的工具影响
                    for job_id in self.env.pending_jobs:
                        if self.env.job_operation_index[job_id] < len(self.env.jobs[job_id]):
                            op_type, _ = self.env.jobs[job_id][self.env.job_operation_index[job_id]]
                            
                            # 如果工具类型已加载，则无需更换
                            if op_type in current_types:
                                impact = 0
                            else:
                                impact = 1  # 需要工具更换
                            
                            if impact < min_tool_impact:
                                min_tool_impact = impact
                                best_action = (machine_id, job_id)
                
                # 如果找到了最佳动作，执行它
                if best_action:
                    machine_id, job_id = best_action
                    _, _, done, _ = self.env.step(best_action)
                    if done:
                        break
                else:
                    # 如果没有找到最佳动作，随机选择
                    machine_id = random.choice(available_machines)
                    job_id = random.choice(self.env.pending_jobs)
                    _, _, done, _ = self.env.step((machine_id, job_id))
                    if done:
                        break
            else:
                break
        
        # 获取调度结果
        makespan = self.env.total_time
        tool_changes = self.env.tool_changes
        
        # 计算多目标适应度（负值，因为我们要最小化这些目标）
        # 适应度越高越好，但我们要最小化makespan和tool_changes，所以取负值
        fitness = -(self.makespan_weight * makespan +
                    self.tool_change_weight * tool_changes * 10)  # 乘以10增加工具更换的影响
        
        return fitness, makespan, tool_changes

    def select_parents(self, k=3):
        """使用锦标赛选择法选择父代"""
        # 随机选择k个个体
        tournament = random.sample(range(len(self.population)), k)

        # 计算每个个体的适应度
        tournament_fitness = []
        for idx in tournament:
            fitness, _, _ = self.evaluate_chromosome(self.population[idx])
            tournament_fitness.append((idx, fitness))

        # 选择适应度最高的个体
        winner_idx = max(tournament_fitness, key=lambda x: x[1])[0]
        return self.population[winner_idx]

    def crossover(self, parent1, parent2):
        """执行顺序交叉(OX)操作，考虑7个槽位和轮询处理方式"""
        if random.random() > self.crossover_rate:
            return parent1, parent2

        # 首先按机器ID对父代进行分组
        parent1_groups = [[] for _ in range(self.num_machines)]
        parent2_groups = [[] for _ in range(self.num_machines)]
        
        for gene in parent1:
            machine_id, job_id = gene
            parent1_groups[machine_id].append(job_id)
            
        for gene in parent2:
            machine_id, job_id = gene
            parent2_groups[machine_id].append(job_id)
        
        # 对每台机器的任务序列执行交叉
        child1_groups = [[] for _ in range(self.num_machines)]
        child2_groups = [[] for _ in range(self.num_machines)]
        
        for machine_id in range(self.num_machines):
            p1_jobs = parent1_groups[machine_id]
            p2_jobs = parent2_groups[machine_id]
            
            if not p1_jobs or not p2_jobs:
                # 如果其中一个父代在该机器上没有任务，直接复制另一个父代的任务
                child1_groups[machine_id] = p1_jobs.copy() if p1_jobs else []
                child2_groups[machine_id] = p2_jobs.copy() if p2_jobs else []
                continue
                
            # 执行部分映射交叉(PMX)
            # 选择交叉点
            length = min(len(p1_jobs), len(p2_jobs))
            if length <= 1:
                # 如果序列太短，直接交换
                child1_groups[machine_id] = p2_jobs.copy()
                child2_groups[machine_id] = p1_jobs.copy()
                continue
                
            start = random.randint(0, length - 1)
            end = random.randint(start, length - 1)
            
            # 创建子代序列
            c1_jobs = [-1] * len(p1_jobs)
            c2_jobs = [-1] * len(p2_jobs)
            
            # 复制交叉段
            for i in range(start, end + 1):
                if i < len(p1_jobs):
                    c1_jobs[i] = p1_jobs[i]
                if i < len(p2_jobs):
                    c2_jobs[i] = p2_jobs[i]
            
            # 填充剩余位置
            for i in range(len(p1_jobs)):
                if i < start or i > end:
                    # 找到一个不在交叉段中的任务
                    for job_id in p2_jobs:
                        if job_id not in c1_jobs:
                            c1_jobs[i] = job_id
                            break
            
            for i in range(len(p2_jobs)):
                if i < start or i > end:
                    # 找到一个不在交叉段中的任务
                    for job_id in p1_jobs:
                        if job_id not in c2_jobs:
                            c2_jobs[i] = job_id
                            break
            
            # 检查是否有未分配的任务（可能由于长度不同）
            unassigned1 = [job_id for job_id in p2_jobs if job_id not in c1_jobs]
            unassigned2 = [job_id for job_id in p1_jobs if job_id not in c2_jobs]
            
            # 将未分配的任务添加到子代末尾
            c1_jobs.extend(unassigned1)
            c2_jobs.extend(unassigned2)
            
            # 检查槽位限制
            if len(c1_jobs) > Config.MAX_SLOTS:
                c1_jobs = c1_jobs[:Config.MAX_SLOTS]
            if len(c2_jobs) > Config.MAX_SLOTS:
                c2_jobs = c2_jobs[:Config.MAX_SLOTS]
                
            child1_groups[machine_id] = c1_jobs
            child2_groups[machine_id] = c2_jobs
        
        # 重建染色体
        child1 = []
        child2 = []
        
        for machine_id in range(self.num_machines):
            for job_id in child1_groups[machine_id]:
                child1.append((machine_id, job_id))
            for job_id in child2_groups[machine_id]:
                child2.append((machine_id, job_id))
        
        # 检查是否所有任务都被分配
        assigned_jobs1 = set(job_id for _, job_id in child1)
        assigned_jobs2 = set(job_id for _, job_id in child2)
        all_jobs = set(range(self.num_jobs))
        
        # 处理未分配的任务
        for job_id in all_jobs - assigned_jobs1:
            # 找一个未满的机器
            machine_counts = [sum(1 for m, j in child1 if m == mid) for mid in range(self.num_machines)]
            available_machines = [mid for mid, count in enumerate(machine_counts) if count < Config.MAX_SLOTS]
            
            if available_machines:
                machine_id = random.choice(available_machines)
            else:
                machine_id = random.randint(0, self.num_machines - 1)
                
            child1.append((machine_id, job_id))
            
        for job_id in all_jobs - assigned_jobs2:
            # 找一个未满的机器
            machine_counts = [sum(1 for m, j in child2 if m == mid) for mid in range(self.num_machines)]
            available_machines = [mid for mid, count in enumerate(machine_counts) if count < Config.MAX_SLOTS]
            
            if available_machines:
                machine_id = random.choice(available_machines)
            else:
                machine_id = random.randint(0, self.num_machines - 1)
                
            child2.append((machine_id, job_id))
        
        return child1, child2



    def mutate(self, chromosome):
        """执行变异操作，考虑7个槽位和轮询处理方式"""
        if random.random() > self.mutation_rate:
            return chromosome

        mutated = copy.deepcopy(chromosome)
        
        # 按机器ID对染色体进行分组
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
        
        # 重建染色体
        new_chromosome = []
        for machine_id, jobs in enumerate(machine_groups):
            for job_id in jobs:
                new_chromosome.append((machine_id, job_id))
        
        # 确保所有任务都被分配
        assigned_jobs = set(job_id for _, job_id in new_chromosome)
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
                
            new_chromosome.append((machine_id, job_id))
            machine_groups[machine_id].append(job_id)
        
        return new_chromosome

    def evolve(self):
        """执行一代进化"""
        # 评估当前种群
        evaluated_population = []
        for chromosome in self.population:
            fitness, makespan, tool_changes = self.evaluate_chromosome(chromosome)
            evaluated_population.append((chromosome, fitness, makespan, tool_changes))

        # 按适应度排序
        evaluated_population.sort(key=lambda x: x[1], reverse=True)

        # 更新历史记录
        avg_fitness = sum(item[1] for item in evaluated_population) / len(evaluated_population)
        self.history['avg_fitness'].append(avg_fitness)
        self.history['best_fitness'].append(evaluated_population[0][1])
        self.history['best_makespan'].append(evaluated_population[0][2])
        self.history['best_tool_changes'].append(evaluated_population[0][3])

        # 更新最优解
        if evaluated_population[0][1] > self.best_fitness:
            self.best_solution = evaluated_population[0][0]
            self.best_fitness = evaluated_population[0][1]
            self.best_makespan = evaluated_population[0][2]
            self.best_tool_changes = evaluated_population[0][3]

        # 保留精英个体
        new_population = [item[0] for item in evaluated_population[:self.elitism_size]]

        # 生成新个体直到种群大小不变
        while len(new_population) < self.population_size:
            # 选择父代
            parent1 = self.select_parents()
            parent2 = self.select_parents()

            # 交叉
            child1, child2 = self.crossover(parent1, parent2)

            # 变异
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            # 添加到新种群
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        # 更新种群
        self.population = new_population

    def run(self, verbose=True):
        """运行遗传算法"""
        start_time = time.time()

        # 初始化种群
        self.initialize_population()

        # 迭代进化
        for generation in range(self.generations):
            self.evolve()

            if verbose and (generation + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Generation {generation + 1}/{self.generations} | "
                      f"Best Fitness: {self.best_fitness:.2f} | "
                      f"Best Makespan: {self.best_makespan:.2f} | "
                      f"Best Tool Changes: {self.best_tool_changes} | "
                      f"Time: {elapsed_time:.2f}s")

        if verbose:
            print("\nOptimization Complete!")
            print(f"Total Runtime: {time.time() - start_time:.2f} seconds")
            print(f"Best Makespan: {self.best_makespan}")
            print(f"Best Tool Changes: {self.best_tool_changes}")
            print(f"Best Fitness: {self.best_fitness}")

        return self.best_solution, self.best_makespan, self.best_tool_changes

    def plot_progress(self, save_dir='ga_progress_charts'):
        """绘制优化进程图并保存到文件夹"""
        # 创建保存文件夹（如果不存在）
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        generations = range(1, len(self.history['best_fitness']) + 1)

        # 生成时间戳作为文件名的一部分，避免覆盖
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 1. 适应度曲线图
        plt.figure(figsize=(10, 6))
        plt.plot(generations, self.history['best_fitness'], 'b-', label='最优适应度')
        plt.plot(generations, self.history['avg_fitness'], 'r--', label='平均适应度')
        plt.xlabel('迭代代数')
        plt.ylabel('适应度')
        plt.title('GA遗传算法 - 迭代过程中的适应度变化')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'fitness_curve_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 完工时间曲线图
        plt.figure(figsize=(10, 6))
        plt.plot(generations, self.history['best_makespan'], 'g-')
        plt.xlabel('迭代代数')
        plt.ylabel('完工时间')
        plt.title('GA遗传算法 - 迭代过程中的最优完工时间')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'makespan_curve_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 工具更换次数曲线图
        plt.figure(figsize=(10, 6))
        plt.plot(generations, self.history['best_tool_changes'], 'm-')
        plt.xlabel('迭代代数')
        plt.ylabel('工具更换次数')
        plt.title('GA遗传算法 - 迭代过程中的最优工具更换次数')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'tool_changes_curve_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. 多目标权衡图
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.history['best_makespan'], self.history['best_tool_changes'],
                              c=generations, cmap='viridis', s=30)
        plt.xlabel('完工时间')
        plt.ylabel('工具更换次数')
        plt.title('GA遗传算法 - 完工时间与工具更换次数的权衡关系')
        plt.grid(True)
        plt.colorbar(scatter, label='迭代代数')
        plt.savefig(os.path.join(save_dir, f'tradeoff_curve_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. 组合图 (所有子图在一起)
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # 适应度曲线
        axs[0, 0].plot(generations, self.history['best_fitness'], 'b-', label='最优适应度')
        axs[0, 0].plot(generations, self.history['avg_fitness'], 'r--', label='平均适应度')
        axs[0, 0].set_xlabel('迭代代数')
        axs[0, 0].set_ylabel('适应度')
        axs[0, 0].set_title('GA遗传算法 - 迭代过程中的适应度变化')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # 完工时间曲线
        axs[0, 1].plot(generations, self.history['best_makespan'], 'g-')
        axs[0, 1].set_xlabel('迭代代数')
        axs[0, 1].set_ylabel('完工时间')
        axs[0, 1].set_title('GA遗传算法 - 迭代过程中的最优完工时间')
        axs[0, 1].grid(True)

        # 工具更换次数曲线
        axs[1, 0].plot(generations, self.history['best_tool_changes'], 'm-')
        axs[1, 0].set_xlabel('迭代代数')
        axs[1, 0].set_ylabel('工具更换次数')
        axs[1, 0].set_title('GA遗传算法 - 迭代过程中的最优工具更换次数')
        axs[1, 0].grid(True)

        # 多目标权衡图
        scatter = axs[1, 1].scatter(self.history['best_makespan'], self.history['best_tool_changes'],
                                    c=generations, cmap='viridis', s=30)
        axs[1, 1].set_xlabel('完工时间')
        axs[1, 1].set_ylabel('工具更换次数')
        axs[1, 1].set_title('GA遗传算法 - 完工时间与工具更换次数的权衡关系')
        axs[1, 1].grid(True)
        fig.colorbar(scatter, ax=axs[1, 1], label='迭代代数')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'all_charts_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"图表已保存到 {save_dir} 文件夹")

    def apply_best_solution(self, env=None):
        """将最优解应用于环境并返回结果"""
        if env is None:
            env = self.env

        # 重置环境
        state = env.reset()

        # 执行最优解中的每个动作
        for action in self.best_solution:
            machine_id, job_id = action

            # 检查任务是否还在待分配列表中
            if job_id in env.pending_jobs:
                # 检查机器槽位是否有空间
                if len(env.machine_slots[machine_id]) < Config.MAX_SLOTS:
                    # 执行动作
                    _, _, done, _ = env.step(action)

                    # 如果所有任务都完成了，跳出循环
                    if done:
                        break

        # 处理剩余未分配任务
        while env.pending_jobs and any(len(slots) < Config.MAX_SLOTS
                                       for slots in env.machine_slots):
            # 找出有空槽的机器
            available_machines = [i for i, slots in enumerate(env.machine_slots)
                                  if len(slots) < Config.MAX_SLOTS]

            # 随机选择一个机器和一个待分配任务
            if available_machines and env.pending_jobs:
                machine_id = random.choice(available_machines)
                job_id = random.choice(env.pending_jobs)

                # 执行动作
                _, _, done, _ = env.step((machine_id, job_id))

                # 如果所有任务都完成了，跳出循环
                if done:
                    break

        return env.total_time, env.tool_changes, env


def run_ga_optimization(data_file, num_machines=3, population_size=100, generations=100,
                        makespan_weight=0.7, tool_change_weight=0.3):
    """运行遗传算法优化调度问题"""
    # 创建环境
    env = JobShopEnv(data_file, num_machines)
    print(f"数据集: {data_file}")
    print(f"任务数: {env.num_jobs}, 工序类型数: {env.num_operation_types}, 机器数: {num_machines}")

    # 创建遗传算法实例
    ga = GeneticAlgorithm(
        env=env,
        population_size=population_size,
        generations=generations,
        makespan_weight=makespan_weight,
        tool_change_weight=tool_change_weight,
        crossover_rate=0.8,
        mutation_rate=0.2,
        elitism_size=int(population_size * 0.1)  # 保留10%的精英个体
    )

    print(f"\n开始遗传算法优化...")
    # 运行遗传算法
    best_solution, best_makespan, best_tool_changes = ga.run(verbose=True)

    # 绘制优化进程
    ga.plot_progress()

    # 将最优解应用于环境
    makespan, tool_changes, env = ga.apply_best_solution()

    print("\n最优调度结果:")
    print(f"完工时间: {makespan}")
    print(f"工具更换次数: {tool_changes}")

    # 打印每台机器处理的任务序列
    env.print_job_sequences()

    # 如果有甘特图工具，可以生成甘特图
    try:
        from gantt_chart_utils import generate_all_charts
        print("\n生成甘特图...")
        generate_all_charts(env, output_dir='ga_charts')
    except ImportError:
        print("\n提示: 甘特图工具未找到，将不会生成可视化图表")

    return best_solution, best_makespan, best_tool_changes, env


# 多目标权重调整分析函数
def analyze_weight_combinations(data_file, num_machines=3, population_size=50, generations=50, step_size=0.1, num_trials=10):
    """分析不同权重组合下的优化结果，可自定义步长测试权重参数
    
    参数:
        data_file: 数据文件路径
        num_machines: 机器数量
        population_size: 种群大小
        generations: 迭代代数
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
        print(f"\n测试权重组合: 完工时间权重={makespan_w}, 工具更换权重={tool_w}")
        
        # 存储多次测试的结果
        trial_makespans = []
        trial_tool_changes = []
        
        for trial in range(num_trials):
            print(f"  正在进行第 {trial+1}/{num_trials} 次测试...")
            
            # 创建环境
            env = JobShopEnv(data_file, num_machines)

            # 创建遗传算法实例
            ga = GeneticAlgorithm(
                env=env,
                population_size=population_size,
                generations=generations,
                makespan_weight=makespan_w,
                tool_change_weight=tool_w,
                crossover_rate=0.8,
                mutation_rate=0.2,
                elitism_size=int(population_size * 0.1)
            )

            # 运行遗传算法
            _, makespan, tool_changes = ga.run(verbose=False)
            
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
    plt.title('遗传算法 - 不同权重组合的帕累托前沿', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加算法名称标识
    plt.text(0.02, 0.98, 'GA算法', transform=plt.gca().transAxes, 
             fontsize=12, fontweight='bold', verticalalignment='top')
    
    # 添加权重说明
    plt.figtext(0.5, 0.01, f'权重从(1.0,0.0)到(0.0,1.0)，步长{step_size}，每种权重测试{num_trials}次取平均值', 
                ha='center', fontsize=10, bbox={'facecolor':'lightgray', 'alpha':0.5, 'pad':5})
    
    plt.tight_layout()
    plt.savefig('ga_pareto_front.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results


# 示例用法
if __name__ == "__main__":
    # 指定数据文件路径
    import os
    # 使用绝对路径访问数据文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "datasets", "data_1.txt")  # 使用绝对路径

    # 运行单次优化
    '''
    
    
    best_solution, best_makespan, best_tool_changes, env = run_ga_optimization(
        data_file=data_file,
        num_machines=3,
        population_size=300,
        generations=100,
        makespan_weight=0.7,
        tool_change_weight=0.3
    )
    
    '''

    # 分析不同权重组合的效果
    weight_results = analyze_weight_combinations(data_file)