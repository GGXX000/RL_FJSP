import numpy as np
import random
import time
import copy
import matplotlib.pyplot as plt
from collections import defaultdict

# 导入您现有的环境类，假设它在同一目录下的envs.py文件中
# 如果需要从其他地方导入，请修改导入路径
from envs import JobShopEnv
from config import Config


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
        print(f"GA 种群大小：{population_size} 迭代次数：{generations}")
        # 历史记录
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_makespan': [],
            'best_tool_changes': []
        }

    def initialize_population(self):
        """初始化种群，每个个体是一个任务分配序列"""
        self.population = []

        for _ in range(self.population_size):
            # 创建一个编码为(机器ID, 任务ID)对的序列
            # 每个任务需要分配到一个机器
            chromosome = []

            # 为每个任务随机分配机器
            for job_id in range(self.num_jobs):
                machine_id = random.randint(0, self.num_machines - 1)
                chromosome.append((machine_id, job_id))

            # 随机打乱任务分配顺序
            random.shuffle(chromosome)

            self.population.append(chromosome)

    def evaluate_chromosome(self, chromosome):
        """评估单个染色体的适应度"""
        # 重置环境
        state = self.env.reset()

        # 执行染色体中的每个行动
        for action in chromosome:
            machine_id, job_id = action

            # 检查任务是否还在待分配列表中
            if job_id in self.env.pending_jobs:
                # 检查机器槽位是否有空间
                if len(self.env.machine_slots[machine_id]) < Config.MAX_SLOTS:
                    # 执行动作
                    _, _, done, _ = self.env.step(action)

                    # 如果所有任务都完成了，跳出循环
                    if done:
                        break

        # 如果还有未分配任务，且机器有空槽位，随机分配
        while self.env.pending_jobs and any(len(slots) < Config.MAX_SLOTS
                                            for slots in self.env.machine_slots):
            # 找出有空槽的机器
            available_machines = [i for i, slots in enumerate(self.env.machine_slots)
                                  if len(slots) < Config.MAX_SLOTS]

            # 随机选择一个机器和一个待分配任务
            if available_machines and self.env.pending_jobs:
                machine_id = random.choice(available_machines)
                job_id = random.choice(self.env.pending_jobs)

                # 执行动作
                _, _, done, _ = self.env.step((machine_id, job_id))

                # 如果所有任务都完成了，跳出循环
                if done:
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
        """执行顺序交叉(OX)操作"""
        if random.random() > self.crossover_rate:
            return parent1, parent2

        # 选择两个交叉点
        length = len(parent1)
        start = random.randint(0, length - 2)
        end = random.randint(start + 1, length - 1)

        # 创建子代
        child1 = [None] * length
        child2 = [None] * length

        # 复制交叉段
        for i in range(start, end + 1):
            child1[i] = parent1[i]
            child2[i] = parent2[i]

        # 填充剩余位置
        self._fill_crossover_child(parent2, child1, start, end)
        self._fill_crossover_child(parent1, child2, start, end)

        return child1, child2

    def _fill_crossover_child(self, parent, child, start, end):
        """填充交叉后的子代剩余位置"""
        # 创建已使用任务的集合
        used_jobs = set(job_id for _, job_id in [gene for gene in child if gene is not None])

        # 从parent中按顺序找出未使用的任务
        remaining_genes = [gene for gene in parent if gene[1] not in used_jobs]

        # 填充child中的剩余位置
        idx = 0
        for i in range(len(child)):
            if child[i] is None:
                child[i] = remaining_genes[idx]
                idx += 1

    def mutate(self, chromosome):
        """执行变异操作"""
        if random.random() > self.mutation_rate:
            return chromosome

        mutated = copy.deepcopy(chromosome)

        # 选择变异类型
        mutation_type = random.choice(['swap', 'machine_change', 'insert'])

        if mutation_type == 'swap':
            # 交换两个位置的任务
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]

        elif mutation_type == 'machine_change':
            # 改变一个任务的分配机器
            idx = random.randint(0, len(mutated) - 1)
            machine_id, job_id = mutated[idx]
            new_machine_id = random.randint(0, self.num_machines - 1)
            while new_machine_id == machine_id:
                new_machine_id = random.randint(0, self.num_machines - 1)
            mutated[idx] = (new_machine_id, job_id)

        elif mutation_type == 'insert':
            # 将一个位置的任务插入到另一个位置
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            gene = mutated.pop(idx2)
            mutated.insert(idx1, gene)

        return mutated

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
        plt.plot(generations, self.history['best_fitness'], 'b-', label='Best Fitness')
        plt.plot(generations, self.history['avg_fitness'], 'r--', label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Over Generations')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'fitness_curve_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 完工时间曲线图
        plt.figure(figsize=(10, 6))
        plt.plot(generations, self.history['best_makespan'], 'g-')
        plt.xlabel('Generation')
        plt.ylabel('Makespan')
        plt.title('Best Makespan Over Generations')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'makespan_curve_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 工具更换次数曲线图
        plt.figure(figsize=(10, 6))
        plt.plot(generations, self.history['best_tool_changes'], 'm-')
        plt.xlabel('Generation')
        plt.ylabel('Tool Changes')
        plt.title('Best Tool Changes Over Generations')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'tool_changes_curve_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. 多目标权衡图
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.history['best_makespan'], self.history['best_tool_changes'],
                              c=generations, cmap='viridis', s=30)
        plt.xlabel('Makespan')
        plt.ylabel('Tool Changes')
        plt.title('Makespan vs Tool Changes Trade-off')
        plt.grid(True)
        plt.colorbar(scatter, label='Generation')
        plt.savefig(os.path.join(save_dir, f'tradeoff_curve_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. 组合图 (所有子图在一起)
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # 适应度曲线
        axs[0, 0].plot(generations, self.history['best_fitness'], 'b-', label='Best Fitness')
        axs[0, 0].plot(generations, self.history['avg_fitness'], 'r--', label='Average Fitness')
        axs[0, 0].set_xlabel('Generation')
        axs[0, 0].set_ylabel('Fitness')
        axs[0, 0].set_title('Fitness Over Generations')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # 完工时间曲线
        axs[0, 1].plot(generations, self.history['best_makespan'], 'g-')
        axs[0, 1].set_xlabel('Generation')
        axs[0, 1].set_ylabel('Makespan')
        axs[0, 1].set_title('Best Makespan Over Generations')
        axs[0, 1].grid(True)

        # 工具更换次数曲线
        axs[1, 0].plot(generations, self.history['best_tool_changes'], 'm-')
        axs[1, 0].set_xlabel('Generation')
        axs[1, 0].set_ylabel('Tool Changes')
        axs[1, 0].set_title('Best Tool Changes Over Generations')
        axs[1, 0].grid(True)

        # 多目标权衡图
        scatter = axs[1, 1].scatter(self.history['best_makespan'], self.history['best_tool_changes'],
                                    c=generations, cmap='viridis', s=30)
        axs[1, 1].set_xlabel('Makespan')
        axs[1, 1].set_ylabel('Tool Changes')
        axs[1, 1].set_title('Makespan vs Tool Changes Trade-off')
        axs[1, 1].grid(True)
        fig.colorbar(scatter, ax=axs[1, 1], label='Generation')

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
def analyze_weight_combinations(data_file, num_machines=3, population_size=50, generations=50):
    """分析不同权重组合下的优化结果"""
    # 权重组合
    weight_combinations = [
        (1.0, 0.0),  # 仅考虑完工时间
        (0.8, 0.2),  # 完工时间为主，兼顾工具更换
        (0.5, 0.5),  # 平衡两个目标
        (0.2, 0.8),  # 工具更换为主，兼顾完工时间
        (0.0, 1.0)  # 仅考虑工具更换
    ]

    results = []

    for makespan_w, tool_w in weight_combinations:
        print(f"\n测试权重组合: 完工时间权重={makespan_w}, 工具更换权重={tool_w}")

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
        _, best_makespan, best_tool_changes = ga.run(verbose=False)

        # 记录结果
        results.append({
            'makespan_weight': makespan_w,
            'tool_weight': tool_w,
            'best_makespan': best_makespan,
            'best_tool_changes': best_tool_changes
        })

        print(f"完工时间: {best_makespan}, 工具更换次数: {best_tool_changes}")

    # 绘制帕累托前沿
    plt.figure(figsize=(10, 6))

    makespan_values = [r['best_makespan'] for r in results]
    tool_values = [r['best_tool_changes'] for r in results]
    weights = [f"({r['makespan_weight']}, {r['tool_weight']})" for r in results]

    plt.scatter(makespan_values, tool_values, s=100, c='blue')

    # 添加权重标签
    for i, txt in enumerate(weights):
        plt.annotate(txt, (makespan_values[i], tool_values[i]),
                     xytext=(5, 5), textcoords='offset points')

    plt.xlabel('Makespan')
    plt.ylabel('Tool Changes')
    plt.title('Pareto Front for Different Weight Combinations')
    plt.grid(True)
    plt.show()

    return results


# 示例用法
if __name__ == "__main__":
    # 指定数据文件路径
    data_file = "datasets/data_1.txt"  # 替换为您的数据文件路径

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