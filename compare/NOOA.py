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
                 makespan_weight=0.7, tool_change_weight=0.3):
        """
        初始化鱼鹰优化算法

        参数:
            env: JobShopEnv实例，用于模拟环境
            population_size: 种群大小（鱼鹰数量）
            max_iterations: 最大迭代次数
            makespan_weight: 完工时间目标权重
            tool_change_weight: 工具更换目标权重
        """
        self.env = env
        self.population_size = population_size
        self.max_iterations = max_iterations
        print(f"NOOA 种群大小：{population_size} 迭代次数：{max_iterations}")
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
        """初始化鱼鹰种群，随机生成初始解"""
        self.ospreys = []

        for _ in range(self.population_size):
            # 每个鱼鹰代表一个调度方案，即(machine_id, job_id)对的序列
            osprey = []

            # 为每个任务创建分配
            for job_id in range(self.num_jobs):
                machine_id = random.randint(0, self.num_machines - 1)
                osprey.append((machine_id, job_id))

            # 随机打乱分配顺序
            random.shuffle(osprey)

            self.ospreys.append(osprey)

    def evaluate_osprey(self, osprey):
        """评估单个鱼鹰的适应度（解的质量）"""
        # 重置环境
        state = self.env.reset()

        # 执行鱼鹰代表的调度动作
        for action in osprey:
            machine_id, job_id = action

            # 检查任务是否仍在待分配列表中且机器有空间
            if job_id in self.env.pending_jobs:
                if len(self.env.machine_slots[machine_id]) < Config.MAX_SLOTS:
                    # 执行动作
                    _, _, done, _ = self.env.step(action)

                    # 如果所有任务都已分配，则跳出循环
                    if done:
                        break

        # 处理剩余未分配任务
        self._process_remaining_tasks()

        # 获取调度结果
        makespan = self.env.total_time
        tool_changes = self.env.tool_changes

        # 计算适应度（负值，因为我们在最小化目标）
        fitness = -(self.makespan_weight * makespan +
                    self.tool_change_weight * tool_changes * 10)

        return fitness, makespan, tool_changes

    def _process_remaining_tasks(self):
        """使用贪婪策略处理剩余未分配的任务"""
        while self.env.pending_jobs and any(len(slots) < Config.MAX_SLOTS
                                            for slots in self.env.machine_slots):
            # 查找有可用槽位的机器
            available_machines = [i for i, slots in enumerate(self.env.machine_slots)
                                  if len(slots) < Config.MAX_SLOTS]

            if available_machines and self.env.pending_jobs:
                # 对于剩余任务，使用贪婪方法分配
                # 查找负载最少的机器
                machine_loads = [sum(op[1] for op in slots) for slots in self.env.machine_slots]
                machine_id = available_machines[np.argmin([machine_loads[m] for m in available_machines])]

                # 选择尽可能减少工具更换的任务
                current_types = set()
                for job_id, op_idx in self.env.machine_slots[machine_id]:
                    if op_idx < len(self.env.jobs[job_id]):
                        current_types.add(self.env.jobs[job_id][op_idx][0])

                best_job = None
                min_tool_impact = float('inf')

                for job_id in self.env.pending_jobs:
                    if self.env.job_operation_index[job_id] < len(self.env.jobs[job_id]):
                        op_type = self.env.jobs[job_id][self.env.job_operation_index[job_id]][0]
                        # 如果类型已加载，则无需工具更换
                        if op_type in current_types or op_type == self.env.current_operation_type[machine_id]:
                            best_job = job_id
                            break
                        else:
                            # 否则，评估影响
                            impact = 1  # 工具更换惩罚
                            if min_tool_impact > impact:
                                min_tool_impact = impact
                                best_job = job_id

                # 如果没有找到任务，选择第一个
                if best_job is None and self.env.pending_jobs:
                    best_job = self.env.pending_jobs[0]

                # 执行动作
                _, _, done, _ = self.env.step((machine_id, best_job))

                # 如果所有任务完成，跳出循环
                if done:
                    break
            else:
                break

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

    def phase1_hunting_fish(self, osprey, iteration, fish_positions):
        """
        第一阶段：识别鱼的位置并猎捕（探索阶段）
        基于OOA论文：模拟鱼鹰在水下攻击鱼的行为
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

        # 根据OOA论文中的公式计算新位置
        # x_new = x_old + r * (fish_pos - I * x_old)
        # 其中r是随机数，I是随机强度（1或2）

        # 选择一个随机点基于鱼的位置进行修改
        modify_point = random.randint(0, min(len(new_osprey), len(selected_fish)) - 1)

        # 70%的概率直接从鱼那里复制（积极猎捕）
        if random.random() < 0.7:
            new_osprey[modify_point] = selected_fish[modify_point]

        # 30%的概率进行随机修改（探索）
        else:
            # 随机选择一种修改类型
            mod_type = random.choice(['swap', 'change_machine', 'insert'])

            if mod_type == 'swap':
                # 交换两个操作
                if len(new_osprey) > 1:
                    idx1, idx2 = random.sample(range(len(new_osprey)), 2)
                    new_osprey[idx1], new_osprey[idx2] = new_osprey[idx2], new_osprey[idx1]

            elif mod_type == 'change_machine':
                # 为任务更换分配的机器
                idx = random.randint(0, len(new_osprey) - 1)
                machine_id, job_id = new_osprey[idx]
                new_machine_id = random.randint(0, self.num_machines - 1)
                while new_machine_id == machine_id and self.num_machines > 1:
                    new_machine_id = random.randint(0, self.num_machines - 1)
                new_osprey[idx] = (new_machine_id, job_id)

            elif mod_type == 'insert':
                # 将一个操作插入到不同位置
                if len(new_osprey) > 1:
                    idx1, idx2 = random.sample(range(len(new_osprey)), 2)
                    if idx1 != idx2:
                        job = new_osprey.pop(idx1)
                        new_osprey.insert(idx2, job)

        return new_osprey

    def phase2_carrying_fish(self, osprey, iteration):
        """
        第二阶段：将鱼携带到合适位置（利用阶段）
        基于OOA论文：局部微调解决方案
        """
        new_osprey = copy.deepcopy(osprey)

        # 使用OOA论文中的公式：
        # x_new = x_old + (lb + r * (ub - lb))/t
        # 其中t是当前迭代次数

        # 对于作业车间调度，我们将实现为：
        # 1. 在保持问题结构的情况下进行小幅扰动
        # 2. 变化强度随迭代次数减少

        # 计算随迭代减少的调整因子
        adjustment_factor = 1 - (iteration / self.max_iterations)

        # 根据调整因子确定要进行的修改数量
        # 在早期迭代中进行更多修改，后期减少
        num_modifications = max(1, int(adjustment_factor * 3))

        for _ in range(num_modifications):
            # 根据调整因子的权重选择修改类型
            if random.random() < adjustment_factor:
                # 早期进行更激进的更改
                mod_type = random.choice(['swap', 'change_machine', 'insert'])
            else:
                # 后期进行更保守的更改
                mod_type = random.choice(['swap', 'reorder'])

            if mod_type == 'swap':
                # 交换两个操作
                if len(new_osprey) > 1:
                    idx1, idx2 = random.sample(range(len(new_osprey)), 2)
                    new_osprey[idx1], new_osprey[idx2] = new_osprey[idx2], new_osprey[idx1]

            elif mod_type == 'change_machine':
                # 更改任务的机器分配
                idx = random.randint(0, len(new_osprey) - 1)
                machine_id, job_id = new_osprey[idx]
                new_machine_id = random.randint(0, self.num_machines - 1)
                new_osprey[idx] = (new_machine_id, job_id)

            elif mod_type == 'insert':
                # 将一个操作插入到不同位置
                if len(new_osprey) > 1:
                    idx1, idx2 = random.sample(range(len(new_osprey)), 2)
                    if idx1 != idx2:
                        job = new_osprey.pop(idx1)
                        new_osprey.insert(idx2, job)

            elif mod_type == 'reorder':
                # 重排一小段操作序列
                if len(new_osprey) > 2:
                    # 随着迭代进行，选择更小的段大小
                    seg_size = max(2, int(adjustment_factor * min(5, len(new_osprey) // 2)))
                    start = random.randint(0, len(new_osprey) - seg_size)
                    segment = new_osprey[start:start + seg_size]
                    random.shuffle(segment)
                    new_osprey[start:start + seg_size] = segment

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
def analyze_weight_combinations(data_file, num_machines=3, population_size=30, max_iterations=50):
    """分析多目标优化的不同权重组合"""
    # 测试的权重组合
    weight_combinations = [
        (1.0, 0.0),  # 仅考虑完工时间
        (0.8, 0.2),  # 主要考虑完工时间
        (0.5, 0.5),  # 平衡权重
        (0.2, 0.8),  # 主要考虑工具更换
        (0.0, 1.0)  # 仅考虑工具更换
    ]

    results = []

    for makespan_w, tool_w in weight_combinations:
        print(f"\n测试权重组合: 完工时间={makespan_w}, 工具更换={tool_w}")

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
        _, best_makespan, best_tool_changes = ooa.run(verbose=False)

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

    plt.xlabel('完工时间')
    plt.ylabel('工具更换次数')
    plt.title('不同权重组合的帕累托前沿')
    plt.grid(True)
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