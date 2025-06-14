import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import os
from datetime import datetime

# 修复中文字体问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def save_gantt_chart(env, filename, show_rounds=False, highlight_task=None, color_by='task'):
    """
    为作业车间调度生成甘特图并保存

    参数:
    env: JobShopEnv环境对象
    filename: 保存文件名
    show_rounds: 是否显示轮次边界
    highlight_task: 可选，要高亮显示的任务ID
    color_by: 着色方式 - 'task'(按任务),'type'(按工序类型),'round'(按轮次)
    """
    # 创建图表
    fig_height = max(8, env.num_machines * 0.8)  # 根据机器数量调整高度
    fig, ax = plt.subplots(figsize=(16, fig_height))

    # 颜色设置
    if color_by == 'task':
        # 为每个任务分配不同颜色
        colors = plt.cm.get_cmap('tab20', env.num_jobs)
    elif color_by == 'type':
        # 为每种工序类型分配不同颜色
        colors = plt.cm.get_cmap('tab20', env.num_operation_types)
    else:  # color_by == 'round'
        # 为每轮分配不同颜色
        max_rounds = max([len(rounds) for rounds in env.machine_rounds]) if env.machine_rounds else 1
        colors = plt.cm.get_cmap('tab20', max_rounds)

    # 获取所有机器的最大完成时间作为图表宽度
    max_time = env.total_time

    # 定义Y轴标记
    y_ticks = np.arange(env.num_machines)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'Machine {i}' for i in range(env.num_machines)])  # 使用英文避免字体问题

    # 设置X轴范围和标签
    ax.set_xlim(0, max_time * 1.02)  # 给右侧留点空间
    ax.set_xlabel('Time')

    # 设置标题
    ax.set_title(f'Job Shop Scheduling - Total Time: {env.total_time}, Tool Changes: {env.tool_changes}')

    # 添加网格线
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # 添加每个任务的工序块
    legend_handles = []  # 图例项
    legend_labels = []  # 图例标签
    added_legends = set()  # 已添加到图例的项

    # 存储轮次边界，用于后续标记
    round_boundaries = {}

    # 遍历每台机器
    for machine_id in range(env.num_machines):
        job_sequence = env.machine_job_sequences[machine_id]
        if not job_sequence:
            continue

        # 按开始时间排序
        sorted_sequence = sorted(job_sequence, key=lambda x: x["start_time"])

        # 收集该机器上的轮次边界时间点
        if show_rounds:
            round_boundaries[machine_id] = []
            current_round = -1
            for job in sorted_sequence:
                if job["round"] > current_round:
                    current_round = job["round"]
                    # 记录新轮次的开始时间
                    round_boundaries[machine_id].append(job["start_time"])

        # 为每个工序绘制矩形
        for job in sorted_sequence:
            job_id = job["job_id"]
            op_type = job["type"]
            start_time = job["start_time"]
            duration = job["end_time"] - job["start_time"]
            round_num = job["round"]

            # 决定颜色
            if color_by == 'task':
                color_idx = job_id % 20  # 使用模20避免索引超出范围
                color = colors(color_idx)
                legend_key = f'Task {job_id}'
            elif color_by == 'type':
                color_idx = op_type % 20  # 使用模20避免索引超出范围
                color = colors(color_idx)
                legend_key = f'Type {op_type}'
            else:  # color_by == 'round'
                color_idx = round_num % 20  # 使用模20避免索引超出范围
                color = colors(color_idx)
                legend_key = f'Round {round_num + 1}'

            # 如果有高亮任务，调整透明度
            alpha = 1.0
            edgecolor = 'black'
            linewidth = 0.5

            if highlight_task is not None:
                if job_id == highlight_task:
                    alpha = 1.0
                    edgecolor = 'red'
                    linewidth = 2
                else:
                    alpha = 0.3

            # 绘制任务工序块
            rect = patches.Rectangle(
                (start_time, machine_id - 0.4),  # (x, y)
                duration,  # width
                0.8,  # height
                linewidth=linewidth,
                edgecolor=edgecolor,
                facecolor=color,
                alpha=alpha
            )
            ax.add_patch(rect)

            # 在工序块中添加文本标签
            if duration > 5:  # 只在足够宽的块中添加标签
                label_text = f'J{job_id}-O{job["operation"] - 1}'
                ax.text(
                    start_time + duration / 2,
                    machine_id,
                    label_text,
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='black',
                    fontweight='bold'
                )

            # 添加图例项（避免重复）
            if legend_key not in added_legends:
                legend_handle = patches.Patch(color=color, label=legend_key)
                legend_handles.append(legend_handle)
                legend_labels.append(legend_key)
                added_legends.add(legend_key)

    # 添加轮次边界线
    if show_rounds:
        for machine_id, boundaries in round_boundaries.items():
            for time_point in boundaries:
                ax.axvline(x=time_point, ymin=(machine_id - 0.4) / env.num_machines,
                           ymax=(machine_id + 0.4) / env.num_machines,
                           color='red', linestyle='--', alpha=0.7)

    # 添加图例（限制数量防止过多）
    if len(legend_handles) > 20:
        # 如果图例项过多，只显示前20个
        ax.legend(legend_handles[:20], legend_labels[:20],
                  title='Legend (showing first 20)', loc='upper right')
    else:
        ax.legend(legend_handles, legend_labels,
                  title='Legend', loc='upper right')

    # 保存图表
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"甘特图已保存至: {filename}")


def create_interactive_gantt(env, filename):
    """
    创建交互式HTML甘特图

    参数:
    env: JobShopEnv环境对象
    filename: 保存的HTML文件名
    """
    try:
        import plotly.figure_factory as ff
        import plotly.io as pio
        import plotly.graph_objects as go
        import pandas as pd
    except ImportError:
        print("需要安装plotly和pandas库以创建交互式甘特图。请运行: pip install plotly pandas")
        return

    # 准备甘特图数据
    gantt_data = []

    # 遍历每台机器上的每个工序
    for machine_id in range(env.num_machines):
        job_sequence = env.machine_job_sequences[machine_id]

        for job in job_sequence:
            job_id = job["job_id"]
            op_idx = job["operation"] - 1
            op_type = job["type"]
            start_time = job["start_time"]
            end_time = job["end_time"]
            round_num = job["round"]

            # 添加到甘特图数据
            gantt_data.append(dict(
                Task=f'Machine {machine_id}',
                Start=start_time,
                Finish=end_time,
                # 使用统一标识符，避免颜色映射问题
                Resource="Task",
                Description=f'J{job_id}O{op_idx} (Type:{op_type}, Round:{round_num + 1})'
            ))

    # 转换为DataFrame
    df = pd.DataFrame(gantt_data)

    # 如果没有数据，添加一个占位数据
    if len(df) == 0:
        df = pd.DataFrame([dict(Task='Machine 0', Start=0, Finish=1, Resource='No Task', Description='No Data')])

    # 尝试用自定义方式创建甘特图
    try:
        # 创建一个空的图形
        fig = go.Figure()

        # 设置颜色
        colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)',
                  'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)']

        # 按机器分组添加条形图
        for i, machine in enumerate(df['Task'].unique()):
            df_machine = df[df['Task'] == machine]

            for j, row in df_machine.iterrows():
                fig.add_trace(go.Bar(
                    x=[row['Finish'] - row['Start']],
                    y=[row['Task']],
                    orientation='h',
                    base=row['Start'],
                    marker_color=colors[i % len(colors)],
                    name=row['Description'],
                    hoverinfo='text',
                    text=row['Description'],
                    showlegend=False
                ))

        # 更新布局
        fig.update_layout(
            title=f'Job Shop Scheduling - Total Time: {env.total_time}, Tool Changes: {env.tool_changes}',
            xaxis_title='Time',
            yaxis_title='Machines',
            barmode='stack',
            height=400,
            xaxis={'type': 'linear'},
            yaxis={'categoryorder': 'category descending'},
            margin=dict(l=10, r=10, t=50, b=10)
        )

        # 保存为HTML文件
        pio.write_html(fig, filename)
        print(f"交互式甘特图已保存至: {filename}")

    except Exception as e:
        print(f"创建交互式甘特图时出错: {e}")
        print("将使用静态图像代替...")
        # 如果交互式甘特图创建失败，保存一个静态版本作为备份
        static_filename = filename.replace('.html', '_static.png')
        save_gantt_chart(env, static_filename)
        print(f"静态备份甘特图已保存至: {static_filename}")


def generate_all_charts(env, output_dir='charts'):
    """生成多种类型的甘特图并保存"""
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 生成静态甘特图 - 按任务着色
    save_gantt_chart(
        env,
        os.path.join(output_dir, f'gantt_by_task_{timestamp}.png'),
        color_by='task'
    )

    # 生成静态甘特图 - 按工序类型着色
    save_gantt_chart(
        env,
        os.path.join(output_dir, f'gantt_by_type_{timestamp}.png'),
        color_by='type'
    )

    # 生成静态甘特图 - 按轮次着色
    save_gantt_chart(
        env,
        os.path.join(output_dir, f'gantt_by_round_{timestamp}.png'),
        color_by='round',
        show_rounds=True
    )

    # 尝试生成交互式甘特图，有错误处理机制
    try:
        create_interactive_gantt(
            env,
            os.path.join(output_dir, f'interactive_gantt_{timestamp}.html')
        )
    except Exception as e:
        print(f"生成交互式甘特图时出错: {e}")
        print("将只使用静态甘特图。")

    print(f"所有甘特图已生成并保存在 {output_dir} 目录中")