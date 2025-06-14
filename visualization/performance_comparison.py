import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.patches as mpatches

def load_and_preprocess_data(csv_path):
    """
    加载CSV数据并进行预处理
    - 选取每个数据集五次测试中的最佳结果
    - 按任务数量分组
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 为每次测试计算一个综合得分（完工时间和工具更换次数的加权和）
    # 这里我们给完工时间更高的权重
    df['GA_score'] = df['GA_makespan'] + 0.3 * df['GA_tool_changes']
    df['OOA_score'] = df['OOA_makespan'] + 0.3 * df['OOA_tool_changes']
    df['RL_score'] = df['RL_makespan'] + 0.3 * df['RL_tool_changes']
    
    # 对每个数据集选择最佳结果
    best_results = []
    for dataset, group in df.groupby('dataset'):
        # 为每个算法找到最佳得分的行
        ga_best = group.loc[group['GA_score'].idxmin()]
        ooa_best = group.loc[group['OOA_score'].idxmin()]
        rl_best = group.loc[group['RL_score'].idxmin()]
        
        # 创建一个新行，包含每个算法的最佳结果
        best_row = {
            'dataset': dataset,
            'num_jobs': ga_best['num_jobs'],  # 所有行的num_jobs应该相同
            'GA_makespan': ga_best['GA_makespan'],
            'GA_tool_changes': ga_best['GA_tool_changes'],
            'GA_runtime': ga_best['GA_runtime'],
            'OOA_makespan': ooa_best['OOA_makespan'],
            'OOA_tool_changes': ooa_best['OOA_tool_changes'],
            'OOA_runtime': ooa_best['OOA_runtime'],
            'RL_makespan': rl_best['RL_makespan'],
            'RL_tool_changes': rl_best['RL_tool_changes'],
            'RL_runtime': rl_best['RL_runtime']
        }
        best_results.append(best_row)
    
    # 创建新的DataFrame
    best_df = pd.DataFrame(best_results)
    
    # 按任务数量分组
    best_df['group'] = pd.cut(
        best_df['num_jobs'], 
        bins=[0, 8, 12, float('inf')], 
        labels=['small', 'mid', 'big']
    )
    
    return best_df

# 修改雷达图中的标签和标题，解决文字重叠问题
def radar_chart_by_group(df, output_path=None):
    """
    为每个任务规模组绘制雷达图，比较三种算法的性能
    维度包括：完工时间（取倒数）、工具更换次数（取倒数）、运行时间（取倒数）、任务完成率
    """
    # 定义雷达图的辅助函数
    def _radar_factory(num_vars, frame='circle'):
        # 计算每个变量的角度
        theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
        
        # 旋转使第一个轴位于顶部
        theta += np.pi/2
        
        def draw_poly_patch(self):
            verts = unit_poly_verts(theta)
            return plt.Polygon(verts, closed=True, edgecolor='k')
        
        def draw_poly_frame(self):
            verts = unit_poly_verts(theta)
            return plt.Polygon(verts, closed=True, edgecolor='k', fill=False)
        
        def unit_poly_verts(theta):
            x0, y0, r = [0.5] * 3
            verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
            return verts
        
        patch_class = type('RadarPatch', (mpatches.Patch,), 
                          {'_path': Path.unit_regular_polygon(num_vars),
                           'draw': draw_poly_patch})
        
        frame_class = type('RadarFrame', (mpatches.Patch,),
                          {'_path': Path.unit_regular_polygon(num_vars),
                           'draw': draw_poly_frame})
        
        return theta, frame_class, patch_class
    
    # 为每个组绘制雷达图
    for group_name, group_df in df.groupby('group'):
        # 计算每个算法在每个维度上的平均值
        ga_makespan = group_df['GA_makespan'].mean()
        ga_tool_changes = group_df['GA_tool_changes'].mean()
        ga_runtime = group_df['GA_runtime'].mean()
        
        ooa_makespan = group_df['OOA_makespan'].mean()
        ooa_tool_changes = group_df['OOA_tool_changes'].mean()
        ooa_runtime = group_df['OOA_runtime'].mean()
        
        rl_makespan = group_df['RL_makespan'].mean()
        rl_tool_changes = group_df['RL_tool_changes'].mean()
        rl_runtime = group_df['RL_runtime'].mean()
        
        # 计算倒数并归一化
        # 对于完工时间和工具更换次数，我们取倒数并归一化到[0,1]
        makespan_values = [1/ga_makespan, 1/ooa_makespan, 1/rl_makespan]
        makespan_max = max(makespan_values)
        makespan_norm = [v/makespan_max for v in makespan_values]
        
        tool_changes_values = [1/ga_tool_changes, 1/ooa_tool_changes, 1/rl_tool_changes]
        tool_changes_max = max(tool_changes_values)
        tool_changes_norm = [v/tool_changes_max for v in tool_changes_values]
        
        runtime_values = [1/ga_runtime, 1/ooa_runtime, 1/rl_runtime]
        runtime_max = max(runtime_values)
        runtime_norm = [v/runtime_max for v in runtime_values]
        
        # 任务完成率都是100%，所以设为1
        completion_rate = [1, 1, 1]
        
        # 准备雷达图数据 - 简化标签文字
        data = [
            ['Makespan', 'Tool\nChanges', 'Runtime', 'Completion\nRate'],
            ('GA', [
                makespan_norm[0],
                tool_changes_norm[0],
                runtime_norm[0],
                completion_rate[0]
            ]),
            ('OOA', [
                makespan_norm[1],
                tool_changes_norm[1],
                runtime_norm[1],
                completion_rate[1]
            ]),
            ('RL', [
                makespan_norm[2],
                tool_changes_norm[2],
                runtime_norm[2],
                completion_rate[2]
            ])
        ]
        
        N = len(data[0])
        theta, frame_class, patch_class = _radar_factory(N)
        
        # 绘制雷达图 - 增加图表大小
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # 绘制背景网格
        for i in range(5):
            r = 0.2 * (i + 1)
            ax.plot(theta, [r] * N, color='gray', linestyle=':')
        
        # 绘制轴线
        for i in range(N):
            ax.plot([theta[i], theta[i]], [0, 1], color='gray', linestyle='-')
        
        # 绘制各算法的雷达图
        colors = ['b', 'g', 'orange']
        for i, (name, values) in enumerate(data[1:]):
            angle = np.linspace(0, 2*np.pi, len(values), endpoint=False)
            angle += np.pi/2  # 旋转使第一个轴位于顶部
            values = np.concatenate((values, [values[0]]))  # 闭合多边形
            angle = np.concatenate((angle, [angle[0]]))  # 闭合多边形
            ax.plot(angle, values, color=colors[i], linewidth=2, label=name)
            ax.fill(angle, values, color=colors[i], alpha=0.25)
        
        # 设置标签 - 调整标签位置
        ax.set_xticks(theta)
        ax.set_xticklabels(data[0], fontsize=12)
        
        # 添加说明文本
        ax.text(0, -0.1, "(All values are inverse, higher is better)", 
                transform=ax.transAxes, ha='center', fontsize=10)
        
        # 添加标题和图例 - 使用完整算法名称在图例中
        ax.set_title(f'Algorithm Performance Comparison - {group_name} Group (Jobs: {group_df["num_jobs"].min()}-{group_df["num_jobs"].max()})', 
                    fontsize=15)
        ax.legend(['Genetic Algorithm (GA)', 'Osprey Algorithm (OOA)', 'Reinforcement Learning (RL)'], 
                 loc='lower right', fontsize=10)
        
        # 增加边距
        plt.tight_layout(pad=2.0)
        
        # 设置y轴范围
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        
        # 添加标题和图例
        ax.set_title(f'Algorithm Performance Comparison - {group_name} Group (Jobs: {group_df["num_jobs"].min()}-{group_df["num_jobs"].max()})', 
                    fontsize=15)
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        # 保存图像
        if output_path:
            plt.savefig(f"{output_path}/radar_chart_{group_name}.png", dpi=300, bbox_inches='tight')
        
        # 移除 plt.show() 调用
        plt.close(fig)  # 关闭图形以释放内存

# 在性能比值图函数中移除 plt.show()
def performance_ratio_by_group(df, output_path=None):
    """
    为每个任务规模组绘制性能比值图
    以GA为基准(1.0)，计算OOA和RL相对于GA的性能比值
    """
    # 计算每个数据集的性能比值
    df['OOA_makespan_ratio'] = df['OOA_makespan'] / df['GA_makespan']
    df['RL_makespan_ratio'] = df['RL_makespan'] / df['GA_makespan']
    df['OOA_tool_changes_ratio'] = df['OOA_tool_changes'] / df['GA_tool_changes']
    df['RL_tool_changes_ratio'] = df['RL_tool_changes'] / df['GA_tool_changes']
    df['OOA_runtime_ratio'] = df['OOA_runtime'] / df['GA_runtime']
    df['RL_runtime_ratio'] = df['RL_runtime'] / df['GA_runtime']
    
    # 为每个组绘制性能比值图
    for group_name, group_df in df.groupby('group'):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 完工时间比值
        makespan_data = {
            'OOA/GA': group_df['OOA_makespan_ratio'],
            'RL/GA': group_df['RL_makespan_ratio']
        }
        makespan_df = pd.DataFrame(makespan_data)
        
        sns.boxplot(data=makespan_df, ax=axes[0])
        axes[0].axhline(y=1, color='r', linestyle='--')
        axes[0].set_title(f'Makespan Ratio - {group_name} Group')
        axes[0].set_ylabel('Ratio (Relative to GA)')
        
        # 2. 工具更换次数比值 - 添加数据绘制
        tool_changes_data = {
            'OOA/GA': group_df['OOA_tool_changes_ratio'],
            'RL/GA': group_df['RL_tool_changes_ratio']
        }
        tool_changes_df = pd.DataFrame(tool_changes_data)
        
        sns.boxplot(data=tool_changes_df, ax=axes[1])
        axes[1].axhline(y=1, color='r', linestyle='--')
        axes[1].set_title(f'Tool Changes Ratio - {group_name} Group')
        axes[1].set_ylabel('Ratio (Relative to GA)')
        
        # 3. 运行时间比值 - 添加数据绘制
        runtime_data = {
            'OOA/GA': group_df['OOA_runtime_ratio'],
            'RL/GA': group_df['RL_runtime_ratio']
        }
        runtime_df = pd.DataFrame(runtime_data)
        
        sns.boxplot(data=runtime_df, ax=axes[2])
        axes[2].axhline(y=1, color='r', linestyle='--')
        axes[2].set_title(f'Runtime Ratio - {group_name} Group')
        axes[2].set_ylabel('Ratio (Relative to GA)')
        
        plt.tight_layout()
        
        # 保存图像
        if output_path:
            plt.savefig(f"{output_path}/performance_ratio_{group_name}.png", dpi=300, bbox_inches='tight')
        
        # 移除 plt.show() 调用
        plt.close(fig)  # 关闭图形以释放内存

# 在胜率图函数中移除 plt.show()
# 修改胜率图函数，确保饼图保持圆形
def win_rate_by_group(df, output_path=None):
    """
    为每个任务规模组统计三种算法之间的两两胜率
    """
    # 为每个组计算胜率
    for group_name, group_df in df.groupby('group'):
        # 初始化胜率计数器
        win_counts = {
            'GA_vs_OOA': {'GA': 0, 'OOA': 0, 'Tie': 0},
            'GA_vs_RL': {'GA': 0, 'RL': 0, 'Tie': 0},
            'OOA_vs_RL': {'OOA': 0, 'RL': 0, 'Tie': 0}
        }
        
        # 计算每个数据集上的胜负
        for _, row in group_df.iterrows():
            # GA vs OOA
            if row['GA_makespan'] < row['OOA_makespan']:
                win_counts['GA_vs_OOA']['GA'] += 1
            elif row['GA_makespan'] > row['OOA_makespan']:
                win_counts['GA_vs_OOA']['OOA'] += 1
            else:
                # 如果完工时间相同，比较工具更换次数
                if row['GA_tool_changes'] < row['OOA_tool_changes']:
                    win_counts['GA_vs_OOA']['GA'] += 1
                elif row['GA_tool_changes'] > row['OOA_tool_changes']:
                    win_counts['GA_vs_OOA']['OOA'] += 1
                else:
                    win_counts['GA_vs_OOA']['Tie'] += 1
            
            # GA vs RL
            if row['GA_makespan'] < row['RL_makespan']:
                win_counts['GA_vs_RL']['GA'] += 1
            elif row['GA_makespan'] > row['RL_makespan']:
                win_counts['GA_vs_RL']['RL'] += 1
            else:
                # 如果完工时间相同，比较工具更换次数
                if row['GA_tool_changes'] < row['RL_tool_changes']:
                    win_counts['GA_vs_RL']['GA'] += 1
                elif row['GA_tool_changes'] > row['RL_tool_changes']:
                    win_counts['GA_vs_RL']['RL'] += 1
                else:
                    win_counts['GA_vs_RL']['Tie'] += 1
            
            # OOA vs RL
            if row['OOA_makespan'] < row['RL_makespan']:
                win_counts['OOA_vs_RL']['OOA'] += 1
            elif row['OOA_makespan'] > row['RL_makespan']:
                win_counts['OOA_vs_RL']['RL'] += 1
            else:
                # 如果完工时间相同，比较工具更换次数
                if row['OOA_tool_changes'] < row['RL_tool_changes']:
                    win_counts['OOA_vs_RL']['OOA'] += 1
                elif row['OOA_tool_changes'] > row['RL_tool_changes']:
                    win_counts['OOA_vs_RL']['RL'] += 1
                else:
                    win_counts['OOA_vs_RL']['Tie'] += 1
        
        # 计算胜率
        total = len(group_df)
        win_rates = {
            'GA_vs_OOA': {
                'GA': win_counts['GA_vs_OOA']['GA'] / total * 100,
                'OOA': win_counts['GA_vs_OOA']['OOA'] / total * 100,
                'Tie': win_counts['GA_vs_OOA']['Tie'] / total * 100
            },
            'GA_vs_RL': {
                'GA': win_counts['GA_vs_RL']['GA'] / total * 100,
                'RL': win_counts['GA_vs_RL']['RL'] / total * 100,
                'Tie': win_counts['GA_vs_RL']['Tie'] / total * 100
            },
            'OOA_vs_RL': {
                'OOA': win_counts['OOA_vs_RL']['OOA'] / total * 100,
                'RL': win_counts['OOA_vs_RL']['RL'] / total * 100,
                'Tie': win_counts['OOA_vs_RL']['Tie'] / total * 100
            }
        }
        
        # 绘制胜率图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # GA vs OOA
        sizes = [win_rates['GA_vs_OOA']['GA'], win_rates['GA_vs_OOA']['OOA'], win_rates['GA_vs_OOA']['Tie']]
        colors = ['blue', 'green', 'gray']
        # 完全不显示任何标签和百分比
        axes[0].pie(sizes, colors=colors, startangle=90)
        axes[0].set_title(f'GA vs OOA - {group_name} Group')
        # 设置aspect为equal确保饼图是圆形的
        axes[0].set_aspect('equal')
        
        # GA vs RL
        sizes = [win_rates['GA_vs_RL']['GA'], win_rates['GA_vs_RL']['RL'], win_rates['GA_vs_RL']['Tie']]
        colors = ['blue', 'orange', 'gray']
        axes[1].pie(sizes, colors=colors, startangle=90)
        axes[1].set_title(f'GA vs RL - {group_name} Group')
        # 设置aspect为equal确保饼图是圆形的
        axes[1].set_aspect('equal')
        
        # OOA vs RL
        sizes = [win_rates['OOA_vs_RL']['OOA'], win_rates['OOA_vs_RL']['RL'], win_rates['OOA_vs_RL']['Tie']]
        colors = ['green', 'orange', 'gray']
        axes[2].pie(sizes, colors=colors, startangle=90)
        axes[2].set_title(f'OOA vs RL - {group_name} Group')
        # 设置aspect为equal确保饼图是圆形的
        axes[2].set_aspect('equal')
        
        # 为每个子图添加单独的图例
        for i, ax in enumerate(axes):
            if i == 0:
                comparison = 'GA_vs_OOA'
                legend_labels = [
                    f"GA: {win_rates[comparison]['GA']:.1f}%", 
                    f"OOA: {win_rates[comparison]['OOA']:.1f}%", 
                    f"Tie: {win_rates[comparison]['Tie']:.1f}%"
                ]
                legend_colors = ['blue', 'green', 'gray']
            elif i == 1:
                comparison = 'GA_vs_RL'
                legend_labels = [
                    f"GA: {win_rates[comparison]['GA']:.1f}%", 
                    f"RL: {win_rates[comparison]['RL']:.1f}%", 
                    f"Tie: {win_rates[comparison]['Tie']:.1f}%"
                ]
                legend_colors = ['blue', 'orange', 'gray']
            else:
                comparison = 'OOA_vs_RL'
                legend_labels = [
                    f"OOA: {win_rates[comparison]['OOA']:.1f}%", 
                    f"RL: {win_rates[comparison]['RL']:.1f}%", 
                    f"Tie: {win_rates[comparison]['Tie']:.1f}%"
                ]
                legend_colors = ['green', 'orange', 'gray']
            
            # 创建自定义图例，放在饼图下方
            patches = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in legend_colors]
            ax.legend(patches, legend_labels, loc='upper center', 
                     fontsize=12, bbox_to_anchor=(0.5, -0.05), 
                     ncol=1, frameon=True)
        
        plt.tight_layout(pad=3.0, rect=[0, 0.1, 1, 0.95])  # 增加底部空间给图例
        
        # 保存图像
        if output_path:
            plt.savefig(f"{output_path}/win_rate_{group_name}.png", dpi=300, bbox_inches='tight')
        
        # 关闭图形以释放内存
        plt.close(fig)
        
        # 打印详细的胜率信息
        print(f"=== {group_name} Group Win Rate Statistics ===")
        print(f"Total datasets: {total}")
        print("\nGA vs OOA:")
        print(f"GA Wins: {win_counts['GA_vs_OOA']['GA']} ({win_rates['GA_vs_OOA']['GA']:.1f}%)")
        print(f"OOA Wins: {win_counts['GA_vs_OOA']['OOA']} ({win_rates['GA_vs_OOA']['OOA']:.1f}%)")
        print(f"Ties: {win_counts['GA_vs_OOA']['Tie']} ({win_rates['GA_vs_OOA']['Tie']:.1f}%)")
        
        print("\nGA vs RL:")
        print(f"GA Wins: {win_counts['GA_vs_RL']['GA']} ({win_rates['GA_vs_RL']['GA']:.1f}%)")
        print(f"RL Wins: {win_counts['GA_vs_RL']['RL']} ({win_rates['GA_vs_RL']['RL']:.1f}%)")
        print(f"Ties: {win_counts['GA_vs_RL']['Tie']} ({win_rates['GA_vs_RL']['Tie']:.1f}%)")
        
        print("\nOOA vs RL:")
        print(f"OOA Wins: {win_counts['OOA_vs_RL']['OOA']} ({win_rates['OOA_vs_RL']['OOA']:.1f}%)")
        print(f"RL Wins: {win_counts['OOA_vs_RL']['RL']} ({win_rates['OOA_vs_RL']['RL']:.1f}%)")
        print(f"Ties: {win_counts['OOA_vs_RL']['Tie']} ({win_rates['OOA_vs_RL']['Tie']:.1f}%)")
        print("\n")

def main():
    # 数据文件路径
    csv_path = "g:\\GA_OOA_RL\\benchmark_results\\run_20250417_121555\\all_results.csv"
    
    # 输出目录
    output_path = "g:\\GA_OOA_RL\\benchmark_results\\run_20250417_121555\\visualizations"
    
    # 创建输出目录
    import os
    os.makedirs(output_path, exist_ok=True)
    
    # 加载和预处理数据
    df = load_and_preprocess_data(csv_path)
    
    # 绘制雷达图
    radar_chart_by_group(df, output_path)
    
    # 绘制性能比值图
    performance_ratio_by_group(df, output_path)
    
    # 统计胜率
    win_rate_by_group(df, output_path)

if __name__ == "__main__":
    main()