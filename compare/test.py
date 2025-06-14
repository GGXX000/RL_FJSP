#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import torch
from tqdm import tqdm

# Import modules for three algorithms
from envs import JobShopEnv
from config import Config
from GA import GeneticAlgorithm
from NOOA import OspreyOptimizationAlgorithm

# Try to import reinforcement learning model modules
try:
    from utils import state_to_graph, get_valid_actions, convert_action_index_to_action
    from GNN_DQN_model import GNN_DQNAgent

    RL_AVAILABLE = True
except ImportError:
    print("Reinforcement learning module import failed, will only compare GA and OOA algorithms")
    RL_AVAILABLE = False


def benchmark_algorithms(data_dir="datasets",
                         output_dir="benchmark_results",
                         num_machines=3,
                         num_runs=3,
                         save_charts=False,
                         ga_population=100,
                         ga_generations=100,
                         ooa_population=100,
                         ooa_iterations=100,
                         makespan_weight=0.7,
                         tool_change_weight=0.3,
                         rl_model_path="models/job_shop_gnn_dqn_final.pth"):
    """
    Run performance benchmark tests for three algorithms on given datasets

    Parameters:
        data_dir: Dataset directory
        output_dir: Result output directory
        num_machines: Number of machines
        num_runs: Number of runs per dataset
        save_charts: Whether to save Gantt charts
        ga_population: Genetic algorithm population size
        ga_generations: Genetic algorithm generations
        ooa_population: Osprey algorithm population size
        ooa_iterations: Osprey algorithm iterations
        makespan_weight: Makespan weight
        tool_change_weight: Tool change weight
        rl_model_path: Reinforcement learning model path
    """
    # Declare RL_AVAILABLE as global variable
    global RL_AVAILABLE

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all data files
    data_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.txt'):
                data_files.append(os.path.join(root, file))

    if not data_files:
        print(f"No data files found in directory {data_dir}")
        return

    print(f"Found {len(data_files)} data files")

    # Prepare result storage
    all_results = []

    # If reinforcement learning is available, load model
    rl_agent = None
    if RL_AVAILABLE:
        # Try to load first dataset to initialize environment
        try:
            # Initialize environment to get graph node information
            sample_env = JobShopEnv(data_files[0], num_machines)
            sample_state = sample_env.reset()
            graph_state = state_to_graph(sample_state, sample_env)

            # Get dimension information
            node_features_dim = graph_state['node_features'].shape[1]
            edge_features_dim = 1  # Simple binary adjacency matrix
            global_state_dim = len(graph_state['global_state'])

            # Calculate action space size
            action_size = Config.get_action_size(num_machines)

            # Load GNN-DQN model
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            rl_agent = GNN_DQNAgent(
                node_features=node_features_dim,
                edge_features=edge_features_dim,
                global_features=global_state_dim,
                action_size=action_size,
                device=device
            )

            # Load pre-trained model
            if os.path.exists(rl_model_path):
                print(f"Loading RL model: {rl_model_path}")
                rl_agent.load(rl_model_path)
            else:
                # Try to load latest model
                alt_model_path = "models/job_shop_gnn_dqn_latest.pth"
                if os.path.exists(alt_model_path):
                    print(f"Loading alternate RL model: {alt_model_path}")
                    rl_agent.load(alt_model_path)
                else:
                    print("No pre-trained RL model found, will skip RL testing")
                    RL_AVAILABLE = False
        except Exception as e:
            print(f"Error initializing RL model: {str(e)}")
            RL_AVAILABLE = False

    # Create runtime timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create results directory for this run
    run_output_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    # Test each data file
    for data_idx, data_file in enumerate(data_files):
        print(f"\nEvaluating dataset {data_idx + 1}/{len(data_files)}: {os.path.basename(data_file)}")

        try:
            # Create environment
            env = JobShopEnv(data_file, num_machines)
            print(f"Dataset characteristics: Jobs={env.num_jobs}, Operation types={env.num_operation_types}")

            # Create directory for current dataset
            dataset_dir = os.path.join(run_output_dir, f"dataset_{data_idx + 1}_{os.path.basename(data_file)}")
            os.makedirs(dataset_dir, exist_ok=True)

            dataset_results = []

            # Multiple runs to get statistics
            for run_idx in range(num_runs):
                print(f"\nRun {run_idx + 1}/{num_runs}")
                run_results = {
                    'dataset': os.path.basename(data_file),
                    'dataset_path': data_file,
                    'run': run_idx + 1,
                    'num_jobs': env.num_jobs,
                    'num_op_types': env.num_operation_types,
                    'num_machines': num_machines,
                    'timestamp': timestamp
                }

                # 1. Run Genetic Algorithm
                print("\nRunning Genetic Algorithm...")
                ga_start_time = time.time()
                try:
                    ga = GeneticAlgorithm(
                        env=JobShopEnv(data_file, num_machines),
                        population_size=ga_population,
                        generations=ga_generations,
                        makespan_weight=makespan_weight,
                        tool_change_weight=tool_change_weight
                    )

                    _, best_makespan, best_tool_changes = ga.run(verbose=False)
                    ga_time = time.time() - ga_start_time

                    # Apply best solution to environment
                    makespan, tool_changes, env_ga = ga.apply_best_solution()

                    # Record GA results
                    run_results.update({
                        'GA_makespan': makespan,
                        'GA_tool_changes': tool_changes,
                        'GA_runtime': ga_time,
                        'GA_completed_jobs': env_ga.completed_jobs,
                        'GA_success': True
                    })

                    # Save Gantt chart
                    if save_charts:
                        chart_dir = os.path.join(dataset_dir, f"run_{run_idx + 1}")
                        os.makedirs(chart_dir, exist_ok=True)
                        try:
                            from gantt_chart_utils import save_gantt_chart
                            save_gantt_chart(
                                env_ga,
                                os.path.join(chart_dir, 'GA_gantt.png'),
                                color_by='task'
                            )
                        except ImportError:
                            print("Gantt chart tools unavailable, skipping chart generation")
                except Exception as e:
                    print(f"GA execution error: {str(e)}")
                    run_results.update({
                        'GA_success': False,
                        'GA_error': str(e)
                    })

                # 2. Run Osprey Optimization Algorithm
                print("\nRunning Osprey Optimization Algorithm...")
                ooa_start_time = time.time()
                try:
                    ooa = OspreyOptimizationAlgorithm(
                        env=JobShopEnv(data_file, num_machines),
                        population_size=ooa_population,
                        max_iterations=ooa_iterations,
                        makespan_weight=makespan_weight,
                        tool_change_weight=tool_change_weight
                    )

                    _, best_makespan, best_tool_changes = ooa.run(verbose=False)
                    ooa_time = time.time() - ooa_start_time

                    # Apply best solution to environment
                    makespan, tool_changes, env_ooa = ooa.apply_best_solution()

                    # Record OOA results
                    run_results.update({
                        'OOA_makespan': makespan,
                        'OOA_tool_changes': tool_changes,
                        'OOA_runtime': ooa_time,
                        'OOA_completed_jobs': env_ooa.completed_jobs,
                        'OOA_success': True
                    })

                    # Save Gantt chart
                    if save_charts:
                        chart_dir = os.path.join(dataset_dir, f"run_{run_idx + 1}")
                        os.makedirs(chart_dir, exist_ok=True)
                        try:
                            from gantt_chart_utils import save_gantt_chart
                            save_gantt_chart(
                                env_ooa,
                                os.path.join(chart_dir, 'OOA_gantt.png'),
                                color_by='task'
                            )
                        except ImportError:
                            pass
                except Exception as e:
                    print(f"OOA execution error: {str(e)}")
                    run_results.update({
                        'OOA_success': False,
                        'OOA_error': str(e)
                    })

                # 3. Run Reinforcement Learning Algorithm (if available)
                if RL_AVAILABLE and rl_agent:
                    torch.cuda.empty_cache()
                    #print("\Clear Cuda Cache...")
                    # 获取更详细的内存使用情况
                    #print(torch.cuda.memory_summary())
                    print("\nRunning Reinforcement Learning Algorithm...")
                    rl_start_time = time.time()
                    try:
                        rl_env = JobShopEnv(data_file, num_machines)

                        # Execute RL policy
                        state = rl_env.reset()
                        graph_state = state_to_graph(state, rl_env)

                        step_count = 0
                        max_steps = rl_env.num_jobs * rl_env.num_operation_types * 2  # Safety limit

                        while step_count < max_steps:
                            valid_actions = get_valid_actions(state, rl_env)
                            if not valid_actions:
                                break

                            # Fix epsilon=0 for testing (no exploration)
                            epsilon_backup = rl_agent.epsilon
                            rl_agent.epsilon = 0
                            action_idx = rl_agent.act(graph_state)
                            rl_agent.epsilon = epsilon_backup

                            action = convert_action_index_to_action(action_idx, valid_actions)
                            next_state, _, done, _ = rl_env.step(action)
                            next_graph_state = state_to_graph(next_state, rl_env)

                            state = next_state
                            graph_state = next_graph_state
                            step_count += 1

                            if done:
                                break

                        rl_time = time.time() - rl_start_time
                        # 获取更详细的内存使用情况
                        #print(torch.cuda.memory_summary())
                        # Record RL results
                        run_results.update({
                            'RL_makespan': rl_env.total_time,
                            'RL_tool_changes': rl_env.tool_changes,
                            'RL_runtime': rl_time,
                            'RL_completed_jobs': rl_env.completed_jobs,
                            'RL_steps': step_count,
                            'RL_success': True
                        })

                        # Save Gantt chart
                        if save_charts:
                            chart_dir = os.path.join(dataset_dir, f"run_{run_idx + 1}")
                            os.makedirs(chart_dir, exist_ok=True)
                            try:
                                from gantt_chart_utils import save_gantt_chart
                                save_gantt_chart(
                                    rl_env,
                                    os.path.join(chart_dir, 'RL_gantt.png'),
                                    color_by='task'
                                )
                            except ImportError:
                                pass
                    except Exception as e:
                        print(f"RL execution error: {str(e)}")
                        run_results.update({
                            'RL_success': False,
                            'RL_error': str(e)
                        })

                # Add run results
                dataset_results.append(run_results)
                all_results.append(run_results)

                # Print current run results summary
                print("\nCurrent run results summary:")
                if run_results.get('GA_success', False):
                    print(
                        f"GA - Makespan: {run_results['GA_makespan']}, Tool Changes: {run_results['GA_tool_changes']}, Runtime: {run_results['GA_runtime']:.2f}s")
                else:
                    print("GA - Execution failed")

                if run_results.get('OOA_success', False):
                    print(
                        f"OOA - Makespan: {run_results['OOA_makespan']}, Tool Changes: {run_results['OOA_tool_changes']}, Runtime: {run_results['OOA_runtime']:.2f}s")
                else:
                    print("OOA - Execution failed")

                if RL_AVAILABLE and run_results.get('RL_success', False):
                    print(
                        f"RL - Makespan: {run_results['RL_makespan']}, Tool Changes: {run_results['RL_tool_changes']}, Runtime: {run_results['RL_runtime']:.2f}s")
                elif RL_AVAILABLE:
                    print("RL - Execution failed")

            # Calculate and print average results for dataset
            if dataset_results:
                print("\nDataset average results:")

                # GA average results
                successful_ga_runs = [r for r in dataset_results if r.get('GA_success', False)]
                if successful_ga_runs:
                    ga_makespan_avg = sum(r['GA_makespan'] for r in successful_ga_runs) / len(successful_ga_runs)
                    ga_tool_changes_avg = sum(r['GA_tool_changes'] for r in successful_ga_runs) / len(
                        successful_ga_runs)
                    ga_runtime_avg = sum(r['GA_runtime'] for r in successful_ga_runs) / len(successful_ga_runs)
                    print(
                        f"GA Average - Makespan: {ga_makespan_avg:.2f}, Tool Changes: {ga_tool_changes_avg:.2f}, Runtime: {ga_runtime_avg:.2f}s")
                else:
                    print("GA - All runs failed")

                # OOA average results
                successful_ooa_runs = [r for r in dataset_results if r.get('OOA_success', False)]
                if successful_ooa_runs:
                    ooa_makespan_avg = sum(r['OOA_makespan'] for r in successful_ooa_runs) / len(successful_ooa_runs)
                    ooa_tool_changes_avg = sum(r['OOA_tool_changes'] for r in successful_ooa_runs) / len(
                        successful_ooa_runs)
                    ooa_runtime_avg = sum(r['OOA_runtime'] for r in successful_ooa_runs) / len(successful_ooa_runs)
                    print(
                        f"OOA Average - Makespan: {ooa_makespan_avg:.2f}, Tool Changes: {ooa_tool_changes_avg:.2f}, Runtime: {ooa_runtime_avg:.2f}s")
                else:
                    print("OOA - All runs failed")

                # RL average results
                if RL_AVAILABLE:
                    successful_rl_runs = [r for r in dataset_results if r.get('RL_success', False)]
                    if successful_rl_runs:
                        rl_makespan_avg = sum(r['RL_makespan'] for r in successful_rl_runs) / len(successful_rl_runs)
                        rl_tool_changes_avg = sum(r['RL_tool_changes'] for r in successful_rl_runs) / len(
                            successful_rl_runs)
                        rl_runtime_avg = sum(r['RL_runtime'] for r in successful_rl_runs) / len(successful_rl_runs)
                        print(
                            f"RL Average - Makespan: {rl_makespan_avg:.2f}, Tool Changes: {rl_tool_changes_avg:.2f}, Runtime: {rl_runtime_avg:.2f}s")
                    else:
                        print("RL - All runs failed")

                # Save dataset results
                df_dataset = pd.DataFrame(dataset_results)
                df_dataset.to_csv(os.path.join(dataset_dir, 'results.csv'), index=False)

                # Create dataset comparison charts
                try:
                    create_dataset_charts(df_dataset, dataset_dir)
                except Exception as e:
                    print(f"Failed to create dataset charts: {str(e)}")

        except Exception as e:
            print(f"Error processing dataset {data_file}: {str(e)}")

    # Save all results to CSV
    if all_results:
        df_all = pd.DataFrame(all_results)
        csv_path = os.path.join(run_output_dir, 'all_results.csv')
        df_all.to_csv(csv_path, index=False)
        print(f"\nAll results saved to: {csv_path}")

        # Create overall comparison charts
        try:
            create_comparison_charts(df_all, run_output_dir)
        except Exception as e:
            print(f"Failed to create comparison charts: {str(e)}")

    return all_results


def create_dataset_charts(df, output_dir):
    """Create charts for a single dataset"""
    if df.empty:
        return

    # Set chart style
    sns.set(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial']  # English font
    plt.rcParams['axes.unicode_minus'] = False  # Properly display minus signs

    # Create charts directory
    charts_dir = os.path.join(output_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)

    # Compare makespan across three algorithms
    plt.figure(figsize=(10, 6))

    # Extract valid data
    data = []

    if 'GA_makespan' in df.columns and df['GA_success'].any():
        ga_data = df[df['GA_success']]['GA_makespan'].tolist()
        data.append(ga_data)
        labels = ['Genetic Algorithm (GA)']

    if 'OOA_makespan' in df.columns and df['OOA_success'].any():
        ooa_data = df[df['OOA_success']]['OOA_makespan'].tolist()
        data.append(ooa_data)
        labels.append('Osprey Optimization Algorithm (OOA)')

    if 'RL_makespan' in df.columns and df['RL_success'].any():
        rl_data = df[df['RL_success']]['RL_makespan'].tolist()
        data.append(rl_data)
        labels.append('Reinforcement Learning (RL)')

    if data:
        plt.boxplot(data, labels=labels)
        plt.title('Algorithm Makespan Comparison')
        plt.ylabel('Makespan')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(charts_dir, 'makespan_boxplot.png'), dpi=300)
        plt.close()

    # Compare tool changes across three algorithms
    plt.figure(figsize=(10, 6))

    data = []

    if 'GA_tool_changes' in df.columns and df['GA_success'].any():
        ga_data = df[df['GA_success']]['GA_tool_changes'].tolist()
        data.append(ga_data)
        labels = ['Genetic Algorithm (GA)']

    if 'OOA_tool_changes' in df.columns and df['OOA_success'].any():
        ooa_data = df[df['OOA_success']]['OOA_tool_changes'].tolist()
        data.append(ooa_data)
        labels.append('Osprey Optimization Algorithm (OOA)')

    if 'RL_tool_changes' in df.columns and df['RL_success'].any():
        rl_data = df[df['RL_success']]['RL_tool_changes'].tolist()
        data.append(rl_data)
        labels.append('Reinforcement Learning (RL)')

    if data:
        plt.boxplot(data, labels=labels)
        plt.title('Algorithm Tool Changes Comparison')
        plt.ylabel('Tool Changes')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(charts_dir, 'tool_changes_boxplot.png'), dpi=300)
        plt.close()

    # Compare runtime across three algorithms
    plt.figure(figsize=(10, 6))

    data = []

    if 'GA_runtime' in df.columns and df['GA_success'].any():
        ga_data = df[df['GA_success']]['GA_runtime'].tolist()
        data.append(ga_data)
        labels = ['Genetic Algorithm (GA)']

    if 'OOA_runtime' in df.columns and df['OOA_success'].any():
        ooa_data = df[df['OOA_success']]['OOA_runtime'].tolist()
        data.append(ooa_data)
        labels.append('Osprey Optimization Algorithm (OOA)')

    if 'RL_runtime' in df.columns and df['RL_success'].any():
        rl_data = df[df['RL_success']]['RL_runtime'].tolist()
        data.append(rl_data)
        labels.append('Reinforcement Learning (RL)')

    if data:
        plt.boxplot(data, labels=labels)
        plt.title('Algorithm Runtime Comparison')
        plt.ylabel('Runtime (seconds)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(charts_dir, 'runtime_boxplot.png'), dpi=300)
        plt.close()

    # Scatter plot: Makespan vs Tool Changes
    plt.figure(figsize=(10, 8))

    # GA scatter
    if 'GA_makespan' in df.columns and 'GA_tool_changes' in df.columns and df['GA_success'].any():
        plt.scatter(df[df['GA_success']]['GA_makespan'],
                    df[df['GA_success']]['GA_tool_changes'],
                    alpha=0.7, label='Genetic Algorithm (GA)', marker='o', s=80, color='#2196F3')

    # OOA scatter
    if 'OOA_makespan' in df.columns and 'OOA_tool_changes' in df.columns and df['OOA_success'].any():
        plt.scatter(df[df['OOA_success']]['OOA_makespan'],
                    df[df['OOA_success']]['OOA_tool_changes'],
                    alpha=0.7, label='Osprey Optimization Algorithm (OOA)', marker='s', s=80, color='#4CAF50')

    # RL scatter
    if 'RL_makespan' in df.columns and 'RL_tool_changes' in df.columns and df['RL_success'].any():
        plt.scatter(df[df['RL_success']]['RL_makespan'],
                    df[df['RL_success']]['RL_tool_changes'],
                    alpha=0.7, label='Reinforcement Learning (RL)', marker='^', s=80, color='#FF9800')

    plt.xlabel('Makespan')
    plt.ylabel('Tool Changes')
    plt.title('Makespan vs Tool Changes Trade-off Analysis')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(charts_dir, 'tradeoff_scatter.png'), dpi=300)
    plt.close()


def create_comparison_charts(df, output_dir):
    """Create comparison charts across all datasets"""
    if df.empty:
        return

    # Set chart style
    sns.set(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial']  # English font
    plt.rcParams['axes.unicode_minus'] = False  # Properly display minus signs

    # Create charts directory
    charts_dir = os.path.join(output_dir, 'comparison_charts')
    os.makedirs(charts_dir, exist_ok=True)

    # Prepare data - group by dataset
    datasets = df['dataset'].unique()

    # 1. Makespan comparison chart
    plt.figure(figsize=(14, 8))

    ga_makespan = []
    ooa_makespan = []
    rl_makespan = []

    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]

        # GA average makespan
        ga_success = dataset_df['GA_success'] if 'GA_success' in dataset_df.columns else False
        if 'GA_makespan' in dataset_df.columns and ga_success.any():
            ga_makespan.append(dataset_df[ga_success]['GA_makespan'].mean())
        else:
            ga_makespan.append(np.nan)

        # OOA average makespan
        ooa_success = dataset_df['OOA_success'] if 'OOA_success' in dataset_df.columns else False
        if 'OOA_makespan' in dataset_df.columns and ooa_success.any():
            ooa_makespan.append(dataset_df[ooa_success]['OOA_makespan'].mean())
        else:
            ooa_makespan.append(np.nan)

        # RL average makespan
        rl_success = dataset_df['RL_success'] if 'RL_success' in dataset_df.columns else False
        if 'RL_makespan' in dataset_df.columns and rl_success.any():
            rl_makespan.append(dataset_df[rl_success]['RL_makespan'].mean())
        else:
            rl_makespan.append(np.nan)

    # Create bar chart
    x = np.arange(len(datasets))
    width = 0.25  # Bar width

    # Filter out NaN values
    valid_ga = ~np.isnan(ga_makespan)
    valid_ooa = ~np.isnan(ooa_makespan)
    valid_rl = ~np.isnan(rl_makespan)

    if valid_ga.any():
        plt.bar(x[valid_ga] - width, np.array(ga_makespan)[valid_ga], width, label='Genetic Algorithm (GA)',
                color='#2196F3',
                alpha=0.8)
    if valid_ooa.any():
        plt.bar(x[valid_ooa], np.array(ooa_makespan)[valid_ooa], width, label='Osprey Optimization Algorithm (OOA)',
                color='#4CAF50', alpha=0.8)
    if valid_rl.any():
        plt.bar(x[valid_rl] + width, np.array(rl_makespan)[valid_rl], width, label='Reinforcement Learning (RL)',
                color='#FF9800',
                alpha=0.8)

    plt.xlabel('Dataset')
    plt.ylabel('Average Makespan')
    plt.title('Average Makespan Comparison of Algorithms Across Datasets')
    plt.xticks(x, [d[:15] + '...' if len(d) > 15 else d for d in datasets], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'makespan_comparison.png'), dpi=300)
    plt.close()

    # 2. Tool changes comparison chart
    plt.figure(figsize=(14, 8))

    ga_tools = []
    ooa_tools = []
    rl_tools = []

    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]

        # GA average tool changes
        ga_success = dataset_df['GA_success'] if 'GA_success' in dataset_df.columns else False
        if 'GA_tool_changes' in dataset_df.columns and ga_success.any():
            ga_tools.append(dataset_df[ga_success]['GA_tool_changes'].mean())
        else:
            ga_tools.append(np.nan)

        # OOA average tool changes
        ooa_success = dataset_df['OOA_success'] if 'OOA_success' in dataset_df.columns else False
        if 'OOA_tool_changes' in dataset_df.columns and ooa_success.any():
            ooa_tools.append(dataset_df[ooa_success]['OOA_tool_changes'].mean())
        else:
            ooa_tools.append(np.nan)

        # RL average tool changes
        rl_success = dataset_df['RL_success'] if 'RL_success' in dataset_df.columns else False
        if 'RL_tool_changes' in dataset_df.columns and rl_success.any():
            rl_tools.append(dataset_df[rl_success]['RL_tool_changes'].mean())
        else:
            rl_tools.append(np.nan)

    # Create bar chart
    valid_ga = ~np.isnan(ga_tools)
    valid_ooa = ~np.isnan(ooa_tools)
    valid_rl = ~np.isnan(rl_tools)

    if valid_ga.any():
        plt.bar(x[valid_ga] - width, np.array(ga_tools)[valid_ga], width, label='Genetic Algorithm (GA)',
                color='#2196F3', alpha=0.8)
    if valid_ooa.any():
        plt.bar(x[valid_ooa], np.array(ooa_tools)[valid_ooa], width, label='Osprey Optimization Algorithm (OOA)',
                color='#4CAF50', alpha=0.8)
    if valid_rl.any():
        plt.bar(x[valid_rl] + width, np.array(rl_tools)[valid_rl], width, label='Reinforcement Learning (RL)',
                color='#FF9800', alpha=0.8)

    plt.xlabel('Dataset')
    plt.ylabel('Average Tool Changes')
    plt.title('Average Tool Changes Comparison of Algorithms Across Datasets')
    plt.xticks(x, [d[:15] + '...' if len(d) > 15 else d for d in datasets], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'tool_changes_comparison.png'), dpi=300)
    plt.close()

    # 3. Runtime comparison chart
    plt.figure(figsize=(14, 8))

    ga_runtime = []
    ooa_runtime = []
    rl_runtime = []

    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]

        # GA average runtime
        ga_success = dataset_df['GA_success'] if 'GA_success' in dataset_df.columns else False
        if 'GA_runtime' in dataset_df.columns and ga_success.any():
            ga_runtime.append(dataset_df[ga_success]['GA_runtime'].mean())
        else:
            ga_runtime.append(np.nan)

        # OOA average runtime
        ooa_success = dataset_df['OOA_success'] if 'OOA_success' in dataset_df.columns else False
        if 'OOA_runtime' in dataset_df.columns and ooa_success.any():
            ooa_runtime.append(dataset_df[ooa_success]['OOA_runtime'].mean())
        else:
            ooa_runtime.append(np.nan)

        # RL average runtime
        rl_success = dataset_df['RL_success'] if 'RL_success' in dataset_df.columns else False
        if 'RL_runtime' in dataset_df.columns and rl_success.any():
            rl_runtime.append(dataset_df[rl_success]['RL_runtime'].mean())
        else:
            rl_runtime.append(np.nan)

    # Create bar chart
    valid_ga = ~np.isnan(ga_runtime)
    valid_ooa = ~np.isnan(ooa_runtime)
    valid_rl = ~np.isnan(rl_runtime)

    if valid_ga.any():
        plt.bar(x[valid_ga] - width, np.array(ga_runtime)[valid_ga], width, label='Genetic Algorithm (GA)',
                color='#2196F3',
                alpha=0.8)
    if valid_ooa.any():
        plt.bar(x[valid_ooa], np.array(ooa_runtime)[valid_ooa], width, label='Osprey Optimization Algorithm (OOA)',
                color='#4CAF50', alpha=0.8)
    if valid_rl.any():
        plt.bar(x[valid_rl] + width, np.array(rl_runtime)[valid_rl], width, label='Reinforcement Learning (RL)',
                color='#FF9800',
                alpha=0.8)

    plt.xlabel('Dataset')
    plt.ylabel('Average Runtime (seconds)')
    plt.title('Average Runtime Comparison of Algorithms Across Datasets')
    plt.xticks(x, [d[:15] + '...' if len(d) > 15 else d for d in datasets], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'runtime_comparison.png'), dpi=300)
    plt.close()

    # 4. Job completion rate comparison chart
    plt.figure(figsize=(14, 8))

    ga_completion = []
    ooa_completion = []
    rl_completion = []

    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        jobs = dataset_df['num_jobs'].iloc[0]  # Get total number of jobs

        # GA average completion rate
        ga_success = dataset_df['GA_success'] if 'GA_success' in dataset_df.columns else False
        if 'GA_completed_jobs' in dataset_df.columns and ga_success.any():
            ga_completion.append(dataset_df[ga_success]['GA_completed_jobs'].mean() / jobs * 100)
        else:
            ga_completion.append(np.nan)

        # OOA average completion rate
        ooa_success = dataset_df['OOA_success'] if 'OOA_success' in dataset_df.columns else False
        if 'OOA_completed_jobs' in dataset_df.columns and ooa_success.any():
            ooa_completion.append(dataset_df[ooa_success]['OOA_completed_jobs'].mean() / jobs * 100)
        else:
            ooa_completion.append(np.nan)

        # RL average completion rate
        rl_success = dataset_df['RL_success'] if 'RL_success' in dataset_df.columns else False
        if 'RL_completed_jobs' in dataset_df.columns and rl_success.any():
            rl_completion.append(dataset_df[rl_success]['RL_completed_jobs'].mean() / jobs * 100)
        else:
            rl_completion.append(np.nan)

    # Create bar chart
    valid_ga = ~np.isnan(ga_completion)
    valid_ooa = ~np.isnan(ooa_completion)
    valid_rl = ~np.isnan(rl_completion)

    if valid_ga.any():
        plt.bar(x[valid_ga] - width, np.array(ga_completion)[valid_ga], width, label='Genetic Algorithm (GA)',
                color='#2196F3',
                alpha=0.8)
    if valid_ooa.any():
        plt.bar(x[valid_ooa], np.array(ooa_completion)[valid_ooa], width, label='Osprey Optimization Algorithm (OOA)',
                color='#4CAF50',
                alpha=0.8)
    if valid_rl.any():
        plt.bar(x[valid_rl] + width, np.array(rl_completion)[valid_rl], width, label='Reinforcement Learning (RL)',
                color='#FF9800',
                alpha=0.8)

    plt.xlabel('Dataset')
    plt.ylabel('Job Completion Rate (%)')
    plt.title('Job Completion Rate Comparison of Algorithms Across Datasets')
    plt.xticks(x, [d[:15] + '...' if len(d) > 15 else d for d in datasets], rotation=45, ha='right')
    plt.ylim(0, 105)  # Limit y-axis range to 0-105%
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'completion_rate.png'), dpi=300)
    plt.close()

    # 5. Radar chart: Overall performance comparison
    # Prepare data - calculate average performance in each dimension for each algorithm
    ga_perf = {}
    ooa_perf = {}
    rl_perf = {}

    # GA performance
    ga_success = df['GA_success'] if 'GA_success' in df.columns else None
    if ga_success is not None and ga_success.any():
        ga_perf['makespan'] = df[ga_success]['GA_makespan'].mean()
        ga_perf['tool_changes'] = df[ga_success]['GA_tool_changes'].mean()
        ga_perf['runtime'] = df[ga_success]['GA_runtime'].mean()
        ga_perf['completion'] = df[ga_success]['GA_completed_jobs'].mean() / df[ga_success]['num_jobs'].mean() * 100

    # OOA performance
    ooa_success = df['OOA_success'] if 'OOA_success' in df.columns else None
    if ooa_success is not None and ooa_success.any():
        ooa_perf['makespan'] = df[ooa_success]['OOA_makespan'].mean()
        ooa_perf['tool_changes'] = df[ooa_success]['OOA_tool_changes'].mean()
        ooa_perf['runtime'] = df[ooa_success]['OOA_runtime'].mean()
        ooa_perf['completion'] = df[ooa_success]['OOA_completed_jobs'].mean() / df[ooa_success]['num_jobs'].mean() * 100

    # RL performance
    rl_success = df['RL_success'] if 'RL_success' in df.columns else None
    if rl_success is not None and rl_success.any():
        rl_perf['makespan'] = df[rl_success]['RL_makespan'].mean()
        rl_perf['tool_changes'] = df[rl_success]['RL_tool_changes'].mean()
        rl_perf['runtime'] = df[rl_success]['RL_runtime'].mean()
        rl_perf['completion'] = df[rl_success]['RL_completed_jobs'].mean() / df[rl_success]['num_jobs'].mean() * 100

    # Only draw radar chart when at least two algorithms have data
    algos_with_data = sum([bool(ga_perf), bool(ooa_perf), bool(rl_perf)])
    if algos_with_data >= 2:
        # Normalize data
        all_perfs = [perf for perf in [ga_perf, ooa_perf, rl_perf] if perf]

        # Find max values
        max_makespan = max(perf.get('makespan', 0) for perf in all_perfs)
        max_tool_changes = max(perf.get('tool_changes', 0) for perf in all_perfs)
        max_runtime = max(perf.get('runtime', 0) for perf in all_perfs)

        # Normalize function - for makespan, tool_changes, runtime lower is better, so convert to inverse metrics
        for perf in all_perfs:
            if 'makespan' in perf:
                perf['makespan_norm'] = 1 - (perf['makespan'] / max_makespan) if max_makespan > 0 else 0
            if 'tool_changes' in perf:
                perf['tool_changes_norm'] = 1 - (perf['tool_changes'] / max_tool_changes) if max_tool_changes > 0 else 0
            if 'runtime' in perf:
                perf['runtime_norm'] = 1 - (perf['runtime'] / max_runtime) if max_runtime > 0 else 0
            if 'completion' in perf:
                perf['completion_norm'] = perf['completion'] / 100

        # Create radar chart
        plt.figure(figsize=(10, 10))

        # Prepare radar chart data
        categories = ['Makespan\n(inverse)', 'Tool Changes\n(inverse)', 'Runtime\n(inverse)', 'Job Completion Rate']

        # Add closing path
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the polygon

        # Initialize radar chart
        ax = plt.subplot(111, polar=True)

        # Set grid and labels
        plt.xticks(angles[:-1], categories)

        # Draw performance for each algorithm
        for i, (algo, perf) in enumerate(zip(['GA', 'OOA', 'RL'], [ga_perf, ooa_perf, rl_perf])):
            if not perf:
                continue

            values = [
                perf.get('makespan_norm', 0),
                perf.get('tool_changes_norm', 0),
                perf.get('runtime_norm', 0),
                perf.get('completion_norm', 0)
            ]
            values += values[:1]  # Close the polygon

            color = '#2196F3' if algo == 'GA' else '#4CAF50' if algo == 'OOA' else '#FF9800'
            algo_name = 'Genetic Algorithm (GA)' if algo == 'GA' else 'Osprey Optimization Algorithm (OOA)' if algo == 'OOA' else 'Reinforcement Learning (RL)'

            ax.plot(angles, values, linewidth=2, linestyle='solid', label=algo_name, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)

        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        plt.title('Algorithm Performance Comparison Across Multiple Dimensions')
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'radar_comparison.png'), dpi=300)
        plt.close()

    # 6. Scatter plot matrix: Makespan vs Tool Changes, grouped by dataset
    try:
        # Create scatter plot matrix
        # First, prepare valid data
        scatter_data = []
        for dataset in datasets:
            dataset_df = df[df['dataset'] == dataset]

            dataset_data = {'dataset': dataset}

            # GA data
            ga_success = dataset_df['GA_success'] if 'GA_success' in dataset_df.columns else None
            if ga_success is not None and ga_success.any() and 'GA_makespan' in dataset_df.columns and 'GA_tool_changes' in dataset_df.columns:
                dataset_data['GA_makespan'] = dataset_df[ga_success]['GA_makespan'].mean()
                dataset_data['GA_tool_changes'] = dataset_df[ga_success]['GA_tool_changes'].mean()

            # OOA data
            ooa_success = dataset_df['OOA_success'] if 'OOA_success' in dataset_df.columns else None
            if ooa_success is not None and ooa_success.any() and 'OOA_makespan' in dataset_df.columns and 'OOA_tool_changes' in dataset_df.columns:
                dataset_data['OOA_makespan'] = dataset_df[ooa_success]['OOA_makespan'].mean()
                dataset_data['OOA_tool_changes'] = dataset_df[ooa_success]['OOA_tool_changes'].mean()

            # RL data
            rl_success = dataset_df['RL_success'] if 'RL_success' in dataset_df.columns else None
            if rl_success is not None and rl_success.any() and 'RL_makespan' in dataset_df.columns and 'RL_tool_changes' in dataset_df.columns:
                dataset_data['RL_makespan'] = dataset_df[rl_success]['RL_makespan'].mean()
                dataset_data['RL_tool_changes'] = dataset_df[rl_success]['RL_tool_changes'].mean()

            if len(dataset_data) > 1:  # At least one algorithm has data
                scatter_data.append(dataset_data)

        if scatter_data:
            scatter_df = pd.DataFrame(scatter_data)

            # Draw scatter plot matrix
            plt.figure(figsize=(12, 10))

            # GA scatter
            if 'GA_makespan' in scatter_df.columns and 'GA_tool_changes' in scatter_df.columns:
                plt.scatter(scatter_df['GA_makespan'],
                            scatter_df['GA_tool_changes'],
                            alpha=0.7, label='Genetic Algorithm (GA)', marker='o', s=80, color='#2196F3')

                # Add data labels
                for i, txt in enumerate(scatter_df['dataset']):
                    plt.annotate(txt[:10] + '...' if len(txt) > 10 else txt,
                                 (scatter_df['GA_makespan'].iloc[i], scatter_df['GA_tool_changes'].iloc[i]),
                                 xytext=(5, 5), textcoords='offset points', fontsize=8)

            # OOA scatter
            if 'OOA_makespan' in scatter_df.columns and 'OOA_tool_changes' in scatter_df.columns:
                plt.scatter(scatter_df['OOA_makespan'],
                            scatter_df['OOA_tool_changes'],
                            alpha=0.7, label='Osprey Optimization Algorithm (OOA)', marker='s', s=80, color='#4CAF50')

                # Add data labels
                for i, txt in enumerate(scatter_df['dataset']):
                    plt.annotate(txt[:10] + '...' if len(txt) > 10 else txt,
                                 (scatter_df['OOA_makespan'].iloc[i], scatter_df['OOA_tool_changes'].iloc[i]),
                                 xytext=(5, 5), textcoords='offset points', fontsize=8)

            # RL scatter
            if 'RL_makespan' in scatter_df.columns and 'RL_tool_changes' in scatter_df.columns:
                plt.scatter(scatter_df['RL_makespan'],
                            scatter_df['RL_tool_changes'],
                            alpha=0.7, label='Reinforcement Learning (RL)', marker='^', s=80, color='#FF9800')

                # Add data labels
                for i, txt in enumerate(scatter_df['dataset']):
                    plt.annotate(txt[:10] + '...' if len(txt) > 10 else txt,
                                 (scatter_df['RL_makespan'].iloc[i], scatter_df['RL_tool_changes'].iloc[i]),
                                 xytext=(5, 5), textcoords='offset points', fontsize=8)

            plt.xlabel('Average Makespan')
            plt.ylabel('Average Tool Changes')
            plt.title('Algorithm Performance Scatter Plot Across Datasets')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, 'dataset_scatter.png'), dpi=300)
            plt.close()
    except Exception as e:
        print(f"Failed to create scatter plot matrix: {str(e)}")

    print(f"Comparison charts saved to {charts_dir} directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run job shop scheduling benchmark tests for three algorithms')
    parser.add_argument('--data_dir', type=str, default='datasets',
                        help='Dataset directory path')
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                        help='Result output directory')
    parser.add_argument('--num_machines', type=int, default=3,
                        help='Number of machines')
    parser.add_argument('--num_runs', type=int, default=5,
                        help='Number of runs per dataset')
    parser.add_argument('--no_charts', action='store_true',
                        help='Do not generate Gantt charts')
    parser.add_argument('--ga_population', type=int, default=100,
                        help='Genetic algorithm population size')
    parser.add_argument('--ga_generations', type=int, default=30,
                        help='Genetic algorithm generations')
    parser.add_argument('--ooa_population', type=int, default=100,
                        help='Osprey algorithm population size')
    parser.add_argument('--ooa_iterations', type=int, default=30,
                        help='Osprey algorithm iterations')
    parser.add_argument('--makespan_weight', type=float, default=0.7,
                        help='Makespan weight')
    parser.add_argument('--tool_change_weight', type=float, default=0.3,
                        help='Tool change weight')
    parser.add_argument('--rl_model_path', type=str, default="models/job_shop_gnn_dqn_final.pth",
                        help='Reinforcement learning model path')

    args = parser.parse_args()

    # Run benchmark test
    results = benchmark_algorithms(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_machines=args.num_machines,
        num_runs=args.num_runs,
        save_charts=not args.no_charts,
        ga_population=args.ga_population,
        ga_generations=args.ga_generations,
        ooa_population=args.ooa_population,
        ooa_iterations=args.ooa_iterations,
        makespan_weight=args.makespan_weight,
        tool_change_weight=args.tool_change_weight,
        rl_model_path=args.rl_model_path
    )

    print("\nTesting complete! Please check the results directory for charts and data.")