import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import os
from datetime import datetime

# Fix font issue (removed Chinese font settings)
plt.rcParams['font.sans-serif'] = ['Arial']  # English font
plt.rcParams['axes.unicode_minus'] = False  # Properly display minus signs


def save_gantt_chart(env, filename, show_rounds=False, highlight_task=None, color_by='task'):
    """
    Generate and save Gantt chart for job shop scheduling

    Parameters:
    env: JobShopEnv environment object
    filename: Filename to save
    show_rounds: Whether to display round boundaries
    highlight_task: Optional, task ID to highlight
    color_by: Coloring method - 'task', 'type' (operation type), 'round'
    """
    # Create chart
    fig_height = max(8, env.num_machines * 0.8)  # Adjust height based on number of machines
    fig, ax = plt.subplots(figsize=(16, fig_height))

    # Color settings
    if color_by == 'task':
        # Assign different colors for each task
        colors = plt.cm.get_cmap('tab20', env.num_jobs)
    elif color_by == 'type':
        # Assign different colors for each operation type
        colors = plt.cm.get_cmap('tab20', env.num_operation_types)
    else:  # color_by == 'round'
        # Assign different colors for each round
        max_rounds = max([len(rounds) for rounds in env.machine_rounds]) if env.machine_rounds else 1
        colors = plt.cm.get_cmap('tab20', max_rounds)

    # Get maximum completion time across all machines as chart width
    max_time = env.total_time

    # Define Y-axis marks
    y_ticks = np.arange(env.num_machines)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'Machine {i}' for i in range(env.num_machines)])

    # Set X-axis range and label
    ax.set_xlim(0, max_time * 1.02)  # Add some space on the right
    ax.set_xlabel('Time')

    # Set title
    ax.set_title(f'Job Shop Scheduling - Total Time: {env.total_time}, Tool Changes: {env.tool_changes}')

    # Add grid lines
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Add operation blocks for each task
    legend_handles = []  # Legend items
    legend_labels = []  # Legend labels
    added_legends = set()  # Items already added to legend

    # Store round boundaries for later marking
    round_boundaries = {}

    # Iterate through each machine
    for machine_id in range(env.num_machines):
        job_sequence = env.machine_job_sequences[machine_id]
        if not job_sequence:
            continue

        # Sort by start time
        sorted_sequence = sorted(job_sequence, key=lambda x: x["start_time"])

        # Collect round boundary time points for this machine
        if show_rounds:
            round_boundaries[machine_id] = []
            current_round = -1
            for job in sorted_sequence:
                if job["round"] > current_round:
                    current_round = job["round"]
                    # Record start time of new round
                    round_boundaries[machine_id].append(job["start_time"])

        # Draw rectangles for each operation
        for job in sorted_sequence:
            job_id = job["job_id"]
            op_type = job["type"]
            start_time = job["start_time"]
            duration = job["end_time"] - job["start_time"]
            round_num = job["round"]

            # Determine color
            if color_by == 'task':
                color_idx = job_id % 20  # Use mod 20 to avoid index out of range
                color = colors(color_idx)
                legend_key = f'Task {job_id}'
            elif color_by == 'type':
                color_idx = op_type % 20  # Use mod 20 to avoid index out of range
                color = colors(color_idx)
                legend_key = f'Type {op_type}'
            else:  # color_by == 'round'
                color_idx = round_num % 20  # Use mod 20 to avoid index out of range
                color = colors(color_idx)
                legend_key = f'Round {round_num + 1}'

            # Adjust transparency if there's a highlighted task
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

            # Draw task operation block
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

            # Add text labels in the operation blocks
            if duration > 5:  # Only add labels to wide enough blocks
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

            # Add legend item (avoid duplicates)
            if legend_key not in added_legends:
                legend_handle = patches.Patch(color=color, label=legend_key)
                legend_handles.append(legend_handle)
                legend_labels.append(legend_key)
                added_legends.add(legend_key)

    # Add round boundary lines
    if show_rounds:
        for machine_id, boundaries in round_boundaries.items():
            for time_point in boundaries:
                ax.axvline(x=time_point, ymin=(machine_id - 0.4) / env.num_machines,
                           ymax=(machine_id + 0.4) / env.num_machines,
                           color='red', linestyle='--', alpha=0.7)

    # Add legend (limit number to prevent overcrowding)
    if len(legend_handles) > 20:
        # If too many legend items, only show first 20
        ax.legend(legend_handles[:20], legend_labels[:20],
                  title='Legend (showing first 20)', loc='upper right')
    else:
        ax.legend(legend_handles, legend_labels,
                  title='Legend', loc='upper right')

    # Save chart
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Gantt chart saved to: {filename}")


def create_interactive_gantt(env, filename):
    """
    Create interactive HTML Gantt chart

    Parameters:
    env: JobShopEnv environment object
    filename: HTML filename to save
    """
    try:
        import plotly.figure_factory as ff
        import plotly.io as pio
        import plotly.graph_objects as go
        import pandas as pd
    except ImportError:
        print(
            "plotly and pandas libraries required for interactive Gantt charts. Please run: pip install plotly pandas")
        return

    # Prepare Gantt chart data
    gantt_data = []

    # Iterate through each operation on each machine
    for machine_id in range(env.num_machines):
        job_sequence = env.machine_job_sequences[machine_id]

        for job in job_sequence:
            job_id = job["job_id"]
            op_idx = job["operation"] - 1
            op_type = job["type"]
            start_time = job["start_time"]
            end_time = job["end_time"]
            round_num = job["round"]

            # Add to Gantt chart data
            gantt_data.append(dict(
                Task=f'Machine {machine_id}',
                Start=start_time,
                Finish=end_time,
                # Use unified identifier to avoid color mapping issues
                Resource="Task",
                Description=f'J{job_id}O{op_idx} (Type:{op_type}, Round:{round_num + 1})'
            ))

    # Convert to DataFrame
    df = pd.DataFrame(gantt_data)

    # If no data, add placeholder data
    if len(df) == 0:
        df = pd.DataFrame([dict(Task='Machine 0', Start=0, Finish=1, Resource='No Task', Description='No Data')])

    # Try to create Gantt chart using custom method
    try:
        # Create an empty figure
        fig = go.Figure()

        # Set colors
        colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)',
                  'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)']

        # Add bar charts grouped by machine
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

        # Update layout
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

        # Save as HTML file
        pio.write_html(fig, filename)
        print(f"Interactive Gantt chart saved to: {filename}")

    except Exception as e:
        print(f"Error creating interactive Gantt chart: {e}")
        print("Using static image instead...")
        # If interactive Gantt chart creation fails, save a static version as backup
        static_filename = filename.replace('.html', '_static.png')
        save_gantt_chart(env, static_filename)
        print(f"Static backup Gantt chart saved to: {static_filename}")


def generate_all_charts(env, output_dir='charts'):
    """Generate multiple types of Gantt charts and save them"""
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate static Gantt chart - colored by task
    save_gantt_chart(
        env,
        os.path.join(output_dir, f'gantt_by_task_{timestamp}.png'),
        color_by='task'
    )

    # Generate static Gantt chart - colored by operation type
    save_gantt_chart(
        env,
        os.path.join(output_dir, f'gantt_by_type_{timestamp}.png'),
        color_by='type'
    )

    # Generate static Gantt chart - colored by round
    save_gantt_chart(
        env,
        os.path.join(output_dir, f'gantt_by_round_{timestamp}.png'),
        color_by='round',
        show_rounds=True
    )

    # Try to generate interactive Gantt chart, with error handling
    try:
        create_interactive_gantt(
            env,
            os.path.join(output_dir, f'interactive_gantt_{timestamp}.html')
        )
    except Exception as e:
        print(f"Error generating interactive Gantt chart: {e}")
        print("Only using static Gantt charts.")

    print(f"All Gantt charts have been generated and saved in the {output_dir} directory")